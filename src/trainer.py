import torch
import torch.optim as optim
import os
import json
import copy
import yaml
import gc
from tqdm import tqdm
from .eval.metrics import evaluate_metrics

class Trainer:
    def __init__(self, config, model, data_loader):
        self.config, self.model, self.data_loader = config, model, data_loader
        self.device = model.device
        self.model.to(self.device)
        
        # 기본 출력 경로 설정 (우선순위: override > config > default 'output')
        self.output_path = config.get('output_path_override', config.get('output_path', 'output'))
        self.hpo_mode = config.get('hpo_mode', False)
        
        if not self.hpo_mode:
            os.makedirs(self.output_path, exist_ok=True)
            # [재현성] 최종 설정 저장 (config.yaml)
            with open(os.path.join(self.output_path, 'config.yaml'), 'w') as f:
                yaml.dump(config, f, indent=4)
        
        # 닫힌 해(fit) 모델인지 SGD(calc_loss) 모델인지 확인
        self.has_fit = hasattr(model, 'fit') and callable(getattr(model, 'fit'))

        # train 설정 위치 유연하게 처리 (root 또는 model.train)
        self.train_cfg = config.get('train', config.get('model', {}).get('train', {}))
        self.is_sgd = bool(self.train_cfg) and hasattr(model, 'calc_loss')

        self._best_val_metrics = None  # HPO 모드에서 중복 inference 방지용

    def run(self):
        """실행 흐름 통합 (fit -> sgd train -> final evaluate)"""
        # Path 1: Closed-form (EASE, iALS, etc.)
        if self.has_fit:
            self.model.fit(self.data_loader)

        # Path 2: SGD Training (MF, LightGCN, MultiVAE, etc.)
        if self.is_sgd:
            self._train_loop()

        # HPO 모드: validation 결과 반환 (또는 설정에 따라 test 결과 반환)
        res = None
        if self.hpo_mode:
            if self._best_val_metrics is not None:
                res = self._best_val_metrics
            else:
                # use_test_for_hpo가 True이면 test set 결과를 반환
                if self.config.get('use_test_for_hpo', False):
                    res = self.evaluate(is_final=True)
                else:
                    res = self.evaluate(is_final=False, all_metrics=True)
        else:
            # 일반 모드: test set 최종 평가
            res = self.evaluate(is_final=True)
            
        # [Cleanup] HPO 모드에서는 다음 trial을 위해 메모리 강제 해제
        if self.hpo_mode:
            # Weight matrix 등 대형 텐서 참조 해제
            if hasattr(self.model, 'weight_matrix'):
                self.model.weight_matrix = None
            if hasattr(self.model, 'train_matrix_cpu'):
                self.model.train_matrix_cpu = None
            
            gc.collect()
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
                
        return res

    def _train_loop(self):
        train_cfg = self.train_cfg
        lr = float(train_cfg.get('lr', 0.001))
        wd = float(train_cfg.get('weight_decay', 0))
        
        # Optimizer 설정
        opt_name = train_cfg.get('optimizer', 'Adam').lower()
        if opt_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=train_cfg.get('momentum', 0.9))
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = train_cfg.get('epochs', 100)
        batch_size = train_cfg.get('batch_size', 1024)
        loader = self.data_loader.get_train_loader(batch_size)
        
        # Early Stopping
        patience = train_cfg.get('patience', 10)
        patience_counter = 0
        best_metric, best_state = -1, None
        
        eval_cfg = self.config.get('evaluation', {})
        m_name = eval_cfg.get('main_metric', 'NDCG')
        m_k = eval_cfg.get('main_metric_k', 20)
        full_m_name = f"{m_name}@{m_k}"

        use_test = self.config.get('use_test_for_hpo', False)
        print(f"Starting SGD Training (Max Epochs: {epochs}, Patience: {patience}, UseTest: {use_test})...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss_tuple, _ = self.model.calc_loss(batch)
                loss = sum(loss_tuple)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Epoch-wise Validation (Fast evaluation: only main metric)
            val_metrics = self.evaluate(is_final=use_test, all_metrics=False)
            val_val = val_metrics.get(full_m_name, 0)
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, {'Test' if use_test else 'Val'} {full_m_name}: {val_val:.4f}")
            
            if val_val > best_metric:
                best_metric = val_val
                best_state = copy.deepcopy(self.model.state_dict())
                
                # Best epoch일 때 전체 메트릭 계산 및 저장
                if not use_test:
                    print("  [Best Model] Calculating full validation metrics...")
                    self._best_val_metrics = self.evaluate(is_final=False, all_metrics=True)
                    if not self.hpo_mode:
                        with open(os.path.join(self.output_path, 'val_metrics.json'), 'w') as f:
                            json.dump({k: float(v) for k, v in self._best_val_metrics.items()}, f, indent=4)
                else:
                    self._best_val_metrics = val_metrics

                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (Best {full_m_name}: {best_metric:.4f})")
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)

    def evaluate(self, loader=None, is_final=False, all_metrics=True):
        self.model.eval()
        eval_cfg = self.config.get('evaluation', {}).copy()
        
        if loader is None:
            batch_size = eval_cfg.get('batch_size', 4096)
            loader = self.data_loader.get_final_loader(batch_size) if is_final else self.data_loader.get_validation_loader(batch_size)
        
        # 훈련 도중 성능 모니터링 시에만 메인 메트릭 하나로 제한 (속도 최적화)
        if not is_final and not all_metrics:
            m_name = eval_cfg.get('main_metric', 'NDCG')
            m_k = eval_cfg.get('main_metric_k', 100) # evaluation.yaml의 NDCG@100에 맞춤
            eval_cfg['metrics'] = [m_name]
            eval_cfg['top_k'] = [m_k]
            
        metrics = evaluate_metrics(self.model, self.data_loader, eval_cfg, self.device, loader, is_final=is_final)
        
        if is_final and not self.hpo_mode:
            with open(os.path.join(self.output_path, 'metrics.json'), 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
        return metrics
