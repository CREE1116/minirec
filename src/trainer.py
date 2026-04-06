import torch
import torch.optim as optim
import os
import json
import copy
import yaml
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

    def run(self):
        """실행 흐름 통합 (fit -> sgd train -> final evaluate)"""
        # Path 1: Closed-form (EASE, iALS, etc.)
        if self.has_fit:
            self.model.fit(self.data_loader)

        # Path 2: SGD Training (MF, LightGCN, MultiVAE, etc.)
        if self.is_sgd:
            self._train_loop()

        # HPO 모드: validation 결과 반환 (test set 노출 금지)
        if self.hpo_mode:
            return self.evaluate(is_final=False)

        # 일반 모드: test set 최종 평가
        return self.evaluate(is_final=True)

    def _train_loop(self):
        train_cfg = self.train_cfg
        lr = float(train_cfg.get('lr', 0.001))
        wd = float(train_cfg.get('weight_decay', 0))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = train_cfg.get('epochs', 100) # 기본 에폭 상향
        batch_size = train_cfg.get('batch_size', 1024)
        loader = self.data_loader.get_train_loader(batch_size)
        
        # Early Stopping 관련 설정
        patience = train_cfg.get('patience', 10)
        patience_counter = 0
        best_metric, best_state = -1, None
        
        eval_cfg = self.config.get('evaluation', {})
        m_name = eval_cfg.get('main_metric', 'NDCG')
        m_k = eval_cfg.get('main_metric_k', 20)
        full_m_name = f"{m_name}@{m_k}"

        print(f"Starting SGD Training (Max Epochs: {epochs}, Patience: {patience})...")

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
            
            # Epoch-wise Validation
            val_metrics = self.evaluate(is_final=False)
            val_val = val_metrics.get(full_m_name, 0)
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, Val {full_m_name}: {val_val:.4f}")
            
            if val_val > best_metric:
                best_metric = val_val
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0 # 카운터 초기화
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (Best Val {full_m_name}: {best_metric:.4f})")
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
            print(f"Loaded Best Model (Val {full_m_name}: {best_metric:.4f})")

    def evaluate(self, loader=None, is_final=False):
        self.model.eval()
        if loader is None:
            batch_size = 2048
            loader = self.data_loader.get_final_loader(batch_size) if is_final else self.data_loader.get_validation_loader(batch_size)
        
        eval_cfg = self.config.get('evaluation', {}).copy()
        
        # [효율화] Validation 모드(is_final=False): 메인 메트릭@K 하나만 계산하여 에폭 당 소요 시간 단축
        if not is_final:
            m_name = eval_cfg.get('main_metric', 'NDCG')
            m_k = eval_cfg.get('main_metric_k', 20)
            eval_cfg['metrics'] = [m_name]
            eval_cfg['top_k'] = [m_k]
        else:
            # [자동화] Final 모드: evaluation.yaml에 정의된 모든 메트릭/Top-K 계산 (이미 eval_cfg에 리스트로 로드됨)
            print(f"Final Evaluation: Calculating all metrics {eval_cfg.get('metrics')} for Top-K {eval_cfg.get('top_k')}...")
            
        metrics = evaluate_metrics(self.model, self.data_loader, eval_cfg, self.device, loader, is_final=is_final)
        
        if is_final and not self.hpo_mode:
            with open(os.path.join(self.output_path, 'metrics.json'), 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
        return metrics
