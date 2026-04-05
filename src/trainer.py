import torch
import torch.optim as optim
import os
import json
import copy
from tqdm import tqdm
from .eval.metrics import evaluate_metrics

class Trainer:
    def __init__(self, config, model, data_loader):
        self.config, self.model, self.data_loader = config, model, data_loader
        self.device = model.device
        self.model.to(self.device)
        self.output_path = config.get('output_path_override', 'output/default')
        os.makedirs(self.output_path, exist_ok=True)
        
        # 닫힌 해(fit) 모델인지 SGD(calc_loss) 모델인지 확인
        self.has_fit = hasattr(model, 'fit') and callable(getattr(model, 'fit'))
        self.is_sgd = 'train' in config and hasattr(model, 'calc_loss')

    def run(self):
        """실행 흐름 통합 (fit -> sgd train -> final evaluate)"""
        # Path 1: Closed-form (EASE, iALS, etc.)
        if self.has_fit:
            self.model.fit(self.data_loader)
        
        # Path 2: SGD Training (MF, LightGCN, MultiVAE, etc.)
        if self.is_sgd:
            self._train_loop()
        
        # Final Path: Always Evaluate
        return self.evaluate(is_final=True)

    def _train_loop(self):
        train_cfg = self.config['train']
        lr = float(train_cfg.get('lr', 0.001))
        wd = float(train_cfg.get('weight_decay', 0))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = train_cfg.get('epochs', 10)
        batch_size = train_cfg.get('batch_size', 1024)
        loader = self.data_loader.get_train_loader(batch_size)
        
        best_metric, best_state = -1, None
        m_name = f"{self.config.get('evaluation', {}).get('main_metric', 'NDCG')}@{self.config.get('evaluation', {}).get('main_metric_k', 10)}"

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss_tuple, _ = self.model.calc_loss(batch)
                loss = sum(loss_tuple)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Epoch-wise Validation
            val_metrics = self.evaluate(is_final=False)
            val_val = val_metrics.get(m_name, 0)
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, Val {m_name}: {val_val:.4f}")
            
            if val_val > best_metric:
                best_metric = val_val
                best_state = copy.deepcopy(self.model.state_dict())
        
        if best_state:
            self.model.load_state_dict(best_state)
            print(f"Loaded Best Model (Val {m_name}: {best_metric:.4f})")

    def evaluate(self, loader=None, is_final=False):
        self.model.eval()
        if loader is None:
            batch_size = 2048
            loader = self.data_loader.get_final_loader(batch_size) if is_final else self.data_loader.get_validation_loader(batch_size)
        
        eval_cfg = self.config.get('evaluation', {}).copy()
        if not is_final: # Validation mode: only compute main metric
            eval_cfg['metrics'] = [eval_cfg.get('main_metric', 'NDCG')]
            eval_cfg['top_k'] = [eval_cfg.get('main_metric_k', 10)]
            
        metrics = evaluate_metrics(self.model, self.data_loader, eval_cfg, self.device, loader, is_final=is_final)
        
        if is_final:
            with open(os.path.join(self.output_path, 'metrics.json'), 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
        return metrics
