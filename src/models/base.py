import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        device_str = config.get('device', 'cpu')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
            
        self.output_path = config.get('output_path_override', config.get('output_path', 'output'))

    def get_train_matrix(self, data_loader, dtype=torch.float32):
        """훈련 데이터를 sparse tensor로 반환 (device 이동 포함)"""
        train_df = data_loader.train_df
        rows = torch.from_numpy(train_df['user_id'].values).long()
        cols = torch.from_numpy(train_df['item_id'].values).long()
        values = torch.ones(len(rows), dtype=dtype)
        
        # sparse_coo_tensor 생성 후 지정된 device로 이동
        mat = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), 
            values, 
            (self.n_users, self.n_items)
        ).to(self.device)
        
        # 내부적으로 사용할 train_matrix (CSR format이 인덱싱에 유리할 수 있음)
        # 하지만 PyTorch의 sparse support는 아직 COO/CSR 혼용이 복잡할 수 있으므로 
        # 일단 COO로 유지하거나 필요시 변환
        return mat.coalesce()

    def forward(self, user_indices):
        raise NotImplementedError

    def calc_loss(self, batch_data):
        raise NotImplementedError
