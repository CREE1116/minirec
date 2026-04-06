import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
            
        self.output_path = config.get('output_path_override', config.get('output_path', 'output'))

    def get_train_matrix(self, data_loader, dtype=torch.float32):
        """훈련 데이터를 sparse tensor로 반환 (device 이동 포함)"""
        train_df = data_loader.train_df
        # NumPy array copy to avoid "not writable" warning
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        values = torch.ones(len(rows), dtype=dtype)
        
        # sparse_coo_tensor 생성 후 지정된 device로 이동
        mat = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), 
            values, 
            (self.n_users, self.n_items)
        ).to(self.device)
        
        return mat.coalesce()

    def forward(self, user_indices):
        raise NotImplementedError

    def calc_loss(self, batch_data):
        raise NotImplementedError
