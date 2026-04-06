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
        train_df = data_loader.train_df
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        X = torch.zeros(self.n_users, self.n_items, dtype=dtype, device=self.device)
        X[rows, cols] = 1.0
        return X

    def forward(self, user_indices):
        raise NotImplementedError

    def calc_loss(self, batch_data):
        raise NotImplementedError
