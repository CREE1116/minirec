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
            
        self.output_path = config.get('output_path_override', 'output/default')

    def forward(self, user_indices):
        raise NotImplementedError

    def calc_loss(self, batch_data):
        raise NotImplementedError
