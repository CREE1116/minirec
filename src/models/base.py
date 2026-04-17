import torch
import torch.nn as nn
import numpy as np

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
        self.train_matrix_cpu = None # Shared Scipy CSR for children

    def _to_torch_sparse(self, scipy_matrix):
        """Convert Scipy sparse matrix to Torch sparse tensor efficiently"""
        scipy_matrix = scipy_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((scipy_matrix.row, scipy_matrix.col)).astype(np.int64))
        values = torch.from_numpy(scipy_matrix.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(scipy_matrix.shape))

    def _get_batch_ratings(self, user_indices, weight_matrix):
        """
        Memory Efficient Inference: Slice sparse on CPU, Multiply on GPU.
        - user_indices: torch.Tensor on self.device
        - weight_matrix: torch.Tensor on self.device (Item x Item)
        """
        if self.train_matrix_cpu is None:
            raise ValueError("train_matrix_cpu must be set before inference.")
        
        user_ids_np = user_indices.cpu().numpy()
        # 1. Slice on CPU (Fast)
        batch_sp = self.train_matrix_cpu[user_ids_np]
        # 2. Move to GPU as Sparse
        batch_torch = self._to_torch_sparse(batch_sp).to(self.device)
        # 3. Sparse-Dense Multiplication on GPU
        return torch.sparse.mm(batch_torch, weight_matrix)

    def get_train_matrix(self, data_loader, dtype=torch.float32):
        """Standard sparse matrix fetch (Default to GPU if possible)"""
        train_df = data_loader.train_df
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        values = torch.ones(len(rows), dtype=dtype)
        
        with torch.sparse.check_sparse_tensor_invariants(False):
            mat = torch.sparse_coo_tensor(
                torch.stack([rows, cols]),
                values,
                (self.n_users, self.n_items)
            ).coalesce()
        
        try:
            mat = mat.to(self.device)
        except Exception:
            print(f"  [Warning] Failed to move sparse matrix to {self.device}. Keeping on CPU.")
        return mat

    def forward(self, user_indices):
        raise NotImplementedError

    def calc_loss(self, batch_data):
        raise NotImplementedError
