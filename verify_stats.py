from src.data.loader import DataLoader
from src.utils.config import load_yaml

config = load_yaml('configs/datasets/gowalla.yaml')
config['data_cache_path'] = './data_cache/'

try:
    dl = DataLoader(config)
except Exception as e:
    print(f"Error: {e}")
