import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from src.utils.stats import get_dataset_stats

def main():
    base_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} not found.")
        return

    dataset_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not dataset_names:
        print("No datasets found in preprocessed directory.")
        return

    all_stats = []
    for d in sorted(dataset_names):
        print(f"Analyzing {d}...", end="\r", flush=True)
        stats = get_dataset_stats(d, base_path)
        if stats:
            all_stats.append(stats)
    
    print("\n" + "="*120)
    df = pd.DataFrame(all_stats)
    print(df.to_string(index=False))
    print("="*120)

if __name__ == "__main__":
    main()
