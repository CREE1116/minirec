import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
import __init__ as minirec

def run_alpha_sweep():
    datasets = [
        'configs/datasets/ml-100k.yaml',
        'configs/datasets/ml-1m.yaml',
        'configs/datasets/steam.yaml'
    ]
    
    alphas = [0.0, 0.1, 0.3, 0.5, 1.0]
    model_cfg_path = 'configs/models/aspire_pure.yaml'
    
    results = []
    
    print(f"\n{'='*80}\n### Starting Alpha Sweep (AspirePure) \n{'='*80}")
    
    for d_path in datasets:
        d_name = os.path.basename(d_path).replace('.yaml', '')
        print(f"\n[Dataset: {d_name}]")
        
        for alpha in alphas:
            print(f"  > Testing alpha={alpha}...", end=" ", flush=True)
            try:
                # Override alpha in HPO-like manner or via config merge
                # We use minirec.run but with custom config override logic
                # Since minirec.run doesn't support direct model_cfg dict yet easily without hporun, 
                # we'll use a temporary yaml or just call the core trainer
                
                from src.utils.config import merge_all_configs, load_yaml
                from src.trainer import Trainer
                from src.data.loader import DataLoader
                from src.models import get_model
                from src.utils.seed import set_seed
                
                set_seed(42)
                d_cfg = load_yaml(d_path)
                m_cfg = load_yaml(model_cfg_path)
                m_cfg['model']['alpha'] = alpha
                
                config = merge_all_configs(d_cfg, m_cfg)
                config['hpo_mode'] = True # To suppress extra file saving
                
                dl = DataLoader(config)
                model = get_model(config, dl)
                trainer = Trainer(config, model, dl)
                metrics = trainer.run()
                
                entry = {
                    'dataset': d_name,
                    'alpha': alpha,
                }
                entry.update(metrics)
                results.append(entry)
                print("Done.")
                
            except Exception as e:
                print(f"Failed. Error: {e}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs('output/analysis', exist_ok=True)
        csv_path = 'output/analysis/alpha_sweep_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✨ Alpha sweep results saved to '{csv_path}'")
        
        # Summary display
        print(f"\n{'='*30} ALPHA SWEEP SUMMARY {'='*30}")
        main_metric = 'NDCG@20'
        if main_metric in df.columns:
            pivot = df.pivot(index='alpha', columns='dataset', values=main_metric)
            print(pivot)

if __name__ == "__main__":
    run_alpha_sweep()
