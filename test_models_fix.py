import os
import sys
sys.path.append(os.getcwd())
import __init__ as minirec

# 1. 대상 데이터셋 (테스트용으로 ml-100k 사용)
dataset_cfg = 'configs/datasets/ml-100k.yaml'

# 2. 검증 대상 모델
test_models = [
    {'name': 'mf',       'cfg': 'configs/models/mf.yaml'},
    {'name': 'lightgcn',  'cfg': 'configs/models/lightgcn.yaml'},
]

print(f"\n{'='*80}\n### Verifying MF & LightGCN Fixes \n{'='*80}")

for m_info in test_models:
    m_name = m_info['name']
    m_cfg = m_info['cfg']
    
    print(f"\n>>> Testing Model: {m_name}")
    try:
        # 단일 실행 확인 (에포크 수를 줄여서 빠르게 확인)
        metrics = minirec.run(
            dataset_cfg=dataset_cfg,
            model_cfg=m_cfg,
            output_path=f"output/verify_fix/{m_name}"
        )
        print(f"\n[Success] {m_name} results: {metrics}")
    except Exception as e:
        print(f"\n[Failed] {m_name} error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}\n Verification Finished!\n{'='*80}")
