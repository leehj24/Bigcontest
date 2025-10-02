
from pathlib import Path
import numpy as np
import torch

from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 더미 빌드
obs_dim, action_size = 21, 4
ds = MDPDataset(
    observations=np.zeros((1, obs_dim), dtype=np.float32),
    actions=np.zeros((1,), dtype=np.int64),
    rewards=np.zeros((1,), dtype=np.float32),
    terminals=np.ones((1,), dtype=np.int64),
)

# v2 방식: Config -> create(...)
cfg = DiscreteCQLConfig()
algo = cfg.create(device=device)
algo.build_with_dataset(ds)

# 쓰기
out = Path("Model/d3rlpy_logs/DiscreteCQL/test_write_read.d3")
out.parent.mkdir(parents=True, exist_ok=True)
algo.save_model(str(out))

# 읽기
cfg2 = DiscreteCQLConfig()
algo2 = cfg2.create(device=device)
algo2.build_with_dataset(ds)
algo2.load_model(str(out))

print("OK: write & read")
