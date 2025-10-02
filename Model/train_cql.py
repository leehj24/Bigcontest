# Model/train_cql.py
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional
import random
import numpy as np
import pandas as pd
import torch
import warnings
import json
import shutil

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig

# 불필요한 Gym 경고 억제(환경 사용 안 함)
warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022")

# -------------------- 경로/HP --------------------
ROOT = Path(__file__).resolve().parents[1]          # .../Bigcontest
MODEL_DIR = ROOT / "Model"
LOGS_ROOT = MODEL_DIR / "d3rlpy_logs"
FIXED_DIR = LOGS_ROOT / "DiscreteCQL"               # 고정 저장 경로(앱이 읽음)
TIMESTAMP_DIR = LOGS_ROOT / f"DiscreteCQL_{time.strftime('%Y%m%d%H%M%S')}"
DATASET_CSV = MODEL_DIR / "rl_dataset.csv"

SEED = int(os.environ.get("SEED", 2025))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 20000))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 256))
SAVE_EVERY  = int(os.environ.get("SAVE_EVERY", 2000))
GAMMA       = float(os.environ.get("GAMMA", 0.99))
OBS_DIM_FALLBACK = int(os.environ.get("OBS_DIM", 21))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------- 유틸 --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTAMP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- 데이터셋 로딩 --------------------
def try_import_build_dataset() -> Tuple[Optional[MDPDataset], Optional[int]]:
    """
    Model/build_rl_dataset.py에서 (MDPDataset, action_size)를 반환하는
    build_mdp_dataset() 또는 build_dataset()을 찾는다.
    """
    mod_path = MODEL_DIR / "build_rl_dataset.py"
    if not mod_path.exists():
        return None, None
    sys.path.insert(0, str(MODEL_DIR))
    try:
        import build_rl_dataset as brd  # type: ignore
    except Exception:
        return None, None

    for fn in ("build_mdp_dataset", "build_dataset"):
        if hasattr(brd, fn):
            try:
                ds, action_size = getattr(brd, fn)()
                if isinstance(ds, MDPDataset):
                    return ds, int(action_size)
            except Exception:
                continue
    return None, None

def csv_to_mdpdataset(csv_path: Path) -> Tuple[MDPDataset, int]:
    """
    유연한 CSV 폴백 로더:
      - 라벨 컬럼 자동 감지:
          action:  ['action','act','action_id','a']
          reward:  ['reward','rew','r']
          terminal:['terminal','done','is_done','terminated','truncated','done_flag']
      - 관측치(Obs) 자동 감지:
          1) 'latent_*' + 도메인 피처 우선
          2) 없으면 숫자형 전체에서 라벨 제외
    """
    df = pd.read_csv(csv_path)
    cols = [c.strip() for c in df.columns]

    cand_action   = ['action', 'act', 'action_id', 'a']
    cand_reward   = ['reward', 'rew', 'r']
    cand_terminal = ['terminal', 'done', 'is_done', 'terminated', 'truncated', 'done_flag']

    def pick(colnames, candidates):
        for k in candidates:
            if k in colnames:
                return k
        return None

    action_col   = pick(cols, cand_action)
    reward_col   = pick(cols, cand_reward)
    terminal_col = pick(cols, cand_terminal)

    if action_col is None or reward_col is None or terminal_col is None:
        missing = []
        if action_col is None:   missing.append("action")
        if reward_col is None:   missing.append("reward")
        if terminal_col is None: missing.append("terminal/done")
        raise ValueError(f"CSV 라벨 컬럼을 찾을 수 없습니다. 누락: {', '.join(missing)}")

    latent_cols = [c for c in cols if c.startswith("latent_")]
    domain_feats = [
        'DLV_SAA_RAT',
        'MCT_UE_CLN_REU_RAT',
        'MCT_UE_CLN_NEW_RAT',
        'RC_M1_SAA_RANK',
        'TREND_RATIO',
    ]
    domain_cols = [c for c in domain_feats if c in cols]

    obs_cols: list[str] = []
    if latent_cols:
        obs_cols.extend(sorted(latent_cols, key=lambda x: int(x.split("_")[1])))
    obs_cols.extend(domain_cols)

    if not obs_cols:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        obs_cols = [c for c in numeric_cols if c not in {action_col, reward_col, terminal_col}]

    if len(obs_cols) == 0:
        raise ValueError("관측치(Obs)로 사용할 수 있는 숫자형 컬럼을 찾지 못했습니다.")

    obs = df[obs_cols].astype(np.float32).values
    act = df[action_col].astype(np.int64).values
    rew = df[reward_col].astype(np.float32).values
    ter = df[terminal_col].astype(np.int64).values
    if ter.dtype == bool:
        ter = ter.astype(np.int64)

    action_size = int(np.max(act)) + 1

    print(f"[INFO] CSV 로드: obs_dim={obs.shape[1]}, "
          f"action_col={action_col}, reward_col={reward_col}, terminal_col={terminal_col}")
    print(f"[INFO] 사용된 Obs 컬럼 수={len(obs_cols)} 예시={obs_cols[:6]}...")

    ds = MDPDataset(observations=obs, actions=act, rewards=rew, terminals=ter)
    return ds, action_size

def load_dataset() -> Tuple[MDPDataset, int]:
    ds, action_size = try_import_build_dataset()
    if ds is not None:
        print(f"[INFO] build_rl_dataset.py 사용. action_size={action_size}")
        return ds, action_size
    if DATASET_CSV.exists():
        ds, action_size = csv_to_mdpdataset(DATASET_CSV)
        print(f"[INFO] CSV 폴백 로드 완료. action_size={action_size}")
        return ds, action_size
    raise FileNotFoundError("데이터셋을 만들 수 없습니다. build_rl_dataset.py 또는 rl_dataset.csv 필요")

# -------------------- 저장 루틴 --------------------
def save_checkpoint(algo, step: int, *, both: bool = True):
    """
    - TIMESTAMP_DIR: 실험 로그 보관
    - FIXED_DIR:     앱이 읽는 '고정 폴더' (최근 체크포인트 동기화)
    - d3rlpy v2: save_params 없음 → config를 직접 json으로 기록
    """
    # 1) 모델 가중치 저장
    ts_path = TIMESTAMP_DIR / f"model_{step}.d3"
    algo.save_model(str(ts_path))

    # 2) params.json 생성 (obs/action + config)
    params_payload = {
        "observation_shape": list(getattr(algo, "observation_shape", []) or []),
        "action_size": int(getattr(algo, "action_size", 0) or 0),
        "config": None,
    }
    try:
        # v2: algo.config가 존재하고, to_dict()를 제공
        if hasattr(algo, "config") and hasattr(algo.config, "to_dict"):
            params_payload["config"] = algo.config.to_dict()
        elif hasattr(algo, "config") and hasattr(algo.config, "to_json"):
            # 드물게 to_json만 있는 경우 보완
            tmp = TIMESTAMP_DIR / "_config_tmp.json"
            algo.config.to_json(tmp)
            params_payload["config"] = json.loads(tmp.read_text(encoding="utf-8"))
            tmp.unlink(missing_ok=True)
    except Exception:
        # config 직렬화 실패 시 최소 정보만 남겨도 로더는 동작
        pass

    # 타임스탬프 폴더에 params.json 기록
    (TIMESTAMP_DIR / "params.json").write_text(
        json.dumps(params_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if both:
        # 3) 고정 폴더 동기화
        FIXED_DIR.mkdir(parents=True, exist_ok=True)
        fx_path = FIXED_DIR / f"model_{step}.d3"
        # 같은 세션에서 다시 save_model 호출해도 되지만, 파일 복사로 빠르게 동기화
        shutil.copyfile(ts_path, fx_path)
        (FIXED_DIR / "params.json").write_text(
            json.dumps(params_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[SAVE] step={step} → {ts_path.name} (고정 폴더 동기화 완료)")

# -------------------- 학습 --------------------
def train():
    set_seed(SEED)
    ensure_dirs()

    # 1) 데이터셋
    dataset, action_size = load_dataset()

    # 2) Config → create → build_with_dataset
    cfg = DiscreteCQLConfig(gamma=GAMMA, batch_size=BATCH_SIZE)  # ✅ 배치 크기는 Config로
    algo = cfg.create(device=DEVICE)
    algo.build_with_dataset(dataset)

    # obs_dim은 build 이후 모델에서 읽는다.
    obs_shape = getattr(algo, "observation_shape", None)
    obs_dim = int(obs_shape[0]) if (obs_shape and len(obs_shape) > 0) else None
    print(f"[INFO] built with dataset: obs_dim={obs_dim}, action_size≈{action_size}, device={DEVICE}, batch_size={BATCH_SIZE}")

    # 3) d3rlpy fit (오프라인 RL) — v2: fit 인자 최소화
    n_epochs = max(1, TOTAL_STEPS // max(1, SAVE_EVERY))
    steps_per_epoch = SAVE_EVERY
    remain_steps = TOTAL_STEPS - n_epochs * steps_per_epoch
    if remain_steps > 0:
        n_epochs += 1  # 마지막 epoch에 잔여 스텝 포함

    cur_step = 0
    for ep in range(n_epochs):
        budget = min(steps_per_epoch, TOTAL_STEPS - cur_step)
        if budget <= 0:
            break

        # ✅ v2: fit에 dataset, n_steps, n_steps_per_epoch만 전달
        algo.fit(
            dataset=dataset,
            n_steps=budget,
            n_steps_per_epoch=budget,
        )
        cur_step += budget
        save_checkpoint(algo, cur_step, both=True)

    print(f"[DONE] total_steps={cur_step}")

if __name__ == "__main__":
    r"""
    실행:
      py -3.13 .\Model\train_cql.py

    환경변수(선택):
      TOTAL_STEPS, BATCH_SIZE, SAVE_EVERY, GAMMA, SEED, OBS_DIM
    """
    train()
