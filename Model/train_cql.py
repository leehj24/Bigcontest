# Model/train_cql.py  (d3rlpy 최신 호환: factory 객체 사용 + build 명시)
import os
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from sklearn.preprocessing import StandardScaler
import joblib
import time

ROOT = Path(__file__).resolve().parents[1]   # Bigcontest
MODEL_DIR = ROOT / "Model"
LOGS_ROOT = MODEL_DIR / "d3rlpy_logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

def echo(msg): print(f"[train_cql] {msg}")

# 0) 입력 파일 확인
rl_csv = MODEL_DIR / "rl_dataset.csv"
if not rl_csv.exists():
    raise FileNotFoundError(f"훈련용 RL 데이터 없음: {rl_csv}")

echo(f"로드: {rl_csv}")
df = pd.read_csv(rl_csv)
echo(f"행 개수: {len(df):,}")

need_latent = [f"latent_{i}" for i in range(16)]
need_feat = ["DLV_SAA_RAT","MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT","RC_M1_SAA_RANK","TREND_RATIO"]
need = ["action","reward","terminal"] + need_latent + need_feat
miss = [c for c in need if c not in df.columns]
if miss:
    raise ValueError(f"컬럼 누락: {miss}")

# 데이터 통계
acts = df["action"].astype(int)
uniq_acts, counts = np.unique(acts, return_counts=True)
echo(f"액션 분포: " + ", ".join([f"{a}:{c}" for a,c in zip(uniq_acts, counts)]))
echo(f"터미널=1 개수: {int(df['terminal'].sum()):,}")
echo(f"보상 통계: mean={df['reward'].mean():.4f} std={df['reward'].std():.4f} min={df['reward'].min():.4f} max={df['reward'].max():.4f}")

# 안전 가드
if len(df) < 500:
    raise RuntimeError(f"데이터가 너무 적습니다(len={len(df)}). rl_dataset.csv 파이프라인 점검 필요.")
if len(uniq_acts) < 2:
    raise RuntimeError("액션이 한 종류뿐입니다. rule_action/데이터 분포 점검 필요.")

# 관측 구성
X_latent = df[need_latent].astype(np.float32)
X_feats  = df[need_feat].astype(np.float32)

# 스케일러 (월지표 5개만)
scaler_path = MODEL_DIR / "standard_scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    echo(f"스케일러 로드: {scaler_path.name}")
else:
    scaler = StandardScaler().fit(X_feats)
    joblib.dump(scaler, scaler_path)
    echo(f"스케일러 피팅 및 저장: {scaler_path.name}")

X_feats_scaled = scaler.transform(X_feats)
obs = np.concatenate([X_latent.values, X_feats_scaled.astype(np.float32)], axis=1).astype(np.float32)
echo(f"관측 차원 확인: obs.shape={obs.shape} (기대 21)")

actions   = acts.values
rewards   = df["reward"].astype(np.float32).values
terminals = df["terminal"].astype(bool).values

dataset = MDPDataset(
    observations=obs,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
)

# CQL 설정: 문자열 대신 '팩토리 객체' 사용!  ⬇⬇
cfg = DiscreteCQLConfig(
    gamma=0.99,
    learning_rate=3e-4,
    batch_size=1024,
    encoder_factory=VectorEncoderFactory(),      # ✅ 객체
    q_func_factory=MeanQFunctionFactory(),       # ✅ 객체
    n_critics=2,
    alpha=1.0,                                   # CQL 보수성 계수
)

device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
algo = DiscreteCQL(cfg, device, 0, False)  # (config, device, seed, enable_ddp)
echo(f"알고리즘 준비 완료 (device={device})")

# 모델 빌드 (명시)
try:
    algo.build_with_dataset(dataset)
    echo("모델 빌드 완료: build_with_dataset()")
except Exception as ex:
    try:
        obs_dim = obs.shape[1]
        action_size = int(np.max(actions)) + 1
        algo.build(observation_shape=(obs_dim,), action_size=action_size)
        echo(f"모델 빌드 완료: build(observation_shape={(obs_dim,)}, action_size={action_size})")
    except Exception as ex2:
        raise RuntimeError(
            f"모델 빌드 실패: primary={type(ex).__name__}: {ex} / fallback={type(ex2).__name__}: {ex2}"
        )

# 로그가 Model/d3rlpy_logs 아래에 생성되도록 작업 디렉터리 이동
os.chdir(MODEL_DIR)
echo(f"작업 디렉터리 이동: {MODEL_DIR}")

# 학습
N_STEPS = int(os.environ.get("TRAIN_STEPS", "30000"))
echo(f"학습 시작: n_steps={N_STEPS:,}, 로그 루트={LOGS_ROOT}")

t0 = time.time()
algo.fit(
    dataset,
    n_steps=N_STEPS,
    n_steps_per_epoch=1000,
    save_interval=1000,
)
echo(f"학습 종료: {(time.time()-t0):.1f}s")

# 산출물 점검
d3_list = sorted(LOGS_ROOT.rglob("model_*.d3"),
                 key=lambda p: int(p.stem.split("_")[-1]))
if not d3_list:
    raise RuntimeError("체크포인트(.d3)가 생성되지 않았습니다. 쓰기 권한/경로 확인.")
latest = d3_list[-1]
echo(f"최신 체크포인트: {latest.relative_to(MODEL_DIR)} (size={latest.stat().st_size:,} bytes)")

print("훈련 완료. 체크포인트는 여기:", LOGS_ROOT)
