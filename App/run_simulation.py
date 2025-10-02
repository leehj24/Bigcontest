# =====================================================================
# App/run_simulation.py  (FULL)
# - Streamlit self-bootstrap
# - .d3 체크포인트 우선 로딩 + 실패 사유 수집
# - torch.load(weights_only=False) 호환 래퍼
# - 스케일러 자동 생성(없으면) + RC_M1_SAA_RANK 합성
# - Q-values 디버그 패널 + 동점 타이브레이크
# - 모델 관측 차원 vs 입력 차원 스키마 검증
# =====================================================================

import os, sys, subprocess, random
from pathlib import Path

if __name__ == "__main__" and os.environ.get("STREAMLIT_BOOTSTRAPPED") != "1":
    # Streamlit로 자기 자신을 실행 (PowerShell에서도 한 줄로)
    file_path = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["STREAMLIT_BOOTSTRAPPED"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", file_path]
    subprocess.run(cmd, env=env, check=False)
    sys.exit(0)

# =====================================================================
# 본 앱 로직
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import d3rlpy
import torch
import joblib
from d3rlpy.algos import DiscreteCQL
from sklearn.preprocessing import StandardScaler

# --- PyTorch 2.6 호환: d3rlpy 내부 torch.load가 weights_only=True로 로드 실패 방지 ---
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 구버전 .d3 호환
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat
# ------------------------------------------------------------------------------------

# 전역 상수
LATENT_DIM = 16
FEATURES = [
    'DLV_SAA_RAT',          # 배달 매출 비중(%)
    'MCT_UE_CLN_REU_RAT',   # 재방문 고객 비율(%)
    'MCT_UE_CLN_NEW_RAT',   # 신규 고객 비율(%)
    'RC_M1_SAA_RANK',       # 매출 순위 구간(1=상위, 6=하위)
    'TREND_RATIO'           # 업종 트렌드 지수(0~100)
]

# 경로 유틸
def get_paths():
    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent  # Bigcontest
    model_dir = root_dir / "Model"
    eda_dir = root_dir / "EDA"
    logs_dir = model_dir / "d3rlpy_logs" / "DiscreteCQL"

    return {
        "ROOT": root_dir,
        "MODEL_DIR": model_dir,
        "EDA_DIR": eda_dir,
        "LOGS_DIR": logs_dir,
        "PARAMS_JSON": logs_dir / "params.json",
        "SCALER_PKL": model_dir / "standard_scaler.pkl",
        "NODE_EMB": model_dir / "node_embeddings.csv",
        "MERGED_CSV": eda_dir / "merged_data.csv",
        "RL_DATASET": model_dir / "rl_dataset.csv",
    }

# 스케일러 확보(없으면 자동 생성) + RC_M1_SAA_RANK 합성
def ensure_scaler(paths: dict) -> StandardScaler:
    scaler_path = paths["SCALER_PKL"]
    if scaler_path.exists():
        return joblib.load(scaler_path)

    # 데이터 소스 확보
    if paths["MERGED_CSV"].exists():
        df = pd.read_csv(paths["MERGED_CSV"], encoding="utf-8-sig")
    elif paths["RL_DATASET"].exists():
        df = pd.read_csv(paths["RL_DATASET"])
    else:
        raise FileNotFoundError(
            "standard_scaler.pkl 없음 + EDA/merged_data.csv, Model/rl_dataset.csv 모두 없어 자동 생성 불가"
        )

    # RC_M1_SAA_RANK 없으면 즉석 합성
    if "RC_M1_SAA_RANK" not in df.columns:
        required = {"RC_M1_SAA", "TA_YM", "HPSN_MCT_ZCD_NM"}
        if not required.issubset(df.columns):
            raise ValueError(
                "스케일러 생성 실패: RC_M1_SAA_RANK가 없고, 합성에 필요한 열(RC_M1_SAA, TA_YM, HPSN_MCT_ZCD_NM) 부족"
            )
        tmp = df.sort_values(
            ["TA_YM", "HPSN_MCT_ZCD_NM", "RC_M1_SAA"],
            ascending=[True, True, False]
        ).copy()
        tmp["__rank"] = tmp.groupby(["TA_YM", "HPSN_MCT_ZCD_NM"]).cumcount() + 1

        def to_bins(s):
            try:
                return pd.qcut(s, q=6, labels=[1,2,3,4,5,6], duplicates="drop").astype(int)
            except Exception:
                # 표본 수 적을 때 예비 등분
                n = len(s)
                if n == 0:
                    return s.astype(int)
                edges = [(i*n)//6 for i in range(7)]
                idx = s.values - 1
                out = [min(max(sum(v >= e for e in edges[1:]) + 1, 1), 6) for v in idx]
                return pd.Series(out, index=s.index).astype(int)

        df["RC_M1_SAA_RANK"] = tmp.groupby(
            ["TA_YM", "HPSN_MCT_ZCD_NM"]
        )["__rank"].transform(to_bins)

    # 스케일러 피팅
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"스케일러 생성 실패: 학습 피처 누락 {missing}")

    X = df[FEATURES].dropna()
    if len(X) == 0:
        raise ValueError("스케일러 생성 실패: 학습 데이터가 비어 있음")

    scaler = StandardScaler().fit(X)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaler

# d3rlpy DiscreteCQL 모델 로드: .d3 최신 → 나머지 → (마지막) .pt
def load_discrete_cql_with_fallback(params_path: Path, model_dir: Path, device: str):
    """
    우선순위: 가장 큰 스텝의 model_*.d3 -> 나머지 d3 -> (마지막) cql_model.pt
    실패 사유를 문자열로 누적해 디버깅에 활용.
    """
    algo = DiscreteCQL.from_json(params_path, device=device)

    logs_dir = model_dir / "d3rlpy_logs" / "DiscreteCQL"
    d3_candidates = sorted(
        logs_dir.glob("model_*.d3"),
        key=lambda p: int(p.stem.split("_")[-1]),
        reverse=True
    )
    pt_candidate = model_dir / "cql_model.pt"

    tried_msgs = []

    for p in d3_candidates:
        try:
            algo.load_model(str(p))
            return algo, f"loaded: {p.name}"
        except Exception as ex:
            tried_msgs.append(f"d3 fail {p.name}: {type(ex).__name__}: {ex}")

    if pt_candidate.exists() and pt_candidate.stat().st_size > 0:
        try:
            algo.load_model(str(pt_candidate))
            return algo, f"loaded (pt): {pt_candidate.name}"
        except Exception as ex:
            tried_msgs.append(f"pt fail {pt_candidate.name}: {type(ex).__name__}: {ex}")

    detail = "\n".join(tried_msgs) if tried_msgs else "(no candidates found)"
    raise RuntimeError("모델 로딩 실패.\n" + detail)

# 필수 파일 확인
def assert_minimum_files(paths: dict):
    problems = []
    if not paths["NODE_EMB"].exists():
        problems.append(str(paths["NODE_EMB"]))
    if not paths["PARAMS_JSON"].exists():
        problems.append(str(paths["PARAMS_JSON"]))

    logs_dir = paths["LOGS_DIR"]
    has_any_ckpt = False
    if logs_dir.exists():
        for p in logs_dir.glob("model_*.d3"):
            if p.stat().st_size > 0:
                has_any_ckpt = True
                break
    pt_path = paths["MODEL_DIR"] / "cql_model.pt"
    if pt_path.exists() and pt_path.stat().st_size > 0:
        has_any_ckpt = True
    if not has_any_ckpt:
        problems.append(f"{logs_dir}\\model_*.d3 또는 {pt_path} 중 하나")

    if problems:
        msg = "다음 필수 파일(중 하나 이상)이 없습니다 또는 비어 있습니다:\n- " + "\n- ".join(problems)
        raise FileNotFoundError(msg)

# AI 전략가
class MarketQuantum:
    def __init__(self, model_dir: Path, params_path: Path, scaler: StandardScaler, node_data_path: Path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model, source_info = load_discrete_cql_with_fallback(params_path, model_dir, self.device)
        self.source_info = source_info

        self.scaler = scaler
        self.df_embeddings = pd.read_csv(node_data_path)

        # 프로젝트 스토리라인과 맞춘 전략명
        self.action_map = {
            0: "전략 0 (현상 유지·내부역량 강화)",
            1: "전략 1 (신규 고객 유치 집중: 콘텐츠·SNS·제휴)",
            2: "전략 2 (재방문·배달 집중: 멤버십·딜리버리 최적화)",
            3: "전략 3 (공격적 확장: 신메뉴+신규·배달 동시 드라이브)",
        }

        self.latent_cols = [f'latent_{i}' for i in range(LATENT_DIM)]
        missing_latent = [c for c in self.latent_cols if c not in self.df_embeddings.columns]
        if missing_latent or 'ENCODED_MCT' not in self.df_embeddings.columns:
            need = missing_latent + (['ENCODED_MCT'] if 'ENCODED_MCT' not in self.df_embeddings.columns else [])
            raise ValueError(f"node_embeddings.csv 컬럼 누락: {need}")

    def get_state_vector(self, mct_id, monthly_data):
        row = self.df_embeddings[self.df_embeddings['ENCODED_MCT'] == mct_id]
        if row.empty:
            return None
        latent_vector = row[self.latent_cols].values[0].astype(np.float32)

        monthly_df = pd.DataFrame([monthly_data], columns=FEATURES)
        scaled_monthly = self.scaler.transform(monthly_df).astype(np.float32).flatten()

        state_vector = np.concatenate([latent_vector, scaled_monthly]).astype(np.float32)
        return state_vector

    def recommend_strategy(self, state_vector):
        if state_vector is None:
            return "분석 실패: 가게 ID를 임베딩에서 찾을 수 없습니다."

        # 기본 argmax
        actions = self.model.predict(np.expand_dims(state_vector, axis=0))
        action_idx = int(actions[0])

        # 동점 타이브레이크(항상 0으로 고정되는 현상 완화)
        try:
            q = self.model.predict_value(np.expand_dims(state_vector, axis=0))[0]
            best = float(np.max(q))
            ties = np.where(np.isclose(q, best, rtol=1e-5, atol=1e-6))[0].tolist()
            if len(ties) > 1:
                action_idx = int(random.choice(ties))
        except Exception:
            pass

        return self.action_map.get(action_idx, f"알 수 없음(action={action_idx})")

# 관측 차원 vs 입력 차원 검증
def _sanity_check_schema(agent: MarketQuantum):
    cols = agent.latent_cols + FEATURES
    try:
        obs_shape = agent.model.observation_shape  # 예: (21,)
        if obs_shape and obs_shape[0] != len(cols):
            st.error(f"관측 차원 불일치: 모델 {obs_shape[0]} vs 현재 입력 {len(cols)}")
    except Exception:
        pass

@st.cache_resource
def load_ai_agent():
    paths = get_paths()
    assert_minimum_files(paths)
    scaler = ensure_scaler(paths)
    agent = MarketQuantum(
        model_dir=paths["MODEL_DIR"],
        params_path=paths["PARAMS_JSON"],
        scaler=scaler,
        node_data_path=paths["NODE_EMB"],
    )
    _sanity_check_schema(agent)
    return agent

# =========================== Streamlit UI ==============================

st.set_page_config(page_title="마켓 퀀텀 AI 전략 시뮬레이터", layout="wide")

# 1) 에이전트 로딩
try:
    agent = load_ai_agent()
    all_store_ids = agent.df_embeddings['ENCODED_MCT'].dropna().unique()
except Exception as e:
    st.error(f"AI 에이전트 로딩 실패: {e}")
    st.stop()

# 2) 제목/설명
st.title("📈 마켓 퀀텀: AI 전략 시뮬레이터")
st.markdown(
    "성동구 상권 디지털 트윈 기반으로 점포별 **미래 성공 확률을 높이는 전략**을 제안합니다."
)
st.caption(
    f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} / Policy: DiscreteCQL / {agent.source_info}"
)
st.divider()

# 3) 사이드바 입력
with st.sidebar:
    st.header("🔍 시뮬레이션 정보 입력")
    target_store_id = st.selectbox(
        "분석할 가게 ID:",
        options=all_store_ids,
        index=0 if len(all_store_ids) else None
    )

    st.subheader("가장 최근 월 지표 입력")
    dlv_rat = st.slider("배달 매출 비중 (%)", 0.0, 100.0, 10.5)
    reu_rat = st.slider("재방문 고객 비율 (%)", 0.0, 100.0, 30.2)
    new_rat = st.slider("신규 고객 비율 (%)", 0.0, 100.0, 15.8)
    saa_rank = st.select_slider("매출 순위 구간(1=상위)", options=[1, 2, 3, 4, 5, 6], value=4)
    trend_ratio = st.slider("현재 업종 트렌드 지수", 0.0, 100.0, 88.7)

    st.divider()
    run_button = st.button("AI 전략 분석 시작", type="primary", use_container_width=True)

# 4) 결과
if run_button:
    latest_monthly_data = [dlv_rat, reu_rat, new_rat, saa_rank, trend_ratio]
    with st.spinner('AI가 디지털 트윈에서 반사실적 시나리오를 탐색 중...'):
        state_vec = agent.get_state_vector(target_store_id, latest_monthly_data)
        recommendation = agent.recommend_strategy(state_vec)

    st.header("📄 AI 전략 시뮬레이션 리포트", divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="분석 대상 가게 ID", value=str(target_store_id))
    with col2:
        st.metric(label="디바이스", value="CUDA" if torch.cuda.is_available() else "CPU")

    st.subheader("✅ AI 최종 전략 제언")
    st.success(f"**{recommendation}**")

    st.info(
        "근거: VGAE 잠재임베딩 + 최신 지표(배달·재방문·신규·매출순위·트렌드) → "
        "오프라인 RL(DiscreteCQL)로 **반사실적 시나리오**를 탐색해 장기 보상 기준으로 최적 전략을 도출합니다.",
        icon="💡"
    )

    # --- Q-values 디버그 패널 ---
    with st.expander("디버그: Q-values / 입력 상태 점검", expanded=False):
        if state_vec is None:
            st.write("임베딩에서 해당 가게를 찾지 못해 상태벡터가 None 입니다.")
        else:
            try:
                q_values = agent.model.predict_value(np.expand_dims(state_vec, axis=0))[0]
                st.write("로드된 체크포인트:", agent.source_info)
                st.write("Q-values:", [float(x) for x in q_values])
                st.write(
                    "state_vector 요약:",
                    {
                        "dim": int(len(state_vec)),
                        "min": float(np.min(state_vec)),
                        "max": float(np.max(state_vec)),
                        "mean": float(np.mean(state_vec)),
                    }
                )
                spread = float(np.max(q_values) - np.min(q_values))
                if spread < 1e-3:
                    st.warning("모든 액션의 Q값이 거의 동일합니다. "
                               "모델 미학습이거나 입력 스케일/피처 순서가 학습 때와 다를 수 있습니다.")
            except Exception as ex:
                st.error(f"Q-values 계산 실패: {type(ex).__name__}: {ex}")

# 사용 팁:
# - PowerShell에서:
#     PS C:\Users\hyunj\Bigcontest\App> python .\run_simulation.py
#   또는 프로젝트 루트에서:
#     PS C:\Users\hyunj\Bigcontest> python .\App\run_simulation.py
# - Gym 경고는 정보성. 필요시 `pip install gymnasium`로 제거 가능
