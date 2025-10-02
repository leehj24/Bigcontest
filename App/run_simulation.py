# ==============================================================================
# self-bootstrap: `python run_simulation.py`로 실행해도 Streamlit로 재실행
# ==============================================================================
import os, sys, subprocess
from pathlib import Path

if __name__ == "__main__" and os.environ.get("STREAMLIT_BOOTSTRAPPED") != "1":
    file_path = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["STREAMLIT_BOOTSTRAPPED"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", file_path]
    subprocess.run(cmd, env=env, check=False)
    sys.exit(0)

# ==============================================================================
# 본 앱 로직
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import d3rlpy
import torch
import joblib
from pathlib import Path
from d3rlpy.algos import DiscreteCQL
from sklearn.preprocessing import StandardScaler

# --- PyTorch 2.6 호환 래퍼: d3rlpy 내부 torch.load가 weights_only=True로 깨지는 문제 방지 ---
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 구버전 .d3 체크포인트 호환
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat
# --------------------------------------------------------------------------------

# 전역 상수
LATENT_DIM = 16
FEATURES = ['DLV_SAA_RAT', 'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT', 'RC_M1_SAA_RANK', 'TREND_RATIO']

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

# 스케일러 확보(없으면 자동 생성) + 랭크 합성까지 내장
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
        raise FileNotFoundError("standard_scaler.pkl 없음 + EDA/merged_data.csv, Model/rl_dataset.csv 모두 없어 자동 생성 불가")

    # RC_M1_SAA_RANK 없으면 즉석 합성 (월별*업종별 매출 기준 6구간: 1=상위, 6=하위)
    if "RC_M1_SAA_RANK" not in df.columns:
        required = {"RC_M1_SAA", "TA_YM", "HPSN_MCT_ZCD_NM"}
        if not required.issubset(df.columns):
            raise ValueError("스케일러 생성 실패: RC_M1_SAA_RANK가 없고, 합성에 필요한 열(RC_M1_SAA, TA_YM, HPSN_MCT_ZCD_NM) 부족")
        tmp = df.sort_values(["TA_YM", "HPSN_MCT_ZCD_NM", "RC_M1_SAA"], ascending=[True, True, False]).copy()
        tmp["__rank"] = tmp.groupby(["TA_YM", "HPSN_MCT_ZCD_NM"]).cumcount() + 1

        def to_bins(s):
            try:
                return pd.qcut(s, q=6, labels=[1,2,3,4,5,6], duplicates="drop").astype(int)
            except Exception:
                n = len(s)
                if n == 0:
                    return s.astype(int)
                edges = [(i*n)//6 for i in range(7)]
                idx = s.values - 1  # 0-index
                out = [min(max(sum(v >= e for e in edges[1:]) + 1, 1), 6) for v in idx]
                return pd.Series(out, index=s.index).astype(int)

        df["RC_M1_SAA_RANK"] = tmp.groupby(["TA_YM", "HPSN_MCT_ZCD_NM"])["__rank"].transform(to_bins)

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

# d3rlpy DiscreteCQL 모델을 견고하게 로드(여러 후보 시도)
def load_discrete_cql_with_fallback(params_path: Path, model_dir: Path, device: str):
    """
    우선순위: model_10000.d3 -> 9000 -> ... -> 1000 -> cql_model.pt
    """
    algo = DiscreteCQL.from_json(params_path, device=device)

    logs_dir = model_dir / "d3rlpy_logs" / "DiscreteCQL"
    cand_steps = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
    d3_candidates = [logs_dir / f"model_{s}.d3" for s in cand_steps]
    pt_candidate = model_dir / "cql_model.pt"

    tried = []

    for p in d3_candidates:
        if p.exists() and p.stat().st_size > 0:
            try:
                algo.load_model(p)  # d3rlpy native
                return algo, f"loaded: {p.name}"
            except Exception as ex:
                tried.append((str(p), str(ex)))

    if pt_candidate.exists() and pt_candidate.stat().st_size > 0:
        try:
            algo.load_model(pt_candidate)
            return algo, f"loaded (pt): {pt_candidate.name}"
        except Exception as ex:
            tried.append((str(pt_candidate), str(ex)))

    detail = "\n".join([f"- {p} :: {e}" for p, e in tried]) or "(no candidates tried)"
    raise RuntimeError("모델 로딩 실패. 시도 내역:\n" + detail)

# 필수 파일 확인(최소 요건)
def assert_minimum_files(paths: dict):
    problems = []
    if not paths["NODE_EMB"].exists():
        problems.append(str(paths["NODE_EMB"]))
    if not paths["PARAMS_JSON"].exists():
        problems.append(str(paths["PARAMS_JSON"]))
    # 체크포인트는 여러 후보 중 하나라도 있으면 통과
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
        # 견고한 로더 사용
        self.model, source_info = load_discrete_cql_with_fallback(params_path, model_dir, self.device)
        self.source_info = source_info

        self.scaler = scaler
        self.df_embeddings = pd.read_csv(node_data_path)

        self.action_map = {
            0: "전략 0 (현상 유지 또는 내부 역량 강화)",
            1: "전략 1 (신규 고객 유치에 집중)",
            2: "전략 2 (기존 고객 또는 배달 서비스에 집중)",
            3: "전략 3 (배달과 신규 고객 모두 공격적으로 확장)",
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
        action_idx = self.model.predict(np.expand_dims(state_vector, axis=0))[0]
        return self.action_map.get(int(action_idx), f"알 수 없음(action={int(action_idx)})")

# 캐시된 에이전트 로더
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
    return agent

# Streamlit 앱
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
st.markdown("성동구 상권 디지털 트윈 기반으로 점포별 **미래 성공 확률을 높이는 전략**을 제안한다.")
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
        "근거: 잠재 임베딩(VGAE) + 최신 지표(배달·재방문·신규·매출순위·트렌드)를 결합해 "
        "오프라인 RL(DiscreteCQL) 정책으로 장기 보상 관점의 최적 행동을 선정.",
        icon="💡"
    )

# 사용 팁:
# - 이제 `python App\run_simulation.py` 혹은 `python .\run_simulation.py`(App 폴더에서) 한 줄로 실행 가능
# - Gym 경고는 정보성. 필요시 `pip install gymnasium`
