# =====================================================================
# App/run_simulation.py  (FULL)
# - PowerShell에서:  py -3.13 .\run_simulation.py
# - Streamlit self-bootstrap: 한 줄로 실행해도 자동으로 streamlit run
# - d3 체크포인트(.d3)를 반드시 우선 로드 (캡션으로 무엇을 로드했는지 명시)
# - 훈련=무스케일 → 추론도 무스케일(스케일 OFF)로 통일  ✅ 핵심 패치
# - Q-values 디버그 패널로 학습/입력 이상 여부 즉시 점검
# =====================================================================

import os, sys, subprocess, random, json
from pathlib import Path

# ---------------------------------------------------------------------
# Self-bootstrap: Python 파일을 직접 실행하면 streamlit로 재실행
# ---------------------------------------------------------------------
if __name__ == "__main__" and os.environ.get("STREAMLIT_BOOTSTRAPPED") != "1":
    file_path = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["STREAMLIT_BOOTSTRAPPED"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", file_path]
    subprocess.run(cmd, env=env, check=False)
    sys.exit(0)

# ---------------------------------------------------------------------
# 본 앱 로직
# ---------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import torch
import d3rlpy
from d3rlpy.algos import DiscreteCQL

# --- PyTorch 2.6+ 호환: d3rlpy 내부 torch.load(weights_only=True) 이슈 회피 ---
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat
# ---------------------------------------------------------------------

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
    app_dir = Path(__file__).resolve().parent           # .../Bigcontest/App
    root_dir = app_dir.parent                           # .../Bigcontest
    model_dir = root_dir / "Model"
    eda_dir = root_dir / "EDA"
    logs_dir = model_dir / "d3rlpy_logs" / "DiscreteCQL"

    return {
        "ROOT": root_dir,
        "MODEL_DIR": model_dir,
        "EDA_DIR": eda_dir,
        "LOGS_DIR": logs_dir,
        "PARAMS_JSON": logs_dir / "params.json",
        "NODE_EMB": model_dir / "node_embeddings.csv",
        "PT_FALLBACK": model_dir / "cql_model.pt",      # 최후의 보루(되도록 비활성 권장)
    }

# 필수 파일 확인
def assert_minimum_files(paths: dict):
    problems = []
    if not paths["NODE_EMB"].exists():
        problems.append(str(paths["NODE_EMB"]))
    if not paths["PARAMS_JSON"].exists():
        problems.append(str(paths["PARAMS_JSON"]))

    logs_dir = paths["LOGS_DIR"]
    has_any_d3 = False
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("model_*.d3")):
            if p.stat().st_size > 0:
                has_any_d3 = True
                break
    if not has_any_d3:
        problems.append(f"{logs_dir}\\model_*.d3 (최소 1개)")

    if problems:
        msg = "다음 필수 파일이 없습니다(또는 비어 있음):\n- " + "\n- ".join(problems)
        raise FileNotFoundError(msg)

# d3rlpy DiscreteCQL 모델 로드: .d3 최신 → (가능하면) .pt는 사용하지 말 것
def load_discrete_cql_with_fallback(params_path: Path, model_dir: Path, device: str):
    """
    우선순위: 가장 큰 스텝의 model_*.d3 -> 나머지 d3
    (최후 수단) cql_model.pt는 되도록 사용하지 않음. 필요 시만 활성화.
    """
    algo = DiscreteCQL.from_json(params_path, device=device)

    logs_dir = model_dir / "d3rlpy_logs" / "DiscreteCQL"
    d3_candidates = sorted(
        logs_dir.glob("model_*.d3"),
        key=lambda p: int(p.stem.split("_")[-1]),
        reverse=True
    )

    tried_msgs = []

    # 1) d3rlpy 네이티브 체크포인트(.d3) 최신부터 로드
    for p in d3_candidates:
        try:
            algo.load_model(str(p))
            return algo, f"loaded: {p.name}"
        except Exception as ex:
            tried_msgs.append(f"d3 fail {p.name}: {type(ex).__name__}: {ex}")

    # 2) (비권장) .pt로 후퇴 — 가능하면 .pt는 이동/삭제하여 사용하지 않길 권장
    # pt_path = model_dir / "cql_model.pt"
    # if pt_path.exists() and pt_path.stat().st_size > 0:
    #     try:
    #         algo.load_model(str(pt_path))
    #         return algo, f"loaded (pt): {pt_path.name}"
    #     except Exception as ex:
    #         tried_msgs.append(f"pt fail {pt_path.name}: {type(ex).__name__}: {ex}")

    detail = "\n".join(tried_msgs) if tried_msgs else "(no candidates found)"
    raise RuntimeError("모델(.d3) 로딩 실패.\n" + detail)

# AI 전략가
class MarketQuantum:
    def __init__(self, model_dir: Path, params_path: Path, node_data_path: Path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model, source_info = load_discrete_cql_with_fallback(params_path, model_dir, self.device)
        self.source_info = source_info

        # 임베딩 로드
        self.df_embeddings = pd.read_csv(node_data_path)

        # 프로젝트 스토리라인에 맞춘 전략 라벨
        self.action_map = {
            0: "전략 0 (현상 유지·내부역량 강화)",
            1: "전략 1 (신규 고객 유치 집중: 콘텐츠·SNS·제휴)",
            2: "전략 2 (재방문·배달 집중: 멤버십·딜리버리 최적화)",
            3: "전략 3 (공격적 확장: 신메뉴+신규·배달 동시 드라이브)",
        }

        # 잠재벡터 컬럼 확인
        self.latent_cols = [f'latent_{i}' for i in range(LATENT_DIM)]
        missing_latent = [c for c in self.latent_cols if c not in self.df_embeddings.columns]
        if missing_latent or 'ENCODED_MCT' not in self.df_embeddings.columns:
            need = missing_latent + (['ENCODED_MCT'] if 'ENCODED_MCT' not in self.df_embeddings.columns else [])
            raise ValueError(f"node_embeddings.csv 컬럼 누락: {need}")

        # ✅ 핵심 패치: 훈련이 '무스케일'이었으므로 추론도 무스케일 고정
        #    (재학습 시 정규화 사용으로 바꾸면, 아래 플래그를 True로만 바꾸면 됨)
        self._use_scaler = False

    def get_state_vector(self, mct_id, monthly_data):
        # 임베딩 찾기
        row = self.df_embeddings[self.df_embeddings['ENCODED_MCT'] == mct_id]
        if row.empty:
            return None
        latent_vector = row[self.latent_cols].values[0].astype(np.float32)

        # 월 지표(입력): 무스케일 원값 사용
        monthly_df = pd.DataFrame([monthly_data], columns=FEATURES)
        if self._use_scaler:
            # (향후 재학습을 정규화로 돌렸다면 여기에 scaler 적용 코드를 넣으면 됨)
            raise RuntimeError("현재 설정은 무스케일 추론입니다. 스케일 사용으로 전환 시 _use_scaler=True와 scaler 적용 코드를 추가하세요.")
        scaled_monthly = monthly_df.values.astype(np.float32).flatten()

        # 최종 상태 벡터 = [latent(16) || features(5)]
        state_vector = np.concatenate([latent_vector, scaled_monthly]).astype(np.float32)
        return state_vector

    def recommend_strategy(self, state_vector):
        if state_vector is None:
            return "분석 실패: 가게 ID를 임베딩에서 찾을 수 없습니다."

        # 기본 argmax
        actions = self.model.predict(np.expand_dims(state_vector, axis=0))
        action_idx = int(actions[0])

        # 동점 타이브레이크(평탄 Q 완화; 정상 모델에선 영향 적음)
        try:
            q = self.model.predict_value(np.expand_dims(state_vector, axis=0))[0]
            best = float(np.max(q))
            ties = np.where(np.isclose(q, best, rtol=1e-5, atol=1e-6))[0].tolist()
            if len(ties) > 1:
                action_idx = int(random.choice(ties))
        except Exception:
            pass

        return self.action_map.get(action_idx, f"알 수 없음(action={action_idx})")

# 관측 차원 검증(선택)
def _sanity_check_schema(agent: MarketQuantum):
    try:
        obs_shape = agent.model.observation_shape  # 예: (21,)
        expected = LATENT_DIM + len(FEATURES)
        if obs_shape and obs_shape[0] != expected:
            st.error(f"관측 차원 불일치: 모델 {obs_shape[0]} vs 현재 입력 {expected}")
    except Exception:
        pass

@st.cache_resource
def load_ai_agent():
    paths = get_paths()
    assert_minimum_files(paths)
    agent = MarketQuantum(
        model_dir=paths["MODEL_DIR"],
        params_path=paths["PARAMS_JSON"],
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
st.markdown("성동구 상권 디지털 트윈 기반으로 점포별 **미래 성공 확률을 높이는 전략**을 제안합니다.")
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
                spread = float(np.max(q_values) - np.min(q_values))
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
                if spread < 1e-3:
                    st.warning("모든 액션의 Q값이 거의 동일합니다. "
                               "훈련/추론 입력 분포 불일치 또는 모델 미학습 가능성이 있습니다.")
                else:
                    st.success(f"Q spread OK: {spread:.4f}")
            except Exception as ex:
                st.error(f"Q-values 계산 실패: {type(ex).__name__}: {ex}")

# 사용 팁:
# - PowerShell:
#     PS C:\Users\hyunj\Bigcontest\App> py -3.13 .\run_simulation.py
#   또는 프로젝트 루트:
#     PS C:\Users\hyunj\Bigcontest> py -3.13 .\App\run_simulation.py
# - 캡션이 반드시 'loaded: model_XXXX.d3' 형태여야 하며, '(pt)'가 보이면 .pt를 치우고 재실행하세요.
