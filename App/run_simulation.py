# App/run_simulation.py
# - PowerShell:  py -3.13 .\App\run_simulation.py
# - Streamlit self-bootstrap (직접 실행해도 streamlit run으로 재기동)
# - d3 체크포인트를 재귀 탐색하며, 손상 파일(작은 파일)은 자동 스킵
# - 훈련이 무스케일이었으므로 추론도 무스케일(스케일 OFF)로 통일
# - Q-values 디버그 패널 제공

import os, sys, subprocess, random
from pathlib import Path

# ------------------------------------------------------------
# Self-bootstrap: python으로 직접 실행하면 streamlit run으로 재실행
# ------------------------------------------------------------
if __name__ == "__main__" and os.environ.get("STREAMLIT_BOOTSTRAPPED") != "1":
    file_path = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["STREAMLIT_BOOTSTRAPPED"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", file_path]
    subprocess.run(cmd, env=env, check=False)
    sys.exit(0)

# ------------------------------------------------------------
# App 본체
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import torch
from d3rlpy.algos import DiscreteCQL

# --- PyTorch 2.6+ 호환: d3rlpy 내부 torch.load(weights_only=True) 회피 ---
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    # 🔧 d3rlpy가 명시적으로 True를 주입해도 무조건 False로 고정
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat
# ---------------------------------------------------------------------------

LATENT_DIM = 16
FEATURES = [
    'DLV_SAA_RAT',          # 배달 매출 비중(%)
    'MCT_UE_CLN_REU_RAT',   # 재방문 고객 비율(%)
    'MCT_UE_CLN_NEW_RAT',   # 신규 고객 비율(%)
    'RC_M1_SAA_RANK',       # 매출 순위 구간(1=상위, 6=하위)
    'TREND_RATIO'           # 업종 트렌드 지수(0~100)
]

# ---------------------- 경로 유틸 ----------------------
def get_paths():
    app_dir = Path(__file__).resolve().parent        # .../Bigcontest/App
    root_dir = app_dir.parent                        # .../Bigcontest
    model_dir = root_dir / "Model"
    logs_root = model_dir / "d3rlpy_logs"           # d3rlpy가 생성하는 루트
    return {
        "ROOT": root_dir,
        "MODEL_DIR": model_dir,
        "LOGS_ROOT": logs_root,
        "NODE_EMB": model_dir / "node_embeddings.csv",
    }

def assert_minimum_files(paths: dict):
    problems = []
    if not paths["NODE_EMB"].exists():
        problems.append(str(paths["NODE_EMB"]))
    # 재귀로 .d3 존재 여부 확인
    has_any_d3 = any(paths["LOGS_ROOT"].rglob("model_*.d3")) if paths["LOGS_ROOT"].exists() else False
    if not has_any_d3:
        problems.append(f"{paths['LOGS_ROOT']}\\**\\model_*.d3 (최소 1개)")
    if problems:
        raise FileNotFoundError("다음 필수 파일이 없습니다(또는 비어 있음):\n- " + "\n- ".join(problems))

# ------------------ 체크포인트 로더 -------------------
def _extract_step_num(p: Path) -> int:
    try:
        return int(p.stem.split("_")[-1])  # model_12345.d3 -> 12345
    except Exception:
        return -1

def load_discrete_cql_with_fallback(logs_root: Path, device: str):
    # ❶ 먼저 정식 저장본(model_final.d3)부터 시도
    final_ckpt = next(logs_root.rglob("model_final.d3"), None)
    if final_ckpt and final_ckpt.exists() and final_ckpt.stat().st_size > 0:
        params = final_ckpt.parent / "params.json"
        if params.exists():
            algo = DiscreteCQL.from_json(params, device=device)
            algo.load_model(str(final_ckpt))
            return algo, f"loaded: {final_ckpt.name} @ {final_ckpt.parent.name}"

    # ❷ (보조) 주기 저장본(model_*.d3)을 사용 — 단, 크기>100KB만
    cands = [p for p in logs_root.rglob("model_*.d3") if p.stat().st_size > 100_000]
    if not cands:
        raise RuntimeError("체크포인트(.d3)를 찾지 못했습니다.")

    cands = sorted(cands, key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime), reverse=True)
    tried = []
    for ckpt in cands:
        params = ckpt.parent / "params.json"
        if not params.exists():
            tried.append(f"{ckpt.name}: params.json 없음")
            continue
        try:
            algo = DiscreteCQL.from_json(params, device=device)
            algo.load_model(str(ckpt))
            return algo, f"loaded: {ckpt.name} @ {ckpt.parent.name}"
        except Exception as ex:
            tried.append(f"{ckpt.name}: {type(ex).__name__}: {ex}")

    raise RuntimeError("모델(.d3) 로딩 실패. " + " ".join(tried[:10]))

# ------------------ 에이전트 -------------------
class MarketQuantum:
    def __init__(self, logs_root: Path, node_data_path: Path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model, self.source_info = load_discrete_cql_with_fallback(logs_root, self.device)

        # 임베딩 로드
        self.df_embeddings = pd.read_csv(node_data_path)

        # 액션 라벨
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

        # ✅ 훈련=무스케일 → 추론도 무스케일 (원값 사용)
        self._use_scaler = False

    def get_state_vector(self, mct_id, monthly_data):
        row = self.df_embeddings[self.df_embeddings['ENCODED_MCT'] == mct_id]
        if row.empty:
            return None
        latent_vector = row[self.latent_cols].values[0].astype(np.float32)

        # 월 지표(입력): 무스케일 원값 사용
        monthly_df = pd.DataFrame([monthly_data], columns=FEATURES)
        if self._use_scaler:
            # (재학습을 정규화로 바꾸면 여기에 scaler 적용 코드를 넣고 True로 전환)
            raise RuntimeError("현재 설정은 무스케일 추론입니다. 스케일 사용 전환 시 scaler 적용 코드를 추가하세요.")
        feat_vec = monthly_df.values.astype(np.float32).flatten()

        state_vector = np.concatenate([latent_vector, feat_vec]).astype(np.float32)
        return state_vector

    def recommend_strategy(self, state_vector):
        if state_vector is None:
            return "분석 실패: 가게 ID를 임베딩에서 찾을 수 없습니다."
        # 기본 argmax
        actions = self.model.predict(np.expand_dims(state_vector, axis=0))
        action_idx = int(actions[0])

        # 동점 타이브레이크 (Q가 납작할 때)
        try:
            q = self.model.predict_value(np.expand_dims(state_vector, axis=0))[0]
            best = float(np.max(q))
            ties = np.where(np.isclose(q, best, rtol=1e-5, atol=1e-6))[0].tolist()
            if len(ties) > 1:
                action_idx = int(random.choice(ties))
        except Exception:
            pass

        return self.action_map.get(action_idx, f"알 수 없음(action={action_idx})")

def _sanity_check_schema(agent: MarketQuantum):
    try:
        obs_shape = agent.model.observation_shape  # (21,) 기대
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
        logs_root=paths["LOGS_ROOT"],
        node_data_path=paths["NODE_EMB"],
    )
    _sanity_check_schema(agent)
    return agent

# ========================== UI ==========================
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

# 사용 팁
# - 실행:   py -3.13 .\App\run_simulation.py
# - 캡션:   loaded: model_*.d3 @ DiscreteCQL_YYYYMMDD... 형태가 떠야 정상
# - 손상된 .d3(아주 작은 파일)는 자동으로 스킵되며, 타임스탬프 폴더도 자동 검색됨
