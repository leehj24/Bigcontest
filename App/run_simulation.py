import streamlit as st
import pandas as pd
import numpy as np
import d3rlpy
import torch
import joblib
from pathlib import Path
from d3rlpy.algos import DiscreteCQL

# ==============================================================================
# AI 전략가 '마켓 퀀텀' 클래스 (기존 코드와 거의 동일)
# ==============================================================================
# @st.cache_resource 데코레이터를 사용해 AI 에이전트를 캐시에 저장합니다.
# 이렇게 하면 앱이 재실행될 때마다 모델을 새로 불러오는 것을 방지하여 속도를 높일 수 있습니다.
@st.cache_resource
def load_ai_agent():
    # 이 app.py 파일의 위치를 기준으로 상위 폴더(Bigcontest)를 찾습니다.
    APP_DIR = Path(__file__).resolve().parent
    MODEL_DIR = APP_DIR.parent / "Model"
    
    class MarketQuantum:
        def __init__(self, model_path, scaler_path, node_data_path, params_path):
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = DiscreteCQL.from_json(params_path, device=self.device)
            self.model.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.df_embeddings = pd.read_csv(node_data_path)
            self.action_map = {
                0: "전략 0 (현상 유지 또는 내부 역량 강화)",
                1: "전략 1 (신규 고객 유치에 집중)",
                2: "전략 2 (기존 고객 또는 배달 서비스에 집중)",
                3: "전략 3 (배달과 신규 고객 모두 공격적으로 확장)"
            }

        def get_state_vector(self, mct_id, monthly_data):
            try:
                latent_cols = [f'latent_{i}' for i in range(16)]
                latent_vector = self.df_embeddings[self.df_embeddings['ENCODED_MCT'] == mct_id][latent_cols].values[0]
            except IndexError:
                return None
            
            feature_names = ['DLV_SAA_RAT', 'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT', 'RC_M1_SAA_RANK', 'TREND_RATIO']
            monthly_df = pd.DataFrame([monthly_data], columns=feature_names)
            scaled_monthly = self.scaler.transform(monthly_df)
            
            state_vector = np.concatenate([latent_vector, scaled_monthly.flatten()]).astype(np.float32)
            return state_vector

        def recommend_strategy(self, state_vector):
            if state_vector is None:
                return "분석 실패: 가게 ID를 찾을 수 없습니다."
            action_idx = self.model.predict(np.expand_dims(state_vector, axis=0))[0]
            return self.action_map.get(int(action_idx), "알 수 없음")

    params_json_path = MODEL_DIR / 'd3rlpy_logs/DiscreteCQL/params.json'
    
    agent = MarketQuantum(
        # d3rlpy는 모델 저장 시 cql_model.pt가 아닌 model_10000.d3 와 같이 저장합니다.
        model_path= MODEL_DIR / 'd3rlpy_logs/DiscreteCQL/model_10000.d3',
        scaler_path= MODEL_DIR / 'standard_scaler.pkl',
        node_data_path= MODEL_DIR / 'node_embeddings.csv',
        params_path=params_json_path
    )
    return agent

# ==============================================================================
# Streamlit 웹 애플리케이션 UI 구성
# ==============================================================================
st.set_page_config(page_title="마켓 퀀텀 AI 전략 시뮬레이터", layout="wide")

# --- 1. AI 에이전트 및 데이터 로딩 ---
try:
    agent = load_ai_agent()
    # ✅ [수정] agent 로딩 후, 함수 외부에서 all_store_ids 변수를 정의합니다.
    all_store_ids = agent.df_embeddings['ENCODED_MCT'].unique()
except Exception as e:
    st.error(f"AI 에이전트 로딩에 실패했습니다: {e}")
    st.stop()


# --- 2. 웹페이지 제목 및 설명 ---
st.title("📈 마켓 퀀텀: AI 전략 시뮬레이터")
st.markdown("성동구 상권의 디지털 트윈을 기반으로 당신의 가게에 최적화된 미래 성공 전략을 제안합니다.")
st.divider()


# --- 3. 사용자 입력 사이드바 ---
with st.sidebar:
    st.header("🔍 시뮬레이션 정보 입력")
    
    target_store_id = st.selectbox(
        "분석할 가게의 ID를 선택하세요:",
        options=all_store_ids, # 이제 이 변수를 정상적으로 사용할 수 있습니다.
        index=0
    )
    
    st.subheader("가게의 최신 월별 데이터를 입력하세요:")
    
    dlv_rat = st.slider("배달 매출 비중 (%)", 0.0, 100.0, 10.5)
    reu_rat = st.slider("재방문 고객 비율 (%)", 0.0, 100.0, 30.2)
    new_rat = st.slider("신규 고객 비율 (%)", 0.0, 100.0, 15.8)
    saa_rank = st.select_slider("매출 순위 구간", options=[1, 2, 3, 4, 5, 6], value=4)
    trend_ratio = st.slider("현재 업종 트렌드 지수", 0.0, 100.0, 88.7)
    
    st.divider()
    run_button = st.button("AI 전략 분석 시작", type="primary", use_container_width=True)


# --- 4. 분석 결과 출력 ---
if run_button:
    with st.spinner('AI가 디지털 트윈 환경에서 수만 가지 미래를 시뮬레이션 중입니다...'):
        latest_monthly_data = [dlv_rat, reu_rat, new_rat, saa_rank, trend_ratio]
        current_state = agent.get_state_vector(target_store_id, latest_monthly_data)
        recommendation = agent.recommend_strategy(current_state)

    st.header("📄 AI 전략 시뮬레이션 리포트", divider='rainbow')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="분석 대상 가게 ID", value=target_store_id)
    
    st.subheader("✅ AI 최종 전략 제언")
    st.success(f"**{recommendation}**")
    
    st.info("**제언 근거:** AI는 현재 가게의 잠재력과 최근 성과, 그리고 성동구 전체 상권의 변화를 종합적으로 분석했습니다. 그 결과, 제안된 전략이 향후 6개월간 누적 수익을 극대화할 확률이 가장 높다고 판단했습니다.", icon="💡")
