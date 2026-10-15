import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("📈 서울 연평균기온 분석 및 예측기")
st.markdown("제공된 서울 기온 데이터를 기반으로 연도별 평균기온을 계산하고, 선형 회귀 분석을 통해 미래 기온을 예측합니다.")

@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 인코딩 UTF-8 로드
    df = pd.read_csv(url, encoding='utf-8')
    
    # 날짜 컬럼 형변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년까지의 데이터만 필터링 (수업 기준 기간)
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일 수 및 평균기온 계산
    yearly_stats = df.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    # 관측일이 300일 미만인 해는 제외
    valid_yearly = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    
    return valid_yearly

try:
    df_valid = load_and_preprocess_data()
    
    # 회귀 모델 학습
    X = df_valid[['연도']]
    y = df_valid['연평균기온']
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # 상관계수 계산
    correlation = df_valid['연도'].corr(df_valid['연평균기온'])
    
    st.subheader("📊 데이터 분석 요약")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("직선을 만든 해의 개수", f"{len(df_valid)}개")
    with col2:
        st.metric("시작 연도", f"{int(df_valid['연도'].min())}년")
    with col3:
        st.metric("끝 연도", f"{int(df_valid['연도'].max())}년")
    with col4:
        st.metric("상관계수 (r)", f"{correlation:.4f}")
        
    st.markdown("---")
    
    st.subheader("📉 연도별 연평균기온 산점도 및 회귀 직선")
    
    df_valid['예측기온'] = model.predict(X)
    
    fig = px.scatter(
        df_valid, x='연도', y='연평균기온', 
        labels={'연도': '연도', '연평균기온': '연평균기온 (°C)'},
        title="서울 연도별 평균기온 추이 및 회귀선"
    )
    fig.update_traces(marker=dict(size=8, color='royalblue'))
    
    # 회귀 직선 추가
    fig.add_trace(
        go.Scatter(
            x=df_valid['연도'], 
            y=df_valid['예측기온'], 
            mode='lines', 
            name='회귀 직선',
            line=dict(color='crimson', width=3)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔮 연도별 기온 예측하기 (슬라이더)")
    selected_year = st.slider("예측할 연도를 선택하세요 (1900 ~ 2100)", min_value=1900, max_value=2100, value=2026)
    
    predicted_temp = model.predict([[selected_year]])[0]
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 30px; border-radius: 16px; text-align: center; color: white; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
            <h3 style="font-size: 1.25rem; font-weight: 500; margin-bottom: 8px;">{selected_year}년 서울 예상 연평균기온</h3>
            <div style="font-size: 3rem; font-weight: 800;">{predicted_temp:.2f} ℃</div>
        </div>
    """, unsafe_allow_html=True)
