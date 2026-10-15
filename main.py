import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌡️ 서울 기온 예측기 (Seoul Temperature Predictor)")
st.markdown("제공된 서울 기온 데이터를 분석하여 연도별 평균기온 추세를 살펴보고, 선형 회귀 분석을 통해 미래 기온을 예측합니다.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
    except Exception as e:
        # Fallback encoding if utf-8 fails
        df = pd.read_csv(url, encoding='cp949')
    
    # Clean column names just in case
    df.columns = df.columns.str.strip()
    
    # Convert '날짜' to datetime
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df['연도'] = df['날짜'].dt.year
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 기준 기간: 2025년까지, 그 이후 데이터 및 관측일이 300일이 안 되는 해는 제외
df_filtered = df_raw[(df_raw['연도'] <= 2025)].copy()

# 연도별 관측일수 및 평균기온 계산
annual_stats = df_filtered.groupby('연도').agg(
    관측일수=('평균기온', 'count'),
    연평균기온=('평균기온', 'mean')
).reset_index()

# 관측일이 300일 이상인 해만 필터링
valid_annual = annual_stats[annual_stats['관측일수'] >= 300].dropna(subset=['연평균기온'])

if valid_annual.empty:
    st.warning("조건에 맞는 유효한 연도별 데이터가 없습니다.")
    st.stop()

start_year = int(valid_annual['연도'].min())
end_year = int(valid_annual['연도'].max())
num_years = int(valid_annual['연도'].count())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("직선을 만든 해의 개수", f"{num_years}년")
with col2:
    st.metric("분석 시작 연도", f"{start_year}년")
with col3:
    st.metric("분석 끝 연도", f"{end_year}년")

st.markdown("---")

X = valid_annual['연도'].values
y = valid_annual['연평균기온'].values

# 선형 회귀 분석 (Least Squares)
slope, intercept = np.polyfit(X, y, 1)
correlation = np.corrcoef(X, y)[0, 1]

# 예측선 생성 데이터
valid_annual['회귀예측기온'] = slope * valid_annual['연도'] + intercept

st.subheader("📈 연도별 평균기온 산점도 및 회귀 직선")

fig = px.scatter(
    valid_annual, 
    x='연도', 
    y='연평균기온', 
    title="서울 연평균기온 추세 및 회귀 분석",
    labels={'연도': '연도', '연평균기온': '연평균기온 (°C)'},
    template='plotly_white'
)

# 산점도 스타일 수정
fig.update_traces(marker=dict(size=8, color='#3b82f6', opacity=0.8))

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=valid_annual['연도'],
        y=valid_annual['회귀예측기온'],
        mode='lines',
        name=f'회귀 직선 (y = {slope:.4f}x + {intercept:.2f})',
        line=dict(color='#ef4444', width=3)
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 상관계수 및 회귀식 정보 표시
st.markdown(f"""
<div class="info-box">
    <strong>📊 통계 분석 결과</strong><br>
    • <strong>상관계수 (Correlation):</strong> {correlation:.4f} ({'강한 양의 상관관계' if correlation > 0.5 else '상관관계 낮음'})<br>
    • <strong>회귀 방정식:</strong> 연평균기온 = {slope:.4f} × 연도 + {intercept:.2f}<br>
    • <em>해석:</em> 시간이 지남에 따라 서울의 연평균기온이 매년 약 {slope*10:.3f}°C (10년 기준) 비율로 변화하는 경향을 보여줍니다.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🔮 연도별 예상 기온 조회하기")
st.markdown("슬라이더를 움직여 특정 연도를 선택하면 해당 연도의 예상 기온을 확인하실 수 있습니다.")

selected_year = st.slider(
    "예측할 연도를 선택하세요 (1900 ~ 2100)",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)

predicted_temp = slope * selected_year + intercept

col_pred1, col_pred2 = st.columns([1, 2])

with col_pred1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; color:white; font-size: 1.2rem;">{selected_year}년 예상 평균기온</h3>
        <h1 style="margin:10px 0 0 0; color:white; font-size: 2.8rem;">{predicted_temp:.2f}°C</h1>
    </div>
    """, unsafe_allow_html=True)

with col_pred2:
    st.markdown(f"""
    #### 💡 예측 정보
    - **선택 연도:** {selected_year}년
    - **회귀식 계산값:** {slope:.4f} × {selected_year} + {intercept:.2f}
    - **참고사항:** 본 예측값은 {start_year}년부터 {end_year}년까지의 과거 관측 데이터 기반 선형 회귀 추세선을 바탕으로 산출된 수치입니다. 기후 변화 및 환경 요인에 따라 실제 기온은 달라질 수 있습니다.
    """)

st.markdown("---")
with st.expander("📦 필요한 라이브러리 목록 (requirements.txt)"):
    st.code("""streamlit
pandas
numpy
plotly
""", language="text")
