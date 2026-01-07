import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import timedelta

st.set_page_config(page_title="AI Production Forecaster", layout="wide")

st.title("📈 AI-Driven Production Forecasting (DCA)")
st.markdown("---")

# تحميل الموديل
@st.cache_resource
def load_prod_model():
    return joblib.load('production_model.pkl')

model = load_prod_model()

# رفع الملف
uploaded_file = st.file_uploader("Upload Production History (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max()
    
    # التنبؤ للمستقبل (12 شهر)
    future_months = 12
    last_idx = len(df)
    future_indices = np.arange(last_idx, last_idx + future_months).reshape(-1, 1)
    
    future_log_preds = model.predict(future_indices)
    future_preds = np.exp(future_log_preds) - 1
    
    # إنشاء تاريخ المستقبل
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, future_months + 1)]
    
    # عرض النتائج
    st.subheader("🚀 Production Forecast (Next 12 Months)")
    
    fig = go.Figure()
    # التاريخ الحقيقي
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Oil_Rate'], name='Historical Production', mode='lines+markers', line=dict(color='blue')))
    # التنبؤ
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='AI Forecast', mode='lines+markers', line=dict(dash='dash', color='red')))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Oil Rate (BOPD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # مؤشرات اقتصادية
    col1, col2 = st.columns(2)
    with col1:
        total_future_oil = np.sum(future_preds) * 30 # تقدير لعدد الأيام
        st.metric("Estimated Future Production (1 Year)", f"{total_future_oil:,.0f} Barrels")
    with col2:
        decline_per_month = (1 - (future_preds[-1] / future_preds[0])) * 100
        st.metric("Annual Decline Rate (Predicted)", f"{decline_per_month:.1f}%")

st.markdown("---")
st.caption("Developed by Eng. Sulaiman Kudaimi | Production Data Science Division")