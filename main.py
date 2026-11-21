# main.py
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error

# ---------- 한글 폰트 & 스타일 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# ---------- CSV 읽기 함수 ----------
@st.cache_data
def read_csv_file(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ---------- 열 이름 표준화 ----------
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ---------- 추세선 추가 함수 ----------
def add_trendline(fig, x, y, color="red", dash="dash"):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2:
        return
    slope, intercept = np.polyfit(clean["x"], clean["y"], 1)
    x_range = np.linspace(clean["x"].min(), clean["x"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=slope * x_range + intercept,
        mode="lines", name="추세선",
        line=dict(color=color, dash=dash),
        hoverinfo="skip"
    ))

# ---------- 학교명 추출 ----------
def get_school_name(filename):
    if not filename:
        return "업로드된 파일"
    base = os.path.basename(filename)
    name = base.split("_")[0].split("-")[0]
    return name.strip() or "우리 학교"

# ================= 메인 =================
st.title("지상부 길이와 지하부길이의 상관관계")

# 사이드바
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")
st.sidebar.markdown("---")
growth_metric = st.sidebar.selectbox(
    "생육 측정 항목 선택",
    ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부 길이(mm)"],
)

# 데이터 로드
with st.spinner("데이터를 불러오는 중..."):
    env_df = read_csv_file(env_file)
    growth_df = read_csv_file(growth_file)

if env_file is not None:
    env_df = std_cols(env_df)
if growth_file is not None:
    growth_df = std_cols(growth_df)

school_name = get_school_name(env_file.name if env_file else growth_file.name if growth_file else None)

if env_df.empty and growth_df.empty:
    st.info("CSV 파일 2개를 사이드바에서 업로드해 주세요.")
    st.stop()

# ================= 탭 =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 피어슨 상관관계",
    "📈 회귀모델 시각화",
    "📉 개체별 비율",
    "🌱 생육조건 영향 해석",
])

# ---------- 탭 1: 피어슨 상관관계 ----------
with tab1:
    try:
        st.header(f"{school_name} — 피어슨 상관관계 분석")
        if not growth_df.empty:
            cols_available = [c for c in ["지상부 길이", "지하부 길이"] if c in growth_df.columns]
            if len(cols_available) >= 2:
                x_col = "지상부 길이"
                y_col = "지하부 길이"
                clean = growth_df[[x_col, y_col]].dropna()
                if len(clean) >= 2:
                    r, p = stats.pearsonr(clean[x_col], clean[y_col])
                    st.metric("피어슨 상관계수 (r)", f"{r:.4f}")
                    st.metric("p-value", f"{p:.4f}")
                    st.info(
                        "💡 피어슨 상관계수(r)는 −1에서 +1 사이 값으로, "
                        "두 변수가 함께 증가하거나 감소하는지(선형 관계)의 정도를 나타내요. "
                        "0에 가까우면 선형 상관관계가 거의 없다는 뜻이에요."
                    )
                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(go.Scatter(
                        x=clean[x_col], y=clean[y_col],
                        mode="markers", name="개체",
                        marker=dict(size=8, color="royalblue", opacity=0.7),
                        hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>"
                    ))
                    add_trendline(fig, clean[x_col], clean[y_col])
                    fig.update_layout(
                        title=f"{x_col} vs {y_col} (r = {r:.3f})",
                        xaxis_title=x_col, yaxis_title=y_col,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=700, height=500
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("상관관계를 계산할 유효한 데이터가 부족해요.")
            else:
                st.info("지상부 길이·지하부 길이 열이 모두 존재하지 않아 상관관계를 계산할 수 없어요.")
        else:
            st.info("생육 결과 CSV를 업로드해 주세요.")

        if not env_df.empty:
            st.markdown("---")
            st.subheader("환경 변수 간 상관관계")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]
            if len(env_num_cols) >= 2:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) >= 2:
                    corr_mat = env_clean.corr(method="pearson")
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=corr_mat.values,
                        x=corr_mat.columns,
                        y=corr_mat.index,
                        colorscale="RdBu",
                        zmid=0,
                        hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>"
                    ))
                    fig_heatmap.update_layout(
                        title="환경 변수 간 피어슨 상관계수",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=600, height=500
                    )
                    st.plotly_chart(fig_heatmap, width="stretch")
                else:
                    st.info("환경 변수 상관관계 계산에 유효한 데이터가 부족해요.")
            else:
                st.info("상관관계 계산이 가능한 환경 숫자 열이 부족해요.")
        else:
            st.info("환경 CSV를 업로드하면 환경 변수 간 상관관계도 볼 수 있어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------- 탭 2: 회귀모델 ----------
with tab2:
    try:
        st.header(f"{school_name} — 회귀분석 모델 시각화")
        if not growth_df.empty:
            needed = ["잎 수", "지상부 길이", "생중량"]
            if all(c in growth_df.columns for c in needed):
                clean = growth_df[needed].dropna()
                if len(clean) >= 10:
                    X1 = clean[["잎 수"]].values
                    X2 = clean[["잎 수", "지상부 길이"]].values
                    y = clean["생중량"].values

                    # 모델 1
                    model1 = LinearRegression()
                    model1.fit(X1, y)
                    pred1 = model1.predict(X1)
                    r2_1 = model1.score(X1, y)
                    mae_1 = mean_absolute_error(y, pred1)

                    # 모델 2
                    model2 = LinearRegression()
                    model2.fit(X2, y)
                    pred2 = model2.predict(X2)
                    r2_2 = model2.score(X2, y)
                    mae_2 = mean_absolute_error(y, pred2)

                    # 교차검증
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_r2_1 = cross_val_score(model1, X1, y, cv=kf, scoring="r2")
                    cv_mae_1 = -cross_val_score(model1, X1, y, cv=kf, scoring="neg_mean_absolute_error")
                    cv_r2_2 = cross_val_score(model2, X2, y, cv=kf, scoring="r2")
                    cv_mae_2 = -cross_val_score(model2, X2, y, cv=kf, scoring="neg_mean_absolute_error")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("모델 1: 잎 수 → 생중량 (단순 선형회귀)")
                        st.metric("기울기", f"{model1.coef_[0]:.4f}")
                        st.metric("절편", f"{model1.intercept_:.4f}")
                        st.metric("훈련 R²", f"{r2_1:.4f}")
                        st.metric("훈련 MAE", f"{mae_1:.4f} g")
                        st.metric("5겹 CV R²", f"{cv_r2_1.mean():.4f} ± {cv_r2_1.std():.4f}")
                        st.metric("5겹 CV MAE", f"{cv_mae_1.mean():.4f} g")
                        st.info(
                            "💡 R²(결정계수)는 0~1 사이의 값으로, 모델이 데이터 변동을 얼마나 설명하는지를 나타내요. "
                            "MAE(평균절대오차)는 예측값과 실제값의 차이 평균으로, 작을수록 예측이 정확해요."
                        )
                    with col2:
                        st.subheader("모델 2: 잎 수 + 지상부 길이 → 생중량 (다중 선형회귀)")
                        st.metric("잎 수 기울기", f"{model2.coef_[0]:.4f}")
                        st.metric("지상부 길이 기울기", f"{model2.coef_[1]:.4f}")
                        st.metric("절편", f"{model2.intercept_:.4f}")
                        st.metric("훈련 R²", f"{r2_2:.4f}")
                        st.metric("훈련 MAE", f"{mae_2:.4f} g")
                        st.metric("5겹 CV R²", f"{cv_r2_2.mean():.4f} ± {cv_r2_2.std():.4f}")
                        st.metric("5겹 CV MAE", f"{cv_mae_2.mean():.4f} g")

                    # 시각화: 실제 vs 예측
                    st.markdown("---")
                    st.subheader("실제 값 vs 예측 값 비교")
                    fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=[
                        "모델 1 (잎 수)", "모델 2 (잎 수 + 지상부 길이)"
                    ])
                    fig_cmp.add_trace(go.Scatter(
                        x=y, y=pred1, mode="markers",
                        marker=dict(size=7, color="orange", opacity=0.7),
                        name="모델 1",
                        hovertemplate="실제: %{x:.2f}g<br>예측: %{y:.2f}g<extra></extra>"
                    ), row=1, col=1)
                    fig_cmp.add_trace(go.Scatter(
                        x=y, y=pred2, mode="markers",
                        marker=dict(size=7, color="green", opacity=0.7),
                        name="모델 2",
                        hovertemplate="실제: %{x:.2f}g<br>예측: %{y:.2f}g<extra></extra>"
                    ), row=1, col=2)
                    max_val = max(y.max(), pred1.max(), pred2.max()) * 1.05
                    for row_idx, col_idx in [(1, 1), (1, 2)]:
                        fig_cmp.add_trace(go.Scatter(
                            x=[0, max_val], y=[0, max_val],
                            mode="lines", name="완벽 예측선",
                            line=dict(color="gray", dash="dot"),
                            hoverinfo="skip"
                        ), row=row_idx, col=col_idx)
                    fig_cmp.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=900, height=450
                    )
                    fig_cmp.update_xaxes(title_text="실제 생중량 (g)", row=1, col=1)
                    fig_cmp.update_yaxes(title_text="예측 생중량 (g)", row=1, col=1)
                    fig_cmp.update_xaxes(title_text="실제 생중량 (g)", row=1, col=2)
                    fig_cmp.update_yaxes(title_text="예측 생중량 (g)", row=1, col=2)
                    st.plotly_chart(fig_cmp, width="stretch")

                    # 대표 개체 10개 예측
                    st.markdown("---")
                    st.subheader("대표 개체 10개 실제 vs 예측 비교")
                    sample_idx = np.linspace(0, len(clean) - 1, 10, dtype=int)
                    sample = clean.iloc[sample_idx].copy()
                    sample["모델1 예측"] = model1.predict(sample[["잎 수"]].values)
                    sample["모델2 예측"] = model2.predict(sample[["잎 수", "지상부 길이"]].values)
                    display_df = sample[["잎 수", "지상부 길이", "생중량", "모델1 예측", "모델2 예측"]]
                    st.dataframe(display_df, width="stretch", hide_index=True)

                    # 예측용 계산기
                    st.markdown("---")
                    st.subheader("예측 계산기 (학습 데이터 범위 안에서)")
                    input_leaves = st.slider("잎 수(장)", 0, int(clean["잎 수"].max()), 50)
                    input_shoot = st.slider("지상부 길이(mm)", 0, int(clean["지상부 길이"].max()), 40)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        p1 = model1.predict([[input_leaves]])[0]
                        st.metric("모델 1 예측 생중량", f"{p1:.2f} g")
                    with col_b:
                        p2 = model2.predict([[input_leaves, input_shoot]])[0]
                        st.metric("모델 2 예측 생중량", f"{p2:.2f} g")
                    st.info("⚠️ 이 예측은 학습 데이터 범위(잎 수 3~157장, 지상부 10~147mm) 안에서만 참고용이에요. 범위를 벗어난 예측은 신뢰하기 어려워요.")
                else:
                    st.info("회귀모델을 만들기에 데이터가 부족해요 (최소 10개 이상 필요).")
            else:
                st.info("잎 수·지상부 길이·생중량 열이 모두 필요해요.")
        else:
            st.info("생육 결과 CSV를 업로드해 주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------- 탭 3: 개체별 비율 막대그래프 ----------
with tab3:
    try:
        st.header(f"{school_name} — 개체별 지상부·지하부 길이 비율")
        if not growth_df.empty and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
            clean = growth_df[["개체번호", "지상부 길이", "지하부 길이"]].dropna()
            if len(clean) >= 2:
                clean = clean.copy()
                clean["비율(지상부/지하부)"] = clean["지상부 길이"] / clean["지하부 길이"].replace(0, np.nan)
                clean = clean.dropna(subset=["비율(지상부/지하부)"])
                if len(clean) > 0:
                    clean = clean.sort_values("비율(지상부/지하부)")
                    top10 = clean.tail(10)
                    bottom10 = clean.head(10)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("비율이 높은 개체 (지상부가 상대적으로 긴 10개)")
                        fig_top = go.Figure()
                        fig_top.add_trace(go.Bar(
                            x=top10["개체번호"].astype(str),
                            y=top10["비율(지상부/지하부)"],
                            marker_color="steelblue",
                            hovertemplate="개체 %{x}<br>비율: %{y:.2f}<extra></extra>"
                        ))
                        fig_top.update_layout(
                            title="지상부/지하부 길이 비율 상위 10개",
                            xaxis_title="개체번호",
                            yaxis_title="비율 (지상부 ÷ 지하부)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            width=450, height=400
                        )
                        st.plotly_chart(fig_top, width="stretch")

                    with col2:
                        st.subheader("비율이 낮은 개체 (지하부가 상대적으로 긴 10개)")
                        fig_bottom = go.Figure()
                        fig_bottom.add_trace(go.Bar(
                            x=bottom10["개체번호"].astype(str),
                            y=bottom10["비율(지상부/지하부)"],
                            marker_color="coral",
                            hovertemplate="개체 %{x}<br>비율: %{y:.2f}<extra></extra>"
                        ))
                        fig_bottom.update_layout(
                            title="지상부/지하부 길이 비율 하위 10개",
                            xaxis_title="개체번호",
                            yaxis_title="비율 (지상부 ÷ 지하부)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            width=450, height=400
                        )
                        st.plotly_chart(fig_bottom, width="stretch")

                    # 전체 개체 비율 분포
                    st.markdown("---")
                    st.subheader("전체 개체 비율 분포 (히스토그램)")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=clean["비율(지상부/지하부)"],
                        nbinsx=20,
                        marker_color="mediumseagreen",
                        opacity=0.7,
                        hovertemplate="비율 %{x:.2f}<br>개체 수 %{y}<extra></extra>"
                    ))
                    fig_hist.update_layout(
                        title="개체별 지상부/지하부 길이 비율 분포",
                        xaxis_title="비율 (지상부 ÷ 지하부)",
                        yaxis_title="개체 수",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=650, height=400
                    )
                    st.plotly_chart(fig_hist, width="stretch")

                    # 분포 요약
                    st.metric("비율 평균", f"{clean['비율(지상부/지하부)'].mean():.2f}")
                    st.metric("비율 중앙값", f"{clean['비율(지상부/지하부)'].median():.2f}")
                    st.metric("비율 표준편차", f"{clean['비율(지상부/지하부)'].std():.2f}")
                    st.info(
                        "💡 비율이 1보다 크면 지상부가 지하부보다 길다는 뜻, "
                        "1보다 작으면 지하부가 더 길다는 뜻이에요. "
                        "지상부와 지하부 생장이 서로 독립적으로 이뤄질 수 있음을 보여주는 지표예요."
                    )

                    # CSV 다운로드
                    clean_down = clean[["개체번호", "지상부 길이", "지하부 길이", "비율(지상부/지하부)"]].copy()
                    csv_data = clean_down.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 개체별 비율 데이터 다운로드 (CSV)",
                        data=csv_data,
                        file_name=f"{school_name}_개체별_비율.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("비율을 계산할 수 있는 데이터가 부족해요.")
            else:
                st.info("유효한 지상부·지하부 길이 데이터가 부족해요.")
        else:
            st.info("생육 결과 CSV를 업로드하고 지상부 길이·지하부 길이 열이 있는지 확인해 주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------- 탭 4: 생육조건 영향 해석 ----------
with tab4:
    try:
        st.header(f"{school_name} — 생육조건 영향 해석")
        if not env_df.empty:
            st.subheader("1. 스마트팜 환경 요약 통계")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]
            if len(env_num_cols) > 0:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) > 0:
                    stats_df = pd.DataFrame({
                        "평균": env_clean.mean().round(2),
                        "표준편차": env_clean.std().round(2),
                        "최소": env_clean.min().round(2),
                        "최대": env_clean.max().round(2),
                    })
                    stats_df.index.name = "항목"
                    stats_df["변동계수(CV) %"] = (stats_df["표준편차"] / stats_df["평균"] * 100).round(2)
                    st.dataframe(stats_df, width="stretch")
                    st.info(
                        "💡 CV(변동계수)는 표준편차를 평균으로 나눈 백분율이에요. "
                        "단위가 다른 항목끼리도 평균 대비 얼마나 들쭉날쭉한지 비교할 수 있어요."
                    )

                # 시간대별 평균
                if "time" in env_df.columns:
                    st.markdown("---")
                    st.subheader("2. 시간대별 평균 온도·습도")
                    time_clean = env_df[["time", "temperature", "humidity"]].dropna()
                    if len(time_clean) >= 2 and "temperature" in time_clean.columns:
                        time_clean = time_clean.copy()
                        time_clean["hour"] = time_clean["time"].dt.hour
                        hourly = time_clean.groupby("hour").agg(
                            온도_평균=("temperature", "mean"),
                            습도_평균=("humidity", "mean")
                        ).reset_index()
                        if len(hourly) > 0:
                            fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_time.add_trace(go.Bar(
                                x=hourly["hour"], y=hourly["온도_평균"],
                                name="온도(℃)",
                                marker_color="indianred",
                                hovertemplate="시간 %{x}시<br>온도 %{y:.1f}℃<extra></extra>"
                            ))
                            fig_time.add_trace(go.Scatter(
                                x=hourly["hour"], y=hourly["습도_평균"],
                                name="습도(%)",
                                mode="lines+markers",
                                marker=dict(color="royalblue", size=8),
                                line=dict(color="royalblue"),
                                hovertemplate="시간 %{x}시<br>습도 %{y:.1f}%<extra></extra>"
                            ), secondary_y=True)
                            fig_time.update_layout(
                                title="시간대별 평균 온도·습도",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                width=700, height=450
                            )
                            fig_time.update_xaxes(title_text="시간 (시)")
                            fig_time.update_yaxes(title_text="온도 (℃)", row=1, col=1)
                            fig_time.update_yaxes(title_text="습도 (%)", row=1, col=1, secondary_y=True)
                            st.plotly_chart(fig_time, width="stretch")

                # 날짜별 추이
                if "time" in env_df.columns:
                    st.markdown("---")
                    st.subheader("3. 날짜별 평균 온도·습도 추이")
                    date_clean = env_df[["time", "temperature", "humidity"]].dropna()
                    if len(date_clean) >= 2:
                        date_clean = date_clean.copy()
                        date_clean["date"] = date_clean["time"].dt.date
                        daily = date_clean.groupby("date").agg(
                            온도_평균=("temperature", "mean"),
                            온도_std=("temperature", "std"),
                            습도_평균=("humidity", "mean"),
                            습도_std=("humidity", "std")
                        ).reset_index()
                        if len(daily) > 0:
                            fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
                            x_dates = [str(d) for d in daily["date"]]
                            fig_daily.add_trace(go.Scatter(
                                x=x_dates, y=daily["온도_평균"],
                                name="온도 평균",
                                mode="lines+markers",
                                marker=dict(size=5, color="indianred"),
                                line=dict(color="indianred"),
                                hovertemplate="날짜 %{x}<br>온도 %{y:.1f}℃<extra></extra>"
                            ))
                            temp_upper = daily["온도_평균"] + daily["온도_std"].fillna(0)
                            temp_lower = daily["온도_평균"] - daily["온도_std"].fillna(0)
                            fig_daily.add_trace(go.Scatter(
                                x=x_dates + x_dates[::-1],
                                y=list(temp_upper) + list(temp_lower)[::-1],
                                fill="toself", fillcolor="indianred",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="온도 변동 범위",
                                opacity=0.2,
                                hoverinfo="skip"
                            ))
                            fig_daily.add_trace(go.Scatter(
                                x=x_dates, y=daily["습도_평균"],
                                name="습도 평균",
                                mode="lines+markers",
                                marker=dict(size=5, color="royalblue"),
                                line=dict(color="royalblue"),
                                hovertemplate="날짜 %{x}<br>습도 %{y:.1f}%<extra></extra>"
                            ), secondary_y=True)
                            fig_daily.update_layout(
                                title="날짜별 평균 온도·습도 추이",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                width=800, height=450
                            )
                            fig_daily.update_xaxes(title_text="날짜")
                            fig_daily.update_yaxes(title_text="온도 (℃)", row=1, col=1)
                            fig_daily.update_yaxes(title_text="습도 (%)", row=1, col=1, secondary_y=True)
                            st.plotly_chart(fig_daily, width="stretch")

            else:
                st.info("환경 데이터에서 숫자 열을 찾을 수 없어요.")
        else:
            st.info("환경 CSV를 업로드해 주세요.")

        st.markdown("---")
        st.subheader("4. 생육조건 영향 해석 요약")
        st.markdown("""
        <div style="font-size: 0.95rem; line-height: 1.6;">
        <b>📌 이 분석의 한계</b><br>
        생육결과데이터에는 개체별 생육 측정일이 없어 환경데이터 측정 시점과 직접 연결이 어려워요.
        개체 위치·처리구 정보도 없어, 측정된 환경 요인만으로 개체별 차이를 설명하기엔 제약이 커요.
        
        <b>📌 환경 데이터에서 눈에 띄는 점</b><br>
        - 온도·습도는 대체로 안정적이었으나 특정 시점에 급상승 구간이 있었어요.<br>
        - EC는 변동 폭이 컸고, 낮은 그룹과 높은 그룹으로 나뉘는 양상이 보였어요.<br>
        - pH는 후반에 4~5 수준으로 급락하는 구간이 반복되었어요.<br>
        
        <b>📌 개체별 차이 가능성</b><br>
        같은 시설 안에서도 국소적 광량·통풍·배양액 공급 차이, 초기 생육 상태, 측정 시점 차이 등이
        개체별 생육 차이로 이어질 수 있어요. 측정값이 매우 작은 개체들은 생장 초기 단계이거나
        측정 오류 가능성도 고려할 수 있어요.
        
        <b>📌 제안</b><br>
        - 개체별 생육 측정일(날짜·시각)을 개체번호와 함께 기록<br>
        - 개체 위치·처리구·광 조건 정보 함께 기록<br>
        - 환경 급변동·이상치 구간에 대한 현장 메모 남기기<br>
        - 광량·배지·배양액 조성·병해충 관찰 등 추가 변수 도입
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
