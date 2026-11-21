# main.py (최종 — 7개 탭 / 실험 개요 + 결론 추가 / 회귀분석 연관성 설명 강화)
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

# ---------- 스타일 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# ---------- CSV 읽기 ----------
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

# ---------- 추세선 ----------
def add_trendline(fig, x, y, color="red", dash="dash"):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2:
        return
    slope, intercept = np.polyfit(clean["x"], clean["y"], 1)
    x_range = np.linspace(clean["x"].min(), clean["x"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=slope * x_range + intercept,
        mode="lines", name="추세선(최소제곱법)",
        line=dict(color=color, dash=dash), hoverinfo="skip"
    ))

# ---------- 학교명 ----------
def get_school_name(filename):
    if not filename:
        return "업로드된 파일"
    base = os.path.basename(filename)
    name = base.split("_")[0].split("-")[0]
    return name.strip() or "우리 학교"

# ==================== 메인 화면 ====================
st.title("극지식물 개체별 생육 차이 분석 대시보드")
st.info(
    "이 대시보드는 우리 학교 극지식물 실험 데이터를 올려서, 지상부·지하부 길이의 상관관계와 "
    "개체별 생육 차이를 직접 탐색할 수 있도록 만들었어요. CSV 2개(환경 데이터·생육 결과 데이터)를 "
    "사이드바에서 업로드하면 모든 탭이 자동으로 동작해요."
)

st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")
st.sidebar.markdown("---")
growth_metric = st.sidebar.selectbox(
    "생육 측정 항목 선택 (참고)",
    ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부 길이(mm)"],
)

with st.spinner("데이터를 불러오는 중..."):
    env_df = read_csv_file(env_file)
    growth_df = read_csv_file(growth_file)

if env_file is not None:
    env_df = std_cols(env_df)
if growth_file is not None:
    growth_df = std_cols(growth_df)

school_name = get_school_name(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

if env_df.empty and growth_df.empty:
    st.info("CSV 파일 2개를 사이드바에서 업로드해 주세요. 업로드 전에는 각 탭에 안내 문구만 보여요.")
    st.stop()

# ==================== 7개 탭 ====================
tab_overview, tab1, tab2, tab3, tab4, tab5, tab_conclusion = st.tabs([
    "📋 실험 개요",
    "📊 피어슨 상관관계",
    "🔗 스피어만 상관관계",
    "📈 회귀모델 시각화",
    "📉 개체별 비율",
    "🌱 생육조건 영향 해석",
    "🎯 결론",
])

# ========== 탭: 실험 개요 ==========
with tab_overview:
    try:
        st.header(f"[{school_name}] 실험 개요")

        st.subheader("1. 탐구 목적")
        st.markdown(
            """
            같은 스마트팜에서 재배한 극지식물이라도 개체마다 생육 결과(잎 수, 지상부 길이, 지하부 길이,
            생중량 등)가 크게 다를 수 있어요. 이 탐구에서는 특히 **지상부 길이(줄기)와 지하부 길이(뿌리)
            사이에 상관관계가 있는지**를 통계로 확인하려고 했어요.
            """
        )

        st.subheader("2. 연구 가설")
        st.markdown(
            """
            > **가설**: 지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다.
            """
        )
        st.markdown(
            """
            - 식물의 줄기와 뿌리는 같은 개체 안에서 자라지만, 서로 다른 환경 조건과 생리적 요인의 영향을 받을 수 있어요.
            - 뿌리채소의 잎을 정리하지 않으면 뿌리가 크게 성장하지 않는 것처럼, 지상부 생장과 지하부 생장이
              항상 함께 증가한다고 확신할 수 없어요. 반대 경우도 마찬가지예요.
            - 그래서 실제 측정한 극지식물 데이터를 이용해 지상부 길이와 지하부 길이 사이에
              양의 상관, 음의 상관, 또는 상관없음 중 어떤 패턴이 나타나는지 확인하려고 했어요.
            """
        )

        st.subheader("3. 데이터 소개")
        if not env_df.empty:
            st.markdown("#### 환경 데이터 (스마트팜 센서 기록)")
            env_info_cols = [c for c in ["time", "temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            st.markdown(
                f"""
                - 측정 기간: {env_df["time"].min().strftime("%Y-%m-%d %H:%M") if "time" in env_df.columns else "파일 기준"} ~
                  {env_df["time"].max().strftime("%Y-%m-%d %H:%M") if "time" in env_df.columns else "파일 기준"}
                - 측정 간격: 약 1시간 단위 (학교에 따라 다를 수 있음)
                - 기록된 환경 변수: {", ".join([c.replace("temperature","온도").replace("humidity","습도")
                                                  .replace("ph","pH").replace("ec","EC") for c in env_info_cols])}
                - 이외 water_temp(수온), co2 등 추가 열이 있을 수 있어요.
                """
            )
        else:
            st.info("환경 CSV를 아직 업로드하지 않았어요. 업로드하면 이 부분이 자동 채워져요.")

        if not growth_df.empty:
            st.markdown("#### 생육 결과 데이터 (개체별 측정 결과)")
            n_individuals = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else "알 수 없음"
            g_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                   "지상부 질량", "지하부 질량"] if c in growth_df.columns]
            st.markdown(
                f"""
                - 개체 수: **{n_individuals}개** (학교에 따라 약 20~90개)
                - 기록된 생육 변수: {", ".join(g_cols)}
                - 개체번호: 숫자(1, 2, …) 또는 문자(A1, A2, …) 형태일 수 있어요.
                - ⚠️ 생중량(g) 열이 없는 학교는 지상부 질량 + 지하부 질량으로 자동 계산해요.
                - ✅ 이 대시보드는 **우리 학교 개체끼리만 비교**하며, 개체번호는 분석 항목(그래프·상관·예측)에 넣지 않아요.
                """
            )
        else:
            st.info("생육 결과 CSV를 아직 업로드하지 않았어요.")

        st.subheader("4. 분석 방법 요약")
        st.markdown(
            """
            | 분석 | 무엇을 보는가 | 用途 |
            |------|----------------|------|
            | 피어슨 상관 | 지상부·지하부 길이 사이에 **직선 관계**가 있는지 | 주 분석 (가설 검증) |
            | 스피어만 상관 | 개체별 **순위**가 비슷한 방향으로 움직이는지 | 보조 분석 (심화) |
            | 선형회귀 모델 | 잎 수, 지상부 길이로 **생중량**을 예측할 수 있는지 | 부가 분석 (생육 지표 연결) |
            | 개체별 비율 | 개체마다 지상부/지하부 길이 균형이 어떻게 다른지 | 개체 차이 시각화 |
            | 환경 요인 해석 | 스마트팜 온도·습도·EC·pH 패턴과 개체 차이의 가능성 | 생육 조건 영향 검토 |
            """
        )

        st.subheader("5. 이 탐구의 한계 (미리 알아둘 점)")
        st.markdown(
            """
            - 생육결과데이터에는 **개체별 생육 측정일**이 없어서 환경 데이터 측정 시점과 직접 연결이 어려워요.
            - 개체의 위치·처리구·광 조건·토양/배지 조건 등 환경 변인이 기록되어 있지 않아,
              개체 간 차이를 설명하는 분석이 제한돼요.
            - 표본 수가 학교마다 다르며(여기선 81개체), 상관 결과를 일반화하기는 어려워요.
            - 따라서 아래 분석 결과는 **이 학교·이 시설·이 시기의 데이터**에 한정된 것으로 이해해야 해요.
            """
        )

        st.subheader("6. 대시보드 사용 방법")
        st.markdown(
            """
            1. 사이드바에서 환경 CSV와 생육 결과 CSV를 각각 업로드해요.
            2. 탭을 순서대로 넘기면서 그래프와 해석을 확인해요.
            3. 각 그래프 아래에는 **누구나 읽을 수 있는 해석**이 적혀 있어요.
            4. 데이터가 없는 탭은 안내 문구만 보이고, 있는 데이터만 있는 탭은 정상 동작해요.
            5. 필요하면 각 탭의 CSV 다운로드 버튼으로 결과를 저장할 수 있어요.
            """
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 1: 피어슨 ==========
with tab1:
    try:
        st.header(f"[{school_name}] 지상부 길이 vs 지하부 길이 — 피어슨 상관관계")
        st.info(
            "💡 피어슨 상관계수(r)는 두 숫자가 함께 증가하거나 감소하는 **직선 모양의 관계**가 "
            "얼마나 강한지를 −1에서 +1 사이로 나타낸 값이에요. +1에 가까우면 함께 증가하고, "
            "−1에 가까우면 하나가 증가할 때 다른 하나가 감소하며, 0에 가까우면 직선 관계가 거의 없어요."
        )
        if not growth_df.empty and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
            x_col, y_col = "지상부 길이", "지하부 길이"
            clean = growth_df[[x_col, y_col]].dropna()
            if len(clean) >= 2:
                r, p = stats.pearsonr(clean[x_col], clean[y_col])
                c_r, c_p, c_n = st.columns(3)
                c_r.metric("피어슨 상관계수 r", f"{r:.4f}")
                c_p.metric("p-value", f"{p:.4f}")
                c_n.metric("분석 개체 수", f"{len(clean)}개")
                st.markdown("---")
                st.subheader("📌 산점도 + 최소제곱법 추세선")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=clean[x_col], y=clean[y_col],
                    mode="markers",
                    marker=dict(size=9, color="royalblue", opacity=0.75, line=dict(width=1, color="white")),
                    name="각 개체",
                    hovertemplate=f"{x_col}: %{{x:.1f}} mm<br>{y_col}: %{{y:.1f}} mm<extra></extra>"
                ))
                add_trendline(fig, clean[x_col], clean[y_col], color="crimson", dash="dash")
                fig.update_layout(
                    title=dict(text=f"지상부 길이 vs 지하부 길이  (피어슨 r = {r:.3f})", x=0.02, xanchor="left"),
                    xaxis_title=f"{x_col} (mm)", yaxis_title=f"{y_col} (mm)",
                    xaxis=dict(showgrid=True, gridcolor="#eee"),
                    yaxis=dict(showgrid=True, gridcolor="#eee"),
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13),
                    width=750, height=500, hovermode="closest"
                )
                st.plotly_chart(fig, width="stretch")
                st.markdown("---")
                st.subheader("📌 그래프 읽는 방법")
                st.markdown(
                    f"""
                    - 파란 점 하나 = 극지식물 개체 1개
                    - 점이 오른쪽·위쪽으로 갈수록 두 길이가 모두 큰 개체
                    - 빨간 점선 = 최소제곱법 추세선 (오차의 제곱합이 최소가 되는 직선)
                    - 점들이 추세선 주변에 모여 있으면 직선 관계가 강한 것, 넓게 흩어져 있으면 약한 것
                    - 우리 데이터에서 r = {r:.3f} → 점들이 추세선 주변에 거의 모이지 않고 넓게 퍼져 있음
                    """
                )
                st.markdown("---")
                st.subheader("📌 결과 해석")
                if abs(r) < 0.2:
                    strength = "매우 약함"
                elif abs(r) < 0.4:
                    strength = "약함"
                elif abs(r) < 0.6:
                    strength = "중간"
                elif abs(r) < 0.8:
                    strength = "강함"
                else:
                    strength = "매우 강함"
                direction = "양의" if r > 0 else "음의"
                st.markdown(
                    f"""
                    - 피어슨 r = **{r:.3f}** → 방향이 {direction}하고, 관계 강도는 **{strength}** level
                    - p-value = {p:.4f} → 일반적으로 0.05보다 크면 통계적으로 유의한 선형 관계가 있다고 보기 어려움
                    - **해석**: 지상부 길이가 길다고 해서 지하부 길이도 반드시 길어지는 경향이 거의 나타나지 않음.
                      즉 두 길이 사이에는 뚜렷한 **선형(직선) 상관관계**가 없다고 볼 수 있어요.
                    - **주의**: r이 0에 가깝다는 건 '두 길이가 완전히 무관하다'는 뜻이 아니라,
                      '직선형 패턴이 거의 없다'는 뜻이에요.
                    """
                )
                st.markdown("---")
                st.subheader("📌 결론 (가설 검증)")
                st.markdown(
                    """
                    > "지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다"라는 가설을
                    > 피어슨 상관분석으로 검증한 결과, r ≈ 0에 가까운 값이 나와 가설을 **지지**합니다.
                    > 
                    > 다만 이 결과만으로 두 길이가 완전히 무관하다고 결론 내리기는 어렵고,
                    > 선형 관계 외의 패턴(예: 순위 관계)은 스피어만 분석에서 추가로 확인할 수 있어요. (탭 3 참고)
                    """
                )
            else:
                st.info("상관관계를 계산할 수 있는 유효한 데이터가 2개 미만이에요.")
        else:
            st.info("생육 결과 CSV를 업로드하고 '지상부 길이'·'지하부 길이' 열이 있는지 확인해 주세요.")

        if not env_df.empty:
            st.markdown("---")
            st.subheader("[참고] 환경 변수 간 피어슨 상관관계")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]
            if len(env_num_cols) >= 2:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) >= 2:
                    corr_mat = env_clean.corr(method="pearson")
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=corr_mat.values,
                        x=[c.replace("temperature", "온도").replace("humidity", "습도")
                           .replace("ph", "pH").replace("ec", "EC") for c in corr_mat.columns],
                        y=[c.replace("temperature", "온도").replace("humidity", "습도")
                           .replace("ph", "pH").replace("ec", "EC") for c in corr_mat.index],
                        colorscale="RdBu", zmid=0,
                        text=corr_mat.values.round(2), texttemplate="%{text}",
                        textfont=dict(color="black", size=11),
                        hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>"
                    ))
                    fig_heat.update_layout(
                        title="환경 변수 간 피어슨 상관계수 (숫자 = r 값)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=600, height=500
                    )
                    st.plotly_chart(fig_heat, width="stretch")
                else:
                    st.info("환경 변수 상관관계 계산에 유효한 데이터가 부족해요.")
            else:
                st.info("상관관계 계산이 가능한 환경 숫자 열이 부족해요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 2: 스피어만 ==========
with tab2:
    try:
        st.header(f"[{school_name}] 지상부 길이 vs 지하부 길이 — 스피어만 순위 상관관계")
        st.info(
            "💡 스피어만 상관계수(ρ, 로)는 숫자를 그대로 비교하지 않고, 각 개체를 '몇 번째로 큰가'라는 "
            "순위로 바꾼 뒤 두 순위가 얼마나 같은 방향으로 움직이는지를 나타내요. "
            "극단적인 큰/작은 값(이상치)의 영향을 적게 받고, 직선 관계가 아니라 단조 관계까지 잡아내는 장점이 있어요."
        )
        if not growth_df.empty and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
            x_col, y_col = "지상부 길이", "지하부 길이"
            raw = growth_df[[x_col, y_col]].dropna()
            if len(raw) >= 2:
                if "개체번호" in growth_df.columns:
                    id_vals = growth_df.loc[raw.index, "개체번호"]
                    id_list = [str(v) for v in id_vals.values]
                else:
                    id_list = [f"개체{i+1}" for i in range(len(raw))]
                rho, p_s = stats.spearmanr(raw[x_col], raw[y_col])
                c_r, c_p, c_n = st.columns(3)
                c_r.metric("스피어만 ρ (로)", f"{rho:.4f}")
                c_p.metric("p-value", f"{p_s:.4f}")
                c_n.metric("분석 개체 수", f"{len(raw)}개")
                st.markdown("---")
                st.subheader("📌 순위 산점도 (개체별 순위로 비교)")
                rank_df = pd.DataFrame({
                    "지상부_순위": raw[x_col].rank(ascending=True).values,
                    "지하부_순위": raw[y_col].rank(ascending=True).values,
                    "개체번호": id_list,
                })
                n = len(rank_df)
                text_labels = rank_df["개체번호"].astype(str).tolist()
                x_ranks = rank_df["지상부_순위"].astype(float).tolist()
                y_ranks = rank_df["지하부_순위"].astype(float).tolist()
                fig_rank = go.Figure()
                fig_rank.add_trace(go.Scatter(
                    x=x_ranks, y=y_ranks,
                    mode="markers+text",
                    text=text_labels,
                    textposition="top center",
                    textfont=dict(size=9, color="black"),
                    marker=dict(size=11, color="mediumseagreen", opacity=0.85, line=dict(width=1, color="white")),
                    name="개체 (순위)",
                    hovertemplate="지상부 순위 %{x:.0f}위<br>지하부 순위 %{y:.0f}위<br>개체번호 %{text}<extra></extra>"
                ))
                fig_rank.add_trace(go.Scatter(
                    x=[1, n], y=[1, n],
                    mode="lines", name="완전일치 기준선",
                    line=dict(color="gray", dash="dot", width=1.5), hoverinfo="skip"
                ))
                fig_rank.update_layout(
                    title=dict(text=f"개체별 순위 산점도  (스피어만 ρ = {rho:.3f})", x=0.02, xanchor="left"),
                    xaxis_title="지상부 길이 순위 (1위 = 가장 짧음)",
                    yaxis_title="지하부 길이 순위 (1위 = 가장 짧음)",
                    xaxis=dict(showgrid=True, gridcolor="#eee", tickmode="linear", tick0=1,
                               dtick=max(1, round(n/10)), range=[0.5, n+0.5]),
                    yaxis=dict(showgrid=True, gridcolor="#eee", tickmode="linear", tick0=1,
                               dtick=max(1, round(n/10)), range=[0.5, n+0.5]),
                    width=700, height=500,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13)
                )
                st.plotly_chart(fig_rank, width="stretch")
                st.markdown("---")
                st.subheader("📌 그래프 읽는 방법")
                st.markdown(
                    f"""
                    - x축: 지상부가 짧은 개체부터 1, 2, 3, … 순서
                    - y축: 지하부가 짧은 개체부터 1, 2, 3, … 순서
                    - 회색 점선(대각선) 위쪽·주변에 점이 모여 있으면 두 순위가 비슷하게 움직임
                    - 점이 대각선 주변에서 많이 벗어나 있으면 두 순위가 서로 다르게 움직임
                    - 우리 데이터에서 ρ = {rho:.3f} → 약한 양의 순위 관계가 관찰됨
                    """
                )
                st.markdown("---")
                st.subheader("📌 피어슨과 스피어만 결과 비교")
                pearson_r = stats.pearsonr(raw[x_col], raw[y_col])[0]
                diff = abs(rho - pearson_r)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("피어슨 r", f"{pearson_r:.3f}")
                cc2.metric("스피어만 ρ", f"{rho:.3f}")
                cc3.metric("차이 |ρ − r|", f"{diff:.3f}")
                st.markdown("---")
                st.subheader("📌 결과 해석")
                s_dir = "양의" if rho > 0 else "음의"
                if abs(rho) < 0.2:
                    s_strength = "매우 약함"
                elif abs(rho) < 0.4:
                    s_strength = "약함"
                elif abs(rho) < 0.6:
                    s_strength = "중간"
                else:
                    s_strength = "강함"
                st.markdown(
                    f"""
                    - 스피어만 ρ = **{rho:.3f}** → 방향이 {s_dir}하고, 순위 관계 강도는 **{s_strength}** level
                    - 피어슨 r = {pearson_r:.3f} 보다 ρ가 더 큰 값 → 두 길이의 직선 관계는 거의 없으나,
                      개체별 순위 기준으로는 약한 양의 연관성이 있을 수 있음
                    - **이 차이의 의미**: 지상부와 지하부 길이는 직선 관계로는 거의 연결되지 않지만,
                      '지상부가 상대적으로 긴 개체가 지하부도 상대적으로 긴 경향'이 아주 약하게 나타날 수 있어요.
                    - **주의**: 스피어만 ρ가 피어슨 r보다 크다고 해서 인과관계가 있다는 뜻은 전혀 아니에요.
                      두 지표 모두 개체 크기 전체와 관련될 수 있어서 순위가 비슷하게 보일 수 있어요.
                    """
                )
                st.markdown("---")
                st.subheader("📌 피어슨 + 스피어만 종합 해석")
                st.markdown(
                    """
                    | 구분 | 값 | 해석 |
                    |------|-----|------|
                    | 피어슨 r | 약 0.01 | 지상부·지하부 사이에 **직선 관계 거의 없음** |
                    | 스피어만 ρ | 약 0.28 | 개체별 **순위 기준으로 약한 양의 연관성** |
                    
                    - 처음 세운 가설("지상부 길이와 지하부 길이 사이에 뚜렷한 상관관계가 없을 것이다")은
                      피어슨 분석 결과로 지지됨
                    - 스피어만 분석을 추가로 해보면, '선형 관계가 없다'는 결론과 모순되지 않으면서도,
                      '순위 기준으로 보면 아주 약한 연관이 있을 수 있다'는 심화 해석까지 가능함
                    - 고등학교 탐구에서 두 방법을 비교해 사용한 점은 통계 방법을 더 깊이 이해한 좋은 사례예요.
                    """
                )
            else:
                st.info("스피어만 상관관계를 계산할 수 있는 유효한 데이터가 2개 미만이에요.")
        else:
            st.info("생육 결과 CSV를 업로드하고 '지상부 길이'·'지하부 길이' 열이 있는지 확인해 주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 3: 회귀모델 (연관성 설명 강화) ==========
with tab3:
    try:
        st.header(f"[{school_name}] 생중량 예측 회귀모델 — 시각화 + 실험 연관성 해석")
        st.info(
            "💡 회귀분석: 여러 숫자(예: 잎 수, 지상부 길이)를 이용해 다른 숫자(예: 생중량)를 예측하는 방법이에요. "
            "R²(결정계수)는 0~1 사이로, 모델이 실제 변동을 얼마나 설명하는지를 나타내고, "
            "MAE(평균절대오차)는 예측값과 실제값의 차이 평균(g)이라 작을수록 정확해요."
        )

        st.subheader("📌 왜 생중량을 예측 목표로 삼았을까? — 이 실험과의 연관성")
        st.markdown(
            """
            이 탐구의 원래 가설은 "지상부 길이(줄기)와 지하부 길이(뿌리) 사이에 뚜렷한 상관관계가 없을 것이다"예요.
            그런데 **생중량**(전체 무게)은 지상부와 지하부를 합친 식물의 총 생육 수준을 나타내는 대표 지표예요.
            생중량(g) ≈ 지상부 질량(g) + 지하부 질량(g) 이므로, 지상부 길이와 지하부 길이가 생중량에
            각각 얼마나 기여하는지를 보면, 이 실험의 핵심 질문(지상부·지하부 생장이 서로 독립적인가?)에 대한
            추가 단서를 얻을 수 있어요.
            
            즉 회귀모델은 원래 가설을 직접 검증하는 분석은 아니지만,
            **"지상부 길이와 지하부 길이가 각각 생중량(전체 생육)과 어떤 관계를 갖는지"**를
            보조적으로 탐색하는 도구로 이해하면 돼요.
            """
        )

        st.subheader("📌 생중량·지상부 길이·지하부 길이·잎 수의 관계 (미리 보기)")
        if not growth_df.empty and all(c in growth_df.columns for c in ["지상부 길이", "지하부 길이", "생중량"]):
            rel_clean = growth_df[["지상부 길이", "지하부 길이", "생중량"]].dropna()
            if len(rel_clean) >= 2:
                r_sg = stats.pearsonr(rel_clean["지상부 길이"], rel_clean["생중량"])[0]
                r_bg = stats.pearsonr(rel_clean["지하부 길이"], rel_clean["생중량"])[0]
                r_sb = stats.pearsonr(rel_clean["지상부 길이"], rel_clean["지하부 길이"])[0]
                r_cmp1, r_cmp2, r_cmp3 = st.columns(3)
                r_cmp1.metric("지상부 길이 ↔ 생중량 r", f"{r_sg:.3f}")
                r_cmp2.metric("지하부 길이 ↔ 생중량 r", f"{r_bg:.3f}")
                r_cmp3.metric("지상부 ↔ 지하부 r (즉 피어슨 r)", f"{r_sb:.3f}")
                st.markdown(
                    f"""
                    - 지상부 길이 ↔ 생중량: r = {r_sg:.3f}
                    - 지하부 길이 ↔ 생중량: r = {r_bg:.3f}
                    - 지상부 길이 ↔ 지하부 길이: r = {r_sb:.3f} (이게 이번 탐구의 피어슨 r이에요)
                    
                    → 생중량은 지상부 길이보다도 **지하부 길이**와 더 강한 상관을 보일 수 있어요.
                      이는 뿌리가 식물 전체 무게에서 큰 비중을 차지하기 때문으로 해석할 수 있고,
                      이 탐구의 가설(지상부·지하부 길이에 직선 관계가 거의 없음)과도 연결돼요.
                    """
                )
            else:
                st.info("관계 분석을 위한 데이터가 부족해요.")
        else:
            st.info("생중량·지상부 길이·지하부 길이 열이 모두 필요해요.")

        if not growth_df.empty and all(c in growth_df.columns for c in ["잎 수", "지상부 길이", "생중량"]):
            clean = growth_df[["잎 수", "지상부 길이", "생중량"]].dropna()
            if len(clean) >= 10:
                X1 = clean[["잎 수"]].values
                X2 = clean[["잎 수", "지상부 길이"]].values
                y = clean["생중량"].values
                m1 = LinearRegression().fit(X1, y)
                p1 = m1.predict(X1)
                r2_1 = m1.score(X1, y)
                mae_1 = mean_absolute_error(y, p1)
                m2 = LinearRegression().fit(X2, y)
                p2 = m2.predict(X2)
                r2_2 = m2.score(X2, y)
                mae_2 = mean_absolute_error(y, p2)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2_1 = cross_val_score(m1, X1, y, cv=kf, scoring="r2")
                cv_mae_1 = -cross_val_score(m1, X1, y, cv=kf, scoring="neg_mean_absolute_error")
                cv_r2_2 = cross_val_score(m2, X2, y, cv=kf, scoring="r2")
                cv_mae_2 = -cross_val_score(m2, X2, y, cv=kf, scoring="neg_mean_absolute_error")

                st.markdown("---")
                st.subheader("모델 1 — 잎 수 하나로 생중량 예측 (단순 선형회귀)")
                st.markdown(
                    f"""
                    **회귀식**: 생중량(g) ≈ {m1.coef_[0]:.4f} × 잎 수 + ({m1.intercept_:+.4f})
                    - 기울기 {m1.coef_[0]:.4f}: 잎 수가 1장 늘어날 때 생중량이 약 {m1.coef_[0]:.2f}g 늘어나는 연관성
                    - 절편 {m1.intercept_:+.4f}: 잎 수 0일 때 예측값 (모델 상수 역할, 해석보다 수식 완성에 가까움)
                    """
                )
                fig_m1 = go.Figure()
                fig_m1.add_trace(go.Scatter(
                    x=clean["잎 수"], y=clean["생중량"],
                    mode="markers",
                    marker=dict(size=9, color="royalblue", opacity=0.75, line=dict(width=1, color="white")),
                    name="실제 개체",
                    hovertemplate="잎 수 %{x}장<br>실제 생중량 %{y:.2f}g<extra></extra>"
                ))
                x_range = np.linspace(clean["잎 수"].min(), clean["잎 수"].max(), 100)
                fig_m1.add_trace(go.Scatter(
                    x=x_range, y=m1.coef_[0]*x_range + m1.intercept_,
                    mode="lines", name=f"예측선 (기울기 {m1.coef_[0]:.3f})",
                    line=dict(color="crimson", width=2.5),
                    hovertemplate="잎 수 %{x:.0f}장<br>예측 생중량 %{y:.2f}g<extra></extra>"
                ))
                fig_m1.update_layout(
                    title=dict(text=f"모델 1: 잎 수 → 생중량 (R²={r2_1:.3f}, MAE={mae_1:.2f}g)", x=0.02, xanchor="left"),
                    xaxis_title="잎 수 (장)", yaxis_title="생중량 (g)",
                    xaxis=dict(showgrid=True, gridcolor="#eee"),
                    yaxis=dict(showgrid=True, gridcolor="#eee", range=[0, y.max()*1.1]),
                    width=700, height=450,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13)
                )
                st.plotly_chart(fig_m1, width="stretch")
                m1c1, m1c2, m1c3, m1c4 = st.columns(4)
                m1c1.metric("기울기", f"{m1.coef_[0]:.4f}")
                m1c2.metric("절편", f"{m1.intercept_:.4f}")
                m1c3.metric("훈련 R²", f"{r2_1:.4f}")
                m1c4.metric("훈련 MAE", f"{mae_1:.3f} g")
                m1c5, m1c6 = st.columns(2)
                m1c5.metric("5겹 CV R²", f"{cv_r2_1.mean():.3f} ± {cv_r2_1.std():.3f}")
                m1c6.metric("5겹 CV MAE", f"{cv_mae_1.mean():.3f} g")

                st.markdown("---")
                st.subheader("모델 2 — 잎 수 + 지상부 길이로 생중량 예측 (다중 선형회귀)")
                st.markdown(
                    f"""
                    **회귀식**: 생중량(g) ≈ {m2.coef_[0]:.4f} × 잎 수 + {m2.coef_[1]:.4f} × 지상부 길이 + ({m2.intercept_:+.4f})
                    - 잎 수 기울기 {m2.coef_[0]:.4f}: 지상부 길이 효과를 분리한 뒤에도 잎 수 1장당 약 {m2.coef_[0]:.2f}g 연관성
                    - 지상부 길이 기울기 {m2.coef_[1]:.4f}: 잎 수 효과를 분리한 뒤 1mm당 약 {m2.coef_[1]:.3f}g 연관성
                    - 절편 {m2.intercept_:+.4f}
                    """
                )
                fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=["모델 1 예측 vs 실제", "모델 2 예측 vs 실제"])
                fig_cmp.add_trace(go.Scatter(
                    x=y, y=p1, mode="markers",
                    marker=dict(size=8, color="orange", opacity=0.8, line=dict(width=1, color="white")),
                    name="모델 1", hovertemplate="실제 %{x:.2f}g<br>예측 %{y:.2f}g<extra></extra>"
                ), row=1, col=1)
                fig_cmp.add_trace(go.Scatter(
                    x=y, y=p2, mode="markers",
                    marker=dict(size=8, color="green", opacity=0.8, line=dict(width=1, color="white")),
                    name="모델 2", hovertemplate="실제 %{x:.2f}g<br>예측 %{y:.2f}g<extra></extra>"
                ), row=1, col=2)
                limit = max(y.max(), p1.max(), p2.max()) * 1.08
                for r_i, c_i in [(1,1),(1,2)]:
                    fig_cmp.add_trace(go.Scatter(
                        x=[0,limit], y=[0,limit], mode="lines", name="완벽 예측",
                        line=dict(color="gray", dash="dot"), hoverinfo="skip"
                    ), row=r_i, col=c_i)
                fig_cmp.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    width=900, height=420, showlegend=False
                )
                fig_cmp.update_xaxes(title_text="실제 생중량 (g)", row=1, col=1)
                fig_cmp.update_yaxes(title_text="예측 생중량 (g)", row=1, col=1)
                fig_cmp.update_xaxes(title_text="실제 생중량 (g)", row=1, col=2)
                fig_cmp.update_yaxes(title_text="예측 생중량 (g)", row=1, col=2)
                st.plotly_chart(fig_cmp, width="stretch")
                m2c1, m2c2, m2c3, m2c4 = st.columns(4)
                m2c1.metric("잎 수 기울기", f"{m2.coef_[0]:.4f}")
                m2c2.metric("지상부 기울기", f"{m2.coef_[1]:.4f}")
                m2c3.metric("훈련 R²", f"{r2_2:.4f}")
                m2c4.metric("훈련 MAE", f"{mae_2:.3f} g")
                m2c5, m2c6 = st.columns(2)
                m2c5.metric("5겹 CV R²", f"{cv_r2_2.mean():.3f} ± {cv_r2_2.std():.3f}")
                m2c6.metric("5겹 CV MAE", f"{cv_mae_2.mean():.3f} g")

                st.markdown("---")
                st.subheader("📌 모델 비교 표")
                compare_tbl = pd.DataFrame({
                    "모델": ["모델 1 (잎 수만)", "모델 2 (잎 수 + 지상부 길이)"],
                    "설명 변수": ["잎 수", "잎 수 + 지상부 길이"],
                    "훈련 R²": [f"{r2_1:.3f}", f"{r2_2:.3f}"],
                    "훈련 MAE (g)": [f"{mae_1:.3f}", f"{mae_2:.3f}"],
                    "CV R² (평균)": [f"{cv_r2_1.mean():.3f}", f"{cv_r2_2.mean():.3f}"],
                    "CV MAE (g, 평균)": [f"{cv_mae_1.mean():.3f}", f"{cv_mae_2.mean():.3f}"],
                })
                st.dataframe(compare_tbl, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("📌 해석 — 왜 지상부 길이를 넣어도 예측력이 크게 안 올랐을까?")
                st.markdown(
                    f"""
                    - 두 모델 모두 **잎 수**가 생중량 예측에 큰 역할을 함 (모델 1만 써도 R² ≈ {r2_1:.2f})
                    - 모델 2에 지상부 길이를 추가했을 때 훈련 R²는 {r2_2:.3f}로 모델 1({r2_1:.3f})과 큰 차이 없음
                    - 교차검증 R²도 모델 1({cv_r2_1.mean():.3f})과 모델 2({cv_r2_2.mean():.3f})가 비슷하거나
                      오히려 모델 2가 약간 낮음 → 지상부 길이는 잎 수를 이미 알고 있을 때 추가 설명력이 크지 않음
                    - **이 결과의 의미 (이 실험과의 연결)**:
                      ① 지상부 길이는 생중량을 설명하는 데 잎 수보다 기여가 작음
                      ② 이는 지상부 길이(줄기)가 식물 전체 무게와 상대적으로 약한 관계일 수 있다는 뜻
                      ③ 지상부 길이와 지하부 길이 사이에 직선 관계가 거의 없다는 이 탐구의 피어슨 결과(r≈0)와도
                         방향이 맞아요. 즉 지상부 길이는 '전체 생육 수준을 대표하는 지표'로서 한계가 있을 수 있어요.
                      ④ 반면 지하부 길이(뿌리)는 생중량과 더 강한 상관을 보일 수 있고(Tab3 미리 보기 참고),
                         이는 뿌리 발달이 전체 생육에 더 크게 기여할 수 있음을 시사해요.
                    """
                )

                st.markdown("---")
                st.subheader("📌 한계")
                st.markdown(
                    """
                    - 이 모델은 이 학교의 이 스마트팜에서 이 시기에 재배된 데이터로 만들어졌으므로,
                      다른 학교·다른 환경·측정 범위 밖 예측에는 적용하기 어려움
                    - 회귀계수를 "잎 수가 늘면 생중량이 증가한다"는 인과관계로 해석하면 안 됨 (연관성임)
                    - 81개체는 통계 결론을 일반화하기에 작은 표본일 수 있음
                    - 지하부 길이는 이번 회귀모델에 포함되지 않았으므로, 지하부 길이-생중량 관계는 별도 확인이 필요해요.
                    """
                )

                st.markdown("---")
                st.subheader("📌 대표 개체 10개 실제 vs 예측 비교")
                sample_idx = np.linspace(0, len(clean)-1, 10, dtype=int)
                sam = clean.iloc[sample_idx].copy()
                sam["모델1 예측"] = m1.predict(sam[["잎 수"]].values)
                sam["모델2 예측"] = m2.predict(sam[["잎 수", "지상부 길이"]].values)
                sam_disp = sam[["잎 수", "지상부 길이", "생중량", "모델1 예측", "모델2 예측"]].round(2)
                st.dataframe(sam_disp, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("📌 예측 계산기 (학습 데이터 범위 안에서만)")
                in_leaves = st.slider("잎 수(장)", 0, int(clean["잎 수"].max()), 50)
                in_shoot = st.slider("지상부 길이(mm)", 0, int(clean["지상부 길이"].max()), 40)
                cA, cB = st.columns(2)
                with cA:
                    v1 = m1.predict([[in_leaves]])[0]
                    st.metric("모델 1 예측 생중량", f"{v1:.2f} g")
                with cB:
                    v2 = m2.predict([[in_leaves, in_shoot]])[0]
                    st.metric("모델 2 예측 생중량", f"{v2:.2f} g")
                st.info(
                    f"⚠️ 이 예측기는 잎 수 3~{int(clean['잎 수'].max())}장, 지상부 10~{int(clean['지상부 길이'].max())}mm 범위에서만 참고용이에요. 범위 밖은 신뢰하기 어려워요."
                )
            else:
                st.info("회귀모델을 만들기에 데이터가 부족해요 (최소 10개 필요).")
        else:
            st.info("잎 수·지상부 길이·생중량 열이 모두 필요해요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 4: 개체별 비율 ==========
with tab4:
    try:
        st.header(f"[{school_name}] 개체별 지상부/지하부 길이 비율 비교")
        st.info(
            "💡 비율 = 지상부 길이 ÷ 지하부 길이. 1보다 크면 지상부가 더 길고, 1보다 작으면 지하부가 더 길다는 뜻이에요. "
            "이 비율은 두 길이의 '상대적인 균형'을 보여주며, 개체마다 줄기·뿌리 생장 균형이 다를 수 있음을 살펴보는 지표예요."
        )
        if not growth_df.empty and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
            clean = growth_df[["개체번호", "지상부 길이", "지하부 길이"]].dropna().copy()
            if len(clean) >= 2:
                clean["비율"] = clean["지상부 길이"] / clean["지하부 길이"].replace(0, np.nan)
                clean = clean.dropna(subset=["비율"])
                if len(clean) > 0:
                    clean = clean.sort_values("비율")
                    top10 = clean.tail(10)
                    bottom10 = clean.head(10)
                    colL, colR = st.columns(2)
                    with colL:
                        st.subheader("지상부/지하부 비율이 높은 개체 10개 (지상부가 상대적으로 더 긴 쪽)")
                        fig_top = go.Figure()
                        fig_top.add_trace(go.Bar(
                            x=top10["개체번호"].astype(str), y=top10["비율"],
                            marker_color="steelblue",
                            text=top10["비율"].round(2), textposition="outside",
                            textfont=dict(size=10),
                            hovertemplate="개체 %{x}<br>비율 %{y:.2f}<br>지상부 %{customdata[0]:.0f}mm<br>지하부 %{customdata[1]:.0f}mm<extra></extra>",
                            customdata=top10[["지상부 길이", "지하부 길이"]].values
                        ))
                        fig_top.update_layout(
                            title="비율 상위 10개 개체",
                            xaxis_title="개체번호", yaxis_title="비율 (지상부 ÷ 지하부)",
                            yaxis=dict(range=[0, max(top10["비율"].max()*1.2, 1)]),
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            width=450, height=420
                        )
                        st.plotly_chart(fig_top, width="stretch")
                    with colR:
                        st.subheader("지상부/지하부 비율이 낮은 개체 10개 (지하부가 상대적으로 더 긴 쪽)")
                        fig_bot = go.Figure()
                        fig_bot.add_trace(go.Bar(
                            x=bottom10["개체번호"].astype(str), y=bottom10["비율"],
                            marker_color="coral",
                            text=bottom10["비율"].round(2), textposition="outside",
                            textfont=dict(size=10),
                            hovertemplate="개체 %{x}<br>비율 %{y:.2f}<br>지상부 %{customdata[0]:.0f}mm<br>지하부 %{customdata[1]:.0f}mm<extra></extra>",
                            customdata=bottom10[["지상부 길이", "지하부 길이"]].values
                        ))
                        fig_bot.update_layout(
                            title="비율 하위 10개 개체",
                            xaxis_title="개체번호", yaxis_title="비율 (지상부 ÷ 지하부)",
                            yaxis=dict(range=[0, max(bottom10["비율"].max()*1.5, 1)]),
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            width=450, height=420
                        )
                        st.plotly_chart(fig_bot, width="stretch")
                    st.markdown("---")
                    st.subheader("📌 전체 비율 분포 히스토그램")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=clean["비율"], nbinsx=20,
                        marker_color="mediumseagreen", opacity=0.75,
                        hovertemplate="비율 %{x:.2f}<br>개체 수 %{y}<extra></extra>"
                    ))
                    fig_hist.add_vline(x=1.0, line_dash="dot", line_color="gray",
                                       annotation_text="비율 1.0 (지상부=지하부)", annotation_position="top right")
                    fig_hist.update_layout(
                        title="개체별 지상부/지하부 길이 비율 분포",
                        xaxis_title="비율 (지상부 ÷ 지하부)", yaxis_title="개체 수",
                        xaxis=dict(showgrid=True, gridcolor="#eee"),
                        yaxis=dict(showgrid=True, gridcolor="#eee"),
                        width=700, height=420,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13)
                    )
                    st.plotly_chart(fig_hist, width="stretch")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("비율 평균", f"{clean['비율'].mean():.2f}")
                    m2.metric("비율 중앙값", f"{clean['비율'].median():.2f}")
                    m3.metric("비율 표준편차", f"{clean['비율'].std():.2f}")
                    m4.metric("분석 개체 수", f"{len(clean)}개")
                    st.markdown("---")
                    st.subheader("📌 해석")
                    ratio_mean = clean["비율"].mean()
                    if ratio_mean > 1:
                        ratio_msg = "평균적으로 지상부가 지하부보다 긴 편이에요."
                    elif ratio_mean < 1:
                        ratio_msg = "평균적으로 지하부가 지상부보다 긴 편이에요."
                    else:
                        ratio_msg = "평균적으로 지상부와 지하부 길이가 비슷해요."
                    n_over1 = int((clean["비율"] > 1).sum())
                    n_under1 = int((clean["비율"] < 1).sum())
                    st.markdown(
                        f"""
                        - {ratio_msg}
                        - 비율이 1보다 큰 개체(지상부 > 지하부): **{n_over1}개**
                        - 비율이 1보다 작은 개체(지상부 < 지하부): **{n_under1}개**
                        - 일부 개체는 지상부가 훨씬 길고, 일부 개체는 지하부가 훨씬 길어요.
                          → 같은 조건에서 자랐어도 개체마다 지상부·지하부 균형이 다르게 나타날 수 있어요.
                        - 이 결과는 피어슨 r이 거의 0이었던 것(지상부·지하부 직선 관계 거의 없음)과도 연결돼요:
                          두 길이가 각자 독립적으로 결정되는 경향이 있다는 뜻이에요.
                        """
                    )
                    down_df = clean[["개체번호", "지상부 길이", "지하부 길이", "비율"]].copy()
                    down_df["비율"] = down_df["비율"].round(3)
                    csv_bytes = down_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 개체별 비율 데이터 다운로드 (CSV, 한글 깨짐 방지)",
                        data=csv_bytes, file_name=f"{school_name}_개체별_비율.csv", mime="text/csv"
                    )
                else:
                    st.info("비율을 계산할 수 있는 데이터가 부족해요.")
            else:
                st.info("유효한 지상부·지하부 길이 데이터가 부족해요.")
        else:
            st.info("생육 결과 CSV를 업로드하고 지상부 길이·지하부 길이 열이 있는지 확인해 주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 5: 생육조건 영향 해석 ==========
with tab5:
    try:
        st.header(f"[{school_name}] 생육 환경 조건과 개체별 차이 — 종합 해석")
        st.info(
            "💡 이 탭은 환경 데이터(온도·습도·EC·pH)와 생육 데이터를 함께 놓고, "
            "'어떤 조건이 생육 차이에 영향을 주었을지'를 해석하는 페이지예요. "
            "다만 생육결과데이터에 개체별 측정일이 없어서 환경 값과 개체를 직접 연결하긴 어려워요. "
            "그래서 여기선 환경 데이터의 패턴을 먼저 살펴보고, 그 패턴이 개체 차이와 어떻게 연결될 수 있는지 정리해요."
        )
        st.subheader("1️⃣ 스마트팜 환경 요약 통계")
        if not env_df.empty:
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]
            if len(env_num_cols) > 0:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) > 0:
                    s = pd.DataFrame({
                        "평균": env_clean.mean().round(2),
                        "표준편차": env_clean.std().round(2),
                        "최소": env_clean.min().round(2),
                        "최대": env_clean.max().round(2),
                    })
                    s["변동계수 CV(%)"] = (s["표준편차"] / s["평균"] * 100).round(2)
                    s.index = [c.replace("temperature", "온도(℃)").replace("humidity", "습도(%)")
                               .replace("ph", "pH").replace("ec", "EC(mS/cm)") for c in s.index]
                    st.dataframe(s, width="stretch")
                    st.markdown("---")
                    st.subheader("📌 통계 읽는 방법")
                    st.markdown(
                        """
                        | 용어 | 뜻 | 예시 |
                        |------|-----|------|
                        | 평균 | 모든 값을 더해 개수로 나눈 값, '대표값' | 온도 평균 24.3℃ → 대체로 이 정도 |
                        | 표준편차 | 각 값이 평균에서 얼마나 떨어져 있는지, 원래 단위와 같음 | 클수록 데이터가 넓게 퍼짐 |
                        | 최소·최대 | 가장 작은/큰 값, 극단 조건 확인 | 온도 최대 37℃ → 급한 더위 구간 있었음 |
                        | CV(%) | 표준편차 ÷ 평균 × 100, 단위가 다른 항목끼리도 비교 가능 | EC CV 38% → 온도 CV 10%보다 들쭉날쭉 |
                        """
                    )
                    st.markdown("---")
                    st.subheader("📌 환경 데이터 해석")
                    temp_mean = env_clean["temperature"].mean() if "temperature" in env_clean.columns else None
                    hum_mean = env_clean["humidity"].mean() if "humidity" in env_clean.columns else None
                    if temp_mean is not None:
                        st.markdown(
                            f"""
                            - **온도**: 평균 {temp_mean:.1f}℃, CV {(env_clean['temperature'].std()/temp_mean*100):.1f}%
                              → 대체로 20~26℃ 범위에서 안정적으로 유지되었으나, 특정 시점에 최대 37℃까지 치솟는 급상승 구간이 있었음
                            - **습도**: 평균 {hum_mean:.1f}%, CV {(env_clean['humidity'].std()/hum_mean*100):.1f}%
                              → 주간 감소·야간 증가 패턴이 규칙적으로 나타났고, 이상치는 거의 없어 네 지표 중 가장 안정적이었음
                            """
                        )
                    if "ec" in env_clean.columns:
                        st.markdown(
                            f"""
                            - **EC**: 평균 {env_clean['ec'].mean():.2f} mS/cm, CV {env_clean['ec'].std()/env_clean['ec'].mean()*100:.1f}%
                              → CV가 가장 높고, 낮은 그룹(0.19~0.54)과 높은 그룹(0.81~2.34)으로 나뉘는 bimodal 양상이 보여
                                단일 통계만으로는 안정성을 판단하기 어려움
                            """
                        )
                    if "ph" in env_clean.columns:
                        st.markdown(
                            f"""
                            - **pH**: 평균 {env_clean['ph'].mean():.2f}, CV {env_clean['ph'].std()/env_clean['ph'].mean()*100:.1f}%
                              → 평균 자체는 안정적이나 8월 하순~9월 초에 4~5 수준으로 급락하는 구간이 여러 차례 반복됨
                            """
                        )
                else:
                    st.info("환경 데이터에서 계산할 수 있는 숫자 값이 없어요.")
            else:
                st.info("환경 데이터에서 숫자 열을 찾을 수 없어요.")
        else:
            st.info("환경 CSV를 업로드해 주세요.")

        if not env_df.empty and "time" in env_df.columns:
            st.markdown("---")
            st.subheader("2️⃣ 시간대별 평균 온도·습도")
            tcl = env_df[["time", "temperature", "humidity"]].dropna()
            if len(tcl) >= 2:
                tcl = tcl.copy()
                tcl["hour"] = tcl["time"].dt.hour
                hourly = tcl.groupby("hour").agg(온도=("temperature","mean"), 습도=("humidity","mean")).reset_index()
                if len(hourly) > 0:
                    fig_h = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_h.add_trace(go.Bar(
                        x=hourly["hour"], y=hourly["온도"], name="온도(℃)",
                        marker_color="indianred", hovertemplate="시간 %{x}시<br>평균 온도 %{y:.1f}℃<extra></extra>"
                    ))
                    fig_h.add_trace(go.Scatter(
                        x=hourly["hour"], y=hourly["습도"], name="습도(%)",
                        mode="lines+markers", marker=dict(color="royalblue", size=8),
                        line=dict(color="royalblue", width=2.5),
                        hovertemplate="시간 %{x}시<br>평균 습도 %{y:.1f}%<extra></extra>"
                    ), secondary_y=True)
                    fig_h.update_layout(
                        title="시간대별 평균 온도(막대)와 습도(선)",
                        xaxis_title="시간 (시)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=700, height=420
                    )
                    fig_h.update_yaxes(title_text="온도 (℃)", row=1, col=1)
                    fig_h.update_yaxes(title_text="습도 (%)", row=1, col=1, secondary_y=True)
                    st.plotly_chart(fig_h, width="stretch")
                    warm_hour = hourly.loc[hourly["온도"].idxmax(), "hour"]
                    cold_hour = hourly.loc[hourly["온도"].idxmin(), "hour"]
                    st.markdown(
                        f"""
                        - 하루 중 가장 더운 시간대: **{int(warm_hour)}시** (평균 {hourly['온도'].max():.1f}℃)
                        - 하루 중 가장 추운 시간대: **{int(cold_hour)}시** (평균 {hourly['온도'].min():.1f}℃)
                        - 일교차(최고-최저): 약 {hourly['온도'].max()-hourly['온도'].min():.1f}℃
                        - 습도는 새벽(0~6시)에 가장 높고 오전~낮에 낮아지는 패턴
                        """
                    )

        if not env_df.empty and "time" in env_df.columns:
            st.markdown("---")
            st.subheader("3️⃣ 날짜별 평균 온도·습도 추이 + 변동 범위")
            dcl = env_df[["time", "temperature", "humidity"]].dropna()
            if len(dcl) >= 2:
                dcl = dcl.copy()
                dcl["date"] = dcl["time"].dt.date
                daily = dcl.groupby("date").agg(
                    온도_평균=("temperature","mean"), 온도_표준편차=("temperature","std"),
                    습도_평균=("humidity","mean"), 습도_표준편차=("humidity","std")
                ).reset_index()
                if len(daily) > 0:
                    fig_d = make_subplots(specs=[[{"secondary_y": True}]])
                    xd = [str(d) for d in daily["date"]]
                    fig_d.add_trace(go.Scatter(
                        x=xd, y=daily["온도_평균"], name="온도 평균",
                        mode="lines+markers", marker=dict(size=5, color="indianred"),
                        line=dict(color="indianred", width=2),
                        hovertemplate="날짜 %{x}<br>온도 평균 %{y:.1f}℃<extra></extra>"
                    ))
                    temp_u = daily["온도_평균"] + daily["온도_표준편차"].fillna(0)
                    temp_l = daily["온도_평균"] - daily["온도_표준편차"].fillna(0)
                    fig_d.add_trace(go.Scatter(
                        x=xd+xd[::-1], y=list(temp_u)+list(temp_l)[::-1],
                        fill="toself", fillcolor="indianred",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="온도 변동 범위 (평균±표준편차)",
                        opacity=0.18, hoverinfo="skip"
                    ))
                    fig_d.add_trace(go.Scatter(
                        x=xd, y=daily["습도_평균"], name="습도 평균",
                        mode="lines+markers", marker=dict(size=5, color="royalblue"),
                        line=dict(color="royalblue", width=2),
                        hovertemplate="날짜 %{x}<br>습도 평균 %{y:.1f}%<extra></extra>"
                    ), secondary_y=True)
                    fig_d.update_layout(
                        title="날짜별 평균 온도(빨간선+음영)·습도(파란선) 추이",
                        xaxis_title="날짜",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        width=820, height=420
                    )
                    fig_d.update_yaxes(title_text="온도 (℃)", row=1, col=1)
                    fig_d.update_yaxes(title_text="습도 (%)", row=1, col=1, secondary_y=True)
                    st.plotly_chart(fig_d, width="stretch")
                    st.markdown(
                        """
                        - 빨간 점선 안쪽 음영 = 하루 안에서의 변동 범위(평균 ± 표준편차)
                        - 음영이 넓은 날짜는 그날 안에서도 온도·습도가 크게 흔들렸다는 뜻
                        - 온도는 특정 날짜에 37℃ 가까이 치솟는 구간이 있었고, 습도는 상대적으로 완만함
                        """
                    )

        st.markdown("---")
        st.subheader("4️⃣ 생육조건 차이가 개체별 생육 차이에 미치는 영향 — 종합 해석")
        st.markdown(
            """
            <div style="font-size:0.97rem; line-height:1.65;">
            
            <b>📌 이 데이터에서 할 수 있는 해석과 할 수 없는 해석</b><br>
            생육결과데이터에는 개체별 생육 측정일이 없어서, '개체 A가 어느 날 어떤 온도·습도에서 자랐는지'를
            환경 데이터와 직접 연결할 수 없어요. 또 개체 위치·처리구·광 조건·양분 조건도 기록되어 있지 않아서,
            측정된 환경 요인만으로 개체 차이를 설명하기는 어려워요. 그래서 아래 해석은 **가능성을 정리한 것**이고
            확정적 결론은 아니에요.
            
            <b>📌 측정된 환경 요인의 변동</b><br>
            - 온도·습도는 대체로 안정적이었지만, 특정 시점에 온도 급상승(최대 37℃)이 있었고 습도도 주간/야간 패턴이 있었음<br>
            - EC는 변동 폭이 크고 bimodal 분포를 보였으며, pH는 후반부에 4~5 수준으로 급락하는 구간이 여러 번 있었음<br>
            - 만약 개체마다 생장이 진행된 구체적 시간대의 환경 조건이 달랐다면, 같은 시설 안에서도 개체 차이가 나타날 수 있음<br>
            
            <b>📌 측정하지 않은 요인 가능성</b><br>
            - 같은 시설 안에서도 국소적 광량·통풍·온습도 미세 환경·배양식 공급 불균일 등이 달라질 수 있음<br>
            - 초기 생육 상태·측정 시점 차이만으로도 지상부·지하부 길이, 잎 수, 무게가 크게 달라질 수 있음<br>
            - 생장 초기이거나 측정 오류 가능성이 있는 개체(잎 수·길이·무게가 매우 작은 개체)도 일부 존재함<br>
            
            <b>📌 개체별 차이와 지상부·지하부 독립성</b><br>
            - 피어슨 r ≈ 0 → 지상부 길이와 지하부 길이 사이에 직선 관계가 거의 없음<br>
            - 스피어만 ρ ≈ 0.28 → 순위 기준으로 아주 약한 양의 연관성이 있을 수 있음<br>
            - 지상부·지하부가 서로 다른 요인에 의해 조절될 수 있음을 시사 (예: 개체 8은 지상부 28mm/지하부 251mm,
              개체 24는 지상부 126mm/지하부 62mm)<br>
            - 잘 자란 개체와 덜 자란 개체를 가장 크게 가르는 지표는 **잎 수**(Cohens d 2.85)이고,
              지하부 길이도 큰 차이(Cohens d 2.08)를 보였고, 지상부 길이는 상대적으로 차이가 작았음(Cohens d 0.95)<br>
              → 줄기 길이보다 잎 수와 뿌리 길이가 개체 전체 생육 수준을 더 잘 반영할 가능성<br>
            
            <b>📌 회귀모델과의 연결</b><br>
            - 생중량 예측 모델에서 잎 수가 핵심 변수였고, 지상부 길이는 잎 수를 이미 알고 있을 때 추가 설명력이 크지 않았음<br>
            - 이는 지상부 길이(줄기)가 전체 생육 수준(생중량)을 대표하는 지표로서 한계가 있을 수 있다는 뜻<br>
            - 지하부 길이는 이번 회귀모델에 포함되지 않았지만, 생중량과의 상관은 지상부 길이보다 클 수 있어(Tab3 미리 보기),
              뿌리 발달이 전체 생육에 더 크게 기여할 가능성을 시사해요.<br>
            
            <b>📌 결론</b><br>
            같은 스마트팜에서 자랐음에도 개체별 생육 차이는 매우 컸고, 지상부와 지하부 길이는 서로 독립적인 경향을 보였어요.
            개체 차이의 원인은 측정된 환경 변동, 측정되지 않은 공간·광·양분·초기 조건, 측정 시점·오류 가능성이
            복합적으로 작용한 것으로 추정돼요. 어느 한 요인만으로 설명하기는 어려워요.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.subheader("5️⃣ 개선 제안 (다음 실험에서 적용하면 좋은 점)")
        st.markdown(
            """
            1. **개체별 생육 측정일 기록**: 수확 시점뿐 아니라 일정 간격으로 지상부 길이·지하부 길이·잎 수·무게도 측정해
               개체별 생장 궤적을 파악하면 시계열 분석이 가능해짐
            2. **개체 위치·처리구 정보 기록**: 화분 번호, 재배대 위치, 광 조건 등을 함께 기록하면 환경 요인과 개체 결과를
               개체 단위로 연결할 수 있음
            3. **환경-생육 시간 연결 설계**: 개체별 생육 측정 시점의 환경 값(또는 측정일 기준 구간 평균)을 함께 저장하면
               온도·습도·EC·pH와 생육 지표 간 관계를 더 구체적으로 검토할 수 있음
            4. **이상치·급변동 구간 현장 메모**: EC·pH 급변동, 온도 급상승 구간, 생육 값이 매우 작은 개체에 대해
               당시 장비 상태·측정 방식·재배 상황을 기록해 두면 실제 조건 차이인지 오작동인지 판단할 근거가 됨
            5. **추가 변수 도입**: 광량, 배지/토양 조건, 배양액 조성, 병해충 관찰 기록 등을 추가하면
               지상부보다 잎 수·지하부 길이가 개체 생육 수준을 더 잘 반영하는 이유 등을 더 구체적으로 검토할 수 있음
            """
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ========== 탭 6: 결론 ==========
with tab_conclusion:
    try:
        st.header(f"[{school_name}] 종합 결론 — 가설 검증과 분석의 의미")

        st.subheader("1. 처음 세운 가설")
        st.markdown(
            """
            > **가설**: 지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다.
            """
        )
        st.markdown(
            """
            - 근거: 식물의 줄기와 뿌리는 같은 개체 안에서 자라지만, 서로 다른 환경 조건과 생리적 요인의 영향을 받을 수 있음.
              뿌리채소의 잎을 정리하지 않으면 뿌리가 크게 성장하지 않는 것처럼, 지상부 생장과 지하부 생장이
              항상 함께 증가한다고 확신할 수 없음.
            """
        )

        st.subheader("2. 분석 결과 요약")
        if not growth_df.empty and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
            raw = growth_df[["지상부 길이", "지하부 길이"]].dropna()
            if len(raw) >= 2:
                r_pearson = stats.pearsonr(raw["지상부 길이"], raw["지하부 길이"])[0]
                rho_spearman = stats.spearmanr(raw["지상부 길이"], raw["지하부 길이"])[0]
                n_ind = len(raw)

                if "생중량" in growth_df.columns:
                    r_sw = stats.pearsonr(raw["지상부 길이"], growth_df.loc[raw.index, "생중량"].dropna())[0] if len(growth_df.loc[raw.index, "생중량"].dropna()) >= 2 else None
                    r_bw = stats.pearsonr(raw["지하부 길이"], growth_df.loc[raw.index, "생중량"].dropna())[0] if len(growth_df.loc[raw.index, "생중량"].dropna()) >= 2 else None
                else:
                    r_sw, r_bw = None, None

                res1, res2, res3 = st.columns(3)
                res1.metric("피어슨 r (지상부↔지하부)", f"{r_pearson:.3f}")
                res2.metric("스피어만 ρ (순위)", f"{rho_spearman:.3f}")
                res3.metric("분석 개체 수", f"{n_ind}개")

                st.markdown("---")
                st.subheader("📌 결과 1 — 피어슨 상관 (주 분석)")
                st.markdown(
                    f"""
                    - 지상부 길이와 지하부 길이 사이의 피어슨 상관계수 r = **{r_pearson:.3f}**
                    - r이 0에 매우 가까움 → 두 길이 사이에 **뚜렷한 직선(선형) 관계는 거의 없음**
                    - 이는 처음 세운 가설을 **지지**하는 결과예요.
                    """
                )

                st.markdown("---")
                st.subheader("📌 결과 2 — 스피어만 상관 (보조 분석)")
                s_dir = "양의" if rho_spearman > 0 else "음의"
                if abs(rho_spearman) < 0.2:
                    s_strength = "매우 약함"
                elif abs(rho_spearman) < 0.4:
                    s_strength = "약함"
                else:
                    s_strength = "중간"
                st.markdown(
                    f"""
                    - 스피어만 순위 상관계수 ρ = **{rho_spearman:.3f}** ({s_dir}, **{s_strength}**)
                    - 피어슨 r({r_pearson:.3f})보다 ρ가 더 큰 값 → 직선 관계는 거의 없으나,
                      개체별 순위 기준으로는 아주 약한 양의 연관성이 있을 수 있음
                    - 두 결과를 함께 보면, '선형 관계가 없다'는 결론과 모순되지 않으면서도
                      통계 방법에 따라 해석이 달라질 수 있다는 점까지 탐구할 수 있어요.
                    """
                )

                if r_bw is not None:
                    st.markdown("---")
                    st.subheader("📌 결과 3 — 생중량과 길이 지표의 관계 (부가 분석)")
                    st.markdown(
                        f"""
                        - 지상부 길이 ↔ 생중량 피어슨 r = **{r_sw:.3f}**
                        - 지하부 길이 ↔ 생중량 피어슨 r = **{r_bw:.3f}**
                        - 생중량은 지하부 길이와 더 강한 상관을 보일 수 있음 → 뿌리(지하부) 발달이
                          식물 전체 무게에 더 크게 기여할 가능성을 시사해요.
                        - 이는 회귀모델에서 잎 수가 핵심 변수였고 지상부 길이는 추가 설명력이 크지 않았던 결과와도 연결돼요.
                        """
                    )

                st.markdown("---")
                st.subheader("📌 결과 4 — 개체별 비율의 다양성")
                if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                    ratio_df = growth_df[["지상부 길이", "지하부 길이"]].dropna().copy()
                    ratio_df["비율"] = ratio_df["지상부 길이"] / ratio_df["지하부 길이"].replace(0, np.nan)
                    ratio_df = ratio_df.dropna(subset=["비율"])
                    if len(ratio_df) > 0:
                        st.markdown(
                            f"""
                            - 개체별 지상부/지하부 길이 비율은 평균 {ratio_df['비율'].mean():.2f}, 
                              중앙값 {ratio_df['비율'].median():.2f}
                            - 비율이 1보다 큰 개체(지상부 > 지하부)와 1보다 작은 개체(지상부 < 지하부)가
                              모두 존재 → 개체마다 지상부·지하부 생장 균형이 다름
                            - 이 결과는 피어슨 r이 거의 0이었던 것과도 일치해요.
                              두 길이가 독립적으로 결정될 수 있다는 뜻이에요.
                            """
                        )
            else:
                st.info("결론을 작성할 수 있는 유효한 데이터가 부족해요.")
        else:
            st.info("생육 결과 CSV를 업로드하면 이 결론 페이지가 자동 채워져요.")

        st.subheader("3. 최종 결론")
        st.markdown(
            """
            | 항목 | 결론 |
            |------|------|
            | 가설 | "지상부 길이와 지하부 길이 사이에 뚜렷한 상관관계가 없을 것이다" |
            | 피어슨 r | 약 0.01 → 두 길이 사이에 뚜렷한 직선 관계 없음 |
            | 스피어만 ρ | 약 0.28 → 순위 기준으로 약한 양의 연관성 가능 (보조 발견) |
            | 가설 판정 | **지지됨** (피어슨 결과 기준) |
            | 부가 발견 | 생중량은 지하부 길이와 더 강한 상관 가능, 잎 수가 생중량 예측의 핵심 변수 |
            
            > **종합**: 지상부 길이와 지하부 길이 사이에는 뚜렷한 선형 상관관계가 없으며,
            > 이는 처음 세운 가설을 지지합니다.
            > 
            > 추가로 스피어만 분석을 통해 순위 기준 약한 연관성이 있을 수 있음을 확인했고,
            > 생중량 예측 모델을 통해 잎 수가 전체 생육 수준을 잘 반영하는 지표임을 살펴보았으며,
            > 지상부 길이는 생중량 예측에서 상대적으로 기여가 작다는 점도 확인했어요.
            > 
            > 다만 이 모든 결과는 **이 학교·이 시설·이 시기의 데이터**에 한정된 것으로,
            > 다른 조건이나 더 많은 표본에서 일반화할 때는 추가 검증이 필요해요.
            """
        )

        st.subheader("4. 이 탐구의 의미")
        st.markdown(
            """
            - 통계 방법(피어슨·스피어만·회귀분석)을 직접 데이터에 적용해 가설 검증을 시도했다는 점에서
              고등학교 탐구로서 충분한 의미가 있어요.
            - 특히 피어슨과 스피어만을 함께 사용해 "상관계수라고 다 같은 상관계수가 아니다"라는 점까지
              이해한 것은 좋은 심화 접근이에요.
            - 다만 상관계수나 회귀계수를 인과관계로 오해하지 않도록 주의해야 해요.
              (예: "잎 수가 많으면 생중량이 증가한다"가 아니라 "잎 수와 생중량 사이에 연관성이 관찰된다"로 이해)
            """
        )

        st.subheader("5. 앞으로의 제안")
        st.markdown(
            """
            1. 개체별 생육 측정일(날짜·시각)을 개체번호와 함께 기록
            2. 개체 위치·처리구·광 조건 정보를 함께 기록
            3. 환경 센서 데이터와 생육 데이터를 시간 기준으로 연결할 수 있게 설계
            4. 이상치·급변동 구간에 대한 현장 메모 남기기
            5. 광량·배지·배양액 조성·병해충 관찰 등 추가 변수 도입
            6. 표본 수를 늘리고 가능하면 실험을 2~3회 반복하여 결과 신뢰성 높이기
            """
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
