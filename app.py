import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="达人匹配推荐系统", page_icon="🎯", layout="centered")

# ── 加载模型和数据 ──────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load("matching_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    df = joblib.load("creator_data.pkl")
    return model, feature_cols, df

try:
    model, FEATURE_COLS, df = load_assets()
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

# ── 推荐函数 ────────────────────────────────────
def recommend_creators(product_category, product_price, product_commission, top_n=5):
    results = []
    for handle in df['handle'].unique():
        creator_data = df[df['handle'] == handle]
        cat_data = creator_data[creator_data['品类'] == product_category]

        creator_avg_gmv        = creator_data['该商品GMV'].mean()
        creator_median_gmv     = creator_data['该商品GMV'].median()
        creator_std_gmv        = creator_data['该商品GMV'].std() if len(creator_data) > 1 else 0
        creator_order_count    = len(creator_data)
        creator_avg_commission = creator_data['佣金率'].mean()

        cat_preference  = len(cat_data) / len(creator_data)
        cat_avg_gmv     = cat_data['该商品GMV'].mean()  if len(cat_data) > 0 else creator_avg_gmv
        cat_std_gmv     = cat_data['该商品GMV'].std()   if len(cat_data) > 1 else creator_std_gmv
        cat_order_count = len(cat_data)
        cat_vs_overall  = cat_avg_gmv - creator_avg_gmv

        price_deviation      = (product_price - creator_avg_gmv) / (creator_avg_gmv + 0.01)
        commission_deviation = product_commission - creator_avg_commission

        features = pd.DataFrame([{
            'creator_avg_gmv':        creator_avg_gmv,
            'creator_median_gmv':     creator_median_gmv,
            'creator_std_gmv':        creator_std_gmv,
            'creator_order_count':    creator_order_count,
            'creator_avg_commission': creator_avg_commission,
            'cat_preference':         cat_preference,
            'cat_avg_gmv':            cat_avg_gmv,
            'cat_std_gmv':            cat_std_gmv,
            'cat_order_count':        cat_order_count,
            'cat_vs_overall':         cat_vs_overall,
            'current_commission':     product_commission,
            'current_qty':            1,
            'price_deviation':        price_deviation,
            'commission_deviation':   commission_deviation,
            'log_gmv':                np.log1p(product_price),
            'hour':                   12,
            'weekday':                1,
        }])

        score = model.predict_proba(features[FEATURE_COLS])[0][1]
        results.append({
            '达人':             handle,
            '匹配分':           round(score, 3),
            '品类偏好度':       f"{cat_preference*100:.0f}%",
            '品类历史均值GMV':  f"${cat_avg_gmv:.1f}",
            '历史订单数':       creator_order_count,
        })

    result_df = (pd.DataFrame(results)
                 .sort_values('匹配分', ascending=False)
                 .head(top_n)
                 .reset_index(drop=True))
    result_df.index += 1
    return result_df

# ── 页面 UI ─────────────────────────────────────
st.title("🎯 达人-商品匹配推荐系统")
st.caption("输入商品信息，自动推荐最匹配的达人")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox(
        "商品品类",
        ["护肤品", "家居清洁", "服装鞋帽", "数码电子", "其他"]
    )

with col2:
    price = st.number_input("商品定价（$）", min_value=1.0, max_value=500.0,
                             value=79.0, step=1.0)

with col3:
    commission = st.number_input("佣金率（%）", min_value=1.0, max_value=30.0,
                                  value=18.0, step=0.5)

top_n = st.slider("推荐达人数量", min_value=3, max_value=10, value=5)

st.divider()

if st.button("🔍 开始匹配", use_container_width=True, type="primary"):
    with st.spinner("匹配中..."):
        results = recommend_creators(category, price, commission, top_n)

    st.success(f"为「{category}｜${price}｜佣金{commission}%」找到以下推荐达人：")
    st.dataframe(results, use_container_width=True)

    st.caption("💡 匹配分越接近1说明匹配度越高；品类偏好度反映该达人历史带货中该品类的占比；最终选人建议结合人工判断。")
