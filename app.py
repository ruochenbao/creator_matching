import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ==================== 加载模型文件 ====================
@st.cache_resource
def load_model():
    model        = joblib.load('matching_model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    df           = joblib.load('creator_data.pkl')
    return model, feature_cols, df

@st.cache_resource
def load_content_profile():
    try:
        profile_by_cat  = joblib.load('content_profile_by_cat.pkl')
        profile_overall = joblib.load('content_profile_overall.pkl')
        return profile_by_cat, profile_overall
    except:
        return None, None

try:
    model, FEATURE_COLS, df = load_model()
    profile_by_cat, profile_overall = load_content_profile()
    st.success("模型加载成功！")
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

# ==================== 页面标题 ====================
st.title("🎯 达人-商品匹配推荐系统")
st.markdown("输入商品信息，自动推荐最匹配的达人，并附上内容效率参考数据。")

# ==================== 输入区 ====================
st.subheader("📦 商品信息输入")

col1, col2, col3 = st.columns(3)

with col1:
    CATEGORIES = ['skincare', 'beauty', 'womenswear', 'fashion', 'home',
                  'food', 'fitness', 'tech', 'toy', 'pet',
                  'menswear', 'health', 'tools', 'book', 'other']
    category = st.selectbox("商品品类", CATEGORIES)

with col2:
    price = st.number_input("商品定价（$）", min_value=1.0, max_value=500.0,
                             value=30.0, step=1.0)

with col3:
    commission = st.number_input("佣金率（%）", min_value=1.0, max_value=50.0,
                                  value=15.0, step=0.5)

top_n = st.slider("推荐达人数量", min_value=1, max_value=10, value=5)

# ==================== 推荐函数 ====================
def recommend(category, price, commission_rate, top_n, df, model, feature_cols):
    df_settled = df[df['订单状态'] == '已结算'].copy()

    # 按达人汇总特征
    creator_stats = df_settled.groupby('handle').agg(
        creator_avg_gmv     = ('该商品GMV', 'mean'),
        creator_median_gmv  = ('该商品GMV', 'median'),
        creator_order_count = ('订单id', 'count'),
        creator_avg_comm    = ('佣金率', 'mean'),
    ).reset_index()

    # 品类偏好度
    cat_col = 'category' if 'category' in df_settled.columns else '品类'
    cat_count   = df_settled.groupby(['handle', cat_col]).size().reset_index(name='cat_count')
    total_count = df_settled.groupby('handle').size().reset_index(name='total_count')
    cat_pref    = cat_count.merge(total_count, on='handle')
    cat_pref['category_preference'] = cat_pref['cat_count'] / cat_pref['total_count']
    cat_pref_filtered = cat_pref[cat_pref[cat_col] == category][['handle', 'category_preference']]

    candidates = creator_stats.merge(cat_pref_filtered, on='handle', how='left')
    candidates['category_preference'] = candidates['category_preference'].fillna(0)

    # 构造特征
    candidates['price_vs_creator_avg'] = (
        (price - candidates['creator_avg_gmv']) / (candidates['creator_avg_gmv'] + 0.01)
    )
    candidates['log_price']       = np.log1p(price)
    candidates['佣金率']           = commission_rate
    candidates['commission_diff'] = commission_rate - candidates['creator_avg_comm']

    # 补充视频特征（如果有的话）
    for col in ['avg_ctr', 'avg_ctor', 'avg_rpm', 'median_rpm', 'video_count']:
        if col in feature_cols:
            candidates[col] = 0

    # 补全所有模型需要但candidates里没有的列
    for col in feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0

    X = candidates[feature_cols].fillna(0)
        candidates['matching_score'] = model.predict_proba(X)[:, 1]

    result = candidates.sort_values('matching_score', ascending=False).head(top_n)
    return result

# ==================== 查询内容效率画像 ====================
def get_content_metrics(handle, category, profile_by_cat):
    if profile_by_cat is None:
        return '-', '-', '-'
    row = profile_by_cat[
        (profile_by_cat['Creator username'] == handle) &
        (profile_by_cat['category'] == category)
    ]
    if len(row) == 0:
        # 没有该品类数据，用该达人整体数据
        row_all = profile_by_cat[profile_by_cat['Creator username'] == handle]
        if len(row_all) == 0:
            return '-', '-', '-'
        avg_ctr  = f"{row_all['avg_ctr'].mean()*100:.2f}%"
        avg_ctor = f"{row_all['avg_ctor'].mean()*100:.2f}%"
        avg_rpm  = f"${row_all['avg_rpm'].mean():.2f}"
        return avg_ctr, avg_ctor, avg_rpm
    avg_ctr  = f"{row['avg_ctr'].values[0]*100:.2f}%"
    avg_ctor = f"{row['avg_ctor'].values[0]*100:.2f}%"
    avg_rpm  = f"${row['avg_rpm'].values[0]:.2f}"
    return avg_ctr, avg_ctor, avg_rpm

# ==================== 执行推荐 ====================
if st.button("🚀 开始匹配", type="primary"):
    with st.spinner("匹配中..."):
        result = recommend(category, price, commission, top_n, df, model, FEATURE_COLS)

    st.subheader(f"🏆 Top {top_n} 推荐达人")

    # 构造展示表格
    rows = []
    for _, row in result.iterrows():
        handle = row['handle']
        ctr, ctor, rpm = get_content_metrics(handle, category, profile_by_cat)

        # 判断CTR数据来源
        has_cat_data = profile_by_cat is not None and len(
            profile_by_cat[
                (profile_by_cat['Creator username'] == handle) &
                (profile_by_cat['category'] == category)
            ]
        ) > 0
        ctr_note = "" if has_cat_data else "（跨品类均值）" if ctr != '-' else ""

        rows.append({
            '达人账号':       handle,
            '匹配分':         f"{row['matching_score']:.3f}",
            '品类偏好度':     f"{row['category_preference']*100:.1f}%",
            '历史带货均值':   f"${row['creator_avg_gmv']:.1f}",
            '历史订单数':     int(row['creator_order_count']),
            'CTR（点击率）':  ctr + ctr_note,
            'CTOR（转化率）': ctor,
            'RPM（千次收益）':rpm,
        })

    display_df = pd.DataFrame(rows)
    st.dataframe(display_df, use_container_width=True)

    # 说明
    st.caption(
        "📌 匹配分：模型预测该达人带此类商品表现优于同品类平均的概率（0-1，越高越好）\n\n"
        "📌 CTR/CTOR/RPM：来自视频数据，反映该达人的内容效率。'-'表示暂无视频数据；"
        "标注'跨品类均值'表示该达人在此品类无视频记录，显示的是其所有品类的平均值。"
    )
