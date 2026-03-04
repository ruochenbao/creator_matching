import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
    # 兼容新旧pkl：统一列名
    if '品类' in df_settled.columns and 'category' not in df_settled.columns:
        df_settled = df_settled.rename(columns={'品类': 'category'})

    creator_stats = df_settled.groupby('handle').agg(
        creator_avg_gmv        = ('该商品GMV', 'mean'),
        creator_median_gmv     = ('该商品GMV', 'median'),
        creator_std_gmv        = ('该商品GMV', 'std'),
        creator_order_count    = ('订单id',    'count'),
        creator_avg_commission = ('佣金率',    'mean'),
    ).reset_index()
    creator_stats['creator_std_gmv'] = creator_stats['creator_std_gmv'].fillna(0)

    # 品类偏好度
    cat_count   = df_settled.groupby(['handle', 'category']).size().reset_index(name='cat_count')
    total_count = df_settled.groupby('handle').size().reset_index(name='total_count')
    cat_pref    = cat_count.merge(total_count, on='handle')
    cat_pref['cat_preference'] = cat_pref['cat_count'] / cat_pref['total_count']
    cat_pref_filtered = cat_pref[cat_pref['category'] == category][['handle', 'cat_preference']]

    candidates = creator_stats.merge(cat_pref_filtered, on='handle', how='left')
    candidates['cat_preference'] = candidates['cat_preference'].fillna(0)

    # 达人在该品类的历史GMV均值
    cat_stats = df_settled[df_settled['category'] == category].groupby('handle').agg(
        cat_avg_gmv     = ('该商品GMV', 'mean'),
        cat_std_gmv     = ('该商品GMV', 'std'),
        cat_order_count = ('订单id',    'count'),
    ).reset_index()
    cat_stats['cat_std_gmv'] = cat_stats['cat_std_gmv'].fillna(0)

    candidates = candidates.merge(cat_stats, on='handle', how='left')
    candidates['cat_avg_gmv']     = candidates['cat_avg_gmv'].fillna(candidates['creator_avg_gmv'])
    candidates['cat_std_gmv']     = candidates['cat_std_gmv'].fillna(candidates['creator_std_gmv'])
    candidates['cat_order_count'] = candidates['cat_order_count'].fillna(0)
    candidates['cat_vs_overall']  = candidates['cat_avg_gmv'] - candidates['creator_avg_gmv']

    # 商品特征
    candidates['current_commission']   = commission_rate
    candidates['current_qty']          = 1
    candidates['price_deviation']      = (price - candidates['creator_avg_gmv']) / (candidates['creator_avg_gmv'] + 0.01)
    candidates['commission_deviation'] = commission_rate - candidates['creator_avg_commission']
    candidates['log_gmv']              = np.log1p(price)
    candidates['hour']                 = 12
    candidates['weekday']              = 1

    for col in feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0

    X = candidates[feature_cols].fillna(0)
    candidates['matching_score']      = model.predict_proba(X)[:, 1]
    candidates['category_preference'] = candidates['cat_preference']

    result = candidates.sort_values('matching_score', ascending=False).head(top_n)
    return result

# ==================== 内容效率查询 ====================
def get_content_metrics(handle, category, profile_by_cat):
    """返回：avg_ctr(float或None), avg_ctor(float或None), avg_rpm(float或None), is_category_specific(bool)"""
    if profile_by_cat is None:
        return None, None, None, False

    # 先找达人+品类的精确数据
    row = profile_by_cat[
        (profile_by_cat['Creator username'] == handle) &
        (profile_by_cat['category'] == category)
    ]
    if len(row) > 0:
        return (row['avg_ctr'].values[0],
                row['avg_ctor'].values[0],
                row['avg_rpm'].values[0],
                True)  # True = 有该品类精确数据

    # 没有该品类，用跨品类整体均值
    row_all = profile_by_cat[profile_by_cat['Creator username'] == handle]
    if len(row_all) > 0:
        return (row_all['avg_ctr'].mean(),
                row_all['avg_ctor'].mean(),
                row_all['avg_rpm'].mean(),
                False)  # False = 跨品类均值

    return None, None, None, False

# ==================== CTR分位数（用于判断强弱）====================
@st.cache_data
def get_ctr_threshold(_profile_by_cat):
    """计算全量达人CTR的前30%分位线"""
    if _profile_by_cat is None:
        return 0.03  # 默认3%
    overall_ctr = _profile_by_cat.groupby('Creator username')['avg_ctr'].mean()
    return overall_ctr.quantile(0.70)  # 前30%的门槛

ctr_threshold = get_ctr_threshold(profile_by_cat)

# ==================== 推荐标签判断 ====================
def get_recommendation_tag(category_preference, avg_ctr, is_category_specific, ctr_threshold):
    """
    三档逻辑：
    1. 有品类历史数据 → 无标注
    2. 无品类历史数据 + CTR强（前30%）→ 新品类可尝试
    3. 无品类历史数据 + 无/弱CTR → 数据不足
    """
    if category_preference > 0:
        return ""  # 正常，不加标注

    if avg_ctr is not None and avg_ctr >= ctr_threshold:
        return "🌟 新品类，内容能力强，可尝试"
    else:
        return "⚠️ 数据不足，仅供参考"

# ==================== 执行推荐 ====================
if st.button("🚀 开始匹配", type="primary"):
    with st.spinner("匹配中..."):
        result = recommend(category, price, commission, top_n, df, model, FEATURE_COLS)

    st.subheader(f"🏆 Top {top_n} 推荐达人")
    # 当所有推荐达人品类偏好度都为0时，显示提示
    if (result['category_preference'] == 0).all():
        st.warning(
            f"⚠️ 当前达人库中暂无 **{category}** 品类的历史带货数据，"
            "以下推荐仅基于价格带匹配，区分度有限。建议结合「推荐备注」列的内容效率数据进行人工判断。"
        )

    rows = []
    for _, row in result.iterrows():
        handle = row['handle']
        avg_ctr, avg_ctor, avg_rpm, is_cat_specific = get_content_metrics(
            handle, category, profile_by_cat
        )

        tag = get_recommendation_tag(
            row['category_preference'], avg_ctr, is_cat_specific, ctr_threshold
        )

        # 格式化显示
        ctr_display  = f"{avg_ctr*100:.2f}%" if avg_ctr is not None else "-"
        ctor_display = f"{avg_ctor*100:.2f}%" if avg_ctor is not None else "-"
        rpm_display  = f"${avg_rpm:.2f}" if avg_rpm is not None else "-"

        # CTR来源备注
        if avg_ctr is not None and not is_cat_specific and row['category_preference'] == 0:
            ctr_display += "（跨品类均值）"

        rows.append({
            '达人账号':        handle,
            '匹配分':          f"{row['matching_score']:.3f}",
            '品类偏好度':      f"{row['category_preference']*100:.1f}%",
            '历史带货均值':    f"${row['creator_avg_gmv']:.1f}",
            '历史订单数':      int(row['creator_order_count']),
            'CTR（点击率）':   ctr_display,
            'CTOR（转化率）':  ctor_display,
            'RPM（千次收益）': rpm_display,
            '推荐备注':        tag,
        })

    display_df = pd.DataFrame(rows)
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

    # 图例说明
    st.markdown("---")
    st.markdown(
        "**图例说明**\n\n"
        "- **匹配分**：模型预测该达人带此类商品表现优于同品类均值的概率（0–1，越高越好）\n"
        "- **CTR/CTOR/RPM**：来自视频数据，反映内容效率。`-` 表示暂无视频数据\n"
        "- 🌟 **新品类，内容能力强，可尝试**：该达人虽无此品类带货记录，但整体内容转化能力位于前30%\n"
        "- ⚠️ **数据不足，仅供参考**：无品类历史数据且内容效率数据缺失或偏低，建议人工核查"
    )
