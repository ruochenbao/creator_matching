import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="达人匹配系统", layout="wide")

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
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

# 预计算品类全局中位数（推理时对无视频数据达人的 CTR 填充用）
@st.cache_resource
def compute_global_medians(_profile_by_cat):
    if _profile_by_cat is None:
        return {}
    gm = _profile_by_cat.groupby('category').agg(
        global_ctr  = ('avg_ctr',  'median'),
        global_ctor = ('avg_ctor', 'median'),
        global_rpm  = ('avg_rpm',  'median'),
    ).reset_index()
    return gm.set_index('category').to_dict('index')

global_medians = compute_global_medians(profile_by_cat)

# ==================== 页面标题 ====================
st.title("达人-商品匹配推荐系统")
st.caption("输入商品信息，模型自动推荐最匹配的达人，并附上推荐理由和内容效率数据。")
st.divider()

# ==================== 输入区 ====================
st.subheader("商品信息")
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    CATEGORIES = ['skincare', 'beauty', 'womenswear', 'fashion', 'home',
                  'food', 'fitness', 'tech', 'toy', 'pet',
                  'menswear', 'health', 'tools', 'book', 'other']
    category = st.selectbox("品类", CATEGORIES)

with col2:
    price = st.number_input("定价（$）", min_value=1.0, max_value=500.0,
                             value=30.0, step=1.0)
with col3:
    commission = st.number_input("佣金率（%）", min_value=1.0, max_value=50.0,
                                  value=15.0, step=0.5)
with col4:
    top_n = st.number_input("推荐达人数", min_value=1, max_value=10, value=5, step=1)

run = st.button("开始匹配", type="primary", use_container_width=False)

# ==================== 工具函数 ====================
def get_feat_ctr(handle, category, profile_by_cat, global_medians):
    """返回用于模型特征的 feat_ctr/ctor/rpm（数值）"""
    gm = global_medians.get(category, {})
    fallback = (gm.get('global_ctr', 0.01), gm.get('global_ctor', 0.01), gm.get('global_rpm', 5.0))
    if profile_by_cat is None:
        return fallback
    row = profile_by_cat[
        (profile_by_cat['Creator username'] == handle) &
        (profile_by_cat['category'] == category)
    ]
    if len(row) > 0:
        return row['avg_ctr'].values[0], row['avg_ctor'].values[0], row['avg_rpm'].values[0]
    row_all = profile_by_cat[profile_by_cat['Creator username'] == handle]
    if len(row_all) > 0:
        w = row_all['video_count']
        wsum = w.sum()
        if wsum > 0:
            return (
                (row_all['avg_ctr']  * w).sum() / wsum,
                (row_all['avg_ctor'] * w).sum() / wsum,
                (row_all['avg_rpm']  * w).sum() / wsum,
            )
    return fallback


def get_content_display(handle, category, profile_by_cat):
    """返回展示用的 CTR/CTOR/RPM 字符串 + 数据来源标注"""
    if profile_by_cat is None:
        return '-', '-', '-', 'no_data'
    row = profile_by_cat[
        (profile_by_cat['Creator username'] == handle) &
        (profile_by_cat['category'] == category)
    ]
    if len(row) > 0:
        return (
            f"{row['avg_ctr'].values[0]*100:.2f}%",
            f"{row['avg_ctor'].values[0]*100:.2f}%",
            f"${row['avg_rpm'].values[0]:.2f}",
            'exact',
        )
    row_all = profile_by_cat[profile_by_cat['Creator username'] == handle]
    if len(row_all) > 0:
        return (
            f"{row_all['avg_ctr'].mean()*100:.2f}%",
            f"{row_all['avg_ctor'].mean()*100:.2f}%",
            f"${row_all['avg_rpm'].mean():.2f}",
            'overall',
        )
    return '-', '-', '-', 'no_data'


def generate_reason(row, category, price):
    """根据特征值自动生成推荐理由标签"""
    tags = []

    # 品类偏好
    pref = row['cat_preference']
    if pref >= 0.3:
        tags.append(f"🎯 该品类主力达人（偏好度 {pref*100:.0f}%）")
    elif pref >= 0.1:
        tags.append(f"✅ 有{category}带货经验（{pref*100:.0f}%）")

    # 价格带匹配
    dev = abs(row['creator_avg_gmv'] - price) / (price + 0.01)
    if dev < 0.15:
        tags.append(f"💰 价格带高度吻合（历史均值 ${row['creator_avg_gmv']:.0f}）")
    elif dev < 0.35:
        tags.append(f"💰 价格带接近（历史均值 ${row['creator_avg_gmv']:.0f}）")

    # 内容效率
    ctr = row['feat_ctr']
    if ctr >= 0.02:
        tags.append("📈 CTR 优秀（> 2%）")
    elif ctr >= 0.01:
        tags.append("📊 CTR 良好（> 1%）")

    # 互动率
    eng = row.get('creator_avg_eng', 0)
    if pd.notna(eng) and eng >= 0.08:
        tags.append("💬 视频互动率高")

    # 带货经验量
    cnt = row['creator_order_count']
    if cnt >= 200:
        tags.append(f"📦 带货订单丰富（{cnt:.0f} 条）")
    elif cnt >= 50:
        tags.append(f"📦 有稳定带货记录（{cnt:.0f} 条）")

    if not tags:
        tags.append("综合历史带货能力匹配")

    return tags


def score_bar(score):
    """把 0-1 的分数转成 emoji 进度条"""
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled)


def score_color(score):
    if score >= 0.6:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"


# ==================== 推荐函数 ====================
def recommend(category, price, commission_rate, top_n, df, model, feature_cols,
              profile_by_cat, global_medians):
    df_settled = df[df['order_status'] == '已结算'].copy()

    creator_stats = df_settled.groupby('handle').agg(
        creator_avg_gmv     = ('gmv',             'mean'),
        creator_median_gmv  = ('gmv',             'median'),
        creator_std_gmv     = ('gmv',             'std'),
        creator_order_count = ('order_id',        'count'),
        creator_avg_comm    = ('commission_rate', 'mean'),
        creator_avg_eng     = ('engagement_rate', 'mean'),
        creator_video_flag  = ('engagement_rate', lambda x: x.notna().mean()),
    ).reset_index()
    creator_stats['creator_std_gmv'] = creator_stats['creator_std_gmv'].fillna(0)
    creator_stats['creator_avg_eng'] = creator_stats['creator_avg_eng'].fillna(
        df_settled['engagement_rate'].mean()
    )

    cat_count   = df_settled.groupby(['handle', 'category']).size().reset_index(name='cat_count')
    total_count = df_settled.groupby('handle').size().reset_index(name='total_count')
    cat_pref    = cat_count.merge(total_count, on='handle')
    cat_pref['cat_preference'] = cat_pref['cat_count'] / cat_pref['total_count']
    cat_pref_f  = cat_pref[cat_pref['category'] == category][['handle', 'cat_preference']]

    candidates = creator_stats.merge(cat_pref_f, on='handle', how='left')
    candidates['cat_preference'] = candidates['cat_preference'].fillna(0)

    candidates['price_deviation'] = (price - candidates['creator_avg_gmv']) / (candidates['creator_avg_gmv'] + 0.01)
    candidates['commission_diff'] = commission_rate - candidates['creator_avg_comm']
    candidates['log_gmv']         = np.log1p(price)
    candidates['commission_rate'] = commission_rate
    candidates['qty']             = 1
    candidates['order_hour']      = 12
    candidates['order_weekday']   = 2

    feat = candidates['handle'].apply(
        lambda h: get_feat_ctr(h, category, profile_by_cat, global_medians)
    )
    candidates['feat_ctr']  = [r[0] for r in feat]
    candidates['feat_ctor'] = [r[1] for r in feat]
    candidates['feat_rpm']  = [r[2] for r in feat]

    X = candidates[feature_cols].fillna(0)
    candidates['matching_score'] = model.predict_proba(X)[:, 1]

    return candidates.sort_values('matching_score', ascending=False).head(top_n).reset_index(drop=True)


# ==================== 展示推荐结果（卡片式） ====================
if run:
    with st.spinner("匹配中..."):
        result = recommend(
            category, price, commission, top_n,
            df, model, FEATURE_COLS,
            profile_by_cat, global_medians
        )

    st.divider()
    st.subheader(f"推荐结果  ·  {category}  /  ${price:.0f}  /  佣金 {commission:.1f}%")
    st.caption(f"共推荐 {len(result)} 位达人，按匹配分从高到低排列")

    for rank, row in result.iterrows():
        handle = row['handle']
        score  = row['matching_score']
        ctr_s, ctor_s, rpm_s, data_src = get_content_display(handle, category, profile_by_cat)
        tags = generate_reason(row, category, price)

        eng = row.get('creator_avg_eng', float('nan'))
        eng_s = f"{eng*100:.1f}%" if pd.notna(eng) and eng > 0 else '-'

        with st.container(border=True):
            # --- 行1：账号 + 匹配分 ---
            h_col, s_col = st.columns([3, 2])
            with h_col:
                st.markdown(f"### #{rank+1} &nbsp; `@{handle}`")
            with s_col:
                st.markdown(
                    f"**匹配分** &nbsp; {score_color(score)} &nbsp; "
                    f"`{score_bar(score)}` &nbsp; **{score:.3f}**"
                )

            # --- 行2：核心指标 ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("品类偏好度", f"{row['cat_preference']*100:.1f}%",
                          help="该达人历史已结算订单中，此品类占比")
            with m2:
                st.metric("历史带货均值", f"${row['creator_avg_gmv']:.1f}",
                          help="达人历史已结算订单的平均 GMV")
            with m3:
                st.metric("历史订单数", f"{int(row['creator_order_count'])} 条",
                          help="已结算订单总数，反映带货经验量")
            with m4:
                st.metric("互动率（均值）", eng_s,
                          help="历史视频（点赞+评论+分享）÷ 观看数 的平均值")

            # --- 行3：内容效率 ---
            c1, c2, c3, c_note = st.columns([1, 1, 1, 2])
            with c1:
                st.metric("CTR 点击率", ctr_s, help="视频被点击进商品详情页的比率")
            with c2:
                st.metric("CTOR 转化率", ctor_s, help="点击后下单的比率")
            with c3:
                st.metric("RPM 千次收益", rpm_s, help="每千次视频播放产生的 GMV")
            with c_note:
                if data_src == 'overall':
                    st.caption("⚠️ 该品类暂无视频记录，显示的是跨品类均值")
                elif data_src == 'no_data':
                    st.caption("⚠️ 暂无视频效率数据")
                else:
                    st.caption(f"✅ 数据来自该达人 {category} 品类的实际视频记录")

            # --- 行4：推荐理由 ---
            st.markdown("**推荐理由：** " + "　".join(tags))

    st.divider()
    st.caption(
        "匹配分说明：模型预测该达人带此品类商品表现优于同品类平均的概率（0–1）\n\n"
        "CTR/CTOR/RPM 使用贝叶斯平滑处理，视频数越少的达人越向品类全局均值回归"
    )
