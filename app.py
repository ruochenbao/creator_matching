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

# 第一行：品类 + 基础参数
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    CATEGORIES = ['skincare', 'beauty', 'womenswear', 'fashion', 'home',
                  'food', 'fitness', 'tech', 'toy', 'pet',
                  'menswear', 'health', 'tools', 'book', 'other']
    category = st.selectbox("品类（Broad Category）", CATEGORIES,
                             help="选择商品所属的大品类，用于模型匹配")
with col2:
    price = st.number_input("定价（$）", min_value=1.0, max_value=500.0,
                             value=30.0, step=1.0)
with col3:
    commission = st.number_input("佣金率（%）", min_value=1.0, max_value=50.0,
                                  value=15.0, step=0.5)
with col4:
    top_n = st.number_input("推荐达人数", min_value=1, max_value=20, value=5, step=1)

# 第二行：规则过滤层（品牌定位 + 筛选条件）
with st.expander("筛选与定位条件（可选）", expanded=False):
    st.caption("以下条件将对模型结果进行规则过滤和软重排序，让推荐更精准。")
    col_a, col_b = st.columns(2)
    with col_a:
        brand_type = st.selectbox(
            "品牌定位类型",
            ["不指定",
             "成分型（Ingredient-led）",
             "生活方式型（Lifestyle）",
             "底妆/视觉型（Makeup-Visual）",
             "套装性价比型（Bundle-Value）"],
            help=(
                "成分型：优先 CTOR 高的达人（看了真的买）\n"
                "生活方式型：优先 CTR 高的达人（内容吸引力强）\n"
                "底妆/视觉型：优先互动率高的达人（视觉传播力）\n"
                "套装性价比型：优先 RPM 高的达人（带货转化效率）"
            )
        )
        is_new_launch = st.checkbox(
            "新品冷启动模式",
            value=False,
            help="新品上线初期，优先选互动率高的达人快速积累声量，适合0→1阶段"
        )
    with col_b:
        min_ctr_pct = st.slider(
            "最低 CTR 要求（%）",
            min_value=0.0, max_value=3.0, value=0.0, step=0.1,
            help="硬过滤：只保留该品类 CTR 达标的达人。设为 0 则不过滤。"
        )
        cat_exp = st.selectbox(
            "品类带货经验要求",
            ["不限", "有过带货记录（> 0%）", "有一定经验（≥ 10%）", "主力达人（≥ 30%）"],
            help=(
                "硬过滤：按达人历史订单中该品类的占比进行筛选\n"
                "不限：包含无该品类记录的达人（模型仍会综合打分）\n"
                "有过记录：至少有1条该品类已结算订单\n"
                "有一定经验：该品类占历史订单 ≥ 10%\n"
                "主力达人：该品类占历史订单 ≥ 30%"
            )
        )

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


def norm_series(s):
    """将 Series 归一化到 [0, 1]，全相等时返回 0.5"""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def apply_rules(candidates, brand_type, min_ctr_pct, cat_exp, is_new_launch, category):
    """
    规则过滤层：
      - 硬过滤（Hard Filter）：直接剔除不达标的达人
      - 软重排序（Soft Rerank）：根据品牌定位调整综合评分，不剔除任何人
    返回：(过滤后的df, 被剔除人数, 过滤说明列表, 重排序信号名称)
    """
    df = candidates.copy()
    original_count = len(df)
    filter_msgs = []

    # ---- 硬过滤 1：CTR 最低要求 ----
    if min_ctr_pct > 0:
        threshold = min_ctr_pct / 100.0
        df = df[df['feat_ctr'] >= threshold].copy()
        filter_msgs.append(f"CTR ≥ {min_ctr_pct:.1f}%")

    # ---- 硬过滤 2：品类带货经验 ----
    if cat_exp == "有过带货记录（> 0%）":
        df = df[df['cat_preference'] > 0].copy()
        filter_msgs.append(f"{category} 品类有过带货记录")
    elif cat_exp == "有一定经验（≥ 10%）":
        df = df[df['cat_preference'] >= 0.10].copy()
        filter_msgs.append(f"{category} 品类经验 ≥ 10%")
    elif cat_exp == "主力达人（≥ 30%）":
        df = df[df['cat_preference'] >= 0.30].copy()
        filter_msgs.append(f"{category} 品类主力（≥ 30%）")

    removed_count = original_count - len(df)

    # ---- 软重排序：品牌定位 ----
    boost_label = None
    df['final_score'] = df['matching_score'].copy()

    if len(df) > 1:
        base_norm = norm_series(df['matching_score'])

        if brand_type != "不指定":
            if "成分型" in brand_type:
                boost = norm_series(df['feat_ctor'])
                boost_label = "CTOR 点击转化率"
            elif "生活方式型" in brand_type:
                boost = norm_series(df['feat_ctr'])
                boost_label = "CTR 内容点击率"
            elif "视觉型" in brand_type or "底妆" in brand_type:
                boost = norm_series(df['creator_avg_eng'])
                boost_label = "视频互动率"
            elif "套装" in brand_type or "Bundle" in brand_type:
                boost = norm_series(df['feat_rpm'])
                boost_label = "RPM 千次收益"
            else:
                boost = None

            if boost is not None:
                df['final_score'] = 0.65 * base_norm + 0.35 * boost
                base_norm = df['final_score']  # 更新 base，供新品叠加

        # ---- 软重排序：新品冷启动 ----
        if is_new_launch:
            eng_boost = norm_series(df['creator_avg_eng'])
            df['final_score'] = 0.70 * base_norm + 0.30 * eng_boost
            if boost_label:
                boost_label = boost_label + " + 互动率（新品加权）"
            else:
                boost_label = "互动率（新品冷启动加权）"

    df = df.sort_values('final_score', ascending=False)
    return df, removed_count, filter_msgs, boost_label


def generate_reason(row, category, price, brand_type="不指定", is_new_launch=False):
    """根据特征值和品牌定位，自动生成推荐理由标签"""
    tags = []

    # 品类偏好
    pref = row['cat_preference']
    if pref >= 0.3:
        tags.append(f"🎯 该品类主力达人（偏好度 {pref*100:.0f}%）")
    elif pref >= 0.1:
        tags.append(f"✅ 有 {category} 带货经验（{pref*100:.0f}%）")

    # 价格带匹配
    dev = abs(row['creator_avg_gmv'] - price) / (price + 0.01)
    if dev < 0.15:
        tags.append(f"💰 价格带高度吻合（历史均值 ${row['creator_avg_gmv']:.0f}）")
    elif dev < 0.35:
        tags.append(f"💰 价格带接近（历史均值 ${row['creator_avg_gmv']:.0f}）")

    # 品牌定位匹配维度
    if "成分型" in brand_type:
        ctor = row.get('feat_ctor', 0)
        if pd.notna(ctor):
            if ctor >= 0.05:
                tags.append(f"🧪 CTOR 高（{ctor*100:.1f}%），点击后高转化，契合成分种草逻辑")
            elif ctor >= 0.02:
                tags.append(f"🧪 CTOR 良好（{ctor*100:.1f}%），适合成分型内容导购")
    elif "生活方式型" in brand_type:
        ctr = row.get('feat_ctr', 0)
        if pd.notna(ctr) and ctr >= 0.015:
            tags.append(f"🌿 CTR 优秀（{ctr*100:.2f}%），内容吸引力强，适合生活方式扩散")
    elif "视觉型" in brand_type or "底妆" in brand_type:
        eng = row.get('creator_avg_eng', 0)
        if pd.notna(eng) and eng >= 0.05:
            tags.append(f"✨ 互动率高（{eng*100:.1f}%），视觉传播力强，适合上妆展示")
    elif "套装" in brand_type or "Bundle" in brand_type:
        rpm = row.get('feat_rpm', 0)
        if pd.notna(rpm) and rpm >= 5:
            tags.append(f"📦 RPM 优秀（${rpm:.1f}），每千次播放带货效率高")
    else:
        # 无品牌定位时展示通用 CTR
        ctr = row['feat_ctr']
        if ctr >= 0.02:
            tags.append("📈 CTR 优秀（> 2%）")
        elif ctr >= 0.01:
            tags.append("📊 CTR 良好（> 1%）")

    # 互动率（非视觉型品牌时作为通用维度展示）
    if "视觉型" not in brand_type and "底妆" not in brand_type:
        eng = row.get('creator_avg_eng', 0)
        if pd.notna(eng) and eng >= 0.08:
            tags.append("💬 视频互动率高")

    # 新品冷启动加成说明
    if is_new_launch:
        eng = row.get('creator_avg_eng', 0)
        if pd.notna(eng) and eng >= 0.05:
            tags.append(f"🚀 高互动率（{eng*100:.1f}%），适合新品冷启动引爆声量")

    # 带货经验量
    cnt = row['creator_order_count']
    if cnt >= 200:
        tags.append(f"📦 带货订单丰富（{int(cnt)} 条）")
    elif cnt >= 50:
        tags.append(f"📦 有稳定带货记录（{int(cnt)} 条）")

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
def recommend(category, price, commission_rate, df, model, feature_cols,
              profile_by_cat, global_medians):
    """运行模型，返回全部候选达人（未截断），附 matching_score"""

    # 列名兼容：同时支持新版（英文）和旧版（中文）creator_data.pkl
    STATUS_COL = 'order_status' if 'order_status' in df.columns else '订单状态'
    GMV_COL    = 'gmv'          if 'gmv'          in df.columns else '该商品GMV'
    COMM_COL   = 'commission_rate' if 'commission_rate' in df.columns else '佣金率'
    OID_COL    = 'order_id'     if 'order_id'     in df.columns else '订单id'
    ENG_COL    = 'engagement_rate' if 'engagement_rate' in df.columns else None
    SETTLED    = '已结算'

    df_settled = df[df[STATUS_COL] == SETTLED].copy()

    agg_dict = {
        'creator_avg_gmv':     (GMV_COL,  'mean'),
        'creator_median_gmv':  (GMV_COL,  'median'),
        'creator_std_gmv':     (GMV_COL,  'std'),
        'creator_order_count': (OID_COL,  'count'),
        'creator_avg_comm':    (COMM_COL, 'mean'),
    }
    if ENG_COL and ENG_COL in df_settled.columns:
        agg_dict['creator_avg_eng']    = (ENG_COL, 'mean')
        agg_dict['creator_video_flag'] = (ENG_COL, lambda x: x.notna().mean())

    creator_stats = df_settled.groupby('handle').agg(**agg_dict).reset_index()
    if 'creator_avg_eng' not in creator_stats.columns:
        creator_stats['creator_avg_eng']    = 0.0
        creator_stats['creator_video_flag'] = 0.0
    creator_stats['creator_std_gmv'] = creator_stats['creator_std_gmv'].fillna(0)
    if ENG_COL and ENG_COL in df_settled.columns:
        creator_stats['creator_avg_eng'] = creator_stats['creator_avg_eng'].fillna(
            df_settled[ENG_COL].mean()
        )
    else:
        creator_stats['creator_avg_eng'] = creator_stats['creator_avg_eng'].fillna(0)

    CAT_COL = 'category' if 'category' in df_settled.columns else '品类'
    cat_count   = df_settled.groupby(['handle', CAT_COL]).size().reset_index(name='cat_count')
    total_count = df_settled.groupby('handle').size().reset_index(name='total_count')
    cat_pref    = cat_count.merge(total_count, on='handle')
    cat_pref['cat_preference'] = cat_pref['cat_count'] / cat_pref['total_count']
    cat_pref_f  = cat_pref[cat_pref[CAT_COL] == category][['handle', 'cat_preference']]

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

    return candidates.reset_index(drop=True)


# ==================== 展示推荐结果（卡片式） ====================
if run:
    with st.spinner("匹配中..."):
        all_candidates = recommend(
            category, price, commission, df, model, FEATURE_COLS,
            profile_by_cat, global_medians
        )

        # 应用规则过滤层
        result_full, removed_count, filter_msgs, boost_label = apply_rules(
            all_candidates, brand_type, min_ctr_pct, cat_exp, is_new_launch, category
        )
        result = result_full.head(top_n).reset_index(drop=True)

    st.divider()

    # --- 结果标题行 ---
    title_parts = [f"**{category}**", f"**${price:.0f}**", f"**佣金 {commission:.1f}%**"]
    st.subheader(f"推荐结果  ·  {category}  /  ${price:.0f}  /  佣金 {commission:.1f}%")

    # --- 过滤与重排序状态提示 ---
    info_parts = [f"共推荐 **{len(result)}** 位达人，按{'综合评分' if boost_label else '匹配分'}从高到低排列"]
    if removed_count > 0:
        info_parts.append(f"硬过滤移除 **{removed_count}** 位不达标达人（{' / '.join(filter_msgs)}）")
    if boost_label:
        info_parts.append(f"软重排序已启用：模型分 65% + **{boost_label}** 35%")
    if is_new_launch and not boost_label:
        info_parts.append("新品冷启动加权：模型分 70% + 互动率 30%")
    for msg in info_parts:
        st.caption(msg)

    if len(result) == 0:
        st.warning("当前筛选条件下没有符合要求的达人，请放宽筛选条件后重试。")
        st.stop()

    for rank, row in result.iterrows():
        handle   = row['handle']
        score    = row['matching_score']
        f_score  = row.get('final_score', score)
        ctr_s, ctor_s, rpm_s, data_src = get_content_display(handle, category, profile_by_cat)
        tags = generate_reason(row, category, price, brand_type, is_new_launch)

        eng = row.get('creator_avg_eng', float('nan'))
        eng_s = f"{eng*100:.1f}%" if pd.notna(eng) and eng > 0 else '-'

        with st.container(border=True):
            # --- 行1：账号 + 匹配分 + 综合分 ---
            h_col, s_col = st.columns([3, 2])
            with h_col:
                st.markdown(f"### #{rank+1} &nbsp; `@{handle}`")
            with s_col:
                if boost_label or is_new_launch:
                    st.markdown(
                        f"**综合分** &nbsp; {score_color(f_score)} &nbsp; "
                        f"`{score_bar(f_score)}` &nbsp; **{f_score:.3f}**"
                        f"&nbsp;&nbsp; *(模型匹配分 {score:.3f})*"
                    )
                else:
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
        "**匹配分**：模型预测该达人带此品类商品表现优于同品类平均的概率（0–1）  \n"
        "**综合分**（启用定位/新品时）：模型分 × 65% + 品牌定位信号 × 35%，新品再叠加互动率权重  \n"
        "CTR / CTOR / RPM 使用贝叶斯平滑处理，视频数越少的达人越向品类全局均值回归"
    )
