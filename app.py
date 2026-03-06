import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
import os
import pathlib

# Fastmoss 候选池固定路径（与 app.py 同目录，支持 Streamlit Cloud 部署）
_FM_AUTO_PATH = str(pathlib.Path(__file__).parent / 'fastmoss_pool.xlsx')

st.set_page_config(page_title="达人匹配系统", layout="wide")

# ==================== 全局样式：缩小 metric 字体 ====================
st.markdown("""
<style>
/* metric 数值字体缩小 */
[data-testid="stMetricValue"] { font-size: 1.0rem !important; }
/* metric 标签字体缩小 */
[data-testid="stMetricLabel"] { font-size: 0.68rem !important; color: #6b6b6b; }
/* metric 容器内边距收紧 */
[data-testid="stMetric"] { padding-top: 2px !important; padding-bottom: 2px !important; }
/* 卡片内部元素行间距压缩 */
[data-testid="stVerticalBlockBorderWrapper"] > div > div {
    gap: 0.25rem !important;
}
/* 卡片自身上下外边距缩小 */
[data-testid="stVerticalBlockBorderWrapper"] {
    margin-bottom: 0.3rem !important;
}
</style>
""", unsafe_allow_html=True)

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

    st.divider()
    _has_auto_fm = os.path.exists(_FM_AUTO_PATH)
    if _has_auto_fm:
        st.caption(
            "✅ **已检测到 Fastmoss 候选池文件**（fastmoss_pool.xlsx），点击「开始匹配」将自动加载扩展候选池。  \n"
            "如需更新数据，可上传新文件覆盖："
        )
    else:
        st.caption(
            "**扩展候选池（可选）**：上传 Fastmoss 带货达人榜 Excel，可在模型库之外发现更多候选达人。  \n"
            "*（也可将文件命名为 `fastmoss_pool.xlsx` 放入部署目录，下次自动加载无需上传）*"
        )
    fm_file = st.file_uploader(
        "Fastmoss 达人榜 Excel（上传后将覆盖自动加载）" if _has_auto_fm else "Fastmoss 达人榜 Excel",
        type=['xlsx'],
        label_visibility='collapsed',
        help="从 Fastmoss「带货达人榜」页面导出的 Excel 文件，用于扩展推荐候选池"
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

    # ---- 软重排序：基础层（始终生效）----
    # matching_score 和 cat_preference 本身都在 [0,1]，直接加权相加
    # 不做 norm_series：避免 norm 把分布拉伸后反而放大 price_deviation 的影响
    # 权重：模型分 70%，品类专长 30%
    # 解决问题：模型倾向于价格匹配（price_deviation 权重高），
    #           导致价格带偏低的品类专家（如高偏好度 skincare 达人）排名靠后
    boost_label = None
    df['final_score'] = df['matching_score'].copy()

    if len(df) > 1:
        # base = 模型分（反映价格/历史综合能力）+ 品类专长补偿
        base = 0.70 * df['matching_score'] + 0.30 * df['cat_preference']
        df['final_score'] = base

        # ---- 软重排序：品牌定位（叠加在 base 之上，用 norm 对齐量纲）----
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
                df['final_score'] = 0.65 * norm_series(base) + 0.35 * boost

        # ---- 软重排序：新品冷启动 ----
        if is_new_launch:
            eng_boost = norm_series(df['creator_avg_eng'])
            df['final_score'] = 0.70 * df['final_score'] + 0.30 * eng_boost
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


def load_fm_pool(uploaded_file):
    """
    加载 Fastmoss 带货达人榜 Excel，返回规则评分后的 DataFrame。
    列顺序固定：昵称/达人ID/性别/分类/粉丝数/商品数/总GMV/视频GMV/直播GMV/fm链接/tt链接
    """
    df = pd.read_excel(uploaded_file)
    # 按位置重命名（避免编码乱码问题）
    fixed_cols = ['nickname', 'handle', 'gender', 'category', 'followers',
                  'product_count', 'total_gmv', 'video_gmv', 'live_gmv',
                  'fastmoss_url', 'tiktok_url']
    df.columns = fixed_cols[:len(df.columns)]

    # 类型清理
    df['handle']    = df['handle'].astype(str).str.strip()
    df['nickname']  = df['nickname'].astype(str).str.strip()
    df['total_gmv'] = pd.to_numeric(df['total_gmv'], errors='coerce').fillna(0)
    df['video_gmv'] = pd.to_numeric(df['video_gmv'], errors='coerce').fillna(0)
    df['live_gmv']  = pd.to_numeric(df['live_gmv'],  errors='coerce').fillna(0)
    df['followers'] = pd.to_numeric(df['followers'],  errors='coerce').fillna(0)

    # 过滤：只保留美妆相关类目
    if 'category' in df.columns:
        beauty_kw = ['美妆', '护肤', 'beauty', 'Beauty']
        mask = df['category'].astype(str).str.contains('|'.join(beauty_kw), na=False)
        df = df[mask].copy()

    # 过滤：必须有视频带货 GMV（affiliate 友好）
    df = df[df['video_gmv'] > 0].copy()

    if len(df) == 0:
        return df

    # 规则评分 -------------------------------------------------------
    # 1. 视频GMV占比（越高 = 越依赖有机内容，而非直播）
    df['video_ratio'] = df['video_gmv'] / (df['total_gmv'] + 1)

    # 2. 粉丝量级适配（偏好 5K-500K 腰部区间）
    def _follower_score(f):
        if   f <   5_000: return 0.25
        elif f <  50_000: return 0.70
        elif f < 200_000: return 1.00
        elif f < 500_000: return 0.85
        elif f < 2_000_000: return 0.60
        else:               return 0.30
    df['follower_score'] = df['followers'].apply(_follower_score)

    # 3. 视频GMV量级（log 归一化带货经验）
    log_max = np.log1p(df['video_gmv'].max()) or 1.0
    df['gmv_score'] = np.log1p(df['video_gmv']) / log_max

    # 综合分：视频占比40% + 粉丝适配35% + 经验25%
    df['cold_score'] = (
        0.40 * df['video_ratio'] +
        0.35 * df['follower_score'] +
        0.25 * df['gmv_score']
    )

    return df.sort_values('cold_score', ascending=False).reset_index(drop=True)


@st.cache_resource
def _try_load_fm_auto(path: str):
    """从固定路径预加载 Fastmoss 候选池，结果缓存至 Streamlit server 重启"""
    try:
        return load_fm_pool(path)
    except Exception:
        return None


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
    st.subheader(f"推荐结果  ·  {category}  /  ${price:.0f}  /  佣金 {commission:.1f}%")

    # --- 过滤与重排序状态提示 ---
    info_parts = [f"共推荐 **{len(result)}** 位达人，按综合评分从高到低排列"]
    if removed_count > 0:
        info_parts.append(f"硬过滤移除 **{removed_count}** 位不达标达人（{' / '.join(filter_msgs)}）")
    if boost_label:
        info_parts.append(f"品牌定位加权：基础分 65% + **{boost_label}** 35%")
    if is_new_launch:
        info_parts.append("新品冷启动加权已叠加（互动率 +30%）")

    cap_col, dl_col = st.columns([4, 1])
    with cap_col:
        for msg in info_parts:
            st.caption(msg)

    if len(result) == 0:
        st.warning("当前筛选条件下没有符合要求的达人，请放宽筛选条件后重试。")
        st.stop()

    # ==================== 预收集展示数据（同时构建下载内容）====================
    src_label = {'exact': '该品类实际数据', 'overall': '跨品类均值', 'no_data': '无数据'}
    card_data = []
    for rank, row in result.iterrows():
        handle  = row['handle']
        score   = row['matching_score']
        f_score = row.get('final_score', score)
        ctr_s, ctor_s, rpm_s, data_src = get_content_display(handle, category, profile_by_cat)
        tags    = generate_reason(row, category, price, brand_type, is_new_launch)
        eng     = row.get('creator_avg_eng', float('nan'))
        eng_s   = f"{eng*100:.1f}%" if pd.notna(eng) and eng > 0 else '-'
        card_data.append(dict(
            rank=rank, row=row, handle=handle,
            score=score, f_score=f_score,
            ctr_s=ctr_s, ctor_s=ctor_s, rpm_s=rpm_s,
            data_src=data_src, tags=tags, eng=eng, eng_s=eng_s,
        ))

    # --- 构建下载 CSV ---
    download_rows = [{
        '排名':             d['rank'] + 1,
        '达人账号':          f"@{d['handle']}",
        '综合分':            f"{d['f_score']:.3f}",
        '模型原始分':         f"{d['score']:.3f}",
        '品类偏好度':         f"{d['row']['cat_preference']*100:.1f}%",
        '历史带货均值($)':    f"${d['row']['creator_avg_gmv']:.1f}",
        '历史订单数':         int(d['row']['creator_order_count']),
        '互动率':            d['eng_s'],
        'CTR点击率':         d['ctr_s'],
        'CTOR转化率':        d['ctor_s'],
        'RPM千次收益($)':    d['rpm_s'],
        '视频数据来源':       src_label.get(d['data_src'], '-'),
        '推荐理由':          ' | '.join(d['tags']),
    } for d in card_data]

    filter_params_rows = {
        '搜索条件': ['品类', '定价', '佣金率', '品牌定位', '最低CTR要求', '品类带货经验', '新品冷启动', '推荐数上限', '实际返回达人数'],
        '设置值':  [
            category,
            f'${price:.0f}',
            f'{commission:.1f}%',
            brand_type,
            f'{min_ctr_pct:.1f}%' if min_ctr_pct > 0 else '不限',
            cat_exp,
            '是' if is_new_launch else '否',
            top_n,
            len(result),
        ],
    }

    buf = io.StringIO()
    buf.write('【搜索条件】\n')
    pd.DataFrame(filter_params_rows).to_csv(buf, index=False)
    buf.write('\n【推荐结果】\n')
    pd.DataFrame(download_rows).to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode('utf-8-sig')

    with dl_col:
        st.download_button(
            label="⬇️ 下载结果 CSV",
            data=csv_bytes,
            file_name=f"达人推荐_{category}_{price:.0f}美元_{pd.Timestamp.now().strftime('%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ==================== 卡片展示（紧凑版）====================
    for d in card_data:
        rank      = d['rank']
        row       = d['row']
        handle    = d['handle']
        score     = d['score']
        f_score   = d['f_score']
        ctr_s     = d['ctr_s']
        ctor_s    = d['ctor_s']
        rpm_s     = d['rpm_s']
        data_src  = d['data_src']
        tags      = d['tags']
        eng_s     = d['eng_s']
        extra = f"&nbsp; *(模型分 {score:.3f})*" if (boost_label or is_new_launch) else ""

        with st.container(border=True):
            # --- 行1：账号 + 综合分（用加粗普通文字，不用 ### 标题）---
            h_col, s_col = st.columns([3, 2])
            with h_col:
                st.markdown(f"**#{rank+1}** &nbsp; `@{handle}`")
            with s_col:
                st.markdown(
                    f"**综合分** &nbsp; {score_color(f_score)} &nbsp; "
                    f"`{score_bar(f_score)}` &nbsp; **{f_score:.3f}**{extra}",
                )

            # --- 行2：8列指标（原两行4列合并为一行）---
            m1, m2, m3, m4, c1, c2, c3, c_note = st.columns(8)
            with m1:
                st.metric("品类偏好度", f"{row['cat_preference']*100:.1f}%",
                          help="该达人历史已结算订单中，此品类占比")
            with m2:
                st.metric("带货均值GMV", f"${row['creator_avg_gmv']:.1f}",
                          help="达人历史已结算订单的平均 GMV")
            with m3:
                st.metric("历史订单数", f"{int(row['creator_order_count'])}条",
                          help="已结算订单总数，反映带货经验量")
            with m4:
                st.metric("互动率", eng_s,
                          help="历史视频（点赞+评论+分享）÷ 观看数 的平均值")
            with c1:
                st.metric("CTR", ctr_s, help="视频被点击进商品详情页的比率")
            with c2:
                st.metric("CTOR", ctor_s, help="点击后下单的比率")
            with c3:
                st.metric("RPM", rpm_s, help="每千次视频播放产生的 GMV")
            with c_note:
                if data_src == 'overall':
                    st.caption("⚠️ 跨品类均值")
                elif data_src == 'no_data':
                    st.caption("⚠️ 无视频数据")
                else:
                    st.caption(f"✅ {category}品类")

            # --- 行3：推荐理由 ---
            st.markdown("**推荐理由：** " + "　".join(tags))

    # ==================== 冷候选池（Fastmoss 达人榜）====================
    # 优先级：手动上传 > 本地自动加载（fastmoss_pool.xlsx）> 不展示
    if fm_file is not None:
        with st.spinner("处理 Fastmoss 候选池（已上传文件）..."):
            fm_pool = load_fm_pool(fm_file)
    elif os.path.exists(_FM_AUTO_PATH):
        with st.spinner("加载 Fastmoss 候选池（本地文件）..."):
            fm_pool = _try_load_fm_auto(_FM_AUTO_PATH)
    else:
        fm_pool = None

    if fm_pool is not None:
        if len(fm_pool) == 0:
            st.info("Fastmoss 文件中未找到美妆类且有视频带货记录的达人，请确认文件格式。")
        else:
            # 排除模型结果中已有的达人
            existing_handles = set(result['handle'].str.lower().tolist())
            fm_filtered = fm_pool[~fm_pool['handle'].str.lower().isin(existing_handles)].copy()

            st.divider()
            st.subheader(f"🔍 扩展候选池  ·  Fastmoss 达人榜（{len(fm_filtered)} 位）")
            st.caption(
                "以下达人来自 Fastmoss 带货达人榜，无历史订单数据，评分基于规则层：**视频GMV占比（40%）× 粉丝量级适配（35%）× 带货经验（25%）**。  \n"
                "仅供发现参考，建议人工核实内容风格与品牌调性是否匹配后再决策。"
            )

            fm_top = fm_filtered.head(top_n).reset_index(drop=True)

            # 构建展示表格
            fm_rows = []
            for _, r in fm_top.iterrows():
                followers = r['followers']
                if followers >= 10_000:
                    followers_str = f"{followers/10_000:.1f}万"
                else:
                    followers_str = f"{int(followers)}"

                fm_rows.append({
                    '排名':       f"#{_ + 1}",
                    '达人账号':   f"@{r['handle']}",
                    '昵称':       r['nickname'],
                    '粉丝数':     followers_str,
                    '视频GMV($)': f"${r['video_gmv']:,.0f}",
                    '视频占比':   f"{r['video_ratio']*100:.0f}%",
                    '候选分':     f"{r['cold_score']:.3f}",
                    'Fastmoss':  r.get('fastmoss_url', ''),
                })

            fm_display = pd.DataFrame(fm_rows)
            st.dataframe(fm_display, use_container_width=True, hide_index=True,
                         column_config={
                             'Fastmoss': st.column_config.LinkColumn('Fastmoss 详情页', display_text='🔗 查看'),
                         })

            # 冷候选池下载
            buf2 = io.StringIO()
            buf2.write('【搜索条件（同主推荐）】\n')
            pd.DataFrame(filter_params_rows).to_csv(buf2, index=False)
            buf2.write('\n【扩展候选池（Fastmoss规则层）】\n')
            fm_display.to_csv(buf2, index=False)
            st.download_button(
                label="⬇️ 下载扩展候选池 CSV",
                data=buf2.getvalue().encode('utf-8-sig'),
                file_name=f"扩展候选池_{category}_{pd.Timestamp.now().strftime('%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    st.divider()
    st.caption(
        "**综合分**：模型预测分（70%）× 品类专长（30%）；启用品牌定位时进一步加权对应信号  \n"
        "模型预测分：预测该达人带此品类商品 GMV 高于品类中位数的概率（0–1）  \n"
        "CTR / CTOR / RPM 使用贝叶斯平滑处理，视频数越少的达人越向品类全局均值回归"
    )
