# -*- coding: utf-8 -*-
"""
东北边境支线机场货邮吞吐量数据分析与预测系统
================================================
课程设计：民航交通工程
研究对象：牡丹江海浪机场 / 佳木斯东郊机场 / 黑河瑷珲机场

【项目特色说明 —— 请保留于提交版本中】
① 区域特殊性：三座机场均为对俄口岸支线机场，货邮结构中外贸货占比显著，
   且受黑龙江明水期/流冰期/封冻期及俄罗斯果蔬需求季节性的叠加影响；
② 气象—货运耦合：将"月平均风速"作为数据字段纳入管理，可拓展相关性分析；
③ 预测方法民航规范性：采用机场规划经典"货运弹性系数法"
   （弹性系数 = 货邮增长率 / 区域GDP增长率），而非黑箱机器学习模型；
④ 教学适配性：代码精简、注释完整、pyecharts 交互渲染，适合课堂演示答辩。

运行方式：
    pip install streamlit pandas pyecharts numpy
    streamlit run main.py
首次运行自动在 ./data/airport_cargo.csv 生成 3 机场 × 6 年 × 12 月
共 216 条演示数据，无需手动准备文件。
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import Line, Pie, HeatMap, Scatter

# =============================================================================
# 0. 通用工具
# =============================================================================
def st_pyecharts(chart, key: str = None):
    """将 pyecharts 图表嵌入 Streamlit（不依赖 streamlit-echarts 组件包）。"""
    try:
        h = int(str(chart.height).replace("px", ""))
    except Exception:
        h = 400
    components.html(chart.render_embed(), height=h, scrolling=False)


def rerun():
    """兼容不同版本 Streamlit 的页面刷新。"""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# =============================================================================
# 1. 数据初始化（首次运行自动生成演示数据集）
# =============================================================================
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "airport_cargo.csv")

AIRPORTS = ["牡丹江海浪机场", "佳木斯东郊机场", "黑河瑷珲机场"]

# 各机场年货邮量级锚点（吨），体现三机场量级差异
BASE_LEVEL = {
    "牡丹江海浪机场": 1250.0,
    "佳木斯东郊机场": 2150.0,
    "黑河瑷珲机场": 130.0,
}

# 季节因子：东北对俄口岸特征
# 牡丹江/佳木斯：9—12 月对俄果蔬、轻工出口旺季；
# 黑河：5—10 月明水期口岸活跃、冬季封冻期货运转陆路，航空货量偏低。
SEASON_FACTOR = {
    "牡丹江海浪机场": [0.82, 0.78, 0.85, 0.92, 0.96, 1.00,
                      1.00, 1.02, 1.18, 1.30, 1.25, 1.08],
    "佳木斯东郊机场": [0.80, 0.76, 0.84, 0.90, 0.95, 0.98,
                      0.98, 1.00, 1.15, 1.28, 1.22, 1.05],
    "黑河瑷珲机场": [0.72, 0.70, 0.78, 0.88, 1.05, 1.15,
                   1.18, 1.12, 1.05, 0.95, 0.85, 0.76],
}

# 年度 GDP 增速基准（%），叠加随机扰动
GDP_GROWTH_BASE = {
    "牡丹江海浪机场": [3.0, 6.8, 3.0, 4.2, 3.9, 4.0],
    "佳木斯东郊机场": [2.5, 6.5, 3.5, 4.5, 4.9, 4.7],
    "黑河瑷珲机场": [3.5, 6.2, 3.0, 4.0, 3.3, 4.4],
}

# 外贸货占比基准（%）：黑河口岸属性最突出，牡丹江次之
FOREIGN_RATIO_BASE = {
    "牡丹江海浪机场": 32.0,
    "佳木斯东郊机场": 24.0,
    "黑河瑷珲机场": 41.0,
}

COLUMNS = [
    "机场名称", "年份", "月份", "货邮吞吐量(吨)", "货机架次",
    "外贸货占比(%)", "区域GDP增速(%)", "月平均风速(m/s)",
]


def _generate_demo_data() -> pd.DataFrame:
    """生成演示数据集并写入 CSV。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(20240501)
    rows = []
    for ap in AIRPORTS:
        base = BASE_LEVEL[ap]
        gdp_seq = GDP_GROWTH_BASE[ap]
        fb = FOREIGN_RATIO_BASE[ap]
        for yi, year in enumerate(range(2020, 2026)):
            # 年度趋势：GDP 增速与货运近似联动，系数 0.55 模拟弱弹性
            annual_trend = 1.0 + (gdp_seq[yi] / 100.0) * 0.55
            for month in range(1, 13):
                season = SEASON_FACTOR[ap][month - 1]
                cargo = base * (annual_trend ** yi) * season
                cargo = max(round(cargo * rng.normal(1.0, 0.055), 2), 1.0)
                # 货机架次：以吨位折算，含随机噪声
                sorties = max(int(cargo / rng.uniform(2.8, 4.2)), 2)
                # 外贸货占比：旺季出口占比上浮
                foreign = fb + (season - 1.0) * 18 + rng.normal(0, 2.5)
                foreign = round(float(np.clip(foreign, 3, 78)), 1)
                # 月平均风速：春冬偏高，北方特征
                wind = 3.6 + 1.4 * np.sin((month - 3) / 12 * 2 * np.pi) ** 2
                wind = round(float(max(wind + rng.normal(0, 0.35), 1.2)), 1)
                rows.append([
                    ap, year, month, cargo, sorties, foreign,
                    round(gdp_seq[yi] + rng.normal(0, 0.25), 2), wind,
                ])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    return df


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return _generate_demo_data()


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")


# =============================================================================
# 2. 页面配置
# =============================================================================
st.set_page_config(
    page_title="东北边境支线机场货邮数据分析与预测",
    page_icon="✈️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size:1.6rem;font-weight:700;color:#0B3D6B;
                 border-left:6px solid #1E6FA8;padding-left:12px;}
    .sub-note {color:#4A5A6A;font-size:0.9rem;margin-top:-6px;}
    div[data-testid="stMetricValue"] {font-size:1.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">东北边境支线机场货邮吞吐量数据分析与预测系统</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-note">牡丹江海浪 · 佳木斯东郊 · 黑河瑷珲 ｜ '
    '对俄口岸支线航空货运 · 弹性系数法预测</div>',
    unsafe_allow_html=True,
)

df = load_data()

with st.sidebar:
    st.header("✈️ 导航")
    page = st.radio(
        "功能模块",
        ["📊 数据概览", "🛠️ 数据管理", "📈 可视化分析", "🎯 弹性系数预测"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("课程设计 · 民航交通工程")
    st.caption(f"数据文件：{DATA_PATH}")


# =============================================================================
# 3. 页面：数据概览
# =============================================================================
if page == "📊 数据概览":
    st.subheader("数据概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录总数", f"{len(df):,}")
    c2.metric("覆盖机场", f"{df['机场名称'].nunique()} 座")
    c3.metric("时间跨度", f"{df['年份'].min()}—{df['年份'].max()}")
    c4.metric("总货邮量（吨）", f"{df['货邮吞吐量(吨)'].sum():,.0f}")

    st.markdown("#### 各机场年度货邮吞吐量汇总（吨）")
    pivot = (
        df.groupby(["机场名称", "年份"])["货邮吞吐量(吨)"]
        .sum().round(1).unstack("年份")
    )
    st.dataframe(pivot, use_container_width=True)

    st.markdown("#### 原始数据预览（前 20 行）")
    st.dataframe(df.head(20), use_container_width=True)


# =============================================================================
# 4. 页面：数据管理（增删改查 + CSV 导入导出）
# =============================================================================
elif page == "🛠️ 数据管理":
    st.subheader("数据管理")
    tab_q, tab_add, tab_edit, tab_del, tab_io = st.tabs(
        ["🔍 查询", "➕ 新增", "✏️ 修改", "🗑️ 删除", "📁 导入/导出"]
    )

    # ---------- 查询 ----------
    with tab_q:
        c1, c2, c3 = st.columns(3)
        with c1:
            q_ap = st.selectbox("机场", ["全部"] + sorted(df["机场名称"].unique()),
                                key="q_ap")
        with c2:
            q_yr = st.selectbox("年份", ["全部"] + sorted(df["年份"].unique()),
                                key="q_yr")
        with c3:
            q_mo = st.selectbox("月份", ["全部"] + list(range(1, 13)), key="q_mo")
        qdf = df.copy()
        if q_ap != "全部":
            qdf = qdf[qdf["机场名称"] == q_ap]
        if q_yr != "全部":
            qdf = qdf[qdf["年份"] == q_yr]
        if q_mo != "全部":
            qdf = qdf[qdf["月份"] == q_mo]
        st.caption(f"命中 {len(qdf)} 条记录")
        st.dataframe(qdf, use_container_width=True)

    # ---------- 新增 ----------
    with tab_add:
        with st.form("add_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                a_ap = st.selectbox("机场名称", AIRPORTS, key="a_ap")
                a_yr = st.number_input("年份", 2020, 2035, 2025, step=1, key="a_yr")
            with c2:
                a_mo = st.number_input("月份", 1, 12, 1, step=1, key="a_mo")
                a_cg = st.number_input("货邮吞吐量(吨)", 0.0, 100000.0, 100.0,
                                       step=1.0, key="a_cg")
            with c3:
                a_st = st.number_input("货机架次", 0, 100000, 30, step=1, key="a_st")
                a_fr = st.number_input("外贸货占比(%)", 0.0, 100.0, 25.0,
                                       step=0.5, key="a_fr")
            c4, c5 = st.columns(2)
            with c4:
                a_gdp = st.number_input("区域GDP增速(%)", -20.0, 30.0, 4.0,
                                        step=0.1, key="a_gdp")
            with c5:
                a_wd = st.number_input("月平均风速(m/s)", 0.0, 30.0, 3.5,
                                       step=0.1, key="a_wd")
            if st.form_submit_button("确认新增", type="primary"):
                dup = (
                    (df["机场名称"] == a_ap)
                    & (df["年份"] == int(a_yr))
                    & (df["月份"] == int(a_mo))
                )
                if dup.any():
                    st.error("该机场该年月已存在记录，请使用“修改”功能。")
                else:
                    df = pd.concat([df, pd.DataFrame([{
                        "机场名称": a_ap, "年份": int(a_yr), "月份": int(a_mo),
                        "货邮吞吐量(吨)": float(a_cg), "货机架次": int(a_st),
                        "外贸货占比(%)": float(a_fr), "区域GDP增速(%)": float(a_gdp),
                        "月平均风速(m/s)": float(a_wd),
                    }])], ignore_index=True)
                    save_data(df)
                    st.success("新增成功。")
                    rerun()

    # ---------- 修改 ----------
    with tab_edit:
        options = list(df.index[:200])
        edit_idx = st.selectbox(
            "选择要修改的记录（显示前 200 条）",
            options=options,
            format_func=lambda i: (
                f"{df.loc[i, '机场名称']} | {int(df.loc[i, '年份'])}-"
                f"{int(df.loc[i, '月份']):02d} | "
                f"{df.loc[i, '货邮吞吐量(吨)']:.1f} 吨"
            ),
            key="edit_sel",
        )
        row = df.loc[edit_idx]
        with st.form("edit_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                e_cg = st.number_input("货邮吞吐量(吨)", 0.0, 100000.0,
                                       float(row["货邮吞吐量(吨)"]), step=1.0)
                e_st = st.number_input("货机架次", 0, 100000,
                                       int(row["货机架次"]), step=1)
            with c2:
                e_fr = st.number_input("外贸货占比(%)", 0.0, 100.0,
                                       float(row["外贸货占比(%)"]), step=0.5)
                e_gdp = st.number_input("区域GDP增速(%)", -20.0, 30.0,
                                        float(row["区域GDP增速(%)"]), step=0.1)
            with c3:
                e_wd = st.number_input("月平均风速(m/s)", 0.0, 30.0,
                                       float(row["月平均风速(m/s)"]), step=0.1)
            if st.form_submit_button("保存修改", type="primary"):
                df.loc[edit_idx, "货邮吞吐量(吨)"] = float(e_cg)
                df.loc[edit_idx, "货机架次"] = int(e_st)
                df.loc[edit_idx, "外贸货占比(%)"] = float(e_fr)
                df.loc[edit_idx, "区域GDP增速(%)"] = float(e_gdp)
                df.loc[edit_idx, "月平均风速(m/s)"] = float(e_wd)
                save_data(df)
                st.success("修改已保存。")
                rerun()

    # ---------- 删除 ----------
    with tab_del:
        del_idx = st.selectbox(
            "选择要删除的记录",
            options=list(df.index),
            format_func=lambda i: (
                f"{df.loc[i, '机场名称']} | {int(df.loc[i, '年份'])}-"
                f"{int(df.loc[i, '月份']):02d} | "
                f"{df.loc[i, '货邮吞吐量(吨)']:.1f} 吨"
            ),
            key="del_sel",
        )
        st.warning("删除操作不可撤销。")
        if st.button("确认删除", type="secondary"):
            df = df.drop(index=del_idx).reset_index(drop=True)
            save_data(df)
            st.success("已删除。")
            rerun()

    # ---------- 导入/导出 ----------
    with tab_io:
        st.markdown("**CSV 批量导入**")
        up = st.file_uploader("上传 CSV（字段需与系统一致，utf-8 编码）",
                              type=["csv"])
        if up is not None:
            try:
                imported = pd.read_csv(up, encoding="utf-8-sig")
                missing = set(COLUMNS) - set(imported.columns)
                if missing:
                    st.error(f"缺少字段：{missing}")
                else:
                    save_data(imported[COLUMNS])
                    st.success(f"导入成功，共 {len(imported)} 条记录。")
                    rerun()
            except Exception as exc:
                st.error(f"导入失败：{exc}")

        st.divider()
        st.markdown("**一键导出**")
        st.download_button(
            label="⬇️ 下载当前数据 CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="airport_cargo_export.csv",
            mime="text/csv",
        )


# =============================================================================
# 5. 页面：可视化分析
# =============================================================================
elif page == "📈 可视化分析":
    st.subheader("可视化分析")

    scope = st.radio(
        "查看范围",
        ["全部机场对比"] + sorted(df["机场名称"].unique()),
        horizontal=True,
        key="viz_scope",
    )
    plot_df = df if scope == "全部机场对比" else df[df["机场名称"] == scope].copy()

    # ---- 5.1 时序趋势图 ----
    st.markdown("#### ① 货邮吞吐量月度时序趋势")
    monthly = (
        plot_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"]
        .sum().reset_index().sort_values(["年份", "月份"])
    )
    monthly["时间"] = (
        monthly["年份"].astype(str) + "-"
        + monthly["月份"].astype(str).str.zfill(2)
    )
    line = Line(init_opts=opts.InitOpts(height="360px"))
    line.add_xaxis(monthly["时间"].tolist())
    line.add_yaxis(
        "货邮吞吐量(吨)",
        monthly["货邮吞吐量(吨)"].round(1).tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2.4),
    )
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axislabel_opts=opts.LabelOpts(rotate=55, interval=5),
        ),
        yaxis_opts=opts.AxisOpts(name="吨"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
    st_pyecharts(line, key=f"line_{scope}")

    # ---- 5.2 货机架次占比饼图 ----
    st.markdown("#### ② 货机架次占比")
    pie = Pie(init_opts=opts.InitOpts(height="340px"))
    if scope == "全部机场对比":
        pie_data = (
            df.groupby("机场名称")["货机架次"].sum()
            .sort_values(ascending=False)
        )
        pie.add("", [(str(k), int(v)) for k, v in pie_data.items()],
                radius=["32%", "62%"],
                label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
    else:
        pie_data = plot_df.groupby("年份")["货机架次"].sum().sort_index()
        pie.add("", [(f"{k}年", int(v)) for k, v in pie_data.items()],
                radius=["32%", "62%"],
                label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
    pie.set_global_opts(
        legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
    )
    st_pyecharts(pie, key=f"pie_{scope}")

    # ---- 5.3 月度货运季节热力图 ----
    st.markdown("#### ③ 月度货运季节热力图")
    heat_df = df if scope == "全部机场对比" else plot_df
    heat_caption = "全部机场合计" if scope == "全部机场对比" else scope

    heat_pivot = (
        heat_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"]
        .sum().unstack("月份")
    )
    years_h = heat_pivot.index.tolist()
    heat_values = []
    for yi, yr in enumerate(years_h):
        for m in range(1, 13):
            v = heat_pivot.loc[yr, m] if m in heat_pivot.columns else 0
            if pd.isna(v):
                v = 0
            heat_values.append([m - 1, yi, round(float(v), 1)])
    vmin = min(v[2] for v in heat_values)
    vmax = max(v[2] for v in heat_values)

    heat = HeatMap(init_opts=opts.InitOpts(height="360px"))
    heat.add_xaxis([f"{m}月" for m in range(1, 13)])
    heat.add_yaxis(
        "年份",
        [str(y) for y in years_h],
        heat_values,
        label_opts=opts.LabelOpts(is_show=True, font_size=9),
    )
    heat.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            min_=vmin, max_=vmax, is_calculable=True,
            orient="horizontal", pos_left="center",
        ),
        xaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False)),
    )
    st.caption(f"热力主体：{heat_caption}")
    st_pyecharts(heat, key=f"heat_{scope}")

    # ---- 5.4 GDP增速—货邮相关性散点图 ----
    st.markdown("#### ④ GDP增速—货邮吞吐量相关性散点图（弹性系数分析用）")
    sc_df = df.copy()
    if scope != "全部机场对比":
        sc_df = sc_df[sc_df["机场名称"] == scope]
    sc_df = sc_df.sample(min(len(sc_df), 500), random_state=42)

    scatter = Scatter(init_opts=opts.InitOpts(height="400px"))
    scatter.add_xaxis(sc_df["区域GDP增速(%)"].round(2).tolist())
    for ap in sorted(sc_df["机场名称"].unique()):
        sub = sc_df[sc_df["机场名称"] == ap]
        scatter.add_yaxis(
            ap,
            sub["货邮吞吐量(吨)"].round(1).tolist(),
            symbol_size=7,
            label_opts=opts.LabelOpts(is_show=False),
        )
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name="区域GDP增速(%)", type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        yaxis_opts=opts.AxisOpts(name="货邮吞吐量(吨)", type_="value"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        legend_opts=opts.LegendOpts(pos_top="2%"),
    )
    st_pyecharts(scatter, key=f"scatter_{scope}")
    st.caption(
        "说明：散点整体向右上方倾斜的程度可作为弹性系数 E>0 的直观佐证；"
        "若某机场散点分散，表明货运量还受到口岸季节、外贸政策等非GDP因素干扰。"
    )


# =============================================================================
# 6. 页面：弹性系数预测
# =============================================================================
elif page == "🎯 弹性系数预测":
    st.subheader("货运弹性系数预测")

    st.markdown(
        r"""
**方法说明（机场规划 / 交通工程经典方法）**

弹性系数 \(E\) 定义为货邮吞吐量增长率与区域 GDP 增长率之比：

$$E = \frac{\text{货邮年均增长率}}{\text{区域GDP年均增长率}}$$

预测递推公式：

$$Y_{t} = Y_{0} \times (1 + E \times M)^{n}$$

其中 \(Y_0\) 为基准年货邮吞吐量，\(M\) 为预测期 GDP 年均增速（小数），
\(n\) 为预测年限。该方法原理清晰、参数可由历史数据率定，是支线机场
货邮量规划中的标准方法之一。
        """
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**① 参数输入**")
        pred_ap = st.selectbox("选择机场", sorted(df["机场名称"].unique()),
                               key="pred_ap")
        ap_df = df[df["机场名称"] == pred_ap]
        recent_year = int(ap_df["年份"].max())
        default_base = float(
            ap_df.loc[ap_df["年份"] == recent_year, "货邮吞吐量(吨)"].sum()
        )

        base_cargo = st.number_input(
            "基准年货邮吞吐量（吨）",
            min_value=1.0, max_value=1_000_000.0,
            value=round(default_base, 1), step=10.0,
            help="默认取该机场最新年份全年合计，可手动覆盖。",
        )
        gdp_growth_pct = st.slider(
            "预测期区域GDP年均增速 M（%）",
            min_value=0.0, max_value=12.0, value=4.0, step=0.1,
        )
        elasticity = st.slider(
            "货运弹性系数 E",
            min_value=0.2, max_value=2.5, value=1.10, step=0.05,
            help="支线机场货邮弹性通常低于干线机场；参考区间 0.6—1.5。",
        )
        horizon = st.slider(
            "预测年限 n（年）", min_value=1, max_value=15, value=5, step=1
        )
        st.caption(
            f"默认基准值取自 {pred_ap} {recent_year} 年合计货邮量；"
            "支线机场货邮弹性参考区间 0.6—1.5。"
        )

    with col_r:
        st.markdown("**② 预测结果**")
        m = gdp_growth_pct / 100.0
        factor = 1 + elasticity * m
        forecast_values = [base_cargo * (factor ** n) for n in range(1, horizon + 1)]
        years_future = [recent_year + i for i in range(1, horizon + 1)]

        c1, c2 = st.columns(2)
        c1.metric(f"{recent_year + horizon} 年预测值",
                  f"{forecast_values[-1]:,.1f} 吨")
        c2.metric("预测期年均增长率", f"{(factor - 1) * 100:.2f}%")

        chart_years = [str(recent_year)] + [str(y) for y in years_future]
        chart_vals = [round(base_cargo, 1)] + [round(v, 1) for v in forecast_values]
        fc = Line(init_opts=opts.InitOpts(height="360px"))
        fc.add_xaxis(chart_years)
        fc.add_yaxis(
            "预测货邮吞吐量(吨)", chart_vals,
            is_smooth=True, symbol_size=8,
            label_opts=opts.LabelOpts(is_show=True, font_size=10),
        )
        fc.set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(name="吨"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
        st_pyecharts(fc, key=f"forecast_{pred_ap}")

        st.dataframe(
            pd.DataFrame({
                "年份": years_future,
                "预测货邮吞吐量(吨)": [round(v, 2) for v in forecast_values],
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.markdown("**③ 弹性系数率定辅助**")
    st.caption(
        "用历史数据估算各机场隐含弹性（货邮增长率/GDP增速）；"
        "支线机场样本量小，结果仅作教学演示参考。"
    )
    cal_rows = []
    for ap in sorted(df["机场名称"].unique()):
        ap_year = (
            df[df["机场名称"] == ap]
            .groupby("年份")
            .agg(货邮=("货邮吞吐量(吨)", "sum"),
                 GDP=("区域GDP增速(%)", "mean"))
            .reset_index()
        )
        ap_year["货邮增长率(%)"] = ap_year["货邮"].pct_change() * 100
        ap_year["隐含弹性"] = (
            ap_year["货邮增长率(%)"] / ap_year["GDP"].replace(0, np.nan)
        )
        ap_year["机场名称"] = ap
        cal_rows.append(ap_year.dropna(subset=["隐含弹性"]))
    if cal_rows:
        cal_df = pd.concat(cal_rows)[
            ["机场名称", "年份", "货邮增长率(%)", "GDP", "隐含弹性"]
        ].round(2).rename(columns={"GDP": "GDP增速(%)"})
        st.dataframe(cal_df, use_container_width=True, hide_index=True)