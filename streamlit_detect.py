import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ========== 1、界面基础美化配置 ==========
st.set_page_config(
    page_title="人群计数检测工具",
    layout="wide",  # 宽屏，图片更大更好看
    page_icon="👨‍👩‍👧‍👦"  # 浏览器标签图标
)

# 顶部标题美化+分割线
st.title(" 图片多人检测与计数系统")
st.divider()  # 分割线，区分标题和功能区


# ========== 2、模型缓存与切换 ==========
@st.cache_resource
def load_yolo11():
    return YOLO("yolo11m.pt")


@st.cache_resource
def load_custom():
    return YOLO("best.pt")


# 侧边栏放设置，主页面只放图片，界面整洁
with st.sidebar:
    st.header("⚙ 参数设置")
    # 模型选择下拉框
    model_type = st.selectbox("选择检测模型", ["通用YOLO11", "人群专用训练模型"])
    # 可视化调节conf、iou，不用改代码就能调
    conf_val = st.slider("置信度 conf", min_value=0.05, max_value=0.6, value=0.15, step=0.05)
    iou_val = st.slider("去重阈值 iou", min_value=0.1, max_value=0.7, value=0.35, step=0.05)

# 加载对应模型
if model_type == "通用YOLO11":
    model = load_yolo11()
else:
    model = load_custom()

# ========== 3、主页面上传图片区域 ==========
col1, col2 = st.columns(2)  # 左右分栏：原图 | 检测结果
upload_img = st.file_uploader("点击上传待检测图片", type=["jpg", "png", "jpeg"])

if upload_img is not None:
    img = Image.open(upload_img)
    with col1:
        st.subheader("原图展示")
        st.image(img, use_column_width=True)

    # 推理检测
    res = model.predict(
        img,
        conf=conf_val,
        iou=iou_val,
        imgsz=1280,
        classes=[0]
    )
    detect_img = res[0].plot()
    people_count = len(res[0].boxes)

    with col2:
        st.subheader("检测结果")
        st.image(detect_img, use_column_width=True)
        # 醒目的人数展示
        st.success(f"✅ 图片内识别到总人数：{people_count} 人")

# 底部备注
st.divider()
st.caption("使用说明：左侧侧边栏切换模型、调整检测参数；上传图片自动识别人体并统计数量")