import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="人群计数检测工具",
    layout="wide"
)
st.title("图片多人检测与计数系统")
st.divider()

# 模型缓存
@st.cache_resource
def load_yolo11():
    return YOLO("yolo11m.pt")
@st.cache_resource
def load_custom():
    return YOLO("best.pt")

# 侧边参数
with st.sidebar:
    st.header("参数设置")
    model_type = st.selectbox("选择检测模型", ["通用YOLO11", "人群专用训练模型"])
    conf_val = st.slider("置信度 conf", min_value=0.05, max_value=1.0, value=0.15, step=0.01)
    iou_val = st.slider("去重阈值 iou", min_value=0.1, max_value=1.0, value=0.35, step=0.01)

# 加载模型
if model_type == "通用YOLO11":
    model = load_yolo11()
else:
    model = load_custom()

# 分栏上传
col1, col2 = st.columns(2)
upload_img = st.file_uploader("点击上传待检测图片", type=["jpg", "png", "jpeg"])
if upload_img is not None:
    img = Image.open(upload_img)
    with col1:
        st.subheader("原图展示")
        st.image(img, use_column_width=True)
    # 推理
    res = model.predict(img, conf=conf_val, iou=iou_val, imgsz=1280, classes=[0])
    detect_img = res[0].plot()
    # 四层过滤计数
    people_num = 0
    min_w = 25
    min_h = 35
    max_aspect = 3.0
    boxes = res[0].boxes
    for box in boxes:
        if box.cls.item() != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w >= min_w and box_h >= min_h:
            aspect_ratio = box_h / box_w
            if aspect_ratio < max_aspect:
                people_num += 1
    with col2:
        st.subheader("检测结果")
        st.image(detect_img, use_column_width=True)
        st.success(f"图片内识别到总人数：{people_num}")
st.divider()
st.caption("使用说明：左侧侧边栏切换模型、调整检测参数；上传图片自动识别人体，内置尺寸、宽高比双重过滤剔除书包、阴影干扰。")