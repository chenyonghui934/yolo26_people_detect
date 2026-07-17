import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 页面基础设置
st.set_page_config(page_title="人数检测工具", layout="wide")
st.title("图片人群计数检测")

# 加载模型（只加载一次，加速）
@st.cache_resource
def load_model():
    return YOLO("yolo11m.pt")
model = load_model()

# 上传图片组件
upload_img = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if upload_img is not None:
    # PIL读取图片，完全不依赖OpenCV
    img = Image.open(upload_img).convert("RGB")

    # 密集人群调参
    results = model(
        img,
        conf=0.10,
        imgsz=1280,
        max_det=1000,
        iou=0.10,
        augment=True,
        agnostic_nms=True,
        classes=[0]
    )

    # YOLO自带画框，不用cv2手动绘制
    result_img = results[0].plot()
    person_count = len(results[0].boxes)

    # 页面展示
    st.subheader(f"检测完成，总人数：{person_count}")
    st.image(result_img, use_column_width=True)
