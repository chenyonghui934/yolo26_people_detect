import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="人群计数", layout="wide")
st.title="多人检测工具"

# 缓存两个模型，只加载一次
@st.cache_resource
def load_yolo11():
    return YOLO("yolo11m.pt")

@st.cache_resource
def load_custom():
    return YOLO("best.pt")

# 下拉选择模型
model_choose = st.selectbox("选择检测模型",["官方YOLO11m通用模型","自定义人群专用模型(同学训练)"])
if model_choose == "官方YOLO11m通用模型":
    model = load_yolo11()
else:
    model = load_custom()

# 上传图片
upload_img = st.file_uploader("上传图片",type=["jpg","png","jpeg"])
if upload_img:
    img = Image.open(upload_img)
    st.image(img, caption="原图")
    # 推理，适配密集人群参数
    res = model.predict(img, conf=0.5, iou=0.12, imgsz=1280, classes=[0])
    detect_img = res[0].plot()
    count = len(res[0].boxes)
    st.image(detect_img, caption=f"检测完成，总人数：{count}")