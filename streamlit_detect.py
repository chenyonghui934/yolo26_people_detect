import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import cv2
import numpy as np
import io

# 页面基础配置
st.set_page_config(page_title="智眸慧眼——课堂实时考勤人数统计系统", layout="wide")
st.title("智眸慧眼——课堂实时考勤人数统计系统")
st.divider()

# 模型只加载一次
@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

# 人数过滤函数
def count_people_box(results):
    people_num = 0
    min_w = 12
    min_h = 15
    max_aspect = 4.5
    boxes = results[0].boxes
    for box in boxes:
        if box.cls.item() != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w >= min_w and box_h >= min_h:
            aspect = box_h / box_w
            if aspect < max_aspect:
                people_num += 1
    return people_num

# 图片检测
def detect_image(image_pil, conf, iou):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    res = model(img_cv, conf=conf, iou=iou)
    num = count_people_box(res)
    result_img = res[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img, num

# ===================== 新版视频处理：只统计人数，不生成视频 =====================
def video_analysis_only_count(video_bytes, conf, iou):
    temp_video = "/tmp/tmp_input.mp4"
    # 清理旧文件
    if os.path.exists(temp_video):
        os.remove(temp_video)
    with open(temp_video, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video)
    if not cap.isOpened():
        os.remove(temp_video)
        raise Exception("视频文件无法读取")

    max_people = 0        # 视频内出现最多人数
    frame_people_list = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 目标追踪
        results = model.track(frame, conf=conf, iou=iou, persist=True, classes=[0])
        current_num = count_people_box(results)
        frame_people_list.append(current_num)
        if current_num > max_people:
            max_people = current_num
        frame_id += 1

    cap.release()
    os.remove(temp_video)

    avg_people = round(np.mean(frame_people_list), 2)
    return {
        "总帧数": frame_id,
        "最大同时在场人数": max_people,
        "平均人数": avg_people,
        "每一帧人数序列": frame_people_list
    }

# 侧边参数
with st.sidebar:
    st.header("参数设置")
    conf_val = st.slider("置信度 conf", min_value=0.01, max_value=0.99, value=0.35, step=0.01)
    iou_val = st.slider("IOU阈值", min_value=0.01, max_value=0.99, value=0.65, step=0.01)

tab1, tab2 = st.tabs(["图片检测", "视频人数统计(无标注视频输出)"])

# 图片检测页面
with tab1:
    upload_img = st.file_uploader("上传课堂图片", type=["jpg", "png", "jpeg"])
    if upload_img is not None:
        img_ori = Image.open(upload_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_ori, caption="原始图片", use_column_width=True)
        if st.button("开始检测图片"):
            res_img, total = detect_image(img_ori, conf_val, iou_val)
            with col2:
                st.image(res_img, caption="检测标注结果", use_column_width=True)
            st.text_area("识别结果", f"检测到课堂到场总人数：{total}")

# 视频页面【重点改动】
with tab2:
    upload_vid = st.file_uploader("上传视频文件", type=["mp4"])
    if upload_vid is not None:
        st.video(upload_vid)  # 仅展示原始上传视频，不生成新视频
        if st.button("开始分析视频，统计在场人数"):
            with st.spinner("逐帧分析视频中，请耐心等待..."):
                vid_data = upload_vid.read()
                res_data = video_analysis_only_count(vid_data, conf_val, iou_val)
            st.success("视频分析完成！")
            st.write(res_data)
            st.info(f"视频内最多同时在场人数：{res_data['最大同时在场人数']}")