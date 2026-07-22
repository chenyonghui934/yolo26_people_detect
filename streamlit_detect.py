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


# 复用你的人数统计过滤函数
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


# 图片检测函数
def detect_image(image_pil, conf, iou):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    res = model(img_cv, conf=conf, iou=iou)
    num = count_people_box(res)
    result_img = res[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img, num


# 视频处理函数：全部内存操作，规避临时文件找不到问题
def detect_video_file(video_bytes, conf, iou):
    temp_video = "temp_upload_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video)
    if not cap.isOpened():
        os.remove(temp_video)
        raise Exception("无法解析上传的视频文件")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise Exception("读取视频分辨率失败")

    output_path = "video_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, conf=conf, iou=iou, persist=True, classes=[0])
        draw_frame = results[0].plot()
        out.write(draw_frame)

    cap.release()
    out.release()
    os.remove(temp_video)

    # 将生成结果读取到内存，再返回二进制数据
    with open(output_path, "rb") as f:
        video_buffer = io.BytesIO(f.read())
    os.remove(output_path)
    return video_buffer


# 侧边参数面板
with st.sidebar:
    st.header("参数设置")
    conf_val = st.slider("置信度 conf", min_value=0.01, max_value=0.99, value=0.35, step=0.01)
    iou_val = st.slider("IOU阈值", min_value=0.01, max_value=0.99, value=0.65, step=0.01)

# 分页标签
tab1, tab2 = st.tabs(["图片检测", "视频文件检测"])

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

# 视频检测页面
with tab2:
    upload_vid = st.file_uploader("上传本地视频文件", type=["mp4", "mov"])
    if upload_vid is not None:
        st.video(upload_vid)
        if st.button("开始分析视频"):
            with st.spinner("视频处理中，请勿刷新页面，长视频需要等待较长时间..."):
                vid_data = upload_vid.read()
                video_buf = detect_video_file(vid_data, conf_val, iou_val)
                st.success("视频检测完成！下方为标注结果视频")
                # 传入内存二进制对象，不再读取本地文件路径
                st.video(video_buf)