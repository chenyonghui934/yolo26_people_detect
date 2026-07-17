import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 页面基础设置
st.set_page_config(page_title="人数检测工具", layout="wide")
st.title("图片人群计数检测")

# 加载模型（只加载一次，加速）
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()

# 上传图片组件
upload_img = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if upload_img is not None:
    # 转opencv图像
    file_bytes = np.asarray(bytearray(upload_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 推理参数（你调好的平衡配置）
    results = model(
        img,
        conf=0.10,
        imgsz=1280,
        max_det=1000,
        iou=0.10,
        augment=True,
        agnostic_nms=True
    )

    person_count = 0
    for res in results:
        boxes = res.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_score = float(box.conf[0])
            cv2.putText(
                img_rgb, f"person {conf_score:.2f}",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1
            )

    # 绘制总人数文字
    cv2.putText(
        img_rgb, f"Total People: {person_count}",
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
        1.1, (255, 0, 0), 2
    )

    # 展示结果
    st.subheader(f"检测完成，总人数：{person_count}")
    st.image(img_rgb, use_column_width=True)