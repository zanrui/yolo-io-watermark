#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from ultralytics import YOLO
import os, time
from iopaint_utils import IOPaintCmdUtil
from yolo_utils import YOLOUtils
import cv2

def _print_date_info(tag, start_date):
    elapsed_time = time.time() - start_date
    print(f"{tag}: {elapsed_time:.2f}s")

# 擦除水印
def detect_and_erase(bboxes, image_path, output_dir, device="cpu"):
    # 初始化IOPaint工具
    iopaint_obj = IOPaintCmdUtil(device=device)

    # 读取图像
    image = cv2.imread(image_path)

    # 创建并保存掩码图像
    mask = iopaint_obj.create_mask(image, bboxes)
    iopaint_obj.erase_watermark(image_path, mask, output_dir)


if __name__ == "__main__":
    # 加载模型
    start_date = time.time()
    model_path = "runs/detect/train/weights/best.pt" 
    image_path = "test.jpg"
    print(f"Load model: {model_path}")
    # 初始化YOLO模型和IOPaint工具
    yolo_obj = YOLOUtils(model_path)
    iopaint_obj = IOPaintCmdUtil(device="cpu")
    _print_date_info(f"Load model: {model_path} success, time consuming", start_date)

    # 预测测试图片
    inference_output_dir = "inference/output"
    os.makedirs(inference_output_dir, exist_ok=True)
    # 调用训练好的模型，并过滤置信度大于0.xx的结果
    results, bboxes = yolo_obj.get_bboxes(image_path, conf=0.01)
    print(f"检测水印的位置和类别：{bboxes}")
    # 保存结果
    for r in results:
        im_array = r.plot(boxes=r.boxes)
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f"{inference_output_dir}/test.jpg")

    # 移除水印
    erase_output_dir = "erase/output"
    os.makedirs(erase_output_dir, exist_ok=True)
    detect_and_erase(bboxes, image_path, erase_output_dir)

    print("\nall done")





