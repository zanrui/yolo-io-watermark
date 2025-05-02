#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:59
# describe：使用yolo提取图片中目标边框的工具

from ultralytics import YOLO

import configs


class YOLOUtils:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_bboxes(self, image, conf=0.75):
        results = self.model(image, conf=conf)
        bboxes = []
        for i in range(len(results[0].boxes)):
            m = results[0].boxes[i].cpu().data.numpy().tolist()[0]
            bboxes.append(m)
        return results, bboxes

    def get_model(self):
        return self.model
    


if __name__ == "__main__":
    model = YOLO(f"{configs.models_dir}/last.pt")
