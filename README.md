## 小红书logo的水印检测demo

本demo使用`labelImg`对`24`张样本图片的水印位置进行标注，[ultralytics-YOLO8](https://github.com/ultralytics/ultralytics)对水印位置进行模型训练&检测。

如果需要使用 [ultralytics-YOLO8](https://github.com/ultralytics/ultralytics) + [IOPaint](https://github.com/Sanster/IOPaint) 进行组合，自动移除yolo识别的目标水印，请点击[yolo8-plus-iopaint](https://github.com/Samge0/yolo8-plus-iopaint)仓库查看。


### 当前开发环境使用的关键依赖版本
```text
python==3.8.18
torch==2.3.0+cu118
torchvision==0.18.0+cu118
ultralytics==8.2.28

# labelImg is used to label the training data
labelImg==1.8.6
```


### 环境配置
- 【推荐】使用vscode的`Dev Containers`模式，参考[.devcontainer/README.md](.devcontainer/README.md)

    复制并根据实际情况调整`Dev Containers`插件的容器编排配置，需要关注的主要是`/root/.cache`的路径映射、是否配置代理`PROXY`
    ```shell
    cp .devcontainer/docker-compose-demo.yml .devcontainer/docker-compose.yml
    ```

- 【可选】其他虚拟环境方式
    - 【二选一】安装torch-cpu版
        ```shell
        pip install torch torchvision
        ```
    - 【二选一】安装torch-cuda版
        ```shell
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ```
    - 【必要】安装依赖
        ```shell
        pip install -r requirements.txt
        ```


### 训练
```shell
python train.py
```


### 推理
```shell
python test.py
```

### 自定义数据集进行训练：
- 安装`labelImg`
    ```shell
    pip install labelImg
    ```

- 启动`labelImg`
    ```shell
    labelImg
    ```

- 清理或备份旧的数据集，将需要训练的新数据图集放到[datasets/data/images](datasets/data/images)目录，参与训练的图片宽高最好一致，训练前需要在[train.py](train.py)中配置`imgsz`图片宽高信息
- 在`labelImg`打开[datasets/data/images](datasets/data/images)的图集进行标注，保存格式选择`YOLO`（建议点击`File -> YOLO`保存全局默认`YOLO`导出后，重新打开`labelImg`，可在后续保存标注时避免频繁切换导出格式）
- 标注完毕后，执行命令将[datasets/data/images](datasets/data/images)拆分为[datasets/data/train](datasets/data/train)、[datasets/data/test](datasets/data/test)、[datasets/data/val](datasets/data/val)
    ```shell
    cd datasets && python Process.py
    ```
- 按前面文档所示，执行`python train.py`进行训练，执行`python test.py`进行推理


### 相关截图
- labelImg标注界面
![image](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/1e1c64f6-5049-4e43-aeb7-5cfedd305bee)

- 训练后的模型预测结果
![image](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/6eaf4f70-f1a9-4605-8f9a-7ecf7abba921)

|img1|img2|
|:--------:|:--------:|
|![test](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/04fdd1d9-6055-4774-a973-e11882a75b15)|![tmp9zby3ksb](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/db2d1deb-00e8-4c89-bdc9-72fdfb1fb658)|
|![tmp2046ojfe](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/8a33950c-5ee3-49f5-91f5-38a3c2e7b32c)|![tmpyv0z5qty](https://github.com/Samge0/yolo8-watermark-xhs/assets/17336101/288efd14-7cc2-4a2d-86a5-a825d8e9a02d)|