import hashlib

import cv2
import requests
from ultralytics import YOLO
import time
import os
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.keras.models import load_model


def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # get cv2 install path
    # cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    # print(cv2_base_dir)
    # get now file
    cap = cv2.CascadeClassifier(
        './cv2_xml_files/haarcascade_frontalface_alt2.xml'
    )

    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    return img


def v8_pose():
    # 加载YOLO姿态检测模型
    model = YOLO('./YOLOmodels/yolov8x-pose.pt')  # 使用预训练的姿态检测模型

    # 打开摄像头
    cap = cv2.VideoCapture('/dev/video0')  # 0表示默认摄像头

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    startTime = datetime.datetime.now()
    emotion_classifier = load_model(
        'classifier/emotion_models/simple_CNN.530-0.65.hdf5',
        custom_objects={
            'BatchNormalization': tf.keras.layers.BatchNormalization,  # TensorFlow Keras 方式
            # 若有其他自定义层（如激活函数、损失函数），也需在此注册
        }
    )
    endTime = datetime.datetime.now()
    print(endTime - startTime)

    # # 获取摄像头默认分辨率
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # # 初始化视频保存
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    # 设置检测参数
    conf_threshold = 0.5  # 置信度阈值

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        # detect face
        frame = discern(frame)

        # 使用YOLO进行姿态检测
        results = model(frame, verbose=False, conf=conf_threshold)  # verbose=False关闭控制台输出


        # 绘制检测结果
        annotated_frame = results[0].plot()  # 自动绘制检测框和关键点

        # 显示实时画面
        cv2.imshow('Pose Detection', annotated_frame)

        # # # 保存处理后的帧到视频
        # # out.write(annotated_frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    # out.release()
    cv2.destroyAllWindows()







if __name__ == '__main__':
    try:
        v8_pose()

    except KeyboardInterrupt:
        print('Exiting...')

