自然灾害类-泥石流目标检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b57aadced9204b619f5c729c4d40836b.png)
6653张（数据存在增强处理），23000标注，提供yolo和voc两种标注方式
标注数量：
debris-flow: 11111
muddy: 634
rock: 11487
图像数量：
image num: 6653
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6df77df695bd43a2bb1c8a14b38a1978.png)

2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a0a9f05a5b44081a64380b1f27fefd1.png)

3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e829dc1a26934f4aa673a62e1456817f.png)

# YOLOv11‑泥石流灾害检测
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f06e4989b9c42f2be16dd87dfd23ddb.png)

## 1、数据集配置 debris.yaml![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bbcc0734601241a49b0370b651ddb254.png)

```yaml
path: ./datasets/debris
train: images/train
val: images/val
test: images/test

names:
  0: debris‑flow
  1: rock
  2: muddy
```
## 2、训练代码 train.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2528ac76e57435f8e0e95b00794d3b2.png)

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="debris.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)
```
## 3、推理代码 predict.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7373bd7b900949fcbd14de090d1d2d0e.png)

```python
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")
res = model.predict(source="test.jpg",conf=0.5,save=True)
```
## 4、PyQt5可视化GUI界面代码
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication,QMainWindow,QPushButton,QFileDialog,QLabel,QTableWidget,QTableWidgetItem,QHeaderView)
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class DebrisDetect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的泥石流等灾害检测系统")
        self.resize(1300,950)
        self.model = YOLO("./runs/detect/train/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        self.label_img=QLabel(self)
        self.label_img.setGeometry(20,70,700,600)

        self.btn_img=QPushButton("选择图片",self)
        self.btn_img.setGeometry(20,10,130,40)
        self.btn_img.clicked.connect(self.detect_img)

        self.btn_video=QPushButton("选择视频",self)
        self.btn_video.setGeometry(170,10,130,40)

        self.btn_cam=QPushButton("摄像头",self)
        self.btn_cam.setGeometry(320,10,130,40)

        self.table=QTableWidget(self)
        self.table.setGeometry(20,690,1240,220)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def detect_img(self):
        file,_=QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;*.png")
        if not file:
            return
        result=self.model.predict(source=file,conf=0.5,save=False)[0]
        img=result.plot()
        self.show(img,self.label_img)
        self.table.setRowCount(0)
        for index,box in enumerate(result.boxes):
            row=self.table.rowCount()
            self.table.insertRow(row)
            cls=self.model.names[int(box.cls)]
            conf=float(box.conf)
            xyxy=[round(float(i),1) for i in box.xyxy[0]]
            self.table.setItem(row,0,QTableWidgetItem(str(index+1)))
            self.table.setItem(row,1,QTableWidgetItem(file))
            self.table.setItem(row,2,QTableWidgetItem(cls))
            self.table.setItem(row,3,QTableWidgetItem(f"{conf*100:.2f}%"))
            self.table.setItem(row,4,QTableWidgetItem(str(xyxy)))

    def show(self,img,label):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c=img.shape
        qimg=QImage(img.data,w,h,c*w,QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(),Qt.KeepAspectRatio))

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=DebrisDetect()
    win.show()
    sys.exit(app.exec_())
```
## 环境安装
```bash
pip install ultralytics pyqt5 opencv‑python
```

