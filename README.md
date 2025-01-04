# Yolo v3 Object Detection in Tensorflow  

Yolo v3 is an algorithm that uses deep convolutional neural networks to detect objects in images and videos in real time. This repository provides a TensorFlow implementation of Yolo v3, along with pretrained weights, inference scripts, and examples.  
<br>  
[Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow)  

---

## 🔧 Getting Started  

This guide will walk you through setting up the project, downloading the necessary resources, and running object detection on images and videos.  

### Prerequisites  

Ensure you have the following installed:  
- **Python 3.6.6** or later  
- **TensorFlow** (for deep learning)  
- **NumPy** (for numerical computing)  
- **Pillow** (for image processing)  
- **OpenCV** (for computer vision)  
- **Seaborn** (for visualization)  

Install the dependencies using the provided requirements file:  
```bash  
pip install -r requirements.txt  
```  

---

### Downloading Official Pretrained Weights  

Download the official weights pretrained on the **COCO dataset**.  
```bash  
wget -P weights https://pjreddie.com/media/files/yolov3.weights  
```  

---

### Save the Weights in TensorFlow Format  

Use the `load_weights.py` script to convert the weights into TensorFlow format.  
```bash  
python load_weights.py  
```  

---

## 🚀 Running the Model  

You can run the model for both images and video files using the `detect.py` script. Make sure to set the **IoU (Intersection over Union)** and **confidence thresholds** for filtering detections.  

### Usage  
```bash  
python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>  
```  

---

### Images Example  

Run an example using sample images:  
```bash  
python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg  
```  

The detections will be saved in the `detections` folder.  

Sample output:  
```
detection_1.jpg  
```  
![Detection Example 1](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detection_1.jpg)  
```
detection_2.jpg  
```  
![Detection Example 2](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detection_2.jpg)  

---

### Video Example  

Run the script with a video file:  
```bash  
python detect.py video 0.5 0.5 data/video/shinjuku.mp4  
```  

The detections will be saved as a video file (`detections.mp4`).  

![Video Detection Example](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detections.gif)  

---

## 🛠️ To-Do List  

- [x] Add pretrained weights and inference scripts  
- [x] Implement detection on images and video  
- [ ] Train the model on custom datasets  
- [ ] Add support for other Yolo versions (v4, v5)  
- [ ] Deploy as a web application using Flask/Django  

---

## 📜 References  

1. **YOLOv3 Paper**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1804.02767)  
2. **Darknet Weights**: [PJ Reddie's YOLO Website](https://pjreddie.com/darknet/yolo/)  
3. **TensorFlow Documentation**: [TensorFlow.org](https://www.tensorflow.org/)  
