# Dont-Blink AI - Printhead Tracking  

## Overview  
This project aims to create an **AI-powered timelapse system for 3D printers**, where the camera automatically takes a picture **only when the printhead is out of the frame**. The goal is to generate clean timelapse videos without the printhead obstructing the view.  

### **Demo Video**  
<p align="center">
  <a href="https://youtu.be/nMrHcGVqUqU">
    <img src="https://img.youtube.com/vi/nMrHcGVqUqU/maxresdefault.jpg" width="700">
  </a>
</p>

---

## **How It Works**  
- A **YOLOV8 model** detects the printhead in each frame.  
- If the printhead **is not in the way of the print**, a **photo is taken**.  
- The system runs in real-time and works alongside the **3D printer's movement logic**, ensuring optimal capture timing.  

---

## First Prototype - Initial Results  

### **Training Data Samples**  
We trained the model using captured images of different printhead positions.  

#### **Example Training Batch**  
<img src="assets/img/train_batch0.jpg" width="550">  

#### **Labeled Validation Batch**  
<img src="assets/img/val_batch0_labels.jpg" width="550">  

---

### **Results of the First Prototype**  
After training, the model's first results show promising detection accuracy. Here’s an overview of detections in sample frames:  

<img src="assets/img/results.png" width="550">  

- The system **accurately avoids taking pictures when the printhead is in the frame**.  
- Works in **real-time**, ensuring smooth operation alongside the 3D printer’s layer movements.  
- Further improvements will focus on **faster detection and better handling of lighting conditions**.  
