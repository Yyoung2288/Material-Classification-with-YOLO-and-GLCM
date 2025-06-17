# Object Detection and Material Classification Using YOLO and GLCM

This project combines YOLO object detection and Gray-Level Co-occurrence Matrix (GLCM) image processing techniques to classify detected objects based on their materials (e.g., metal, glass, plastic, paper).

## Description

The project uses a YOLO deep learning model to detect objects within images. After detecting objects, grayscale image processing methods and GLCM-based features (contrast, homogeneity, energy, and correlation) are extracted to classify the material of each object.

## Methodology

1. **YOLO Object Detection**:

   * YOLO (`my_yolo.cfg`, `my_yolo_best.weights`) identifies objects and their bounding boxes.

2. **Image Preprocessing**:

   * Detected objects are cropped and converted to grayscale.

3. **Feature Extraction using GLCM**:

   * GLCM matrices are generated to analyze texture characteristics of objects.

4. **Material Classification**:

   * GLCM features are used to classify the material type.

## Technologies Used

* C++
* OpenCV (Image Processing)
* YOLOv4 (Deep Learning)
* GLCM (Feature Extraction)

## Model Weights

Due to GitHub's file size limitations, the trained YOLO model weights are not included in this repository.
You can download them from the following link:

[Download YOLOv4 Weights](https://drive.google.com/drive/folders/1ZmPFxEZ_VvaQVP2O_kExBN98cEGbEjKV?usp=sharing)

## Limitations

* Classification performance is dependent on the quality and diversity of training data.
* GLCM parameters may require further tuning for more diverse real-world datasets.
* The current material classification logic is based on manually defined heuristic thresholds. While it demonstrates reasonable accuracy on controlled samples, it lacks adaptability and scalability. Future versions could leverage supervised learning models trained on GLCM feature vectors to improve classification robustness.

## Potential Improvements

* Enhance material classification using supervised learning (e.g., SVM, Random Forest).
* Implement automated tuning of GLCM parameters using optimization techniques.

## Application Scenario

* Automated waste sorting
* Recycling industry
* Manufacturing quality control

## Demonstration

[Demo Video](https://youtu.be/7BL6nNVUg5g?si=NFSCnNQaenV-8Ck4)
