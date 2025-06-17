# Object Detection and Material Classification Using YOLO and GLCM

This project presents a hybrid computer vision pipeline that integrates deep learning-based object detection (YOLO) with traditional texture analysis (GLCM) to classify detected objects by material type, such as metal, glass, plastic, and paper. The project was implemented entirely in C++ using OpenCV and demonstrates a practical exploration of combining statistical and learned features.

## Motivation

Accurate material classification is essential in applications such as waste sorting, recycling, and manufacturing inspection. While deep learning models like YOLO are effective at detecting objects and their locations, they may not always distinguish materials with similar shapes but different textures. This project aims to bridge this gap by incorporating Gray-Level Co-occurrence Matrix (GLCM) features into the classification pipeline, leveraging fine-grained texture information to enhance material discrimination.

## Method Overview

1. **Object Detection with YOLO**
   - A YOLOv4-tiny model is used to detect objects and generate bounding boxes from input images.
   - Custom-trained weights (`my_yolo_best.weights`) were used on a small-scale material dataset.

2. **Image Preprocessing**
   - Detected objects are cropped and converted to grayscale for texture analysis.

3. **GLCM Feature Extraction**
   - Four common GLCM features are computed: contrast, homogeneity, energy, and correlation.
   - GLCMs are calculated with offsets such as (1, 0) and (0, 1) to capture directional texture patterns.

4. **Material Classification**
   - GLCM features are passed through a manually defined rule-based decision logic to classify material types.
   - In current implementation, simple thresholds are used. No learning algorithm was applied at this stage.

## Technologies and Tools

- **Programming Language**: C++
- **Libraries**: OpenCV (image processing), YOLOv4-tiny (deep learning inference)
- **Model**: YOLOv4-tiny with custom configuration

## Dataset and Model Weights

- The dataset used is composed of images manually labeled into 4 material categories.
- Trained model weights are hosted externally due to GitHub file size limitations.

[Download YOLOv4 Weights](https://drive.google.com/drive/folders/1ZmPFxEZ_VvaQVP2O_kExBN98cEGbEjKV?usp=sharing)

## Limitations

- The current material classification logic is heuristic and lacks adaptability.
- No formal supervised classifier (e.g., SVM or neural networks) was trained on the GLCM feature vectors.
- Dataset size and diversity are limited, impacting generalization to real-world conditions.
- Feature selection and GLCM parameters are manually defined and may not be optimal.

## Potential Improvements

- Integrate supervised learning models trained on GLCM vectors (e.g., SVM, Random Forest).
- Optimize GLCM parameters via hyperparameter search.
- Extend the dataset to include more material categories and real-world variation.
- Combine deep feature embeddings (e.g., CNN activations) with GLCM for fusion-based classification.

## Application Scenarios

- Intelligent waste sorting systems
- Industrial material inspection
- Real-time edge computing for resource-constrained devices

## Demonstration

[Watch Demonstration Video](https://youtu.be/7BL6nNVUg5g?si=NFSCnNQaenV-8Ck4)

## Author

Liu Tz-Yang (劉子揚)  
Department of Computer Science and Engineering  
Yuan Ze University
