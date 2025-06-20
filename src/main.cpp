#include <opencv2/opencv.hpp>
#include <iostream>
#include "yolo_detector.h"
#include "glcm_analyzer.h"
#include "material_classifier.h"

using namespace cv;
using namespace std;

int main() {
    try {
        YoloDetector detector("my_yolo.cfg", "my_yolo_best.weights", "obj.names");
        GLCMAnalyzer glcmAnalyzer;
        MaterialClassifier materialClassifier;

        Mat frame = imread("image/all.jpg");
        if (frame.empty()) {
            cout << "無法讀取圖像" << endl;
            return -1;
        }

        vector<DetectionResult> detections = detector.detect(frame, 0.5, 0.4);

        for (const auto& detection : detections) {
            Mat object = frame(detection.box).clone();
            GLCMFeatures features = glcmAnalyzer.analyzeTexture(object);
            string material = materialClassifier.classifyMaterial(features);
            string label = detection.className + ":" + 
                          to_string((int)(detection.confidence * 100)) + "%" +
                          " - " + material;
            rectangle(frame, detection.box, Scalar(0, 255, 0), 3);
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int topLabel = max(detection.box.y, labelSize.height);
            rectangle(frame, Point(detection.box.x, topLabel - round(1.5 * labelSize.height)),
                     Point(detection.box.x + round(1.5 * labelSize.width), topLabel + baseLine), 
                     Scalar(255, 255, 255), FILLED);
            putText(frame, label, Point(detection.box.x, topLabel), 
                   FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
        }

        imshow("YOLO Object Detection with Material Classification", frame);
        waitKey(0);
        destroyAllWindows();
        
    } catch (const exception& e) {
        cerr << "錯誤: " << e.what() << endl;
        return -1;
    }

    return 0;
}

