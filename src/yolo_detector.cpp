#include "yolo_detector.h"
#include <iostream>
#include <fstream>

YoloDetector::YoloDetector(const string& configFile, const string& weightsFile, const string& classesFile) {
    classNames = loadClassNames(classesFile);
    net = readNetFromDarknet(configFile, weightsFile);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
}

YoloDetector::~YoloDetector() {
}

vector<string> YoloDetector::loadClassNames(const string& classFile) {
    vector<string> classNames;
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

vector<string> YoloDetector::getOutputLayerNames(const Net& net) {
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<String> layersNames = net.getLayerNames();
    vector<string> names(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    return names;
}

vector<DetectionResult> YoloDetector::detect(const Mat& image, float confThreshold, float nmsThreshold) {
    vector<DetectionResult> results;
    
    Mat blob;
    blobFromImage(image, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, getOutputLayerNames(net));

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        DetectionResult result;
        result.box = boxes[idx];
        result.classId = classIds[idx];
        result.confidence = confidences[idx];
        
        if (!classNames.empty() && result.classId < (int)classNames.size()) {
            result.className = classNames[result.classId];
        } else {
            result.className = "unknown";
        }
        
        results.push_back(result);
    }
    
    return results;
} 