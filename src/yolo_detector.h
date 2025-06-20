#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

struct DetectionResult {
    Rect box;
    int classId;
    float confidence;
    string className;
};

class YoloDetector {
public:
    YoloDetector(const string& configFile, const string& weightsFile, const string& classesFile);
    ~YoloDetector();
    
    vector<DetectionResult> detect(const Mat& image, float confThreshold = 0.5, float nmsThreshold = 0.4);
    
private:
    Net net;
    vector<string> classNames;
    
    vector<string> loadClassNames(const string& classFile);
    vector<string> getOutputLayerNames(const Net& net);
};

#endif // YOLO_DETECTOR_H 