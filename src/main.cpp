#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> loadClassNames(const string& classFile) {
    vector<string> classNames;
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

vector<string> getOutputLayerNames(const Net& net) {
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<String> layersNames = net.getLayerNames();
    vector<string> names(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    return names;
}

Mat calculateGLCM(const Mat& src, int dx, int dy) {
    Mat glcm = Mat::zeros(256, 256, CV_32F);
    
    for (int y = 0; y < src.rows - dy; ++y) {
        for (int x = 0; x < src.cols - dx; ++x) {
            int i = src.at<uchar>(y, x);
            int j = src.at<uchar>(y + dy, x + dx);
            glcm.at<float>(i, j)++;
        }
    }
    glcm /= sum(glcm)[0];
    return glcm;
}


void calculateGLCMFeatures(const Mat& glcm, double& contrast, double& homogeneity, double& energy, double& correlation) {
    contrast = 0.0;
    homogeneity = 0.0;
    energy = 0.0;
    correlation = 0.0;

    double mean_i = 0.0, mean_j = 0.0, std_i = 0.0, std_j = 0.0;


    for (int i = 0; i < glcm.rows; ++i) {
        for (int j = 0; j < glcm.cols; ++j) {
            mean_i += i * glcm.at<float>(i, j);
            mean_j += j * glcm.at<float>(i, j);
        }
    }

    for (int i = 0; i < glcm.rows; ++i) {
        for (int j = 0; j < glcm.cols; ++j) {
            std_i += (i - mean_i) * (i - mean_i) * glcm.at<float>(i, j);
            std_j += (j - mean_j) * (j - mean_j) * glcm.at<float>(i, j);
        }
    }
    std_i = sqrt(std_i);
    std_j = sqrt(std_j);


    for (int i = 0; i < glcm.rows; ++i) {
        for (int j = 0; j < glcm.cols; ++j) {
            float value = glcm.at<float>(i, j);
            contrast += (i - j) * (i - j) * value;
            homogeneity += value / (1.0 + abs(i - j));
            energy += value * value;
            if (std_i * std_j != 0) {
                correlation += (i - mean_i) * (j - mean_j) * value / (std_i * std_j);
            }
        }
    }
}

string classifyMaterial(double contrast, double homogeneity, double energy, double correlation) {
    if (contrast > 600 && contrast < 2000 &&
        homogeneity > 0.1 && homogeneity < 0.3 &&
        energy > 0.1 && energy < 0.3 &&
        correlation > 0.3 && correlation < 0.6) {
        return "metal";
    }
    else if (contrast > 200 && contrast < 700 &&
        homogeneity > 0.5 && homogeneity < 0.7 &&
        energy > 0.3 && energy < 0.6 &&
        correlation > 0.4 && correlation < 0.7) {
        return "glass";
    }
    else if (contrast > 100 && contrast < 500 &&
        homogeneity > 0.7 && homogeneity < 0.9 &&
        energy > 0.4 && energy < 0.7 &&
        correlation > 0.6 && correlation < 0.9) {
        return "plastic";
    }
    else if (contrast > 50 && contrast < 300 &&
        homogeneity > 0.8 && homogeneity < 0.95 &&
        energy > 0.4 && energy < 0.8 &&
        correlation > 0.7 && correlation < 0.95) {
        return "paper";
    }
    else
        return "can't classify";
}

int main() {
    string classesFile = "obj.names";
    vector<string> classNames = loadClassNames(classesFile);

    String modelConfiguration = "my_yolo.cfg";
    String modelWeights = "my_yolo_best.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat frame = imread("image/all.jpg");
    if (frame.empty()) {
        cout << "無法讀取圖像" << endl;
        return -1;
    }

    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, getOutputLayerNames(net));

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
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
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
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
        Rect box = boxes[idx];
        int classId = classIds[idx];
        string label = format("%.2f", confidences[idx]);
        if (!classNames.empty()) {
            CV_Assert(classId < (int)classNames.size());
            label = classNames[classId] + ":" + label;
        }

        Mat object = frame(box).clone();
        cvtColor(object, object, COLOR_BGR2GRAY);

        Mat glcm = calculateGLCM(object, 1, 0);
        double contrast, homogeneity, energy, correlation;
        calculateGLCMFeatures(glcm, contrast, homogeneity, energy, correlation);

        string material = classifyMaterial(contrast, homogeneity, energy, correlation);
        label += " - " + material;
        rectangle(frame, box, Scalar(0, 255, 0), 3);

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int topLabel = max(box.y, labelSize.height);
        rectangle(frame, Point(box.x, topLabel - round(1.5 * labelSize.height)),
            Point(box.x + round(1.5 * labelSize.width), topLabel + baseLine), Scalar(255, 255, 255), FILLED);
        putText(frame, label, Point(box.x, topLabel), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
    }
    
    imshow("YOLO Object Detection", frame);
    waitKey(0);

    return 0;
}

