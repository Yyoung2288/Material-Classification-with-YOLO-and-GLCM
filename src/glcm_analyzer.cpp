#include "glcm_analyzer.h"
#include <cmath>

GLCMAnalyzer::GLCMAnalyzer() {
}

GLCMAnalyzer::~GLCMAnalyzer() {
}

GLCMFeatures GLCMAnalyzer::analyzeTexture(const Mat& image, int dx, int dy) {
    Mat grayImage;
    if (image.channels() == 3) {
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }
    
    Mat glcm = calculateGLCM(grayImage, dx, dy);
    
    GLCMFeatures features;
    calculateGLCMFeatures(glcm, features.contrast, features.homogeneity, 
                         features.energy, features.correlation);
    
    return features;
}

Mat GLCMAnalyzer::calculateGLCM(const Mat& src, int dx, int dy) {
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

void GLCMAnalyzer::calculateGLCMFeatures(const Mat& glcm, double& contrast, double& homogeneity, 
                                        double& energy, double& correlation) {
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