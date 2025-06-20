#ifndef GLCM_ANALYZER_H
#define GLCM_ANALYZER_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct GLCMFeatures {
    double contrast;
    double homogeneity;
    double energy;
    double correlation;
};

class GLCMAnalyzer {
public:
    GLCMAnalyzer();
    ~GLCMAnalyzer();
    
    GLCMFeatures analyzeTexture(const Mat& image, int dx = 1, int dy = 0);
    
private:
    Mat calculateGLCM(const Mat& src, int dx, int dy);
    void calculateGLCMFeatures(const Mat& glcm, double& contrast, double& homogeneity, 
                               double& energy, double& correlation);
};

#endif // GLCM_ANALYZER_H 