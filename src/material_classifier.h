#ifndef MATERIAL_CLASSIFIER_H
#define MATERIAL_CLASSIFIER_H

#include "glcm_analyzer.h"
#include <string>

using namespace std;

class MaterialClassifier {
public:
    MaterialClassifier();
    ~MaterialClassifier();
    
    string classifyMaterial(const GLCMFeatures& features);
    string classifyMaterial(double contrast, double homogeneity, double energy, double correlation);
    
private:
};

#endif // MATERIAL_CLASSIFIER_H 