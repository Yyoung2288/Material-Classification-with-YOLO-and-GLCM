#include "material_classifier.h"

MaterialClassifier::MaterialClassifier() {
}

MaterialClassifier::~MaterialClassifier() {
}

string MaterialClassifier::classifyMaterial(const GLCMFeatures& features) {
    return classifyMaterial(features.contrast, features.homogeneity, 
                           features.energy, features.correlation);
}

string MaterialClassifier::classifyMaterial(double contrast, double homogeneity, 
                                           double energy, double correlation) {
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
    else {
        return "can't classify";
    }
} 