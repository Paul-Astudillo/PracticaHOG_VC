#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <regex>
#include <map>
//#include "matplotlibcpp.h"

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

//Función para calcular la matriz de confusión
Mat computeConfusionMatrix(const vector<int>& trueLabels, const vector<int>& predictedLabels, int numClasses) {
    Mat confusionMatrix = Mat::zeros(numClasses, numClasses, CV_32S);
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        confusionMatrix.at<int>(trueLabels[i], predictedLabels[i])++;
    }
    return confusionMatrix;
}

// Función para extraer la etiqueta del nombre de archivo
int extractLabel(const string& filename) {
    regex labelRegex(".*-(\\d+)\\.png");
    smatch match;
    if (regex_match(filename, match, labelRegex)) {
        return stoi(match[1]);
    } else {
        throw runtime_error("Etiqueta no encontrada en el nombre de archivo: " + filename);
    }
}

// Función para dibujar la matriz de confusión
void drawConfusionMatrix(const Mat& confusionMatrix, const map<int, string>& etiquetaClases) {
    int numClasses = confusionMatrix.rows;
    int cellSize = 150; // Tamaño de cada celda
    int margin = 150; // Espacio para etiquetas
    Mat image = Mat::zeros(numClasses * cellSize + margin, numClasses * cellSize + margin, CV_8UC3);
    
    // Dibujar celdas y números
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            Rect cellRect(margin + j * cellSize, margin + i * cellSize, cellSize, cellSize);
            rectangle(image, cellRect, Scalar(255, 255, 255), FILLED);
            putText(image, to_string(confusionMatrix.at<int>(i, j)), cellRect.tl() + Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
        }
    }

    // Dibujar etiquetas de clases
    int baseline;
    for (const auto& [label, name] : etiquetaClases) {
        Size textSize = getTextSize(name, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
        putText(image, name, Point(margin + label * cellSize + (cellSize - textSize.width) / 2, margin - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(image, name, Point(10, margin + label * cellSize + (cellSize + textSize.height) / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    }

    // Mostrar la imagen
    imshow("Matriz de Confusión", image);
    waitKey(0);
}


// Etiquetas de las clases
map<int, string> etiquetaClases = {
    {0, "Facebook"},
    {1, "amazon"},
    {2, "snapchat"},
    {3, "netflix"}
};

int main() {
    // Cargar el modelo SVM entrenado
    Ptr<SVM> svm = SVM::load("hog_svm_model.yml");

    string testDir = "./dataset/Testing";
    vector<int> trueLabels, predictedLabels;

    // Leer los archivos de imágenes del directorio de prueba
    for (const auto& entry : fs::directory_iterator(testDir)) {
        string filename = entry.path().string();
        
        // Cargar la imagen
        Mat image = imread(filename, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "No se pudo leer la imagen: " << filename << endl;
            continue;
        }
        resize(image, image, Size(64, 128)); // redimensionar la imagen a 64x128

        // Extraer la etiqueta del nombre de archivo
        int trueLabel = extractLabel(filename);

        // Preprocesar la imagen y extraer características HOG
        HOGDescriptor hog;
        vector<float> descriptors;
        hog.compute(image, descriptors);
        Mat descriptorMat(descriptors);

        // Predecir la etiqueta usando el modelo SVM
        int predictedLabel = svm->predict(descriptorMat.reshape(1, 1));
        // Almacenar las etiquetas verdadera y predicha
        trueLabels.push_back(trueLabel);
        predictedLabels.push_back(predictedLabel);
    }

    // Calcular y mostrar la matriz de confusión
    int numClasses = etiquetaClases.size();
    Mat confusionMatrix = computeConfusionMatrix(trueLabels, predictedLabels, numClasses);

    cout << "Matriz de Confusión:" << endl;
    cout << confusionMatrix << endl;

    // Mostrar la matriz de confusión con nombres de clases
    cout << "Matriz de Confusión con Nombres de Clases:" << endl;
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            cout << confusionMatrix.at<int>(i, j) << " ";
        }
        cout << "| " << etiquetaClases[i] << endl;
    }
    for (int j = 0; j < numClasses; ++j) {
        cout << etiquetaClases[j] << " ";
    }
    cout << endl;

      // Mostrar la matriz de confusión
    drawConfusionMatrix(confusionMatrix, etiquetaClases);

    return 0;
}
