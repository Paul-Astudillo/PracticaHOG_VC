#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void calcularHOG(const Mat& imagen, vector<float>& descriptores) {
    HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    hog.compute(imagen, descriptores);
}

void cargarImagenesYEtiquetas(const string& rutaDataset, vector<Mat>& imagenes, vector<int>& etiquetas) {
    ifstream archivo(rutaDataset);
    string linea;
    while (getline(archivo, linea)) {
        istringstream iss(linea);
        string rutaImagen;
        int etiqueta;
        if (!(iss >> rutaImagen >> etiqueta)) { break; }
        Mat imagen = imread(rutaImagen, IMREAD_GRAYSCALE);

        resize(imagen, imagen, Size(64, 128)); 
        imagenes.push_back(imagen);
        etiquetas.push_back(etiqueta);
    }
}

int main(int argc, char* argv[]) {
//imágenes y etiquetas
    vector<Mat> imagenes;
    vector<int> etiquetas;
    cargarImagenesYEtiquetas("dataset.txt", imagenes, etiquetas);


    //descriptores HOG
    vector<vector<float>> descriptoresHOG;
    for (const auto& imagen : imagenes) {
        vector<float> descriptores;
        calcularHOG(imagen, descriptores);
        descriptoresHOG.push_back(descriptores);
    }

//convertir descriptores y etiquetas a matrices
    if (!descriptoresHOG.empty()) {
        Mat matrizHOG(static_cast<int>(descriptoresHOG.size()), static_cast<int>(descriptoresHOG[0].size()), CV_32F);
        for (size_t i = 0; i < descriptoresHOG.size(); ++i) {
            memcpy(matrizHOG.ptr<float>(i), descriptoresHOG[i].data(), descriptoresHOG[i].size() * sizeof(float));
        }
        Mat matrizEtiquetas(static_cast<int>(etiquetas.size()), 1, CV_32S, etiquetas.data());

        // entrenar el clasificador SVM
        Ptr<SVM> svm = SVM::create();
        svm->setKernel(SVM::LINEAR);
        svm->setType(SVM::C_SVC);
        svm->setC(1);
        svm->train(matrizHOG, ROW_SAMPLE, matrizEtiquetas);

    
        svm->save("hog_svm_model.yml");

        cout << "modelo entrenado y guardado como 'hog_svm_model.yml'" << endl;
    } else {
        cerr << "Error: No se calcularon descriptores HOG." << endl;
    }

    return 0;
}
