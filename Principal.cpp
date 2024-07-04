#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;


void calcularHOG(const Mat& imagen, vector<float>& descriptores) {
    HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    hog.compute(imagen, descriptores);
}

int main(int argc, char* argv[]) {
    
    string rutaModelo = "./hog_svm_model.yml";
    string rutaImagen = "./face.png"; //

    // etiquetas
    map<int, string> etiquetaClases = {
        {0, "Facebook"},
        {1, "amazon"},
        {2, "snapchat"},
        {3, "netflix"}
    };

    //cargar el modelo entrenado
    Ptr<SVM> svm = SVM::load(rutaModelo);

    Mat imagen = imread(rutaImagen, IMREAD_GRAYSCALE);
    if (imagen.empty()) {
        cerr << "error al cargar la imagen: " << rutaImagen << endl;
        return 1;
    }
    resize(imagen, imagen, Size(64, 128)); // redimensionar la imagen a 64x128

    //HOG
    vector<float> descriptores;
    calcularHOG(imagen, descriptores);
    Mat descriptoresMat(1, static_cast<int>(descriptores.size()), CV_32F, descriptores.data());

    //etiqueta
    int etiquetaPredicha = static_cast<int>(svm->predict(descriptoresMat));

    //REsultados
    namedWindow("Imagen", WINDOW_AUTOSIZE);
    imshow("Imagen", imagen);
    cout << "Etiqueta Predicha: " << etiquetaPredicha << " (" << etiquetaClases[etiquetaPredicha] << ")" << endl;

    waitKey(0); 
    return 0;
}

