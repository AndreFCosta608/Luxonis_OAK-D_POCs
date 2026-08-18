// poc_volumetric3D/main.cpp
// Captura de 2 vistas (frente e trás, giro de 180° manual). O objeto é isolado
// por SEGMENTAÇÃO DE INSTÂNCIA REAL (YOLOv8-seg, rodando via ONNX Runtime em
// C++ puro -- sem Python, sem TensorFlow, sem subprocess) em vez de subtração
// de fundo. Merge das duas metades usa a rotação de 180° já conhecida (o giro
// é garantido fisicamente pelo usuário) + correção fina por centroide.
//
// PREPARO (uma vez só, offline, fora do app -- não roda em tempo de execução):
//   pip install ultralytics
//   yolo export model=yolov8n-seg.pt format=onnx imgsz=640
// Isso gera "yolov8n-seg.onnx". Coloca esse arquivo na pasta do executável
// (ou ajusta MODEL_PATH abaixo).
#include <iostream>
#include <exception>
#include <atomic>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

#include "open3d/Open3D.h"
#include "open3d/visualization/visualizer/VisualizerWithKeyCallback.h"

#include <onnxruntime_cxx_api.h>

// ─────────────────────────────────────────────
//  Constantes de Configuração
// ─────────────────────────────────────────────
static constexpr int    FRAMES_PER_CAPTURE   = 10;    // frames de profundidade por captura (mediana)
static constexpr float  VOXEL_SIZE           = 0.002f; // 2mm
static constexpr float  DEPTH_MIN            = 0.15f;
static constexpr float  DEPTH_MAX            = 0.60f;
static constexpr int    WARMUP_FRAMES        = 30;
static constexpr int    NUM_VIEWS            = 2;      // frente + trás (180°, fixo)

static const std::string MODEL_PATH   = "yolov8n-seg.onnx";
static constexpr int    NET_INPUT_SIZE = 640;   // yolov8n-seg exportado com imgsz=640
static constexpr int    COCO_CAR_INDEX = 2;     // "car" na ordem padrão das 80 classes COCO
static constexpr float  CONF_THRESHOLD = 0.25f; // baixe se a miniatura não for detectada

enum class ScanState {
    IDLE,        // aguardando ESPAÇO pra próxima vista
    CAPTURING,   // coletando os FRAMES_PER_CAPTURE frames de profundidade
    SEGMENTING,  // rodando YOLOv8-seg + montando a nuvem (thread separada)
    READY_MERGE, // as NUM_VIEWS vistas capturadas, aguardando [G]
    MESHING,     // juntando as vistas e reconstruindo a superfície
    DONE         // malha pronta
};

struct SharedState {
    std::mutex cloudMutex;
    std::mutex meshMutex;
    std::mutex rgbMutex;

    cv::Mat lastRGB;
    cv::Mat lastDepthRaw; // mm, CV_16UC1 -- pra visualização de diagnóstico ao vivo

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> capturedClouds; // 1 por vista
    std::shared_ptr<open3d::geometry::TriangleMesh> finalMesh;

    std::atomic<bool>      running{true};
    std::atomic<ScanState> scanState{ScanState::IDLE};

    std::atomic<int> viewIndex{0};            // próxima vista a capturar (0=frente, 1=trás)
    std::atomic<int> captureFrameProgress{0}; // 0..FRAMES_PER_CAPTURE

    std::atomic<bool> captureRequested{false}; // usuário apertou ESPAÇO
    std::atomic<bool> mergeRequested{false};   // usuário apertou G

    std::atomic<bool> displayDirty{true};

    std::string lastSegInfo; // última linha de status da segmentação (protegida por rgbMutex)
};

// ─────────────────────────────────────────────
//  Mediana por pixel de N frames de profundidade (reduz ruído do sensor)
// ─────────────────────────────────────────────
cv::Mat computeMedianDepth(const std::vector<cv::Mat>& frames) {
    CV_Assert(!frames.empty());
    const int rows = frames[0].rows;
    const int cols = frames[0].cols;
    const int n    = static_cast<int>(frames.size());

    const uint16_t minMm = static_cast<uint16_t>(DEPTH_MIN * 1000.0f);
    const uint16_t maxMm = static_cast<uint16_t>(DEPTH_MAX * 1000.0f);

    cv::Mat result(rows, cols, CV_16UC1, cv::Scalar(0));
    std::vector<uint16_t> samples;
    samples.reserve(n);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            samples.clear();
            for (int i = 0; i < n; ++i) {
                uint16_t v = frames[i].at<uint16_t>(y, x);
                if (v >= minMm && v <= maxMm) samples.push_back(v);
            }
            if (!samples.empty()) {
                std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
                result.at<uint16_t>(y, x) = samples[samples.size() / 2];
            }
        }
    }
    return result;
}

std::shared_ptr<open3d::geometry::PointCloud> rawCloudFromDepth(
    const cv::Mat& depthMat, const cv::Mat& colorMat,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic)
{
    open3d::geometry::Image depthImg, colorImg;
    depthImg.Prepare(depthMat.cols, depthMat.rows, 1, 2);
    std::memcpy(depthImg.data_.data(), depthMat.data, depthMat.total() * depthMat.elemSize());

    colorImg.Prepare(colorMat.cols, colorMat.rows, 3, 1);
    cv::Mat colorRGB;
    cv::cvtColor(colorMat, colorRGB, cv::COLOR_BGR2RGB);
    std::memcpy(colorImg.data_.data(), colorRGB.data, colorRGB.total() * colorRGB.elemSize());

    auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(colorImg, depthImg, 1000.0, DEPTH_MAX, false);
    return open3d::geometry::PointCloud::CreateFromRGBDImage(*rgbd, intrinsic);
}

// ─────────────────────────────────────────────
//  Segmentador: YOLOv8-seg via ONNX Runtime, C++ puro
// ─────────────────────────────────────────────
struct LetterboxInfo { float scale; int padX; int padY; int newW; int newH; };

cv::Mat letterbox(const cv::Mat& src, int targetSize, LetterboxInfo& info) {
    int w = src.cols, h = src.rows;
    float scale = std::min(static_cast<float>(targetSize) / w, static_cast<float>(targetSize) / h);
    int newW = static_cast<int>(std::round(w * scale));
    int newH = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    int padX = (targetSize - newW) / 2;
    int padY = (targetSize - newH) / 2;

    cv::Mat out(targetSize, targetSize, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));

    info = {scale, padX, padY, newW, newH};
    return out;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

class CarSegmenter {
public:
    explicit CarSegmenter(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "car_segmenter"),
          session_(nullptr)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, modelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        auto inName = session_.GetInputNameAllocated(0, allocator);
        inputName_ = inName.get();
        size_t numOutputs = session_.GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outName = session_.GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(outName.get());
        }
        std::cout << "[YOLO] Modelo carregado. Input: " << inputName_
                  << " | Outputs: " << outputNames_.size() << "\n";
    }

    // Retorna máscara CV_8UC1 (255=objeto, 0=fundo) no tamanho da imagem de
    // entrada. foundCar indica se a classe "car" foi detectada (senão, usa
    // fallback: maior ativação entre qualquer classe).
    cv::Mat segment(const cv::Mat& colorBGR, bool& foundCar, float& confidence, std::string& classUsed) {
        LetterboxInfo lb;
        cv::Mat padded = letterbox(colorBGR, NET_INPUT_SIZE, lb);

        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<float> inputTensorValues(3 * NET_INPUT_SIZE * NET_INPUT_SIZE);
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c)
            channels[c] = cv::Mat(NET_INPUT_SIZE, NET_INPUT_SIZE, CV_32F,
                                   inputTensorValues.data() + c * NET_INPUT_SIZE * NET_INPUT_SIZE);
        cv::split(rgb, channels);

        std::array<int64_t, 4> inputShape{1, 3, NET_INPUT_SIZE, NET_INPUT_SIZE};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size());

        const char* inputNames[]  = { inputName_.c_str() };
        std::vector<const char*> outputNamesCstr;
        for (auto& n : outputNames_) outputNamesCstr.push_back(n.c_str());

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                     outputNamesCstr.data(), outputNamesCstr.size());

        // Identifica qual saída é a de detecções (rank 3) e qual é o proto de máscara (rank 4)
        Ort::Value* detTensor = nullptr;
        Ort::Value* protoTensor = nullptr;
        for (auto& out : outputs) {
            auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 3) detTensor = &out;
            else if (shape.size() == 4) protoTensor = &out;
        }
        if (!detTensor || !protoTensor) {
            std::cout << "[YOLO] AVISO: formato de saida inesperado do modelo.\n";
            foundCar = false; confidence = 0.0f; classUsed = "erro";
            return cv::Mat::ones(colorBGR.size(), CV_8UC1) * 255;
        }

        auto detShape = detTensor->GetTensorTypeAndShapeInfo().GetShape();   // [1, 116, numAnchors]
        auto protoShape = protoTensor->GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, mh, mw]
        int numChannels = static_cast<int>(detShape[1]); // 4 + numClasses + maskDim
        int numAnchors  = static_cast<int>(detShape[2]);
        int maskDim     = static_cast<int>(protoShape[1]);
        int protoH      = static_cast<int>(protoShape[2]);
        int protoW      = static_cast<int>(protoShape[3]);
        int numClasses  = numChannels - 4 - maskDim;

        const float* det   = detTensor->GetTensorData<float>();
        const float* proto = protoTensor->GetTensorData<float>();

        auto detAt = [&](int channel, int anchor) { return det[channel * numAnchors + anchor]; };

        // Melhor anchor pra classe "car"
        int bestCarAnchor = -1; float bestCarScore = 0.0f;
        // Melhor anchor entre QUALQUER classe (fallback)
        int bestAnyAnchor = -1; float bestAnyScore = 0.0f; int bestAnyClass = -1;

        for (int a = 0; a < numAnchors; ++a) {
            for (int c = 0; c < numClasses; ++c) {
                float score = detAt(4 + c, a);
                if (score > bestAnyScore) { bestAnyScore = score; bestAnyAnchor = a; bestAnyClass = c; }
                if (c == COCO_CAR_INDEX && score > bestCarScore) { bestCarScore = score; bestCarAnchor = a; }
            }
        }

        int chosenAnchor;
        if (bestCarAnchor >= 0 && bestCarScore >= CONF_THRESHOLD) {
            chosenAnchor = bestCarAnchor;
            foundCar = true; confidence = bestCarScore; classUsed = "car";
        } else if (bestAnyAnchor >= 0) {
            chosenAnchor = bestAnyAnchor;
            foundCar = false; confidence = bestAnyScore; classUsed = "fallback#" + std::to_string(bestAnyClass);
        } else {
            foundCar = false; confidence = 0.0f; classUsed = "nenhum";
            return cv::Mat::ones(colorBGR.size(), CV_8UC1) * 255; // nada detectado: não filtra
        }

        // Combina os coeficientes de máscara do anchor escolhido com o proto
        cv::Mat maskLow(protoH, protoW, CV_32F, cv::Scalar(0));
        for (int y = 0; y < protoH; ++y) {
            for (int x = 0; x < protoW; ++x) {
                float sum = 0.0f;
                for (int m = 0; m < maskDim; ++m) {
                    float coeff = detAt(4 + numClasses + m, chosenAnchor);
                    sum += coeff * proto[m * protoH * protoW + y * protoW + x];
                }
                maskLow.at<float>(y, x) = sigmoidf(sum);
            }
        }

        cv::Mat maskNet;
        cv::resize(maskLow, maskNet, cv::Size(NET_INPUT_SIZE, NET_INPUT_SIZE));

        cv::Rect validRegion(lb.padX, lb.padY, lb.newW, lb.newH);
        validRegion &= cv::Rect(0, 0, NET_INPUT_SIZE, NET_INPUT_SIZE);
        cv::Mat cropped = maskNet(validRegion);

        cv::Mat maskOrig;
        cv::resize(cropped, maskOrig, colorBGR.size());

        cv::Mat maskBin;
        cv::threshold(maskOrig, maskBin, 0.5, 255, cv::THRESH_BINARY);
        maskBin.convertTo(maskBin, CV_8UC1);
        return maskBin;
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string inputName_;
    std::vector<std::string> outputNames_;
};

// ─────────────────────────────────────────────
//  Junta as 2 vistas (rotação de 180° conhecida + correção fina por centroide)
//  e reconstrói a malha final. SEM ICP: não há sobreposição real entre frente
//  e trás pra ele se apoiar.
// ─────────────────────────────────────────────
void performMergeAndMesh(SharedState& state) {
    std::shared_ptr<open3d::geometry::PointCloud> view0, view1;
    {
        std::lock_guard<std::mutex> lk(state.cloudMutex);
        if (state.capturedClouds.size() < 2) {
            state.scanState = ScanState::DONE;
            return;
        }
        view0 = state.capturedClouds[0];
        view1 = std::make_shared<open3d::geometry::PointCloud>(*state.capturedClouds[1]);
    }

    if (view0->points_.empty() || view1->points_.empty()) {
        std::cout << "[AVISO] Uma das vistas ficou vazia -- verifique a segmentacao.\n";
        state.scanState = ScanState::DONE;
        return;
    }

    Eigen::Vector3d centroid0 = view0->GetCenter();
    Eigen::Vector3d centroid1 = view1->GetCenter();

    Eigen::Matrix3d rotY180;
    rotY180 << -1.0, 0.0,  0.0,
                0.0, 1.0,  0.0,
                0.0, 0.0, -1.0;

    view1->Rotate(rotY180, centroid1);
    view1->Translate(centroid0 - centroid1);

    auto merged = std::make_shared<open3d::geometry::PointCloud>(*view0);
    *merged += *view1;

    merged = merged->VoxelDownSample(VOXEL_SIZE);
    merged->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.01, 30));
    merged->OrientNormalsConsistentTangentPlane(30);

    double avgSpacing = 0.0;
    {
        std::vector<double> dists = merged->ComputeNearestNeighborDistance();
        if (!dists.empty()) {
            for (double d : dists) avgSpacing += d;
            avgSpacing /= static_cast<double>(dists.size());
        } else {
            avgSpacing = VOXEL_SIZE * 2.0;
        }
    }
    std::vector<double> radii = { avgSpacing * 1.5, avgSpacing * 3.0, avgSpacing * 6.0 };
    auto mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*merged, radii);
    mesh->ComputeVertexNormals();
    mesh->RemoveDuplicatedVertices();
    mesh->RemoveDegenerateTriangles();

    {
        std::lock_guard<std::mutex> lk(state.meshMutex);
        state.finalMesh = mesh;
    }
    open3d::io::WriteTriangleMesh("scan_2vistas.ply", *mesh);
    std::cout << "✓ Malha salva: scan_2vistas.ply\n";

    state.scanState = ScanState::DONE;
}

// Roda em thread separada (não trava a captura nem a UI): segmenta, monta a
// nuvem, aplica correção de orientação, guarda e avança o estado.
void performSegmentAndBuildCloud(SharedState& state, std::shared_future<std::shared_ptr<CarSegmenter>> segmenterFuture,
                                  cv::Mat medianDepth, cv::Mat colorFrame,
                                  open3d::camera::PinholeCameraIntrinsic intrinsic)
{
    // Só bloqueia aqui (thread de segmentação, já desacoplada da leitura de
    // frames) se o modelo ainda não tiver terminado de carregar -- na prática
    // isso já deve estar pronto muito antes do usuário terminar a primeira captura.
    std::shared_ptr<CarSegmenter> segmenter = segmenterFuture.get();

    bool foundCar; float confidence; std::string classUsed;
    cv::Mat mask = segmenter->segment(colorFrame, foundCar, confidence, classUsed);

    std::string info = "Segmentacao: classe=" + classUsed + " confianca=" +
                        std::to_string(confidence).substr(0, 4);
    {
        std::lock_guard<std::mutex> lk(state.rgbMutex);
        state.lastSegInfo = info;
    }
    std::cout << "[YOLO] " << info << "\n";

    medianDepth.setTo(0, mask == 0);

    auto raw = rawCloudFromDepth(medianDepth, colorFrame, intrinsic);

    Eigen::Matrix4d flipX = Eigen::Matrix4d::Identity();
    flipX(1, 1) = -1.0;
    flipX(2, 2) = -1.0;
    raw->Transform(flipX);

    {
        std::lock_guard<std::mutex> lk(state.cloudMutex);
        state.capturedClouds.push_back(raw);
    }
    state.displayDirty = true;

    int nextView = state.viewIndex.load() + 1;
    state.viewIndex = nextView;
    state.scanState = (nextView >= NUM_VIEWS) ? ScanState::READY_MERGE : ScanState::IDLE;
}

// ─────────────────────────────────────────────
//  Thread de Captura
// ─────────────────────────────────────────────
void captureThread(SharedState& state, dai::Device& device, std::shared_future<std::shared_ptr<CarSegmenter>> segmenterFuture)
{
    auto qRgb   = device.getOutputQueue("rgb",   8, false);
    auto qDepth = device.getOutputQueue("depth", 8, false);

    double fx = 457.0, fy = 457.0, cx = 320.0, cy = 200.0;
    try {
        dai::CalibrationHandler calib = device.readCalibration();
        auto M = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, 640, 400);
        fx = M[0][0]; fy = M[1][1]; cx = M[0][2]; cy = M[1][2];
        std::cout << "[CALIB] fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << " (lido do dispositivo)\n";
    } catch (const std::exception& e) {
        std::cout << "[CALIB] AVISO: falha ao ler calibracao (" << e.what() << ") -- usando valores aproximados\n";
    }
    open3d::camera::PinholeCameraIntrinsic intrinsic(640, 400, fx, fy, cx, cy);

    int warmup = 0;
    std::vector<cv::Mat> depthBuffer;
    cv::Mat lastColorForCapture;

    while (state.running) {
        auto rgbFrame   = qRgb->get<dai::ImgFrame>();
        auto depthFrame = qDepth->get<dai::ImgFrame>();
        if (!rgbFrame || !depthFrame) continue;
        if (warmup++ < WARMUP_FRAMES) continue;

        cv::Mat colorMat = rgbFrame->getCvFrame();
        cv::Mat depthMat = depthFrame->getCvFrame();
        if (!colorMat.isContinuous()) colorMat = colorMat.clone();
        if (!depthMat.isContinuous()) depthMat = depthMat.clone();

        {
            std::lock_guard<std::mutex> lk(state.rgbMutex);
            state.lastRGB = colorMat.clone();
            state.lastDepthRaw = depthMat.clone();
        }

        ScanState cs = state.scanState.load();

        if (cs == ScanState::IDLE && state.captureRequested.load()) {
            state.scanState = ScanState::CAPTURING;
            depthBuffer.clear();
            state.captureFrameProgress = 0;
        }
        else if (cs == ScanState::CAPTURING) {
            depthBuffer.push_back(depthMat.clone());
            lastColorForCapture = colorMat.clone();

            int progress = static_cast<int>(depthBuffer.size());
            state.captureFrameProgress = progress;

            if (progress >= FRAMES_PER_CAPTURE) {
                cv::Mat medianDepth = computeMedianDepth(depthBuffer);
                state.captureRequested = false;
                depthBuffer.clear();
                state.scanState = ScanState::SEGMENTING;

                std::thread(performSegmentAndBuildCloud, std::ref(state), segmenterFuture,
                            medianDepth.clone(), lastColorForCapture.clone(), intrinsic).detach();
            }
        }
        else if (cs == ScanState::READY_MERGE && state.mergeRequested.load()) {
            state.mergeRequested = false;
            state.scanState = ScanState::MESHING;
            std::thread(performMergeAndMesh, std::ref(state)).detach();
        }
    }
}

// ─────────────────────────────────────────────
//  Thread do Painel de Controle OpenCV
// ─────────────────────────────────────────────
void previewThread(SharedState& state)
{
    cv::namedWindow("OAK-D — Captura 2 Vistas (YOLOv8-seg)", cv::WINDOW_NORMAL);
    cv::resizeWindow("OAK-D — Captura 2 Vistas (YOLOv8-seg)", 640, 400);
    cv::namedWindow("DEBUG — Profundidade ao vivo", cv::WINDOW_NORMAL);
    cv::resizeWindow("DEBUG — Profundidade ao vivo", 640, 400);

    while (state.running) {
        cv::Mat frame;
        cv::Mat depthRaw;
        std::string segInfo;
        {
            std::lock_guard<std::mutex> lk(state.rgbMutex);
            if (!state.lastRGB.empty()) frame = state.lastRGB.clone();
            if (!state.lastDepthRaw.empty()) depthRaw = state.lastDepthRaw.clone();
            segInfo = state.lastSegInfo;
        }

        if (!depthRaw.empty()) {
            cv::Mat depthVis;
            double minMm = DEPTH_MIN * 1000.0, maxMm = DEPTH_MAX * 1000.0;
            depthRaw.convertTo(depthVis, CV_8UC1, -255.0 / (maxMm - minMm), 255.0 * maxMm / (maxMm - minMm));
            cv::Mat mask = (depthRaw == 0);
            depthVis.setTo(0, mask);
            cv::Mat depthColor;
            cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);
            depthColor.setTo(cv::Scalar(0, 0, 0), mask);
            cv::imshow("DEBUG — Profundidade ao vivo", depthColor);
        }

        if (!frame.empty()) {
            ScanState cs = state.scanState.load();
            int view = state.viewIndex.load();

            switch (cs) {
                case ScanState::IDLE: {
                    std::string info = "Vista " + std::to_string(view + 1) + "/" + std::to_string(NUM_VIEWS) +
                                        " -- posicione o objeto e aperte [ESPACO]";
                    cv::putText(frame, info, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 165, 255}, 2);
                    break;
                }
                case ScanState::CAPTURING: {
                    int prog = state.captureFrameProgress.load();
                    std::string info = "CAPTURANDO vista " + std::to_string(view + 1) + "/" + std::to_string(NUM_VIEWS) +
                                        " -- frame " + std::to_string(prog) + "/" + std::to_string(FRAMES_PER_CAPTURE);
                    cv::putText(frame, info, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 2);
                    break;
                }
                case ScanState::SEGMENTING:
                    cv::putText(frame, "RODANDO YOLOv8-seg... aguarde", {10, 25},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 140, 255}, 2);
                    break;
                case ScanState::READY_MERGE:
                    cv::putText(frame, "AS 2 VISTAS COLETADAS -- [G] Gerar malha", {10, 25},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 0}, 2);
                    break;
                case ScanState::MESHING:
                    cv::putText(frame, "PROCESSANDO MALHA... aguarde", {10, 25},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 140, 255}, 2);
                    break;
                case ScanState::DONE:
                    cv::putText(frame, "MALHA PRONTA -- [Q] Sair", {10, 25},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 2);
                    break;
            }

            if (!segInfo.empty()) {
                cv::putText(frame, segInfo, {10, 50}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {200, 200, 0}, 1);
            }

            cv::putText(frame, "[ESPACO] Capturar | [G] Gerar malha | [Q] Sair",
                        {10, frame.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {255, 255, 255}, 1);

            cv::imshow("OAK-D — Captura 2 Vistas (YOLOv8-seg)", frame);
        }

        int key = cv::waitKey(30);
        if (key == ' ') {
            if (state.scanState.load() == ScanState::IDLE) {
                state.captureRequested = true;
            }
        }
        else if (key == 'g' || key == 'G') {
            if (state.scanState.load() == ScanState::READY_MERGE) {
                state.mergeRequested = true;
            }
        }
        else if (key == 'q' || key == 'Q') {
            state.running = false;
            break;
        }
    }
    cv::destroyAllWindows();
}

// ─────────────────────────────────────────────
//  Visualizador Open3D — nuvens acumulando, depois a malha aparece por cima
// ─────────────────────────────────────────────
void visualizationThread(SharedState& state)
{
    open3d::visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("OAK-D — Nuvem / Malha (2 vistas)", 1024, 768);
    vis.GetRenderOption().background_color_ = {1.0, 1.0, 1.0};
    vis.GetRenderOption().point_size_ = 2.0;

    auto displayCloud = std::make_shared<open3d::geometry::PointCloud>();
    auto displayMesh  = std::make_shared<open3d::geometry::TriangleMesh>();

    bool cloudAdded = false;
    bool meshAdded  = false;

    while (state.running && vis.PollEvents()) {
        if (state.displayDirty.load()) {
            std::lock_guard<std::mutex> lk(state.cloudMutex);
            displayCloud->Clear();
            for (auto& c : state.capturedClouds) *displayCloud += *c;
            state.displayDirty = false;

            if (!cloudAdded) { vis.AddGeometry(displayCloud); cloudAdded = true; }
            else { vis.UpdateGeometry(displayCloud); }

            if (meshAdded) { vis.RemoveGeometry(displayMesh); meshAdded = false; }
        }

        {
            std::lock_guard<std::mutex> lk(state.meshMutex);
            if (state.finalMesh && !state.finalMesh->vertices_.empty() && !meshAdded) {
                *displayMesh = *state.finalMesh;
                vis.AddGeometry(displayMesh);
                meshAdded = true;
            }
        }

        vis.UpdateRender();
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    vis.DestroyVisualizerWindow();
    state.running = false;
}

dai::Pipeline buildPipeline()
{
    dai::Pipeline pipeline;
    auto camRgb    = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft  = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo    = pipeline.create<dai::node::StereoDepth>();
    auto xoutRgb   = pipeline.create<dai::node::XLinkOut>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("rgb");
    xoutDepth->setStreamName("depth");

    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setPreviewSize(640, 400);
    camRgb->setInterleaved(false);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    camRgb->setPreviewKeepAspectRatio(false);
    camRgb->setFps(15);

    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    monoLeft->setFps(15);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    monoRight->setFps(15);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(true);
    stereo->setSubpixel(false);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    stereo->setOutputSize(640, 400);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    camRgb->preview.link(xoutRgb->input);
    stereo->depth.link(xoutDepth->input);

    return pipeline;
}

int main()
{
    SharedState state;

    dai::Pipeline pipeline = buildPipeline();
    dai::Device device(pipeline, dai::UsbSpeed::HIGH);

    // Carrega o modelo em paralelo -- a leitura de frames começa imediatamente
    // abaixo, sem esperar nada. Só quando o usuário realmente capturar (thread
    // de segmentação, já desacoplada) é que se espera o carregamento terminar,
    // se ainda não tiver terminado.
    std::shared_future<std::shared_ptr<CarSegmenter>> segmenterFuture = std::async(
        std::launch::async,
        [] {
            std::cout << "[YOLO] Carregando modelo em paralelo...\n";
            return std::make_shared<CarSegmenter>(MODEL_PATH);
        }
    ).share();

    std::cout << "[START] Captura 2 vistas (frente/tras) com segmentacao YOLOv8-seg. "
              << FRAMES_PER_CAPTURE << " frames/captura\n";

    std::thread tCapture([&]{ captureThread(state, device, segmenterFuture); });
    std::thread tPreview([&]{ previewThread(state); });

    visualizationThread(state);

    state.running = false;
    tCapture.join();
    tPreview.join();

    return 0;
}
