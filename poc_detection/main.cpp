#include <iostream>
#include <chrono>
#include <string>
#include <vector>

// Transmissão da OAK-D (DepthAI)
#include <depthai/depthai.hpp>

// Processamento Gráfico (OpenCV)
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

int main() {
    // =========================================================================
    // CONFIGURAÇÕES GERAIS E VARIÁVEIS DE TUNING
    // =========================================================================
    const int LIMITE_APRENDIZADO = 150;   // Quantidade de frames para calibrar o fundo
    const int VALOR_THRESHOLD    = 16;    // Sensibilidade do subtrator (maior = menos ruído)
    const double AREA_MINIMA_OBJ = 400.0;  // Tamanho mínimo em pixels para ser considerado objeto

    // Parâmetros de calibração da OAK-D obtidos no teu log para o cálculo 3D
    const float FOCAL_LENGTH = 644.06f;

    // =========================================================================
    // CONFIGURAÇÃO DO PIPELINE DA OAK-D (DEPTHAI)
    // =========================================================================
    dai::Pipeline pipeline;

    // 1. Nós dos Sensores
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();

    // 2. Nós de Saída para o Host (Annunnaki)
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("rgb");
    xoutDepth->setStreamName("depth");

    // 3. Propriedades da Câmara Colorida
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setPreviewSize(640, 400); // Resolução leve e perfeitamente compatível
    camRgb->setInterleaved(false);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    // 4. Propriedades do Stereo Depth
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setCamera("left");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // ALINHAMENTO CRÍTICO: Alinha o mapa de profundidade com o ecrã da câmara colorida (CAM_A)
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);

    // 5. Vinculação dos Nós (Links)
    camRgb->preview.link(xoutRgb->input);
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->depth.link(xoutDepth->input);

    // Inicializa o dispositivo OAK-D carregando o pipeline criado
    dai::Device device(pipeline);

    // Cria as filas de recepção no Host (Não bloqueantes para manter a fluidez)
    auto qRgb = device.getOutputQueue("rgb", 4, false);
    auto qDepth = device.getOutputQueue("depth", 4, false);

    // =========================================================================
    // INICIALIZAÇÃO DOS MÓDULOS OpenCV
    // =========================================================================
    cv::Ptr<cv::BackgroundSubtractorMOG2> subtrator =
        cv::createBackgroundSubtractorMOG2(LIMITE_APRENDIZADO, VALOR_THRESHOLD, false);

    cv::Mat frame, depthFrame, mascaraBinaria;

    int frameContador = 0;
    float fps = 0.0f;
    auto tempoAnterior = std::chrono::high_resolution_clock::now();

    std::cout << "[SUCESSO] Pipeline OAK-D iniciado. Monitorizando mesa..." << std::endl;

    // =========================================================================
    // LOOP PRINCIPAL DE PROCESSAMENTO DO HOST
    // =========================================================================
    while (true) {
        // ---------------------------------------------------------------------
        // PASSO 1: Captura Real dos Frames da OAK-D via DepthAI
        // ---------------------------------------------------------------------
        auto imgFrameRgb = qRgb->get<dai::ImgFrame>();
        auto imgFrameDepth = qDepth->get<dai::ImgFrame>();

        if (imgFrameRgb) {
            frame = imgFrameRgb->getCvFrame();
        }
        if (imgFrameDepth) {
            depthFrame = imgFrameDepth->getCvFrame();
        }

        if (frame.empty()) {
            continue;
        }

        frameContador++;

        // ---------------------------------------------------------------------
        // PASSO 2: Cálculo Matemático do FPS
        // ---------------------------------------------------------------------
        auto tempoAtual = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duracao = tempoAtual - tempoAnterior;
        tempoAnterior = tempoAtual;

        float fpsInstantaneo = 1.0f / duracao.count();
        fps = (fps * 0.9f) + (fpsInstantaneo * 0.1f); // Filtro passa-baixa para estabilizar leitura

        // ---------------------------------------------------------------------
        // PASSO 3: Subtração de Fundo e Morfologia Clássica
        // ---------------------------------------------------------------------
        subtrator->apply(frame, mascaraBinaria);

        // Operações morfológicas para limpar poeira de pixeis isolados
        cv::morphologyEx(mascaraBinaria, mascaraBinaria, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        cv::morphologyEx(mascaraBinaria, mascaraBinaria, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));

        // ---------------------------------------------------------------------
        // PASSO 4: Análise de Contornos, Centro de Massa e Projeção 3D
        // ---------------------------------------------------------------------
        if (frameContador > 30) { // Evita os segundos iniciais ruidosos do MOG2
            std::vector<std::vector<cv::Point>> contornos;
            cv::findContours(mascaraBinaria, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contornos.size(); i++) {
                if (cv::contourArea(contornos[i]) > AREA_MINIMA_OBJ) {

                    // Desenha o Bounding Box
                    cv::Rect bbox = cv::boundingRect(contornos[i]);
                    cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

                    // Computa os momentos para encontrar o Centro de Massa (Centróide)
                    cv::Moments m = cv::moments(contornos[i]);
                    if (m.m00 != 0) {
                        int cX = static_cast<int>(m.m10 / m.m00);
                        int cY = static_cast<int>(m.m01 / m.m00);

                        // Marca o centróide com um círculo azul
                        cv::circle(frame, cv::Point(cX, cY), 5, cv::Scalar(255, 0, 0), -1);

                        // Se o mapa de profundidade estiver pronto, projeta as coordenadas 3D reais
                        if (!depthFrame.empty()) {
                            uint16_t depthMm = depthFrame.at<uint16_t>(cY, cX);
                            if (depthMm > 0) {
                                float z_metros = depthMm / 1000.0f;
                                float c_x = frame.cols / 2.0f;
                                float c_y = frame.rows / 2.0f;

                                // Equação de Projeção Perspectiva Métrica
                                float x_real = ((cX - c_x) * z_metros) / FOCAL_LENGTH;
                                float y_real = ((cY - c_y) * z_metros) / FOCAL_LENGTH;

                                // Escreve a telemetria 3D sobre o objeto no ecrã
                                std::string coordTexto = "XYZ: [" + cv::format("%.2f", x_real) + ", " +
                                                         cv::format("%.2f", y_real) + ", " +
                                                         cv::format("%.2f", z_metros) + "]m";
                                cv::putText(frame, coordTexto, cv::Point(bbox.x, bbox.y - 10),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // PASSO 5: Renderização das Regras de Cores e Interface
        // ---------------------------------------------------------------------
        cv::Scalar corInterface;
        std::string statusSistema;

        // Firula Visual: Menor que o limite = Vermelho. Maior = Amarelo.
        if (frameContador < LIMITE_APRENDIZADO) {
            corInterface = cv::Scalar(0, 0, 255); // Vermelho puro em BGR
            statusSistema = "STATUS: CALIBRANDO BACKGROUND";
        } else {
            corInterface = cv::Scalar(0, 255, 255); // Amarelo puro em BGR
            statusSistema = "STATUS: MONITORAMENTO ATIVO";
        }

        std::string textoContador = "FRAME: " + std::to_string(frameContador) + " / " + std::to_string(LIMITE_APRENDIZADO);
        std::string textoFps      = "HOST FPS: " + cv::format("%.1f", fps);

        // Desenha as informações diretamente no ecrã colorida desde o frame 0
        cv::putText(frame, statusSistema, cv::Point(20, 40),  cv::FONT_HERSHEY_SIMPLEX, 0.7, corInterface, 2);
        cv::putText(frame, textoContador,  cv::Point(20, 70),  cv::FONT_HERSHEY_SIMPLEX, 0.6, corInterface, 1);
        cv::putText(frame, textoFps,       cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, corInterface, 1);

        // ---------------------------------------------------------------------
        // PASSO 6: Exibição das Janelas
        // ---------------------------------------------------------------------
        cv::imshow("ChessBOT POC - Saida Colorida", frame);
        cv::imshow("ChessBOT POC - Mascara Binaria (MOG2)", mascaraBinaria);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
