# poc_volumetric3D/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(poc_volumetric3D VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Busca as dependências do sistema
find_package(depthai REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# Busca a Open3D local
find_package(Open3D REQUIRED NO_DEFAULT_PATH
    PATHS "${CMAKE_CURRENT_LIST_DIR}/../../open3d/lib/cmake/Open3D")

# ONNX Runtime -- ajuste ONNXRUNTIME_ROOT se não detectar automaticamente.
# Se você já usa ONNX Runtime no Smith Chart Tool, aponte pro mesmo diretório
# (o que contém include/ e lib/), ex:
#   cmake -S. -Bbuild -DONNXRUNTIME_ROOT=/caminho/pra/onnxruntime-linux-x64-1.xx.x
set(ONNXRUNTIME_ROOT "" CACHE PATH "Diretório da instalação do ONNX Runtime (contendo include/ e lib/)")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Defina -DONNXRUNTIME_ROOT=/caminho/pra/onnxruntime ao rodar cmake (veja comentário acima)")
endif()
include_directories("${ONNXRUNTIME_ROOT}/include")
link_directories("${ONNXRUNTIME_ROOT}/lib")

# Libs comuns aos dois executáveis
set(COMMON_LIBS
    PRIVATE
    depthai::core
    depthai::opencv
    ${OpenCV_LIBS}
    -Wl,--no-as-needed
    Open3D::Open3D
    -Wl,--as-needed
    OpenSSL::SSL
    OpenSSL::Crypto
    c++        # <--- Resolve os erros de std::__1
    c++abi     # <--- Resolve os erros de suporte da libc++
    onnxruntime
)

# ── Versão simples (showroom): 1 vista, sem ArUco, sem merge ──
add_executable(poc_volumetric3D main.cpp)
target_link_libraries(poc_volumetric3D ${COMMON_LIBS})

# main_plus.cpp (4 vistas + ArUco + merge) fica guardado no repositório mas fora
# do build por enquanto -- descomente as duas linhas abaixo se quiser retomar:
# add_executable(poc_volumetric3D_plus main_plus.cpp)
# target_link_libraries(poc_volumetric3D_plus ${COMMON_LIBS})
