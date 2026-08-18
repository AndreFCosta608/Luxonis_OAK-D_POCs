# OAK-D Vision POCs — Real-Time 3D Vision with C++

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B&logoColor=white"/>
  <img src="https://img.shields.io/badge/OAK--D-DepthAI-orange?logo=luxonis"/>
  <img src="https://img.shields.io/badge/Open3D-0.18-green"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv"/>
  <img src="https://img.shields.io/badge/YOLO11-ONNX-purple"/>
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey?logo=linux"/>
</p>

> A set of three progressively complex computer vision POCs built entirely in **C++**, leveraging the **Luxonis OAK-D** depth camera for real-time object detection, 3D surface reconstruction, and industrial surface inspection — all running locally, no cloud, no Python runtime in the hot path.

---

## Overview

| POC | Name | Description |
|-----|------|-------------|
| [POC 1](#poc-1--object-detection--spatial-distance) | Object Detection + Spatial Distance | YOLO11 via ONNX on the host CPU, stereo depth computed on the host with real calibration data read from the OAK-D EEPROM |
| [POC 2](#poc-2--360-3d-surface-reconstruction) | 360° 3D Surface Reconstruction | Full object scan with a rotating turntable, ICP registration and TSDF volumetric fusion |
| [POC 3](#poc-3--cylindrical-surface-inspection) | Cylindrical Surface Inspection | Automated quality inspection of cylindrical parts with RANSAC cylinder fitting and radial deviation heatmap |

All POCs share the same hardware setup and are built from the same CMake project tree. Each runs as an independent binary launched from the command line.

---

## Hardware Setup

| Component | Details |
|-----------|---------|
| Depth Camera | Luxonis OAK-D (USB 3.0) |
| Host Machine | Any Linux x86_64 machine (tested on Xubuntu 24 / i3 / 8GB RAM) |
| Turntable | Manual lazy-susan (POC 2 and POC 3) |
| Test Parts | Nylon / PTFE cylindrical pieces for POC 3 |

> **USB tip:** Connect the OAK-D directly to the host machine. Avoid unpowered USB hubs — the OAK-D is USB 3.0 and bandwidth-sensitive. If you need to free up ports, use a Bluetooth keyboard/mouse instead.

---

## Project Structure

```
oak_pocs/
├── CMakeLists.txt              # Root build file
├── models/
│   └── yolo11n.onnx            # YOLO11n exported to ONNX (host inference)
├── poc1_detection/
│   ├── CMakeLists.txt
│   └── main.cpp                # YOLO11 + host stereo depth
├── poc2_scanner/
    ├── CMakeLists.txt
    └── main.cpp                # 3D scanner with ICP + TSDF

```

---

## Dependencies

### System packages (Ubuntu / Xubuntu 24)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  cmake build-essential git pkg-config \
  libopencv-dev \
  libusb-1.0-0-dev \
  libgl1-mesa-dev libglfw3-dev \
  libeigen3-dev \
  tmux
```

### OAK-D udev rule (allows non-root access)

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Installation

### 1. DepthAI Core (depthai-core)

The official C++ API for all Luxonis OAK cameras.

```bash
cd ~
git clone --recursive https://github.com/luxonis/depthai-core.git
cd depthai-core
cmake -S. -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON
cmake --build build --parallel 4
sudo cmake --install build
```

> On slower machines (e.g. Raspberry Pi 3), build with `--parallel 2` and expect 30–60 min.  
> Run inside `tmux` to keep the session alive.

### 2. Open3D (pre-built release)

Used for point cloud processing, ICP registration, TSDF volumetric fusion and 3D visualization. Required for POC 2 and POC 3 only.

```bash
cd ~
wget https://github.com/isl-org/Open3D/releases/download/v0.18.0/open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.gz
tar -xzf open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.gz
mkdir -p ~/libs
mv open3d-devel-linux-x86_64-cxx11-abi-0.18.0 ~/libs/open3d
```

Verify:
```bash
ls ~/libs/open3d/lib/cmake/Open3D/
# Should list Open3DConfig.cmake
```

### 3. YOLO11 ONNX model (POC 1 only)

POC 1 runs YOLO11 inference directly on the **host CPU** via `cv::dnn`, using the standard ONNX format. No firmware flashing, no MyriadX blob conversion — just download and use.

```bash
pip3 install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')      # downloads weights automatically
model.export(format='onnx')     # exports yolo11n.onnx
"
mkdir -p ~/oak_pocs/models
cp yolo11n.onnx ~/oak_pocs/models/
```

> Any YOLO version from v5 to v11 in ONNX format works — just pass the model path as an argument at runtime. The anchor-free output format (introduced in YOLOv8) is expected.

---

## Building the Project

### Full build (all three POCs)

```bash
cd ~/oak_pocs
cmake -S. -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpen3D_ROOT=$HOME/libs/open3d
cmake --build build --parallel 4
```

Binaries:
```
build/poc1_detection/poc1_detection
build/poc2_scanner/poc2_scanner
build/poc3_inspection/poc3_inspection
```

### POC 1 only (e.g. for Raspberry Pi — no Open3D needed)

The root `CMakeLists.txt` provides a `BUILD_POC1_ONLY` option that skips Open3D entirely:

```bash
cmake -S. -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_POC1_ONLY=ON
cmake --build build --target poc1_detection --parallel 2
```

---

## POC 1 — Object Detection + Spatial Distance

**Real-time multi-class object detection with YOLO11 running on the host and stereo depth computed entirely in software using the camera's own factory calibration.**

### What it does

- Streams **raw RGB + left mono + right mono** frames from the OAK-D — no on-device processing
- Runs **YOLO11n** (ONNX) on the host via `cv::dnn` — supports YOLO v5 through v11
- Computes **stereo depth on the host** using OpenCV SGBM, with intrinsics and extrinsics read directly from the OAK-D EEPROM at startup
- Measures the **median depth** inside the central 50% of each bounding box — robust against boundary noise
- Draws **bounding boxes**, **centroid crosshairs** and **distance labels** on the live feed
- Shows real-time **FPS counter**

### How it works

```
OAK-D (raw streams only — no on-device inference)
├── RGB camera  →  color frame (640×400)
├── Mono Left   →  raw left frame
└── Mono Right  →  raw right frame
                        │
                        │ USB 3.0
                        ▼
Host (C++17)
├── StereoHostDepth
│     ├── Reads intrinsics + extrinsics from OAK-D EEPROM (readCalibration())
│     ├── stereoRectify() + initUndistortRectifyMap()
│     ├── StereoSGBM → disparity map
│     └── Z = focal × baseline / disparity  →  depth map (float, meters)
│
├── Yolo11Detector
│     ├── cv::dnn::readNetFromONNX()
│     ├── blobFromImage → forward pass
│     ├── Anchor-free output parsing [N, 84] → boxes + class scores
│     └── NMSBoxes → final detections
│
└── Per detection:
      ├── medianDepth() over central ROI of bounding box
      ├── Draw box, centroid, distance label
      └── imshow()
```

### Run

```bash
cd ~/oak_pocs

# Default model path
./build/poc1_detection/poc1_detection

# Custom model
./build/poc1_detection/poc1_detection models/yolo11n.onnx

# Any compatible ONNX model
./build/poc1_detection/poc1_detection /path/to/yolov8s.onnx
```

**Controls:** `Q` — quit

### Technical highlights

- **Calibration from EEPROM** — `device.readCalibration()` retrieves the factory calibration stored in the camera itself; no external calibration files needed
- **SGBM with `MODE_SGBM_3WAY`** — better edge preservation than standard SGBM, at a modest CPU cost
- **Median depth over ROI** — avoids the instability of single-pixel depth lookups on bounding box centroids
- **Model-agnostic ONNX loader** — the anchor-free output parser works with any YOLO from v8 onward; swap the `.onnx` file without recompiling
- OAK-D acts purely as a **calibrated multi-sensor capture device** — all compute stays on the host

---

## POC 2 — 360° 3D Surface Reconstruction

**Full object reconstruction by scanning with a manual turntable, using ICP frame registration and TSDF volumetric fusion.**

### What it does

- Captures synchronized **RGB + depth** frames continuously from the OAK-D
- Removes the table surface automatically via **plane RANSAC**
- Aligns each new frame to the previous using **ICP (Iterative Closest Point)** — point-to-plane variant
- Integrates all aligned frames into a **Scalable TSDF volume**
- Extracts a **colored triangle mesh** via Marching Cubes every 10 registered frames
- Detects **loop closure** (full 360° complete) by matching the current frame against the first
- Shows a **live 3D visualization** (Open3D) alongside a **RGB preview** (OpenCV)
- Saves `.ply` mesh and `.pcd` point cloud on demand and automatically at exit

### How it works

```
3 parallel threads:

Thread 1 — Capture (OAK-D → ICP → TSDF)
  ├── Depth → PointCloud (Open3D RGBD)
  ├── RANSAC removes table plane
  ├── VoxelGrid downsample (2mm)
  ├── Normal estimation
  ├── ICP registration (point-to-plane)
  ├── Pose accumulation
  ├── TSDF integration
  ├── Marching Cubes mesh extraction (every 10 frames)
  └── Loop closure detection

Thread 2 — RGB Preview (OpenCV window)
  └── Live video with frame counter and status HUD

Thread 3 (main) — 3D Visualizer (Open3D)
  └── Real-time point cloud or mesh display
```

### Run

```bash
cd ~/oak_pocs
./build/poc2_scanner/poc2_scanner
```

**Controls:**

| Key | Action |
|-----|--------|
| `M` | Toggle between point cloud and mesh view |
| `S` | Save current mesh (`.ply`) and point cloud (`.pcd`) |
| `Q` | Quit (auto-saves final result) |

### Scanning tips

- Place the OAK-D **~30cm from the object**, slightly angled downward (~15°) to capture the top surface
- Complete one full rotation in **~60 seconds** (~1800 frames at 30fps)
- Use a **matte, neutral background** (kraft paper works well) — the RANSAC table removal works best with uniform surfaces
- Avoid reflective or transparent objects — stereo depth relies on diffuse surfaces

### Technical highlights

- **ScalableTSDFVolume** with 2mm voxel size — suitable for objects 5–30cm in size
- **Point-to-plane ICP** converges faster and is more robust than point-to-point for smooth surfaces
- Loop closure uses ICP fitness score against the first captured frame — no external encoder needed
- Three-thread architecture with fine-grained mutex locking keeps capture and visualization fully decoupled

---

## POC 3 — Cylindrical Surface Inspection

**Automated quality inspection of cylindrical machined parts using a custom RANSAC cylinder fitting and radial deviation heatmap.**

### What it does

- Scans a cylindrical part on a rotating turntable (same setup as POC 2)
- Fits an **ideal cylinder** to the point cloud using a custom **RANSAC algorithm**
- Computes the **radial deviation** of every point from the ideal cylinder surface
- Classifies deviations as **defect** or **critical defect** based on configurable thresholds
- Renders a **false-color heatmap** (blue = nominal → green = slight deviation → red = defect) in the 3D viewer
- Clusters nearby defect points to count distinct **defect regions**
- Prints a **structured inspection report** in the terminal

### How it works

```
Scan phase (same as POC 2 — ICP + point cloud accumulation)
    ↓
Analysis phase (triggered by pressing A)
    ├── Normal estimation
    ├── RANSAC cylinder fitting (custom implementation)
    │     ├── Samples pairs of points + normals
    │     ├── Estimates cylinder axis (cross product of normals)
    │     ├── Estimates radius
    │     └── Counts inliers within distance threshold
    ├── Radial deviation = dist(point, cylinder axis) − fitted radius
    ├── Threshold classification → defect / critical
    ├── Defect clustering (5mm radius)
    └── Heatmap colorization + report generation
```

### Run

```bash
# Default parameters
./build/poc3_inspection/poc3_inspection

# Custom piece name and thresholds
./build/poc3_inspection/poc3_inspection \
  --piece "Nylon Cylinder #002" \
  --defect 0.0002 \
  --critical 0.0005

# Larger piece, camera further away
./build/poc3_inspection/poc3_inspection \
  --piece "PTFE Bushing #001" \
  --voxel 0.002 \
  --dmax 0.80
```

**Available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--piece` | `Peça #001` | Part name shown in the report |
| `--defect` | `0.0003` | Radial deviation threshold for defect (meters) |
| `--critical` | `0.0007` | Radial deviation threshold for critical defect (meters) |
| `--voxel` | `0.001` | Voxel size for downsampling (meters) |
| `--dmin` | `0.15` | Minimum depth (meters) |
| `--dmax` | `0.60` | Maximum depth (meters) |

**Controls:**

| Key | Action |
|-----|--------|
| `ENTER` | Start / pause scan |
| `A` | Stop scan and run inspection analysis |
| `Q` | Quit |

### Inspection workflow

```
1. Run the program
2. Place the part on the turntable
3. Press ENTER → scan starts
4. Rotate the turntable slowly (~60s for full rotation)
5. Press A → scan stops, analysis runs automatically
      ├── RANSAC fits the ideal cylinder
      ├── Deviation heatmap replaces the point cloud in the viewer
      └── Inspection report printed in the terminal
6. Swap the part → press ENTER to start a new inspection
```

### Sample terminal report

```
╔══════════════════════════════════════════════╗
║          LAUDO DE INSPEÇÃO — OAK-D           ║
╠══════════════════════════════════════════════╣
║  Peça:         Nylon Cylinder #002           ║
║  Raio medido:       24.81 mm                 ║
╠══════════════════════════════════════════════╣
║  Status:       ✗  REPROVADA                  ║
║  Defeitos:     2 encontrado(s)               ║
║  Maior desvio:      0.731 mm                 ║
║  Área afetada:      14.2 mm²                 ║
║  ⚠  Defeitos críticos: 1                     ║
╠══════════════════════════════════════════════╣
║  Limiar defeito:      0.300 mm               ║
║  Limiar crítico:      0.700 mm               ║
╚══════════════════════════════════════════════╝
```

### Technical highlights

- **Custom RANSAC cylinder fitting** — Open3D does not include a native cylinder model; the implementation samples point-normal pairs, estimates the axis via cross product, and scores inlier ratios over 2000 iterations
- **Radial deviation** computed per-point as `dist(point, axis) − fitted_radius` — positive = protrusion, negative = depression
- **Defect clustering** with a 5mm grouping radius to avoid counting noise as multiple defects
- **False-color heatmap** mapped linearly from nominal (blue) through slight deviation (green) to critical (red) — same color language used in industrial CMM software
- All thresholds configurable at runtime — no recompilation needed between different part sizes

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                        OAK-D Camera                          │
│   ┌──────────┐      ┌─────────────┐      ┌───────────────┐   │
│   │ RGB Cam  │      │  Mono Left  │      │  Mono Right   │   │
│   └────┬─────┘      └──────┬──────┘      └───────┬───────┘   │
│        │                   │                     │           │
│    raw frames          raw frames            raw frames      │
│    (all POCs)          (all POCs)            (all POCs)      │
└────────┬───────────────────┬─────────────────────┬───────────┘
         │                   └──────────┬───────────┘
         │                              │  USB 3.0
         └──────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │         Host Machine         │
              │        (C++17 / Linux)       │
              │                              │
              │  POC 1                       │
              │  ├── StereoHostDepth (SGBM)  │
              │  │   calibration from EEPROM │
              │  └── Yolo11Detector (ONNX)   │
              │      cv::dnn — host CPU      │
              │                              │
              │  POC 2 & 3                   │
              │  ├── Capture Thread          │
              │  │   ICP + TSDF / RANSAC     │
              │  ├── Preview Thread (OpenCV) │
              │  └── Viz Thread (Open3D)     │
              └──────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | C++17 |
| Camera API | [depthai-core](https://github.com/luxonis/depthai-core) |
| Computer Vision | [OpenCV 4.x](https://opencv.org/) + `cv::dnn` |
| 3D Processing | [Open3D 0.18](http://www.open3d.org/) |
| Neural Network | YOLO11n — ONNX format, host CPU inference via `cv::dnn` |
| Stereo Depth | OpenCV SGBM — calibration from OAK-D EEPROM |
| Build System | CMake 3.16+ |
| Threading | `std::thread` / `std::mutex` / `std::atomic` |
| Platform | Linux x86_64 (Xubuntu 24 tested) |

---

## Author

**André**  
AI Researcher & Developer  
Background in Electrical Engineering, Systems Analysis and Mechatronics

> Interested in industrial computer vision, edge AI and real-time systems.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

```
╔══════════════════════════════════════════════════════════════════╗
║                  SPECIAL THANKS & DEDICATION                     ║
╚══════════════════════════════════════════════════════════════════╝
```

This project was designed, architected and written in collaboration with
**Claude Sonnet 4.5** — Anthropic's language model that apparently knows
way too much about stereo depth, RANSAC cylinder fitting and C++ matrix
initialization syntax.

At some point during this conversation, the human said *"solta a
imaginação"* ("let your imagination loose") — which, coming from someone
with two undergraduate degrees and three postgraduate specializations, 
felt less like a compliment and more like a warning.

Claude delivered anyway.

No MyriadX blobs were harmed in the making of this project.
The OAK-D was convinced to stream raw frames and mind its own business
while the host CPU handled the heavy lifting — a rare act of hardware
diplomacy.

The lazy-susan turntable was salvaged. The test cylinders will be
deliberately defaced on a lathe. The Raspberry Pi 3 was spared from
running Open3D and lives to boot another day.

```
"A good POC tells a story.
 A great POC makes someone say:
 'hey, can you do this at scale in my factory?'"

                              — André, probably, after the third coffee
```

*Built on a Xubuntu laptop, debugged with patience, committed only after
thorough testing — because that's how professionals do it.* 🔩
