# Trabajo Práctico - Visión por Computadora: Reconstrucción 3D y Estimación de Pose

Implementación completa del trabajo práctico de Robótica Móvil centrado en visión estéreo, reconstrucción 3D y estimación de pose utilizando el dataset EuRoC MAV.

## Estructura del Proyecto

```
tp-vision/
├── src/
│   ├── camera_calibration.py      # Ejercicio 1: Calibración de cámara estéreo
│   ├── stereo_pipeline.py          # Ejercicios 2a-2h: Pipeline completo de visión estéreo
│   ├── trajectory_estimation.py   # Ejercicio 2j: Estimación de trayectoria
│   ├── feature_mapping.py         # Ejercicios 2f,2i: Mapeo con ground-truth (opcional)
│   ├── ground_truth_loader.py     # Carga y transformación de ground-truth
│   └── extract_groundtruth_from_rosbag.py  # Extracción de GT desde rosbag
├── calibration/                   # Datos de calibración
├── docs/                          # Informe LaTeX
├── Dockerfile                     # Para reproducibilidad
├── requirements.txt               # Dependencias Python
├── run_all.sh                     # Script para ejecutar todos los ejercicios
└── README.md                      # Este archivo
```

## Requisitos

- Python 3.8+
- OpenCV 4.8+
- NumPy, Matplotlib, SciPy
- rosbags (para leer archivos ROS2)

## Instalación

```bash
pip install -r requirements.txt
```

O usando Docker:

```bash
docker build -t tp-vision .
docker run -it -v $(pwd):/workspace tp-vision
```

## Uso

Ejecutar el pipeline completo:

```bash
./run_all.sh data/MH_01_easy results
```

O ejecutar ejercicios individuales:

```bash
# Calibración
python3 src/camera_calibration.py

# Pipeline estéreo completo
python3 src/stereo_pipeline.py --rosbag data/MH_01_easy --output_dir results

# Estimación de trayectoria
python3 src/trajectory_estimation.py --rosbag data/MH_01_easy --output_dir results
```

## Dataset

Se utiliza el dataset EuRoC MAV. Los rosbags deben descargarse desde:
http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/

## Resultados

Los resultados se generan en la carpeta `results/`:
- Imágenes rectificadas y visualizaciones
- Nubes de puntos (.ply)
- Trayectoria estimada
- Métricas de error

## Informe

El informe LaTeX se encuentra en `docs/informe.tex`. Para compilar:

```bash
cd docs
pdflatex informe.tex
pdflatex informe.tex
```

