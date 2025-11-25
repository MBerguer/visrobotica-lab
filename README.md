# Trabajo Práctico - Visión por Computadora: Reconstrucción 3D y Estimación de Pose

Implementación completa del trabajo práctico de Robótica Móvil centrado en visión estéreo, reconstrucción 3D y estimación de pose utilizando el dataset EuRoC MAV.

## Guía Rápida

```bash
# 1. Clonar e instalar dependencias
git clone <repository-url>
cd visrobotica-lab
pip install -r requirements.txt

# 2. Descargar rosbag (2.5 GB, puede tardar 10-30 min)
mkdir -p data/MH_01_easy
cd data/MH_01_easy
curl -L -o MH_01_easy_with_camera_info.db3 \
  "http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/MH_01_easy_with_camera_info/MH_01_easy_with_camera_info.db3"
curl -L -o metadata.yaml \
  "http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/MH_01_easy_with_camera_info/metadata.yaml"
cd ../..

# 3. Ejecutar todos los ejercicios (5-10 min)
./run_all.sh data/MH_01_easy results

# 4. Compilar el informe PDF
cd docs
pdflatex informe.tex
pdflatex informe.tex
```

**Requisitos de espacio:** ~3 GB (rosbag + resultados)

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

## Instalación Completa

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd visrobotica-lab
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O usando Docker:

```bash
docker build -t tp-vision .
docker run -it -v $(pwd):/workspace tp-vision
```

### 3. Descargar el dataset

El proyecto requiere un rosbag del dataset EuRoC MAV. Los rosbags están disponibles en formato ROS2:

```bash
# Crear directorio para datos
mkdir -p data/MH_01_easy

# Descargar el rosbag (aproximadamente 2.5 GB - puede tardar varios minutos)
cd data/MH_01_easy
curl -L -o MH_01_easy_with_camera_info.db3 \
  "http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/MH_01_easy_with_camera_info/MH_01_easy_with_camera_info.db3"

# Descargar metadata (requerido para ROS2)
curl -L -o metadata.yaml \
  "http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/MH_01_easy_with_camera_info/metadata.yaml"

cd ../..
```

**Nota:** El archivo del rosbag es grande (~2.5 GB). La descarga puede tardar 10-30 minutos dependiendo de la velocidad de conexión.

### 4. Ejecutar todos los ejercicios

Una vez descargado el rosbag, ejecutar el pipeline completo:

```bash
./run_all.sh data/MH_01_easy results
```

Este script ejecuta automáticamente:
- Calibración de cámara estéreo
- Pipeline estéreo completo (ejercicios 2a-2h)
- Estimación de trayectoria (ejercicio 2j)
- Mapeo de features (ejercicio 2f opcional)
- Mapeo denso (ejercicio 2i opcional)

**Tiempo estimado:** 5-10 minutos dependiendo del hardware.

### 5. Compilar el informe PDF

Después de ejecutar los ejercicios, las imágenes se generan en `results/`. Para compilar el PDF con todas las imágenes:

```bash
cd docs
pdflatex informe.tex
pdflatex informe.tex  # Segunda pasada para referencias cruzadas
```

El PDF compilado estará en `docs/informe.pdf`.

## Uso Avanzado

### Ejecutar ejercicios individuales

```bash
# Calibración
python3 src/camera_calibration.py

# Pipeline estéreo completo
python3 src/stereo_pipeline.py --rosbag data/MH_01_easy --output_dir results

# Estimación de trayectoria
python3 src/trajectory_estimation.py --rosbag data/MH_01_easy --output_dir results

# Mapeo de features
python3 src/feature_mapping.py --rosbag data/MH_01_easy --output_dir results --mode sparse

# Mapeo denso
python3 src/feature_mapping.py --rosbag data/MH_01_easy --output_dir results --mode dense
```

## Dataset

Se utiliza el dataset EuRoC MAV en formato ROS2. Los rosbags están disponibles en:
http://fs01.cifasis-conicet.gov.ar:90/~pire/datasets/euroc_rosbag2/

El formato esperado es un directorio con:
- `*.db3` - Archivo de datos SQLite del rosbag
- `metadata.yaml` - Metadata del rosbag (requerido para ROS2)

## Resultados

Los resultados se generan en la carpeta `results/`:
- Imágenes rectificadas (`rectified_left.png`, `rectified_right.png`)
- Visualizaciones de features (`features.png`)
- Visualizaciones de matches (`matches_all.png`, `matches_filtered.png`, `matches_ransac.png`)
- Mapa de disparidad (`disparity_map.png`)
- Visualización de trayectoria (`trajectory.png`)
- Nubes de puntos (.ply) - sparse y dense
- Métricas de error de trayectoria

**Nota:** Las imágenes generadas son necesarias para compilar el informe PDF completo.

## Informe

El informe LaTeX se encuentra en `docs/informe.tex`. Para compilar:

```bash
cd docs
pdflatex informe.tex
pdflatex informe.tex
```

