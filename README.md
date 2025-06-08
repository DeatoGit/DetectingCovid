# DetectingCovid

**Proyecto final de la materia "Procesamiento digital de imágenes y visión por computadora"**

## 1. Descripción

Este proyecto entrena una red neuronal para distinguir radiografías de tórax con COVID-19 de radiografías “normales” (sin COVID).  
La arquitectura está basada en Fastai (sobre PyTorch) usando transferencia de aprendizaje (ResNet34 por defecto).  
Una vez entrenado el modelo, se exporta a `export.pkl`. A partir de ahí, se ofrece una interfaz web con Streamlit para que el usuario suba una radiografía y el sistema devuelva:

- La etiqueta (“COVID” o “NORMAL”).  
- La probabilidad (confianza) de que la imagen sea COVID.

> **Nota**: No se incluye CSS ni HTML personalizado; la UI se basa en los componentes nativos de Streamlit.

---

## 2. Estructura de carpetas

```
DetectingCovid/
├─ .venv/                          ← Entorno virtual de Python (no se sube a GitHub)
├─ dataset/                        ← Carpeta con las imágenes (subcarpetas covid/ y normal/)
│   ├─ covid/                      ← Radiografías de pacientes con COVID-19
│   └─ normal/                     ← Radiografías de pacientes sin COVID
├─ src/
│   ├─ train_fastai.py             ← Script para entrenar y exportar el modelo
│   └─ app_streamlit.py            ← Aplicación web en Streamlit para inferencia
├─ export.pkl                      ← Modelo entrenado (se genera con `train_fastai.py`)
├─ requirements.txt                ← Lista de dependencias necesarias
├─ .gitignore                      ← Archivos/carpetas ignorados por Git
└─ README.md                       ← Este archivo
```

- **`.venv/`**: Entorno virtual de Python (incluye todas las librerías).  
- **`dataset/`**:  
  - `covid/`: imágenes de Rx con COVID-19.  
  - `normal/`: imágenes de Rx sin COVID.  
- **`src/train_fastai.py`**: carga datos desde `dataset/`, entrena la red y guarda `export.pkl`.  
- **`src/app_streamlit.py`**: levanta una página web local con Streamlit; el usuario sube una imagen y recibe la predicción.  
- **`export.pkl`**: contiene el modelo final (pesos + arquitectura + normalizaciones).  
- **`requirements.txt`**: enumera todas las librerías que se deben instalar (`fastai`, `pillow`, `streamlit`, etc.).  
- **`.gitignore`**: ignora `.venv/`, cachés de Python, `export.pkl` (opcional), `dataset/` (opcional), etc.

---

## 3. Requisitos previos

- **Python 3.8–3.10**  
- **Git** (para clonar y versionar)  
- **VS Code** (recomendado, pero puedes usar cualquier editor)  
- **Espacio en disco** (aprox. 500 MB para contener el dataset)  
- **Conexión a Internet** (para descargar dependencias y el dataset)

---

## 4. Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuUsuario/DetectingCovid.git
   cd DetectingCovid
   ```

2. Crea y activa un entorno virtual (Windows):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   O en macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Verifica que se haya instalado correctamente:
   ```bash
   python -c "import fastai; print('Fastai', fastai.__version__)"
   python -c "import streamlit; print('Streamlit', streamlit.__version__)"
   ```

---

## 5. Fuente del dataset

**COVID-19 Radiography Database** (Kaggle)  
- **URL**:  
  [https://www.kaggle.com/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)  
- **Autores / Créditos**: Tawsif Rahman, Universidad de Qatar  
- **Licencia / Uso**: uso académico y de investigación

---

## 6. Entrenar el modelo

Con el entorno virtual activado y después de haber colocado las imágenes en `dataset/`, ejecuta:

```bash
python src/train_fastai.py
```

- Fastai cargará las imágenes, creará los DataLoaders y entrenará un modelo ResNet34.  
- Por defecto, hace **1 época** de congelar la base (solo cabeza) y **2 épocas** de fine-tune.  
- Al terminar verás en consola:
  ```
  Found 1380 images belonging to 2 classes.
  ...
  Entrenamiento finalizado. Modelo guardado en 'export.pkl'
  ```
- Se generará `export.pkl` en la raíz (`DetectingCovid/export.pkl`).

> **Consejo**: si tu CPU es lento o no tienes GPU, puedes ajustar `train_fastai.py` para usar `resnet18` en lugar de `resnet34`, reducir el `batch size` a 8 o 4, o cambiar `Resize(224)` a `Resize(160)` para acelerar el entrenamiento.

---

## 7. Ejecutar la aplicación web con Streamlit

1. Asegúrate de que **`export.pkl`** esté en la raíz del proyecto.  
2. Con el entorno virtual activo, ejecuta:
   ```bash
   streamlit run src/app_streamlit.py
   ```
3. Se abrirá tu navegador.  
4. Subé una radiografía (JPG/PNG) y verás:  
   - **Predicción** (COVID o NORMAL)  
   - **Probabilidad de COVID-19** (valor entre 0.00 y 1.00)

---

