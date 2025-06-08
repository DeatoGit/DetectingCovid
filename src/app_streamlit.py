import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib

st.title("Detector de COVID-19 en Radiografías de Tórax")
st.write("""
  Sube una imagen de radiografía (formato JPG o PNG) y el modelo te dirá 
  si hay indicios de COVID-19 o no, con un nivel de confianza.
""")

@st.cache_resource
def cargar_modelo():
    ruta_modelo = pathlib.Path(__file__).parent.parent / "export.pkl"
    return load_learner(ruta_modelo)

learner = cargar_modelo()

uploaded_file = st.file_uploader("Elige una radiografía...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiografía cargada", use_column_width=True)
    st.write("")

    # Fastai se encarga de redimensionar y normalizar la imagen internamente:
    pred_label, pred_idx, outputs = learner.predict(image)
    # outputs es un tensor con tantas posiciones como clases. 
    # Por ejemplo, si hay 2 clases ("covid","normal"), outputs = [p_normal, p_covid]

    # Para acceder a la probabilidad de "covid", debemos buscar su índice:
    idx_covid = learner.dls.vocab.o2i["covid"]
    prob_covid = float(outputs[idx_covid])

    st.markdown(f"**Predicción:** {pred_label.upper()}")
    st.markdown(f"**Probabilidad de COVID-19:** {prob_covid:.4f}")
    st.write("""
      - Si la probabilidad es cercana a 1.0, el modelo está seguro de que es COVID-19.
      - Si es cercana a 0.0, el modelo considera que NO es COVID-19.
    """)

