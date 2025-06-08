from fastai.vision.all import *
import pathlib

# 1) Definir la ruta al dataset
#    Aquí usamos pathlib para que funcione en Windows/macOS/Linux sin cambiar "/\".
path = pathlib.Path(__file__).parent.parent / "dataset"

# 2) Configurar el DataBlock
#    - ImageBlock: le dice a Fastai que estamos trabajando con imágenes.
#    - CategoryBlock: que es clasificación por carpetas (la carpeta se llama "covid" o "normal").
#    - get_items: busca todas las imágenes en 'dataset'.
#    - splitter: separa aleatoriamente un 20% para validación.
#    - get_y: la etiqueta (label) viene del nombre de la carpeta padre ("covid" o "normal").
#    - item_tfms: resize a 224×224 (tamaño estándar de ImageNet).
#    - batch_tfms: augmentations por defecto (rotaciones, flips, etc.) para no sobreajustar.
covid_data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(mult=1.0)
)

# 3) Crear DataLoaders (da el objeto con loaders de train y valid)
dls = covid_data.dataloaders(path, bs=16, num_workers=0)
  

# 4) Mostrar algunos ejemplos (opcional, para verificar que cargue bien)
dls.show_batch(max_n=6)  # Esto abre una ventana gráfica con 6 imágenes y sus etiquetas

# 5) Definir el learner (modelo) usando un pre-entrenado de torchvision
#    En este caso, usamos 'resnet34' por ser ligero y rápido de entrenar.
#    metrics: accuracy (exactitud) y RocAucBinary (curva ROC-AUC para clasificación binaria).
learn = vision_learner(dls, resnet34, metrics=[accuracy, RocAucBinary()])

# 6) Entrenamiento “rápido”:
#    - Frozen: entrena solo la última cabeza (congelando la base pre-entrenada).
#    - Luego fine-tune: descongela todo y entrena un par de épocas más.
#    Aquí haremos 3 épocas en total (1 frozen + 2 fine-tune).
learn.fine_tune(2)

# 7) Guardar el modelo exportado para inferencia
#    Esto crea 'export.pkl' con todo lo necesario para cargar el modelo más tarde.
learn.export(pathlib.Path(__file__).parent.parent / "export.pkl")

print("Entrenamiento finalizado. Modelo guardado en 'export.pkl'")
