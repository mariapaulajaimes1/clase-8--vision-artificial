import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yolov5

# Cargar modelo de YOLOv5
try:
    model = yolov5.load('./yolov5s.pt')
except Exception:
    st.error("Error al cargar el modelo. Asegúrate de que el archivo yolov5s.pt esté en la carpeta.")
    st.stop()

# Configurar parámetros del modelo
model.conf = 0.25
model.iou = 0.45

st.title("🖼️ Detección de Objetos en Imágenes")
with st.sidebar:
    st.header("⚙️ Parámetros de Configuración")
    model.iou = st.slider('📏 IoU (Intersección sobre Unión)', 0.0, 1.0, model.iou)
    st.write('**IoU seleccionado:**', model.iou)
    model.conf = st.slider('🕵️‍♀️ Confianza', 0.0, 1.0, model.conf)
    st.write('**Confianza seleccionada:**', model.conf)

picture = st.camera_input("📸 Capturar foto")

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Redimensionar imagen si es muy grande
    max_width = 800
    height, width, _ = cv2_img.shape
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, int(height * scale))
        cv2_img = cv2.resize(cv2_img, new_size)

    # Realizar inferencia
    results = model(cv2_img)
    predictions = results.pred[0]

    # Obtener resultados
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    col1, col2 = st.columns(2)

    with col1:
        results.render()  # Renderiza las detecciones en la imagen
        st.image(cv2_img, channels='BGR')

    with col2:
        st.header("📊 Resultados de Detección")
        label_names = model.names
        category_count = {}
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = []
        for category, count in category_count.items():
            label = label_names[int(category)]
            data.append({"Categoría": label, "Cantidad": count})
        
        data_df = pd.DataFrame(data)
        st.dataframe(data_df.head(10))  # Mostrar solo las primeras 10 filas
