import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo preentrenado
model = yolov5.load('yolov5s.pt')

# Configurar parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45  # Umbral de IoU NMS
model.agnostic = False  # NMS independiente de clases
model.multi_label = False  # Etiquetas múltiples por caja
model.max_det = 1000  # Máximo número de detecciones por imagen

# Estilo general con CSS
st.markdown(
    """
    <style>
    .title {
        color: #4CAF50;
        font-size: 35px;
        font-weight: bold;
    }
    .subtitle {
        color: #2C3E50;
        font-size: 22px;
        font-weight: bold;
    }
    .sidebar-text {
        color: #007BFF;
        font-size: 18px;
    }
    .detection-box {
        color: #34495E;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y configuración en la barra lateral
st.markdown('<div class="title">🖼️ Detección de Objetos en Imágenes</div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="subtitle">⚙️ Parámetros de Configuración</div>', unsafe_allow_html=True)
    model.iou = st.slider('📏 IoU (Intersección sobre Unión)', 0.0, 1.0, model.iou)
    st.write('**IoU seleccionado:**', model.iou)

    model.conf = st.slider('🕵️‍♀️ Confianza', 0.0, 1.0, model.conf)
    st.write('**Confianza seleccionada:**', model.conf)

# Captura de imagen con la cámara
picture = st.camera_input("📸 Capturar foto")

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Realizar inferencia
    results = model(cv2_img)

    # Obtener predicciones
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    col1, col2 = st.columns(2)

    with col1:
        # Mostrar cajas de detección en la imagen
        results.render()
        st.image(cv2_img, channels='BGR')

    with col2:
        st.markdown('<div class="subtitle">📊 Resultados de Detección</div>', unsafe_allow_html=True)

        # Obtener nombres de etiquetas
        label_names = model.names
        # Contar categorías
        category_count = {}
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = []        
        # Imprimir recuento de categorías y etiquetas
        for category, count in category_count.items():
            label = label_names[int(category)]            
            data.append({"Categoría": label, "Cantidad": count})
        data_df = pd.DataFrame(data)
        
        # Agrupar los datos por categoría y sumar las cantidades
        df_sum = data_df.groupby('Categoría')['Cantidad'].sum().reset_index()

        # Mostrar los datos en un formato de tabla
        st.dataframe(df_sum)

# Fin del código con emojis y estilo mejorado para una experiencia más visual e intuitiva
