import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo preentrenado
model = yolov5.load('yolov5s.pt')

# Configuración de parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45   # Umbral de IoU NMS

# Estilos personalizados
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        color: #4B0082;
        text-align: center;
    }
    .sidebar-title {
        font-size: 24px;
        color: #2F4F4F;
    }
    .sidebar-slider {
        margin-bottom: 20px;
    }
    .data-table {
        border: 2px solid #4B0082;
        border-radius: 10px;
        padding: 10px;
        background-color: #f0f8ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la aplicación
st.markdown('<h1 class="title">VEO VEEEOOO...🧐</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="title">Puedo detectar los objetos de tus imagenes</h2>', unsafe_allow_html=True)

# Barra lateral para parámetros
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">Parámetros de Configuración</h2>', unsafe_allow_html=True)
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0, value=model.iou, key='iou', help='Ajusta el umbral de IoU para las detecciones')
    st.write('**IoU:**', model.iou)

    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0, value=model.conf, key='confidence', help='Ajusta el umbral de confianza para las detecciones')
    st.write('**Confianza:**', model.conf)

# Captura de imagen
picture = st.camera_input("Capturar foto", label_visibility='visible')

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Realiza la inferencia
    results = model(cv2_img)

    # Procesa los resultados
    predictions = results.pred[0]
    boxes = predictions[:, :4] 
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    col1, col2 = st.columns(2)

    with col1:
        # Mostrar la imagen con detecciones
        results.render()
        st.image(cv2_img, channels='BGR', caption='Imagen con Detecciones', use_column_width=True)

    with col2:
        # Obtener nombres de las etiquetas
        label_names = model.names
        category_count = {}

        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = []        
        # Imprimir conteos de categorías y etiquetas
        for category, count in category_count.items():
            label = label_names[int(category)]            
            data.append({"Categoría": label, "Cantidad": count})
        
        data2 = pd.DataFrame(data)

        # Agrupar los datos por categoría
        df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index()

        # Mostrar la tabla de resultados
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.write(df_sum)
        st.markdown('</div>', unsafe_allow_html=True)
