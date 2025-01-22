import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.ensemble import RandomForestClassifier
import pathlib
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

theme_plotly = None

st.set_page_config(page_title="Prediccion-Ventas", page_icon="img/logo2.png", layout="wide")

#""" imagen de background"""
def add_local_background_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stApp{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )
add_local_background_image("img/fondo.jpg")

#""" imagen de sidebar"""
def add_local_sidebar_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stSidebar{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )

add_local_sidebar_image("img/fondo1.jpg")

#-------------- animacion con css de los botones de footer ------------------------

with open('asset/styles.css') as f:
        css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)



st.button("Predicción venta de marcas de Vehiculo x Sucursal" ,key="pulse")    
#st.header('Predicción venta de marcas de Vehiculo x Sucursal')
st.write("---") #st.write("##")
st.button("con modelo de Machine Learning", key="inpulse")
st.write("---")
st.info('App creada para aplicar modelos de machine learning')

# Set up input widgets
st.logo(image="img/logo3.png",
      size='large')

with st.expander('Datos'):
  st.write('detalle de datos')
  df = pd.read_excel('data/data.xlsx')
  df
# filtrando por x
  #st.write(' - X - ')
X_raw = df.drop('marca', axis=1)
#X_raw
# filtrando por y
#st.write(' - y - ')
y_raw = df['marca']
#y_raw

with st.expander('Visualizacion de datos'):
  chart_data = pd.DataFrame(df, columns=['auto', 'suv', 'camioneta'])
  st.line_chart(chart_data)

  
# Input features / caracteristicas de entrada
with st.sidebar:
  st.button('Seleccione datos a filtrar', key="topulse")
  sucursal = st.selectbox('Sucursal', ('NORTE', 'SUR', 'ESTE', 'OESTE'))
  auto = st.slider('auto', 0, 50)
  suv = st.slider('suv', 0, 50)
  camioneta = st.slider('camioneta', 0, 50)
  vta = st.selectbox('forma de venta', ('CONTADO', 'FINANCIADO'))
  
  # Create a DataFrame for the input features / se crea un DataFrame para las caracteristica de entrada
  
  data = {'sucursal': sucursal,
          'auto': auto,
          'suv': suv,
          'camioneta': camioneta,
          'forma-vta': vta}
  input_df = pd.DataFrame(data, index=[0])
  input_marca = pd.concat([input_df, X_raw], axis=0)

# se muestra por pantalla los datos seleccinados en la barra lateral
with st.expander('Datos seleccionado en barra lateral x sucursal'):
  st.write('Sucursal elegiada y datos')
  input_df
  st.write('datos combinados de Sucursal')
  input_marca


# Data preparation / preparacion de datos
# Encode X - codificando x 
encode = ['sucursal', 'forma-vta']
df_marca = pd.get_dummies(input_marca, prefix=encode)



X = df_marca[1:]
input_row = df_marca[:1]

# Encode y - codificando y
target_mapper = {'TOYOTA': 0,
                 'FORD': 1,
                 'RENAULT': 2,
                 'VOLKSWAGEN': 3,
                 'FIAT': 4,
                }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

# Model training and inference / modelo de entrenamiento
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions / se aplica el modelo de prediccion 
prediction = clf.predict(input_row)
pred_probabilidad = clf.predict_proba(input_row)
pred_probabilidad = pred_probabilidad*100

df_pred_probabilidad = pd.DataFrame(pred_probabilidad)

df_pred_probabilidad.columns = ['TOYOTA',
                 'FORD',
                 'RENAULT',
                 'VOLKSWAGEN',
                 'FIAT']
df_pred_probabilidad.rename(columns={0: 'TOYOTA',
                                 1: 'FORD',
                                 2: 'RENUALT',
                                 3: 'VOLKSWAGEN',
                                 4: 'FIAT'                                   
                                 })

# Display predicted marca / muestra la prediccion de la marca
st.write("---")
st.subheader('Predicción de Probabilidad por Marca (%)')
st.dataframe(df_pred_probabilidad,
             column_config={
               'TOYOTA': st.column_config.ProgressColumn(
                 'TOYOTA',
                 format= '%.2f',
                 width='medium',
                 min_value=0,
                 max_value=100
               ),
               'FORD': st.column_config.ProgressColumn(
                 'FORD',
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=100
               ),
               'RENAULT': st.column_config.ProgressColumn(
                 'RENAULT',
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=100
               ),
               'VOLKSWAGEN': st.column_config.ProgressColumn(
                 'VOLKSWAGEN',
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=100
               ),
                'FIAT': st.column_config.ProgressColumn(
                 'FIAT',
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=100
               ),
             }, hide_index=True)


marca_df = np.array(['TOYOTA', 'FORD', 'RENAULT','VOLKSWAGEN', 'FIAT'])
container = st.container (border=True)
container.success(str(marca_df[prediction][0]))


# DataFrame display- -mostrar los datos basicos de los  calculos
with st.expander('Ver datos (Dataframe)-descagar en formato csv'):
    #st.dataframe(df_filtered)
    st.dataframe(df)
    st.dataframe(df_pred_probabilidad)



  #st.write("---")
st.write("&copy; - derechos reservados -  2024 -  Walter Gómez - FullStack Developer - Data Science - Business Intelligence")
  #st.write("##")
left, right = st.columns(2, gap='medium', vertical_alignment="center")
with left:
      url="https://www.linkedin.com/in/walter-gomez-fullstack-developer-datascience-businessintelligence-finanzas-python/"            
      st.link_button("Mi LinkedIn", url, use_container_width= True)
with right: 
      url1= "https://walter-portfolio-animado.netlify.app/"      
      st.link_button("Mi Portfolio", url1, use_container_width= True)
    
