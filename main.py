import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#======================================================================
# CARGAR ARCHIVO
#======================================================================
archivo = st.sidebar.file_uploader("Seleccione un archivo", type=['csv', 'xls', 'xlsx', 'json'])
#Se obtiene la extension del archivo
if archivo:
     archivo_split = os.path.splitext(archivo.name)
     nombre_archivo = archivo_split[0]
     ext_archivo = archivo_split[1]

     if(ext_archivo == '.csv'):
          df = pd.read_csv(archivo)
     elif(ext_archivo == '.json'):          
          df = pd.read_json(archivo)
          df.to_csv('temp.csv')
          df = pd.read_csv('temp.csv')
     elif(ext_archivo == '.xlsx' or ext_archivo == '.xls'):
          df = pd.read_excel(archivo)
          df.to_csv('temp.csv')
          df = pd.read_csv('temp.csv')
               
     st.sidebar.write('Nombre del archivo: ', nombre_archivo)
     st.sidebar.write('Archivo de tipo: ', ext_archivo)

     #======================================================================
     # SE ELIGE EL AGORITMO A UTILIZAR
     #======================================================================          
     algoritmos = ["Regresion lineal","Regresion polinomial", "Clasificador gaussiano"]
     opcion = st.selectbox("Seleccione algoritmo", algoritmos)
     encabezados = df.columns.values.tolist()

     #======================================================================
     # REGRESION LINEAL
     #======================================================================          
     if opcion == 'Regresion lineal':
          st.write(df)                    
          opcionX = st.selectbox('Escoja un atributo para X: ', encabezados)
          opcionY = st.selectbox('Escoja un atributo para Y: ', encabezados)

          x = np.asarray(df[opcionX]).reshape(-1,1)
          y = np.asarray(df[opcionY]).reshape(-1,1)

          regr = LinearRegression()
          regr.fit(x,y)
          y_pred = regr.predict(x)
          param = st.number_input('Ingrese el parametro a predecir')   

          prediccion = regr.predict([[param]])                                      
          st.metric(label='Prediccion: ', value=prediccion)

          coef = regr.coef_
          st.metric(label='Coeficiente: ', value=coef)

          err = mean_squared_error(y, y_pred)
          st.metric(label='Error cuadrático medio: ', value=err)

          varianza = r2_score(y, y_pred)
          st.metric(label='Varianza: ', value=varianza)

          fig, ax = plt.subplots()
          plt.scatter(x,y, color='black')
          plt.plot(x,y_pred, color = 'blue', linewidth=3)
          plt.ylim(min(y),max(y))
          st.pyplot(fig)

     #======================================================================
     # REGRESION POLINOMIAL
     #======================================================================          
     elif opcion == 'Regresion polinomial':
          st.write(df)                    
          opcionX = st.selectbox('Escoja un atributo para X: ', encabezados)
          opcionY = st.selectbox('Escoja un atributo para Y: ', encabezados)

          x = np.asarray(df[opcionX]).reshape(-1,1)
          y = np.asarray(df[opcionY]).reshape(-1,1)

          fig, ax = plt.subplots()
          plt.scatter(x,y)
          st.pyplot(fig)

          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

          nb_degree = st.number_input('Ingrese el grado: ')   
          nb_degree = int(nb_degree)
          polynomial_features = PolynomialFeatures(degree = nb_degree)

          x_train_pol = polynomial_features.fit_transform(x_train)
          x_test_pol = polynomial_features.fit_transform(x_test)

          #Defino el algoritmo
          model = linear_model.LinearRegression()

          #Entreno el modelo
          model.fit(x_train_pol, y_train)

          #Prediccion
          y_pred = model.predict(x_test_pol)

          st.subheader('Resultados:')

          #Graficamos los datos junto con el modelo
          fig1, ax1 = plt.subplots()
          plt.scatter(x_test, y_test)
          plt.plot(x_test, y_pred, color='red', linewidth=3)
          st.pyplot(fig1)

          #Datos del modelo
          st.write('Valor de la pendiente o coeficiente a')
          st.write(model.coef_)

          st.write('Valor de la inteserccion o coeficiente b')
          st.write(model.intercept_)
          
          st.metric(label='Precision del modelo', value=(model.score(x_train_pol, y_train)))
          #st.write(model.score(x_train_pol, y_train))
           
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))                   
          st.metric(label='Error cuadrático medio: ', value=rmse)
          
          r2 = r2_score(y_test, y_pred)          
          st.metric(label='Varianza: ', value=r2)
     
          #Prediccion
          st.subheader('Prediccion:')          
          pred2 = st.number_input('Ingrse el número de la predicción: ')

          st.write(model.predict(polynomial_features.fit_transform([[int(pred2)]])))
          
     #======================================================================
     # REGRESION POLINOMIAL
     #======================================================================          
     #elif opcion == 'Clasificador gaussiano':







     






