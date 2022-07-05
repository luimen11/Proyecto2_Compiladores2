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
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier


#st.beta_set_page_config(page_title='Segundo Proyecto')
#======================================================================
# CARGAR ARCHIVO
#======================================================================
archivo = st.sidebar.file_uploader("Seleccione un archivo", type=['csv', 'xls', 'xlsx', 'json'])

if archivo is None:
     st.title('Compiladores 2')
     st.header('Segundo proyecto')
     st.subheader('Curso de vacaciones junio 2022')
#Se obtiene la extension del archivo
elif archivo:
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
     algoritmos = ["Seleccione...","Regresion lineal","Regresion polinomial", "Clasificador gaussiano", "Clasificador de arboles de decision", "Redes neuronales"]
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
          if nb_degree != 0:               
               polynomial_features = PolynomialFeatures(degree = nb_degree)

               # x_train_pol = polynomial_features.fit_transform(x_train)
               # x_test_pol = polynomial_features.fit_transform(x_test)

               x1 = polynomial_features.fit_transform(x)

               #Defino el algoritmo
               model = linear_model.LinearRegression()

               #Entreno el modelo
               #model.fit(x_train_pol, y_train)               
               model.fit(x1, y)               

               #Prediccion
               #y_pred = model.predict(x_test_pol)               
               y_pred = model.predict(x1)               

               st.subheader('Resultados:')

               #Graficamos los datos junto con el modelo
               fig1, ax1 = plt.subplots()
               #plt.scatter(x_test, y_test)               
               plt.scatter(x, y)               
               #plt.plot(x_test, y_pred, color='red', linewidth=3)               
               plt.plot(x, y_pred, color='red', linewidth=3)               
               st.pyplot(fig1)

               #Datos del modelo
               st.write('Valor de la pendiente o coeficiente a')
               st.write(model.coef_)

               st.write('Valor de la inteserccion o coeficiente b')
               st.write(model.intercept_)
               
               st.metric(label='Precision del modelo', value=(model.score(x1, y)))
               #st.metric(label='Precision del modelo', value=(model.score(x_train_pol, y_train)))
               #st.write(model.score(x_train_pol, y_train))
               
               #rmse = np.sqrt(mean_squared_error(y_test, y_pred))                                  
               rmse = np.sqrt(mean_squared_error(y, y_pred))                                  
               st.metric(label='Error cuadrático medio: ', value=rmse)
               
               #r2 = r2_score(y_test, y_pred)                         
               r2 = r2_score(y, y_pred)                         
               st.metric(label='Varianza: ', value=r2)
     
               #Prediccion
               st.subheader('Prediccion:')          
               pred2 = st.number_input('Ingrse el número de la predicción: ')
               pred2 = int(pred2)
               if pred2 != 0:
                    st.write(model.predict(polynomial_features.fit_transform([[pred2]])))
          
     #======================================================================
     # Clasificador gaussiano
     #======================================================================          
     elif opcion == 'Clasificador gaussiano':
          st.write(df) 
          le = preprocessing.LabelEncoder()
          encabezados = df.columns.values.tolist()
          dic_enc = {}
          print(encabezados)
          last = encabezados.pop()
          for a in encabezados:
               if(a!='#' and a!='NO' and a!='No.' and a!='NO.' and a!='Num'and a!=last):
                    temp = np.asarray(df[a])    
                    dic_enc[a] = le.fit_transform(temp)
          
          play = np.asarray(df[last])
          label = le.fit_transform(play)
          features = list(zip(*dic_enc.values()))
          #print(features)
          #st.write(features)
          st.table(features)
          model = GaussianNB()

          model.fit(features, label)

          valores = st.text_input("Ingrese los valores a predecir separados por comas:")

          if(valores != ''):
               valor_num = valores.split(',')
               lista = []
               for a in valor_num:
                    lista.append(int(a))

               #predicted = model.predict([[2,1,1,0]])          
               predicted = model.predict([lista])          
               st.metric(label='Prediccion ', value=predicted)

     #======================================================================
     # Clasificador de arboles de decision
     #======================================================================  
     elif opcion == "Clasificador de arboles de decision":
          st.write(df) 
          le = preprocessing.LabelEncoder()
          encabezados = df.columns.values.tolist()
          dic_enc = {}
          print(encabezados)
          last = encabezados.pop()
          for a in encabezados:
               if(a!='#' and a!='NO' and a!='No.' and a!='NO.' and a!='Num'and a!=last):
                    temp = np.asarray(df[a])    
                    dic_enc[a] = le.fit_transform(temp)
          
          play = np.asarray(df[last])
          label = le.fit_transform(play)

          features = list(zip(*dic_enc.values()))
          
          #st.write(features)
          st.table(features)
          
          clf = DecisionTreeClassifier()
          
          clf.fit(features, label)


          target = list(df[last].unique())
          fig1, ax1 = plt.subplots()
          plot_tree(clf, filled=True, class_names=target)
          #plt.show()
          st.pyplot(fig1)
              
          valores = st.text_input("Ingrese los valores a predecir separados por comas:")

          if(valores != ''):
               valor_num = valores.split(',')
               lista = []
               for a in valor_num:
                    lista.append(int(a))
               
               st.write(lista)
               predicted = clf.predict([lista])          
               st.metric(label='Prediccion ', value=predicted)

     #======================================================================
     # Redes neuronales
     #======================================================================  
     elif opcion == "Redes neuronales":
          st.write(df) 
          le = preprocessing.LabelEncoder()
          encabezados = df.columns.values.tolist()
          
          dic_enc = {}          
          last = encabezados.pop()

          for a in encabezados:
               if(a!='#' and a!='NO' and a!='No.' and a!='NO.' and a!='Num'and a!=last):
                    temp = np.asarray(df[a])    
                    dic_enc[a] = le.fit_transform(temp)
          
          play = np.asarray(df[last])
          label = le.fit_transform(play)
          features = list(zip(*dic_enc.values()))
                    
          st.table(features)

          #Entrenando el modelo
          x_train, x_test, y_train, y_test = train_test_split(features,label, test_size = 0.2)

          capas = st.text_input("Ingrese el tamaño de las capas separadas por comas:")

          if(capas != ''):
               valores = capas.split(',')
               lista = []
               for a in valores:
                    lista.append(int(a))
               tupla = tuple(lista)

               iter = st.text_input("Ingrese el numero de iteraciones:")

               if(iter != ''):
                    iter = int(iter)                                 
                                                  
                    mlp = MLPClassifier(hidden_layer_sizes=tupla, max_iter=iter)
                    mlp.fit(features, label)
                    prediction = mlp.predict(features)
                    st.write('Prediccion: ')
                    st.write(prediction)
                    #st.metric(label='Resultado de la prediccion ', value=prediction)


                    valores_pred = st.text_input("Ingrese los valores a predecir seprados por comas:")

                    if(valores_pred != ''):
                         v = valores_pred.split(',')
                         lista_pred = []
                         for a in v:
                              lista_pred.append(int(a))
                         
                         predicted = mlp.predict([lista_pred])          
                         #st.metric(label='Prediccion ', value=predicted)
                         st.write('Prediccion: ')
                         st.write(predicted)











     






