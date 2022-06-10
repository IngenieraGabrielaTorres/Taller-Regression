#Taller 
#Gariela Torres
#ID:1001970935
#ID:502193
#correo:gabriela.torresr@upb.edu.co
#Cel:3234708201
#Diplomado de PYTHON APLICADO A LA INGENIERIA 
#Docente:Roberto Paez Salgado
#Modulo 2

# Importamos la libreria necesarias 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# leer el archivos csv con panda
datos = pd.read_excel("AirQualityUCI.xlsx")

# Declaramos las variable a evaluar de X  y Y
x = datos["NO2(GT)"]

y = datos["T"]

x,y = np.array(x).reshape(-1,1), np.array(y)

var_X = x[:8000]
var_Y = y[:8000]

pruebaX = x[8000:]
pruebaY = y[8000:]

# Regresion 
  
model = LinearRegression().fit(var_X,var_Y)

#  R-cuadrado
r_sq_train = model.score(var_X,var_Y)

r_sq_test = model.score(pruebaX,pruebaY)

# Predicion valores a Futuro
y_predict = model.predict(pruebaX)

# Grafica de dispersion 
plt.scatter(pruebaX,pruebaY)
plt.plot(pruebaX, y_predict)
plt.show()

# Imprimir
print("El R es:", r_sq_train)
print("Predicion", r_sq_test)
print("")
