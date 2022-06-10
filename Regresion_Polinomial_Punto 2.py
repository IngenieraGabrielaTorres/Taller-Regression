#Taller 
#Gariela Torres
#ID:1001970935
#ID:502193
#correo:gabriela.torresr@upb.edu.co
#Cel:3234708201
#Diplomado de PYTHON APLICADO A LA INGENIERIA 
#Docente:Roberto Paez Salgado
#Modulo 2

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ----------------Regresion Polinomial --------------------
# leer el archivos csv con panda
datos = pd.read_excel("AirQualityUCI.xlsx")

# Declaramos las variable a evaluar de X  y Y
x = datos["NO2(GT)"]

y = datos["T"]

var_X = x[:8000]
var_Y = y[:8000]

pruebaX = x[8000:]
pruebaY = y[8000:]

# Modelo polinomial​

mymodel = np.poly1d(np.polyfit(var_X, var_Y, 3))

# Definimos el espaciamiento para la linea
myline = np.linspace(100, 1000, 8000)

# Graficamos la línea de regresión polinomial

plt.scatter(var_X,var_Y)
plt.plot(var_X, mymodel(myline))
plt.show()
r2 = r2_score(var_Y, mymodel(var_X))
print("El R cuadrado es:")
print(r2)