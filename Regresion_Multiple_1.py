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
from sklearn import linear_model

# leer el archivos csv con pandas
df_cars = pd.read_csv("cars.csv")

# Generamos una lista de Marcas de los vehiculos
condicionList = [
    (df_cars["Car"] == "Toyoty"),
    (df_cars["Car"] == "mitsubishi"),
    (df_cars["Car"] == "Skoda"),
    (df_cars["Car"] == "Fiat"),
    (df_cars["Car"] == "Mini"),
    (df_cars["Car"] == "VW"),
    (df_cars["Car"] == "Mercedes"),
    (df_cars["Car"] == "Ford"),
    (df_cars["Car"] == "Audi"),
    (df_cars["Car"] == "Hyundai"),
    (df_cars["Car"] == "Suzuki"),
    (df_cars["Car"] == "Honda"),
    (df_cars["Car"] == "Hundai"),
    (df_cars["Car"] == "Opel"),
    (df_cars["Car"] == "BMW"),
    (df_cars["Car"] == "Mazda"),
    (df_cars["Car"] == "Volvo"),

   
]
# clasificamos con valores numericos la diferente marca de los vehiculos en una lista
 
eleccionLista = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


df_cars['marcaCars'] = np.select(condicionList, eleccionLista, default= "not_specified")

# Creamos un nuevo dataframe
new_df = pd.DataFrame()

new_df["marca"] = df_cars["Car"].drop_duplicates()

new_df["marcaCars"] = eleccionLista

# Declaramos variable independientes 
x = df_cars[['Volume', 'Weight', 'CO2']]

# valor de la variable dependiente
y = df_cars["marcaCars"]

x = np.array(x)
y = np.array(y)

regr = linear_model.LinearRegression()
regr.fit(x,y)

# Inprimimos los coeficientes
print(regr.coef_)

# Regresion_ Multiple 

predicted_Car = regr.predict([[2000, 1746, 117]])

marcaCarro=int(np.round(predicted_Car,decimals = 0))

nombre = new_df[new_df["marcaCars"].isin([marcaCarro])]
print(df_cars)

# inprimimos el valor de la marca
print("la Marca Selecionada  es: ",nombre["marca"].values[0])
