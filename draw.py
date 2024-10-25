import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

# Configura el analizador de argumentos
parser = argparse.ArgumentParser(description="Graficar datos desde múltiples archivos CSV.")
parser.add_argument('files', type=str, nargs='+', help="Rutas a los archivos CSV")
args = parser.parse_args()

# Lista de colores para las líneas
colors = plt.cm.viridis(np.linspace(0, 1, len(args.files)))  # Crea una lista de colores

# Lee y grafica cada archivo
for i, file in enumerate(args.files):
    # Lee el archivo CSV sin encabezado
    data = pd.read_csv(file, header=None)

    # Asegúrate de que hay suficientes filas en el archivo
    if data.shape[0] < 2:
        print(f"El archivo {file} no tiene suficientes datos.")
        continue

    # Asumimos que la primera columna es x y la segunda columna es y
    x = data.iloc[:, 0].astype(float)  # Primera columna como eje x
    y = data.iloc[:, 1].astype(float)  # Segunda columna como eje y

    # Verifica las dimensiones de x e y
    print(f"Archivo: {file} | Longitud de x: {len(x)} | Longitud de y: {len(y)}")

    # Dibuja los datos en una gráfica con una línea continua sin puntos
    plt.figure()  # Crea una nueva figura
    plt.plot(x, y, linestyle='-', color=colors[i], label=os.path.basename(file))

    # Configura la gráfica
    plt.xlabel('Generaciones')
    plt.ylabel('Error global')
    plt.title(f'Datos de {os.path.basename(file)}')
    plt.legend()  # Muestra la leyenda
    plt.grid(True)
    plt.show()  # Muestra la gráfica actual

# Crea una gráfica de fusión después de mostrar cada gráfica individual
plt.figure()  # Nueva figura para la fusión
for i, file in enumerate(args.files):
    data = pd.read_csv(file, header=None)
    x = data.iloc[:, 0].astype(float)
    y = data.iloc[:, 1].astype(float)
    plt.plot(x, y, linestyle='-', color=colors[i], label=os.path.basename(file))

# Configura la gráfica de fusión
plt.xlabel('Generaciones')
plt.ylabel('Error global')
plt.title('Comparación de redes neuronales')
plt.legend()  # Muestra la leyenda
plt.grid(True)
plt.show()  # Muestra la gráfica de fusión
