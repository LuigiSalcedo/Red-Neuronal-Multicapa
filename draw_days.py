import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Configura el analizador de argumentos
parser = argparse.ArgumentParser(description="Graficar datos de consumo desde múltiples archivos CSV.")
parser.add_argument('files', type=str, nargs='+', help="Rutas a los archivos CSV")
args = parser.parse_args()

# Lee y grafica cada archivo
for file in args.files:
    # Lee el archivo CSV
    data = pd.read_csv(file, header=None)  # Sin encabezados

    # Asegúrate de que hay suficientes columnas
    if data.shape[1] < 3:
        print(f"El archivo {file} no tiene suficientes columnas.")
        continue

    # Asumimos que las columnas son: 'Horas del día', 'Consumo Esperado', 'Consumo dado por la Red'
    x = data[0]  # Primera columna (Horas del día)
    y1 = data[1]  # Segunda columna (Consumo Esperado)
    y2 = data[2]  # Tercera columna (Consumo dado por la Red)

    # Verifica las dimensiones de x e y
    print(f"Archivo: {file} | Longitud de x: {len(x)} | Longitud de y1: {len(y1)} | Longitud de y2: {len(y2)}")

    # Dibuja los datos en una gráfica
    plt.figure()  # Crea una nueva figura
    plt.plot(x, y1, linestyle='-', color='blue', label='Consumo Esperado')
    plt.plot(x, y2, linestyle='-', color='orange', label='Consumo dado por la Red')

    # Configura la gráfica
    plt.xlabel('Horas del día')
    plt.ylabel('Consumo')
    plt.title(f'Datos de Consumo de {os.path.basename(file)}')
    plt.legend()  # Muestra la leyenda
    plt.grid(True)
    plt.show()  # Muestra la gráfica actual
