package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"rn-multi/files"
	"slices"
)

type RedInput struct {
	X1 float64
	X2 float64
	Y  float64
}

type RedOutput struct {
	RedInput
	Output float64
}

// Función para inicializar pesos y sesgos aleatoriamente en el rango [-1, 1]
func initRandomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = -1 + rand.Float64()*2
		}
	}
	return matrix
}

func initRandomVector(size int) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()
	}
	return vector
}

// Función de activación Sigmoide
func logsig(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función Sigmoide
func dLogsig(x float64) float64 {
	return logsig(x) * (1 - logsig(x))
}

// Función de activación tangente hiperbólica
func tansig(x float64) float64 {
	return (2.0 / (1.0 + math.Exp(-2*x))) - 1.0
}

// Derivada de la función tangente hiperbólica
func dTansig(x float64) float64 {
	return 1 - math.Pow(tansig(x), 2)
}

func main() {
	file, err := os.Create("output.txt")
	if err != nil {
		panic(err)
	}
	outputs := make(map[RedInput]float64)
	writer := csv.NewWriter(file)
	writer.Write([]string{"DÍA DE LA SEMANA", "HORA DEL DÍA", "CONSUMO (1 ALTO -1 BAJO)", "SALIDA DE LA RED"})
	// Datos de entrada y configuración de la red
	dato := files.LoadData()
	red := []int{2, 16, 1} // Capas de la red: 1 capa oculta con 3 neuronas

	f := len(dato) // Filas de dato
	tol := 1e-5    // Tolerancia
	n := 0.8       // Coeficiente de entrenamiento

	// Inicialización de pesos y sesgos
	w := make([][][]float64, len(red)-1)
	b := make([][]float64, len(red)-1)
	delta := make([][]float64, len(red)-1)

	for i := 0; i < len(red)-1; i++ {
		w[i] = initRandomMatrix(red[i+1], red[i])
		b[i] = initRandomVector(red[i+1])
		delta[i] = make([]float64, red[i+1])
	}

	epocas := 0
	promeg := 99.0
	salidas := [][]float64{}
	errorG := []float64{99}

	// Bucle de entrenamiento
	for promeg >= tol && epocas <= 10000 {
		for p := 0; p < f; p++ {
			Y := make([][]float64, len(red)-1)

			// Propagación hacia adelante
			for i := 0; i < len(red)-1; i++ {
				var a []float64
				if i == 0 {
					a = make([]float64, red[i+1])
					for j := 0; j < red[i+1]; j++ {
						sum := b[i][j]
						for k := 0; k < red[i]; k++ {
							sum += w[i][j][k] * dato[p][k]
						}
						a[j] = logsig(sum)
					}
				} else {
					a = make([]float64, red[i+1])
					for j := 0; j < red[i+1]; j++ {
						sum := b[i][j]
						for k := 0; k < red[i]; k++ {
							sum += w[i][j][k] * Y[i-1][k]
						}
						a[j] = tansig(sum)
					}
				}
				Y[i] = a
			}
			// fmt.Println("Patrón:", dato[p], "Sálida:", Y[len(Y)-1])
			// writer.Write([]string{fmt.Sprintf("%f", dato[p][0]), fmt.Sprintf("%f", dato[p][1]), fmt.Sprintf("%f", dato[p][2]), fmt.Sprintf("%f", Y[len(Y)-1][0])})
			// Guardar salida
			outputs[RedInput{X1: dato[p][0], X2: dato[p][1], Y: dato[p][2]}] = Y[len(Y)-1][0]
			salidas = append(salidas, Y[len(Y)-1])

			// Calcular error
			e := make([]float64, red[len(red)-1])
			for j := 0; j < len(e); j++ {
				e[j] = dato[p][j+2] - Y[len(Y)-1][j]
			}
			delta[len(delta)-1] = e

			// Propagación del error hacia atrás
			for k := len(delta) - 2; k >= 0; k-- {
				for j := 0; j < len(delta[k]); j++ {
					sum := 0.0
					for m := 0; m < len(delta[k+1]); m++ {
						sum += w[k+1][m][j] * delta[k+1][m]
					}
					delta[k][j] = sum
				}
			}

			// Actualización de sesgos y pesos
			for i := 0; i < len(b); i++ {
				for j := 0; j < len(b[i]); j++ {
					b[i][j] += n * delta[i][j]
				}
			}

			for i := 0; i < len(w); i++ {
				for j := 0; j < len(w[i]); j++ {
					for k := 0; k < len(w[i][j]); k++ {
						if i == 0 {
							w[i][j][k] += n * delta[i][j] * dLogsig(Y[i][j]) * dato[p][k]
						} else {
							w[i][j][k] += n * delta[i][j] * dTansig(Y[i][j]) * Y[i-1][k]
						}
					}
				}
			}
		}

		// Calcular error cuadrático medio
		err := 0.0
		for _, e := range delta {
			for _, v := range e {
				err += v * v
			}
		}
		errorG = append(errorG, err/2.0)
		promeg = errorG[len(errorG)-1] / float64(len(delta))

		epocas++
	}
	sorted_outputs := make([]RedOutput, 0, len(outputs))
	for k, v := range outputs {
		sorted_outputs = append(sorted_outputs, RedOutput{RedInput: k, Output: v})
	}

	slices.SortFunc(sorted_outputs, func(a, b RedOutput) int {
		if a.X1 < b.X1 {
			return -1
		}

		if a.X1 > b.X1 {
			return 1
		}

		if a.X2 < b.X2 {
			return -1
		}

		if a.X2 > b.X2 {
			return 1
		}

		return 0
	})
	for o := range sorted_outputs {
		writer.Write([]string{fmt.Sprintf("%f", sorted_outputs[o].X1), fmt.Sprintf("%f", sorted_outputs[o].X2), fmt.Sprintf("%f", sorted_outputs[o].Y), fmt.Sprintf("%f", sorted_outputs[o].Output)})
	}
	// Mostrar resultados
	fmt.Println("Epocas:", epocas)
	fmt.Println("Error global:", errorG[len(errorG)-1])
	fmt.Println("Promedio error global:", promeg)
	err_totales_finales := 0
	for _, o := range sorted_outputs {
		if o.Y > 0 && o.Output < 0 {
			fmt.Println("Error en:", o.X1, o.X2, o.Y, o.Output)
			err_totales_finales++
		}
	}
	fmt.Println("Total de salida:", len(sorted_outputs))
	fmt.Println("Errores totales finales:", err_totales_finales)
	// fmt.Println("Salidas:", salidas)
	writer.Flush()
}
