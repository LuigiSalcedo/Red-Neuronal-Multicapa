package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"rn-multi/files"
	"slices"

	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type RedInput struct {
	X1 float64
	X2 float64
	Y  float64
}

type BlindInput struct {
	X1 int
	X2 int
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
	GenRed([]int{2, 12, 8, 1}, "RedInicial")
	fmt.Println("Red Inicial terminada")
	GenRed([]int{2, 10, 7, 6, 1}, "RedPropuesta")
	fmt.Println("Red Propuesta terminada")
}

func GenRed(inputRed []int, name string) {
	file, _ := os.Create(name + "_output_data.csv")
	outputWriter := csv.NewWriter(file)
	plot := plot.New()
	plot.X.Label.Text = "Generaciones"
	plot.Y.Label.Text = "Errores"
	plot.X.Min = 0
	plot.Y.Min = 0
	plot.Y.Max = 0
	points := make(plotter.XYs, 0, 10)
	file, err := os.Create(name + "_output.txt")
	if err != nil {
		panic(err)
	}
	outputs := make(map[RedInput]float64)
	outputByDay := make(map[BlindInput]RedOutput)
	days := map[int]string{
		1: "Lunes",
		2: "Martes",
		3: "Miércoles",
		4: "Jueves",
		5: "Viernes",
		6: "Sábado",
		7: "Domingo",
	}
	writer := csv.NewWriter(file)
	writer.Write([]string{"DÍA DE LA SEMANA", "HORA DEL DÍA", "CONSUMO", "SALIDA DE LA RED"})
	// Datos de entrada y configuración de la red
	dato := files.LoadData()
	red := inputRed // Capas de la red: 1 capa oculta con 3 neuronas

	f := len(dato) // Filas de dato
	tol := 1e-8    // Tolerancia
	n := 0.15      // Coeficiente de entrenamiento

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
	// salidas := [][]float64{}
	errorG := []float64{0.1}

	// Bucle de entrenamiento
	for promeg >= tol && epocas <= 8000 {
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
			outputByDay[BlindInput{X1: int(dato[p][0]), X2: int(dato[p][1])}] = RedOutput{RedInput: RedInput{X1: dato[p][0], X2: dato[p][1], Y: dato[p][2]}, Output: Y[len(Y)-1][0]}
			// salidas = append(salidas, Y[len(Y)-1])

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
		if epocas%2 == 0 {
			points = append(points, plotter.XY{X: float64(epocas), Y: promeg})
			outputWriter.Write([]string{fmt.Sprintf("%d", epocas), fmt.Sprintf("%.5f", errorG[len(errorG)-1])})
		}
		// fmt.Println(promeg)
		errorG = append(errorG, err/2.0)
		promeg = errorG[len(errorG)-1] / float64(len(delta))
		if promeg > plot.Y.Max {
			plot.Y.Max = promeg
		}
		// a := ""
		epocas++
		plot.X.Max = float64(epocas)
		// fmt.Scanf("%s", &a)
	}
	line, err := plotter.NewLine(points)
	if err != nil {
		panic(err)
	}
	line.Color = color.RGBA{R: 255, A: 0, B: 0}
	plot.Add(line)

	err = plot.Save(4*vg.Inch, 4*vg.Inch, "Resultados errores.png")
	if err != nil {
		panic(err)
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
		if o.Y-o.Output > 0.25 {
			fmt.Println("Error en:", o.X1, o.X2, o.Y, o.Output)
			err_totales_finales++
		}
	}
	fmt.Println("Total de salida:", len(sorted_outputs))
	fmt.Println("Errores totales finales:", err_totales_finales)
	// fmt.Println("Salidas:", salidas)
	for i := 1; i <= 7; i++ {
		file, _ := os.Create(fmt.Sprintf("%s_"+name+".csv", days[i]))
		byDayWriter := csv.NewWriter(file)
		for j := 1; j <= 24; j++ {
			byDayWriter.Write([]string{fmt.Sprintf("%d", j), fmt.Sprintf("%.5f", outputByDay[BlindInput{X1: i, X2: j}].Y), fmt.Sprintf("%.5f", outputByDay[BlindInput{X1: i, X2: j}].Output)})
		}
		byDayWriter.Flush()
	}
	writer.Flush()
	outputWriter.Flush()
}
