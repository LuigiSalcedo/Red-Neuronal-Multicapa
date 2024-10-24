package files

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadData() [][]float64 {
	file, err := os.Open("refactor_data_formated.csv")
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	data := make([][]float64, 0, 168)
	for lineData, err := reader.Read(); err == nil; lineData, err = reader.Read() {

		v1, err := strconv.ParseFloat(lineData[0], 64)
		if err != nil {
			panic(err)
		}
		v2, err := strconv.ParseFloat(lineData[1], 64)
		if err != nil {
			panic(err)
		}
		v3, err := strconv.ParseFloat(lineData[2], 64)
		if err != nil {
			panic(err)
		}
		data = append(data, []float64{v1, v2, v3})
	}
	return data
}
