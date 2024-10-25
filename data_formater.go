package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func main() {
	file, _ := os.Open("input_data.csv")
	reader := csv.NewReader(file)
	output := make([][]string, 0, 169)
	for record, err := reader.Read(); err == nil; record, err = reader.Read() {
		value, _ := strconv.ParseFloat(record[2], 64)
		data := fmt.Sprintf("%.5f", (value / 3))

		output = append(output, []string{record[0], record[1], data})
	}
	file.Close()
	file, _ = os.Create("scaled_data.csv")
	writer := csv.NewWriter(file)
	writer.WriteAll(output)
	fmt.Println("Data scaled")
}
