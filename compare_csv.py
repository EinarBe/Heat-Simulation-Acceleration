import csv
import sys

def read_csv(file_path):
    """Reads a CSV file and returns its content as a list of lists with floats."""
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        return [[float(cell) for cell in row] for row in reader]

def compare_csv(file1, file2):
    """Compares two CSV files element by element."""
    data1 = read_csv(file1)
    data2 = read_csv(file2)
    
    if len(data1) != len(data2):
        print("Files have different number of rows!")
        return
    
    total_elements = 0
    wrong_elements = 0
    
    for i in range(len(data1)):
        if len(data1[i]) != len(data2[i]):
            print(f"Row {i} has different number of columns!")
            continue
        
        for j in range(len(data1[i])):
            total_elements += 1
            if abs(data1[i][j] - data2[i][j]) > 0.1:
                wrong_elements += 1
                print(f"Difference at row {i}, column {j}: {data1[i][j]} vs {data2[i][j]}")
    
    print(f"Total elements: {total_elements}, Wrong elements: {wrong_elements}")
    if total_elements > 0:
        print(f"Error rate: {wrong_elements / total_elements:.2%}")

compare_csv("heat_output.csv", "heat_output_cuda.csv")
