def calculate_column_means(file_name):
    column_1_sum = 0
    column_2_sum = 0
    total_rows = 0

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                elements = line.split()
                if len(elements) == 2:
                    try:
                        column_1_sum += abs(float(elements[0]))
                        column_2_sum += abs(float(elements[1]))
                        total_rows += 1
                    except ValueError:
                        print(f"Invalid value in line: {line}. Skipping...")

        if total_rows > 0:
            column_1_mean = column_1_sum / total_rows
            column_2_mean = column_2_sum / total_rows
            return column_1_mean, column_2_mean
        else:
            return None, None

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None, None


column_1_mean, column_2_mean = calculate_column_means("output.txt")

if column_1_mean is not None and column_2_mean is not None:
    print(f"Mean of column 1: {column_1_mean}")
    print(f"Mean of column 2: {column_2_mean}")
else:
    print("Unable to calculate the means. Check if the file exists and contains valid data.")