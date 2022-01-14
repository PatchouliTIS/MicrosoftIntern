import sys

def align(input_file_1, input_file_2, out_file):
    
    input_list_1 = []
    input_list_2 = []

    with open(input_file_1, "r", encoding="utf-8") as f:
        for line in f:
            row = line.split("\t")
            for item in row:
                input_list_1.append(float(item))

    with open(input_file_2, "r", encoding="utf-8") as f:
        for line in f:
            row = line.split("\t")
            for item in row:
                input_list_2.append(float(item))

    assert len(input_list_1) == len(input_list_2)
    diff_list = []
    for i in range(len(input_list_1)):
        diff_list.append(abs(input_list_1[i] - input_list_2[i]))

    max_diff = max(diff_list)
    avg_diff = sum(diff_list) / len(diff_list)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("Count: " + str(len(diff_list)/64) + "\n")
        f.write("Avg difference: " + str(avg_diff) + "\n")
        f.write("Max difference: " + str(max_diff) + "\n")

if __name__ == "__main__":
    align(sys.argv[1], sys.argv[2], sys.argv[3])