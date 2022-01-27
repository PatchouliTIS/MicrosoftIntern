import json
import sys


def cal_hamming_distance(input_x, input_y):
    if (len(input_x) != len(input_x)):
        return(0, 0)

    dist = 0
    for i in range(len(input_x)):
        if int(input_x[i]) != int(input_y[i]):
            dist += 1

    return (dist, len(input_x))


def cal_intersaction(input_x, input_y):
    if (len(input_x) != len(input_x)):
        return(0, 0)

    input_x_set = set()
    for x in input_x:
        input_x_set.add(int(x))

    input_y_set = set()
    for y in input_y:
        input_y_set.add(int(y))

    return (len(input_x_set.intersection(input_y_set)), len(input_x))


def process(deep_extract_results_path, deep_extract_results_on_bw_path, output_path):

    result_file = open(deep_extract_results_path, "r", encoding="utf-8")
    result_bw_file = open(deep_extract_results_on_bw_path,
                          "r", encoding="utf-8")

    result_raw_lines = result_file.readlines()
    result_bw_raw_lines = result_bw_file.readlines()

    with open(output_path, "w", encoding="utf8") as output:
        output.write(f"Total Query Count: {len(result_raw_lines)}\n")

        tops = [10, 5, 3, 1]

        for idx in range(2):
            dist_tops = []
            total_dist_tops = []

            intersaction_tops = []
            total_intersaction_tops = []

            for _ in tops:
                dist_tops.append(0)
                total_dist_tops.append(0)
                intersaction_tops.append(0)
                total_intersaction_tops.append(0)

            attribute_for_result = "index_" + str(idx + 1)
            attribute_for_result_on_bw = "scores_" + str(idx + 1) + "_indices"

            for i in range(len(result_raw_lines)):
                input_x = json.loads(result_raw_lines[i])[attribute_for_result]
                input_y = json.loads(result_bw_raw_lines[i])[
                    attribute_for_result_on_bw][0]

                for index, num in enumerate(tops):
                    (temp_dist, temp_total) = cal_hamming_distance(
                        input_x[:num], input_y[:num])
                    dist_tops[index] += temp_dist
                    total_dist_tops[index] += temp_total

                    (temp_dist, temp_total) = cal_intersaction(
                        input_x[:num], input_y[:num])
                    intersaction_tops[index] += temp_dist
                    total_intersaction_tops[index] += temp_total

            for index, num in enumerate(tops):
                output.write(
                    f"The Hamming Distance of Top {num} Results for Index {idx + 1} is: {dist_tops[index]}/{total_dist_tops[index]}\n")

            for index, num in enumerate(tops):
                output.write(
                    f"The Intersaction of Top {num} Results for Index {idx + 1} is: {intersaction_tops[index]}/{total_intersaction_tops[index]}\n")


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2], sys.argv[3])
