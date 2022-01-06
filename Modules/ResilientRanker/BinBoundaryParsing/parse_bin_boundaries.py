import sys, os

def parse_one_boundary(boudary_lines):
    feature = int(boudary_lines[0].strip()[14:])
    if feature == 0:
        return "0::"
    else:
        num_bin = int(boudary_lines[1].split(",", 2)[1].split(":")[1].strip())
        boundaries = "".join([l.strip() for l in boudary_lines[2:]])  # join all lines for boundaries
        boundaries = boundaries.replace("inf,", "1e20")  # replace inf with 1e20 which is greater than the max value for long type
        boundaries = boundaries.replace(" ", "") # remove spaces
        return f"{feature}:{num_bin}:{boundaries}"

def parse_boundaries(input_file):
    with open(input_file) as f:
        all_lines = f.readlines()

    head_index = [i for i in range(len(all_lines)) if all_lines[i].startswith("----- feature")]
    tail_index = head_index[1:]
    last_line = len(all_lines) - 1
    while last_line >= 0 and (not all_lines[last_line].startswith("=========================")):
        last_line = last_line - 1
    if last_line > 0:
        tail_index.append(last_line)
    else:
        raise Exception("does not found last line which is expected to be ====================================")

    return [parse_one_boundary(all_lines[head_index[i]:tail_index[i]]) for i in range(len(head_index))]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python parse_bin_boundaries.py <raw boundary file> <parsed boundary file>")
    else:
        parsed_boundaries = parse_boundaries(sys.argv[1])

        with open(sys.argv[2], "w") as f:
            f.writelines("\n".join(parsed_boundaries))