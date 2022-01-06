import sys, os

def parse_one_feature(feature_lines):
    id = feature_lines[0].split(":")[1].split("]")[0]
    name_index = [i for i in range(4) if feature_lines[i+1].startswith("Name=")]
    type = "feature"
    if not name_index:
        name_index = [i for i in range(4) if feature_lines[i+1].startswith("Line1=")]
        type = "freeform"
    feature_name = feature_lines[name_index[0] + 1].strip().split("=", 1)[1]
    return f"{id}:{type}:{feature_name}"

def parse_ini(ini_file):
    with open(ini_file) as f:
        all_lines = f.readlines()

    head_index = [i for i in range(len(all_lines)) if all_lines[i].startswith("[Input:")]

    return [parse_one_feature(all_lines[i:i+5]) for i in head_index] #["1:feature1","2:feature2"]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python parse_smooth_ini.py <ini file> <filtered list file>")
    else:
        parsed_features = parse_ini(sys.argv[1], sys.argv[2])

        with open(sys.argv[2], "w") as lf:
            lf.writelines("\n".join(parsed_features))