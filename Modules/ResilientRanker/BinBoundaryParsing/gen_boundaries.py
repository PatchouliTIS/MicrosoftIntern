import sys, os
import parse_smooth_ini as ini_parser
import parse_bin_boundaries as bin_parser

def load_header(header_file):
    with open(header_file) as f:
        return [l.strip() for l in f.readlines()]

def gen_boundaries(raw_boundary, smoothed_ini, renamed_header):
    parsed_features = ini_parser.parse_ini(smoothed_ini)   # [feature id]:[feature or freeform]:[feature name]
    parsed_boundaries = bin_parser.parse_boundaries(raw_boundary) # [feature id]:[#bin]:[bins]
    target_features = load_header(renamed_header) # list of features

    feature_id_lookup = {f.split(":")[2].strip():int(f.split(":")[0].strip()) for f in parsed_features if f.split(":")[1] == "feature"}

    def get_feature_boundary(feature):
        if feature in feature_id_lookup:
            i = feature_id_lookup[feature]
            return parsed_boundaries[i].split(":")[2].strip()
        else:
            print(f"feature {feature} does not exist in smooth ini")
            return "1e20"  # set this value so that all features will be converted to 0
    
    return [get_feature_boundary(f) for f in target_features]


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("python gen_boundaries.py <raw boundary file> <smoothed ini> <renamed header> <generated boundary file>")
    else:
        bin_boundaries = gen_boundaries(sys.argv[1], sys.argv[2], sys.argv[3])

        # output bin boundaries as TSV file
        with open(sys.argv[4], "w") as f:
            f.writelines("\t".join(bin_boundaries))