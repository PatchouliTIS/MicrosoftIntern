import sys

def seriaze_desc_schema(desc):
    """ append 12 spaces in front, and add comma in end
    """
    return ",\n".join([" " * 12 + d for d in desc])

def seriaze_desc_extrac(desc):
    """ append 12 spaces in front except the first item, and add comma in end
    """
    return ",\n".join([" " * 12 + d if i > 0 else d for i, d in enumerate(desc)])

def desc_to_view(desc):
    view_lines = [
        "CREATE VIEW SearchLog SCHEMA ( ",
        seriaze_desc_schema(desc),
        ")",
        "PARAMS (",
        "	inputdata string,",
        "	arg string",
        ")",
        "AS BEGIN",
        "searchlog = ",
        f"    EXTRACT {seriaze_desc_extrac(desc)}",
        "    FROM @inputdata",
        "    USING DefaultTextExtractor(@arg);",
        "END;",
    ]
    return "\n".join(view_lines)

def generate_view(meta_list, num_features, target_file, target_column, target_type):
    metas = meta_list.split(",") if len(meta_list) > 0 else []
    meta_desc = [f"{m}: string" for m in metas]
    feature_desc = [f"{target_column}{i+1}: {target_type}" for i in range(num_features)]
    desc = meta_desc + feature_desc
    with open(target_file, "w") as f:
        f.writelines(desc_to_view(desc))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python generate_view.py <meta list> <number of features> <target file>  [<target column>] [<target type>]")
    else:
        generate_view(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "feature", sys.argv[5] if len(sys.argv) > 5 else "long")