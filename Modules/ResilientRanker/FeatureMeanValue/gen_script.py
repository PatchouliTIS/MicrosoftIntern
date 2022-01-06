import sys, os

def gen_features_statement(meta_list):
    return "\n".join([
            "features = ",
            f"SELECT *.Except({meta_list});"])

# we assume num_features > 1
def gen_sums_statement(num_features):
    lines = [
        "sums = ",
        "SELECT SUM(feature1) AS sum1,"]
    lines.extend([f"       SUM(feature{i + 2}) AS sum{i + 2}{',' if i < num_features - 2 else ''}" for i in range(num_features - 1)])
    lines.append("FROM features;")
    return "\n".join(lines)

# we assume num_features > 1
def gen_cnts_statement(num_features):
    lines = [
        "cnts = ",
        "SELECT COUNTIF(feature1 > 0) AS cnt1,"]
    lines.extend([f"       COUNTIF(feature{i + 2} > 0) AS cnt{i + 2}{',' if i < num_features - 2 else ''}" for i in range(num_features - 1)])
    lines.append("FROM features;")
    return "\n".join(lines)

# we assume num_features > 1
def gen_mean_statement(num_features):
    lines = [
        "mean = ",
        "SELECT Utils.Divide(sums.sum1, cnts.cnt1) AS avg1,"]
    lines.extend([f"       Utils.Divide(sums.sum{i + 2}, cnts.cnt{i + 2}) AS avg{i + 2}{',' if i < num_features - 2 else ''}" for i in range(num_features - 1)])
    lines.extend([
        "FROM sums CROSS JOIN cnts;",
        "OUTPUT TO @@Output_avg@@;"
    ])
    return "\n".join(lines)

def gen_sqr1_statement(num_features):
    lines = ["SELECT Utils.Square(features.feature1, mean.avg1) AS sqr1,"]
    lines.extend([f"       Utils.Square(features.feature{i + 2}, mean.avg{i + 2}) AS sqr{i + 2}{',' if i < num_features - 2 else ''}" for i in range(num_features - 1)])
    lines.append("FROM features CROSS JOIN mean;")
    return "\n".join(lines)

def gen_sqr2_statement(num_features):
    lines = [
        "sum_sqr = ",
        "SELECT SUM(sqr1) AS sum1,"
    ]
    lines.extend([f"       SUM(sqr{i + 2}) AS sum{i + 2}{',' if i < num_features - 2 else ';'}" for i in range(num_features - 1)])
    return "\n".join(lines)

def gen_sart_statement(num_features):
    lines = ["SELECT Utils.DivideMinus1Root(sum_sqr.sum1, cnts.cnt1) AS sqrt1,"]
    lines.extend([f"       Utils.DivideMinus1Root(sum_sqr.sum{i + 2}, cnts.cnt{i + 2}) AS sqrt{i + 2}{',' if i < num_features - 2 else ''}" for i in range(num_features - 1)])
    lines.extend([
        "FROM sum_sqr CROSS JOIN cnts;",
        "OUTPUT TO @@Output_sqrt@@;"
    ])
    return "\n".join(lines)

def gen_script(meta_list, num_features, output_script):
    all_statements = [
        gen_features_statement(meta_list),
        gen_sums_statement(num_features),
        gen_cnts_statement(num_features),
        gen_mean_statement(num_features),
        gen_sqr1_statement(num_features),
        gen_sqr2_statement(num_features),
        gen_sart_statement(num_features)
    ]

    output = "\n\n".join(all_statements)

    with open(output_script, "w") as f:
        f.writelines(output)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python gen_script.py <meta list> <number of features> <output script>")
    else:
        gen_script(sys.argv[1], int(sys.argv[2]), sys.argv[3])