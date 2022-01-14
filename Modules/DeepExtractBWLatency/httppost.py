import sys
import time
import json
import requests

def latency_stat(latency_set, failed_query_num, summaryPath):

    # latency stats
    latency_set.sort()
    count = len(latency_set)
    latency_set = latency_set[:count - 10]
    count = len(latency_set)
    n50 = (count - 1) * 0.5 + 1
    n90 = (count - 1) * 0.9 + 1
    n95 = (count - 1) * 0.95 + 1
    n99 = (count - 1) * 0.99 + 1
    n999 = (count - 1) * 0.999 + 1

    avg = sum(latency_set) / count
    p50 = latency_set[int(n50) - 1]
    p90 = latency_set[int(n90) - 1]
    p95 = latency_set[int(n95) - 1]
    p99 = latency_set[int(n99) - 1]
    p999 = latency_set[int(n999) - 1]

    with open(summaryPath, "w", encoding="utf8") as w:
        w.write("====== latency stats ======\n")
        w.write("\tAverage Latency is " + str(avg) + "\n")
        w.write("\tP50 Latency is " + str(p50) + "\n")
        w.write("\tP90 Latency is " + str(p90) + "\n")
        w.write("\tP95 Latency is " + str(p95) + "\n")
        w.write("\tP99 Latency is " + str(p99) + "\n")
        w.write("\tP999 Latency is " + str(p999) + "\n")
        w.write("\tFailed Query Num is " + str(failed_query_num) + "\n")


def process(inputPath, outputPath, summaryPath, url_type):

    if url_type == 0:
        requrl = "http://CH4AA1090604004:10100/brainwave/models/DeepExtractV2_20211104"
    else:
        requrl = "http://BRAINWAVE-EXP-VIP.FPGAApplianceS10-Prod-CHI02.CHI02.ap.gbl/brainwave/models/DeepExtractV2_20211104"
    
    whole_latency_set = []
    failed_query_num = 0

    with open(inputPath, "r", encoding="utf8") as f:
        with open(outputPath, "w", encoding="utf8") as w:
            for line in f:
                request = line.strip()
                start_time = time.time()
                resp = requests.post(requrl, request)
                ll = (time.time() - start_time) * 1000
                whole_latency_set.append(ll)
                res_dict = resp.json()
                print("res_dict: ", res_dict)
                if res_dict['Succ'] != True:
                    failed_query_num += 1
                new_line = json.dumps(res_dict) + "\n"                
                w.write(new_line)                

    latency_stat(whole_latency_set, failed_query_num, summaryPath)

if __name__ == "__main__":
  process(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))