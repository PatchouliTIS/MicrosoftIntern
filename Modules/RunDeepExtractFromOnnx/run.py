import json
import onnxruntime as ort
import sys


def process(input_path, model_path, output_path):
    sessOption = ort.SessionOptions()
    sessOption.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sessOption)

    with open(output_path, 'w', encoding='utf8') as output:
        # output.write('TraceId\tTRTResults\n')
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                input_data = json.loads(line)

                res = sess.run([sess.get_outputs()[x].name for x in range(len(sess.get_outputs()))],
                               {sess.get_inputs()[0].name: input_data["tokenIds"],
                                sess.get_inputs()[1].name: input_data["segmentMasks"],
                                sess.get_inputs()[2].name: input_data["masks"],
                                sess.get_inputs()[
                                   3].name: input_data["sentMasks"]
                                })
                json_res = {}
                json_res['index_1'] = res[0][0].tolist()
                json_res['score_1'] = res[1][0].tolist()
                json_res['index_2'] = res[2][0].tolist()
                json_res['score_2'] = res[3][0].tolist()

                new_line = json.dumps(json_res) + '\n'
                output.write(new_line)


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2], sys.argv[3])
