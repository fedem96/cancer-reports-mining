import argparse
import json

from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
import numpy as np
import torch

from utils.serialization import load

app = Flask(__name__)
CORS(app)


parser = argparse.ArgumentParser(description='Run predictions on http server')
parser.add_argument("-m", "--model", help="model to use", default=None, type=str, required=True)
parser.add_argument("-p", "--port", help="port to listen on", default=None, type=int)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load(args.model)
model.eval()
torch.set_grad_enabled(False)


@app.route('/info')
def info_API():
    return jsonify({"model": args.model})


@app.route('/tokenize')
def tokenize_API():
    return jsonify({"tokens": tokenize(request.args.getlist('report'))})

@app.route('/encode')
def encode_API():
    var = request.args.getlist('variable')
    val = request.args.getlist('value')
    return jsonify({"encoded_value": model.labels_codec[var].encode(val)})

@app.route('/decode')
def decode_API():
    var = request.args.getlist('variable')
    val = request.args.getlist('value')
    return jsonify({"decoded_value": model.labels_codec[var].decode(val)})


def tokenize(reports):
    return model.tokenizer.tokenize_batch(model.preprocessor.preprocess_batch(reports))


@app.route('/predict', methods=['GET', 'POST'])
def predict_API():
    if request.method == 'POST':
        reports = json.loads(request.data)['reports']
    else:
        reports = request.args.getlist('report')
    predictions = predict(reports)
    tokens = tokenize(reports)
    tokens_indices = [tokens_array.tolist() for tokens_array in model.token_codec.encode_batch(tokens)]
    return jsonify({
        **{k: v.cpu().tolist() for k,v in predictions.items() if k in {"reports_importance", "tokens_importance"}},
        "classifications": {k: {model.labels_codec[k].decode(i): v for i,v in enumerate(v.cpu().squeeze().softmax(0).tolist())} for k,v in predictions.items() if k in model.get_validation_classifications()},
        "regressions": {k: model.labels_codec[k].decode(v.cpu().squeeze().item()) for k,v in predictions.items() if k in model.get_validation_regressions()},
        "tokens": [
            [{"text": t, "index": i} for t,i in zip(record_tokens, record_tokens_indices)]
            for record_tokens, record_tokens_indices in zip(tokens, tokens_indices)
        ]
    })

def predict(reports):
    # reports_tensor = model.tensorize(reports) # TODO: !
    reports = model.token_codec.encode_batch(model.tokenizer.tokenize_batch(model.preprocessor.preprocess_batch(reports)))
    reports_tensor = torch.tensor(numpyze([reports]).astype(np.int16), device=model.current_device())
    if all(reports_tensor.flatten() == 0):
        return {}
    return model(reports_tensor, explain=True)

def numpyze(records):
    reports_lengths = [[len(report) for report in record] for record in records]
    records_sizes = [len(record) for record in records]
    max_report_length = max([max(lengths) for lengths in reports_lengths])
    max_record_size = max(records_sizes)
    return np.stack(  # stack records to create dataset
        [
            np.pad(  # pad record
                np.stack([  # stack reports to create record
                    np.pad(report.astype(np.uint16), (0, max_report_length - len(report)))  # pad report
                    for report in record
                ]
                ), ((0, max_record_size - len(record)), (0, 0)))
            for record in records
        ]
    )


if __name__ == '__main__':
    app.run(port=args.port)
