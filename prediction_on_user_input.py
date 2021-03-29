import argparse
import json

import numpy as np
import torch

from utils.serialization import load

parser = argparse.ArgumentParser(description='Prediction on user input')
parser.add_argument("-m", "--model", help="model to use", default=None, type=str, required=True)
# parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
args = parser.parse_args()
print("args:", vars(args))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load(args.model)
model.eval()
torch.set_grad_enabled(False)
classifications, regressions = list(model.classifiers.keys()), list(model.regressors.keys())

while True:
    print("Write a report of the patient, then press enter. You can enter multiple reports in this way.")
    print("Insert an empty report to start the prediction based on entered reports.")

    encoded_record = []
    while True:
        report = input()
        if report == "":
            break
        encoded_record.append(torch.tensor(model.encode_report(report), device=device))

    out = model([encoded_record])
    # print("['" + "',\n'".join(record) + "']")
    for cls_var in classifications:
        print(cls_var)
        prediction_idx = out[cls_var].argmax().item()
        prediction = model.labels_codec.codecs[cls_var].decode(prediction_idx)
        print("prediction logits: {}".format(out[cls_var].cpu().numpy()))
        print("prediction index: {}".format(prediction_idx))
        print("prediction: {}".format(prediction))
        if 'all_features' in out:
            record_features = out['all_features'][0]  # [0] because we want the first (and only) record of the batch
            for idx_fr, fr in enumerate(model.features_reducers):
                num_equals = [(fr(record_features, dim=0) == record_features[i]).sum().item() for i in range(len(encoded_record))]
                if sum(num_equals) == 0:
                    print("this reduce method does not support insights on importance of reports")
                else:
                    print("importance of reports: {}".format(num_equals))
                    print("most important report (0-based index): {}".format(np.argmax(num_equals)))
        else:
            print("this model does not support insights on importance of reports")
        print()

    for reg_var in regressions:
        print(reg_var)
        encoded_prediction = out[reg_var].item()
        prediction = model.labels_codec.codecs[reg_var].decode(encoded_prediction)
        print("prediction:  {}, encoded prediction:  {}".format(prediction, encoded_prediction))
        if 'all_features' in out:
            record_features = out['all_features'][0] # [0] because we want the first (and only) record of the batch
            for idx_fr, fr in enumerate(model.features_reducers):
                num_equals = [(fr(record_features, dim=0) == record_features[i]).sum().item() for i in range(len(encoded_record))]
                if sum(num_equals) == 0:
                    print("this reduce method does not support insights on importance of reports")
                else:
                    print("importance of reports: {}".format(num_equals))
                    print("most important report: {}".format(np.argmax(num_equals)))
        else:
            print("this model does not support insights on importance of reports")
        print()
    print()
