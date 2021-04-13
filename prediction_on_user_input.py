import argparse

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

    record = []
    encoded_record = []
    while True:
        report = input()
        if report == "":
            break
        record.append(report)
        encoded_record.append(model.encode_report(report))

    max_len = max([len(enc_rep) for enc_rep in encoded_record])
    out = model(torch.stack([torch.tensor(np.pad(enc_rep, (0,max_len-len(enc_rep))), device=device) for enc_rep in encoded_record]).unsqueeze(0))
    # print("['" + "',\n'".join(record) + "']")
    for cls_var in classifications:
        print(cls_var)
        prediction_idx = out[cls_var].argmax().item()
        prediction = model.labels_codec.codecs[cls_var].decode(prediction_idx)
        print("prediction logits: {}".format(out[cls_var].cpu().numpy()))
        print("prediction index: {}".format(prediction_idx))
        print("prediction: {}".format(prediction))
        if 'all_features' in out:  # TODO: make compatible with new source code
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
        if 'reports_importance' in out:
            reports_importance = out['reports_importance'][0][:len(record)]
            print("importance of reports: {}".format(reports_importance))
            print("most important report (0-based index): {}".format(np.argmax(reports_importance)))
            if 'tokens_importance' in out:
                tokens_importance = out['tokens_importance'][0][:len(record)]
                for i, report in enumerate(record):
                    report_tokens_str = model.tokenizer.tokenize(model.preprocessor.preprocess(report))
                    print("report {}".format(i))
                    print(list(zip(report_tokens_str, tokens_importance[i].cpu().numpy())))
                    print()
            else:
                print("this model does not support insights on importance of tokens")
        else:
            print("this model does not support insights on importance of reports")
        print()
    print()
