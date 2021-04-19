import argparse
import json
import os

from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

from utils.constants import *
from utils.dataset import Dataset

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser(description='Get reports from http server')
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-ds", "--data-seed", help="seed for random data shuffling", default=None, type=int)
parser.add_argument("-gb", "--group-by",
                    help="list of (space-separated) grouping attributes to make multi-report predictions.",
                    default=None, nargs="+", type=str, metavar=('ATTR1', 'ATTR2'))
parser.add_argument("-id", "--id-column", help="identifier column name", default=None, type=str, required=True)
parser.add_argument("-p", "--port", help="port to listen on", default=None, type=int)
parser.add_argument("-s", "--set", help="set of the dataset", choices=["training", "validation", "test"], default="validation", type=str)
args = parser.parse_args()

if args.group_by is None:
    print("at the moment, only group by is supported")
    exit(0)

input_cols = ["diagnosi", "macroscopia", "notizie"]
classifications_labels_cols = ["grading", "metastasi", "modalita_N", "modalita_T", "morfologia_icdo3", "sede_icdo3", "stadio_N", "stadio_T", "tipo_T"]
regressions_labels_cols = ["cerb", "dimensioni","ki67", "mib1", "numero_sentinella_asportati", "numero_sentinella_positivi", "recettori_estrogeni", "recettori_progestin"]

dataset = Dataset(args.dataset_dir, args.set + "_set.csv", input_cols)

if args.group_by is not None:
    dataset.lazy_group_by(args.group_by)
    dataset.compute_lazy()

if args.data_seed is not None:
    np.random.seed(args.data_seed)

record_idx_of_patient = {}
for i, record in enumerate(dataset.dataframe):
    id_paz = record.iloc[0][args.id_column].item()
    record_idx_of_patient[id_paz] = i

patients_ids = list(set(record_idx_of_patient.keys()))


@app.route('/info')
def info_API():
    return jsonify({"dataset": args.dataset_dir, "set": args.set})


@app.route('/random')
def get_random_record_API():
    patient_id = patients_ids[np.random.randint(0, len(patients_ids))]
    return get_record_of_patient(patient_id)


@app.route('/patient/<patient_id>')
def get_record_of_patient_API(patient_id):
    patient_id = int(patient_id)
    if patient_id not in patients_ids:
        return jsonify([])
    return get_record_of_patient(patient_id)


def get_record_of_patient(patient_id):
    record = dataset.dataframe[record_idx_of_patient[patient_id]]
    return jsonify({
        "id_paz": patient_id,
        "reports": [
            {col: report[1][col] for col in input_cols}
            for report in record.iterrows()
        ],
        "classifications_labels": {
            col: make_serializable(record[col].unique()[0]) for col in classifications_labels_cols
        },
        "regressions_labels": {
            col: make_serializable(record[col].unique()[0]) for col in regressions_labels_cols
        }
    })


def make_serializable(value):
    if isinstance(value, (np.ndarray, np.generic) ) and np.isnan(value):
        return None
    if value != value:
        return None
    if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):

        return int(value)

    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
        return float(value)

    elif isinstance(value, (np.complex_, np.complex64, np.complex128)):
        return {'real': value.real, 'imag': value.imag}

    elif isinstance(value, (np.ndarray,)):
        return value.tolist()

    elif isinstance(value, (np.bool_)):
        return bool(value)

    elif isinstance(value, (np.void)):
        return None

    return value


if __name__ == '__main__':
    app.run(port=args.port)
