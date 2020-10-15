import argparse
import re

import numpy as np
import pandas as pd

from utils.constants import *
from utils.idf import InverseFrequenciesCounter
from utils.tokenizing import TokenCodecCreator
from utils.utilities import *


parser = argparse.ArgumentParser(description='Clean and prepare the new dataset (the one with breast and colon cancer types only)')
parser.add_argument("-c", "--codec", help="filename where to save the token codec", default=TOKEN_CODEC, type=str)
parser.add_argument("-i", "--idf", help="filename where to save Inverse Document Frequencies", default=IDF, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory for the cleaned dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-r", "--raw-csv", help="raw csv file to process", default=NEW_DATA_FILE, type=str)
args = parser.parse_args()

df = pd.read_csv(args.raw_csv, delimiter="|", quotechar="'", encoding="ISO-8859-1", dtype={"anno_referto": "Int64"})
df = df.rename(columns={"Year_of_diagnosis": "anno_diagnosi"})
print("cleaning data")

''' clean "tipo_T" column '''
df.loc[df.index[df["tipo_T"].apply(lambda v: v not in {"P", "PY"})], "tipo_T"] = np.NaN

''' clean "metastasi" column '''
df.loc[df.index[df["metastasi"].apply(lambda v: v in {1.0, "1", "X"})], "metastasi"] = 1
df.loc[df.index[df["metastasi"].apply(lambda v: v in {0.0, "0"})], "metastasi"] = 0
df.loc[df.index[df["metastasi"].isnull()], "metastasi"] = 0
df.loc[df.index[df["metastasi"].apply(lambda v: v in {"4", "9"})], "metastasi"] = np.NaN
df["metastasi"] = df["metastasi"].astype("Int64")

''' clean "modalita_T" column '''
df.loc[df.index[df["modalita_T"].apply(lambda v: v not in {"E", "R"})], "modalita_T"] = np.NaN

''' clean "modalita_N" column '''
df.loc[df.index[df["modalita_N"].apply(lambda v: v not in {"E", "R"})], "modalita_N"] = np.NaN

''' clean "stadio_N" column '''
neg_regex = re.compile("n.*e.*g", re.I)
df.loc[df.index[df["stadio_N"].apply(lambda v: None != neg_regex.match(str(v)))], "stadio_N"] = 0

''' clean "dimensioni" column '''
df.drop(df[df["dimensioni"] == " bb"].index, inplace=True)
for column in ["dimensioni"]:
    df[column] = df[column].str.replace(",", ".").astype(float)

''' convert "numero_sentinella_asportati" and "numero_sentinella_positivi" '''
for column in ["numero_sentinella_asportati", "numero_sentinella_positivi"]:
    df[column] = df[column].astype("Int64")

''' clean "mib1", "cerb" and "ki67" columns '''
pos_regex = re.compile("p.*o.*s", re.I)
special_regex = re.compile("[+<>%-]")
for column in ["mib1", "cerb", "ki67"]:
    df.loc[df.index[df[column].apply(lambda v: neg_regex.search(str(v)) != None)], column] = 0
    df.loc[df.index[df[column].apply(lambda v: pos_regex.search(str(v)) != None)], column] = 100
    df.loc[df.index[df[column] == "N.V."], column] = np.NaN
    df[column] = df[column].apply(lambda v: special_regex.sub("", str(v)).replace(",", "."))
    df.loc[df.index[df[column] == ""], column] = np.NaN
    df[column] = df[column].astype(float)

print("dtypes after cleaning:")
print(df.dtypes)

print("removing invalid data")
for column in ["recettori_estrogeni_%", "ki67"]:
    # df.drop(df[df[column] < 0].index, inplace=True)
    df.drop(df[df[column] > 100].index, inplace=True)

df.drop(df[df["grading"] == "A"].index, inplace=True)
df.drop(df[df["grading"] == "B"].index, inplace=True)
cols = ["diagnosi", "macroscopia", "notizie"]
replace_nulls(df, {col: "" for col in cols})
for col in cols:
    df[col] = df[col].apply(lambda text: text.strip())
df.drop(df[(df["diagnosi"] == "") & (df["macroscopia"] == "") & (df["notizie"] == "")].index, inplace=True)

print("asserting data validity")
for column in ["recettori_estrogeni_%", "recettori_progestin_%", "mib1", "cerb", "ki67"]:
    assert df[column].min() >= 0 and df[column].max() <= 100

# print("floats to strings")
# for column in df.columns:
#     if df[column].dtype == float:
#         df[column] = df[column].apply(lambda v: str(v).rstrip('0').rstrip('.') if v is not None and v != np.NaN else "")

print("splitting dataset by year of diagnosis")
dfTrain, dfTest = train_test_split(df, "anno_diagnosi", 2015)
dfTrain, dfVal = train_test_split(dfTrain, "anno_diagnosi", 2014)
# total:     121415 samples
# train:      82372 samples (67.8%) from 2003 to 2013
# validation: 19528 samples (16.1%) of 2014
# test:       19515 samples (16.1%) of 2015

print("saving cleaned csv")
if not os.path.exists(args.dataset_dir): os.makedirs(args.dataset_dir)
dfTrain.to_csv(os.path.join(args.dataset_dir, TRAINING_SET), index=False)
dfVal.to_csv(os.path.join(args.dataset_dir, VALIDATION_SET), index=False)
dfTest.to_csv(os.path.join(args.dataset_dir, TEST_SET), index=False)

print("generating codec")
# training_set = pd.read_csv(os.path.join(args.dataset_dir, TRAINING_SET), usecols=cols)
replace_nulls(df, {col: "" for col in cols})
texts = merge_and_extract(df, cols)
TokenCodecCreator().create_codec(texts).save(os.path.join(args.dataset_dir, args.codec))

print("generating idf")
InverseFrequenciesCounter().train(texts).save(os.path.join(args.dataset_dir, args.idf))

print("done")
