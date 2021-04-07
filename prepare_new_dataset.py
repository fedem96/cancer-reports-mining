import argparse
import json
import re
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandasql import sqldf

from utils.constants import *
from utils.idf import InverseFrequenciesCounter
from utils.tokenizing import TokenCodecCreator
from utils.utilities import *


parser = argparse.ArgumentParser(description='Clean and prepare the new dataset (the one with breast and colon cancer types only)')
parser.add_argument("-c", "--codec", help="filename where to save the token codec", default=TOKEN_CODEC, type=str)
parser.add_argument("-i", "--idf", help="filename where to save Inverse Document Frequencies", default=IDF, type=str)
parser.add_argument("-s", "--stats", help="subdirectory where to save the statistics", default=STATS, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory for the cleaned dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-r", "--raw-csv", help="raw csv file to process", default=NEW_DATA_FILE, type=str)
args = parser.parse_args()

print("reading raw data")

df = pd.read_csv(args.raw_csv, delimiter="|", quotechar="'", encoding="ISO-8859-1", dtype={"anno_referto": "Int64"})
df = df.rename(columns={"Year_of_diagnosis": "anno_diagnosi"})
print("cleaning data")

''' remove invalid rows '''
df.drop(df[df["id_isto"] == 5068716].index, inplace=True)

''' remove invalid columns '''
df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

''' clean "sede_icdo3" column '''
df["sede_icdo3"] = df["sede_icdo3"].apply(lambda s: s.upper())

''' clean "dimensioni" column '''
df.loc[df.index[df["dimensioni"] == " bb"], "dimensioni"] = np.NaN

for column in ["dimensioni"]:
    df[column] = df[column].str.replace(",", ".").astype(float)

''' clean "tipo_T" column '''
df.loc[df.index[df["tipo_T"].apply(lambda v: v not in {"P", "PY"})], "tipo_T"] = np.NaN

''' clean "metastasi" column '''
df.loc[df.index[df["metastasi"].apply(lambda v: v in {1.0, "1"})], "metastasi"] = 1
df.loc[df.index[df["metastasi"].apply(lambda v: v in {0.0, "0"})], "metastasi"] = 0
# df.loc[df.index[df["metastasi"].isnull()], "metastasi"] = 0
df.loc[df.index[df["metastasi"].apply(lambda v: v in {"4", "9", "X"})], "metastasi"] = np.NaN
df["metastasi"] = df["metastasi"].astype("Int64")

''' clean "modalita_T" column '''
df.loc[df.index[df["modalita_T"].apply(lambda v: v not in {"E", "R"})], "modalita_T"] = np.NaN

''' clean "modalita_N" column '''
df.loc[df.index[df["modalita_N"].apply(lambda v: v not in {"E", "R"})], "modalita_N"] = np.NaN

''' clean "stadio_T" column '''
df.loc[df.index[df["stadio_T"].apply(lambda v: v == "X" or v == "A")], "stadio_T"] = np.NaN
df["stadio_T"] = df["stadio_T"].apply(lambda v: str(v).replace(" ", "").upper() if pd.notna(v) else None)

''' clean "stadio_N" column '''
df.loc[df.index[df["stadio_N"].apply(lambda v: v == "X")], "stadio_N"] = np.NaN
neg_regex = re.compile("n.*e.*g", re.I)
df.loc[df.index[df["stadio_N"].apply(lambda v: None != neg_regex.match(str(v)))], "stadio_N"] = 0
df["stadio_N"] = df["stadio_N"].apply(lambda v: str(v).replace(" ", "").upper() if pd.notna(v) else None)

''' convert "numero_sentinella_asportati" and "numero_sentinella_positivi" '''
for column in ["numero_sentinella_asportati", "numero_sentinella_positivi"]:
    df[column] = df[column].astype("Int64")

mask_invalid_rows = df.index[(df["numero_sentinella_asportati"].notnull()) &
                             (df["numero_sentinella_positivi"].notnull()) &
                             (df["numero_sentinella_asportati"] < df["numero_sentinella_positivi"])]
df.loc[mask_invalid_rows, "numero_sentinella_asportati"] = np.NaN
df.loc[mask_invalid_rows, "numero_sentinella_positivi"] = np.NaN

''' clean "grading" column '''
df.loc[df.index[df["grading"] == "A"], "grading"] = np.NaN
df.loc[df.index[df["grading"] == "B"], "grading"] = np.NaN
df.drop(df[df["grading"] == "0"].index, inplace=True)

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
    df.loc[df.index[df[column] > 100], column] = np.NaN
    # df.drop(df[df[column] > 100].index, inplace=True)

input_cols = ["notizie", "macroscopia", "diagnosi"]
replace_nulls(df, {col: "" for col in input_cols})
for col in input_cols:
    df[col] = df[col].apply(lambda text: text.strip())
df.drop(df[(df["diagnosi"] == "") & (df["macroscopia"] == "") & (df["notizie"] == "")].index, inplace=True)

print("solving ambiguities")
df.drop_duplicates(subset=["id_isto"], inplace=True, keep=False) # TODO

print("asserting data validity")
assert sqldf("select count(*) from df group by id_isto having count(*) > 1").sum().item() == 0
for column in ["recettori_estrogeni_%", "recettori_progestin_%", "mib1", "cerb", "ki67"]:
    # a percentage can't be smaller than 0 or greater than 100
    assert df[column].min() >= 0 and df[column].max() <= 100
for col in set(df.columns) - {"id_paz", "anno_diagnosi", "anno_referto", "id_isto", "notizie", "macroscopia", "diagnosi"}:
    # assure that each patient has no columns with multiple values
    assert sqldf("select ((count(distinct \"{}\") > 0) + (count(*) - count(\"{}\") > 0)) s from df group by id_paz having s>1".format(col, col)).sum().item() == 0

print("splitting dataset into train, validation and test")
# splits by year
testPatients = set(df[df['anno_diagnosi'] == 2015]['id_paz'].unique())
valPatients = set(df[df['anno_diagnosi'] == 2014]['id_paz'].unique()) - testPatients
trainPatients = set(df['id_paz'].unique()) - testPatients - valPatients

# # version with random splits
# patients = np.array(list(set(df['id_paz'].unique())))
# perm = np.random.permutation(len(patients))
# trainPatients = set(patients[perm][:17000])
# valPatients = set(patients[perm][17000:21000])
# testPatients = set(patients[perm][21000:])

assert len(valPatients.intersection(testPatients)) == 0
assert len(valPatients.intersection(trainPatients)) == 0
assert len(trainPatients.intersection(testPatients)) == 0

set(df['id_paz'].unique())

dfTrain = df[df['id_paz'].isin(trainPatients)].copy()
dfVal = df[df['id_paz'].isin(valPatients)].copy()
dfTest = df[df['id_paz'].isin(testPatients)].copy()

# total:      25184 patients                                                                      115865 reports
# train:      17565 patients (69.75%) without reports neither in 2014 nor in 2015                  77862 reports
# validation:  3783 patients (15.02%) with at least one report in 2014 (but not in 2015)           18994 reports
# test:        3836 patients (15.23%) with at least one report in 2015                             19009 reports

print("saving cleaned csv")
if not os.path.exists(args.dataset_dir): os.makedirs(args.dataset_dir)
dfTrain.to_csv(os.path.join(args.dataset_dir, TRAINING_SET), index=False)
dfVal.to_csv(os.path.join(args.dataset_dir, VALIDATION_SET), index=False)
dfTest.to_csv(os.path.join(args.dataset_dir, TEST_SET), index=False)

print("calculating statistics")
stats_dir = os.path.join(args.dataset_dir, args.stats)
bar_columns = ["anno_diagnosi", "sede_icdo3", "morfologia_icdo3", "tipo_T", "metastasi", "modalita_T", "modalita_N", "stadio_T", "stadio_N", "grading", "anno_referto"]
hist_columns = ["dimensioni", "recettori_estrogeni_%", "recettori_progestin_%", "numero_sentinella_asportati", "numero_sentinella_positivi", "mib1", "cerb", "ki67"]
years = range(2003, 2016)
dataframes = [df] + [df[df["anno_diagnosi"] == year] for year in years] + [dfTrain, dfVal, dfTest]
images_dirs = [stats_dir] + [os.path.join(stats_dir, str(year)) for year in years] + [os.path.join(stats_dir, "train"), os.path.join(stats_dir, "val"), os.path.join(stats_dir, "test")]
for dataframe, images_dir in zip(dataframes, images_dirs):
    if not os.path.exists(images_dir): os.makedirs(images_dir)
    dataframe[set(dataframe.columns) - set(input_cols) - {"id_paz", "id_isto", "anno_diagnosi", "anno_referto"}].notnull().mean().sort_index().plot(kind="bar")
    plt.title("frazione valori non nulli")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "notnull-bar.png"))
    plt.clf()
    for col in bar_columns:
        plot_bar(dataframe, col, output_file=os.path.join(images_dir, col + "-bar.png"))
    for col in hist_columns:
        plot_hist(dataframe, col, output_file=os.path.join(images_dir, col + "-hist.png"))

    with open(os.path.join(images_dir, "stats.json"), 'w') as outfile:
        json.dump({"num_rows": len(dataframe), "not_null": {k: int(v) for k,v in dict(dataframe.notnull().sum()).items()}}, outfile, indent=2)

print("generating codec")
# training_set = pd.read_csv(os.path.join(args.dataset_dir, TRAINING_SET), usecols=cols)
replace_nulls(dfTrain, {col: "" for col in input_cols})
texts = merge_and_extract(dfTrain, input_cols)
TokenCodecCreator().create_codec(texts).save(os.path.join(args.dataset_dir, args.codec))

print("generating idf")
InverseFrequenciesCounter().train(texts).save(os.path.join(args.dataset_dir, args.idf))

print("done")
