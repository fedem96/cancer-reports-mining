import argparse

from tokenizing.tokenizer import Tokenizer
from utils.constants import *
from utils.preprocessing import Preprocessor
from utils.utilities import *


parser = argparse.ArgumentParser(description='Clean and prepare the old dataset (the one with all cancer types)')
parser.add_argument("-d", "--dataset-dir", help="directory for the cleaned dataset", default=os.path.join(DATASETS_DIR, OLD_DATASET), type=str)
parser.add_argument("-hist", "--histologies", help="histologies file", default=OLD_HISTOLOGIES_FILE, type=str)
parser.add_argument("-neop", "--neoplasms", help="neoplasms file", default=OLD_NEOPLASMS_FILE, type=str)
args = parser.parse_args()

print("reading raw data")

df_hists = pd.read_csv(args.histologies, encoding="ISO-8859-1",
                    dtype={"id_neopl": 'Int64', "id_tot": 'Int64', "diagnosi": str, "macroscopia": str, "notizie": str},
                    usecols=["id_neopl", "id_tot", "diagnosi", "macroscopia", "notizie"])

dates_format = {"anno_archivio": "%y", "data_incidenza": "%Y-%m-%d", "data_intervento": "%Y-%m-%d"}
df_neopls = pd.read_csv(args.neoplasms, encoding="ISO-8859-1",
                        dtype={"id_neopl": 'Int64', "id_tot": 'Int64', "sede_icdo3": str, "morfologia_icdo3": str},
                        usecols=["id_neopl", "id_tot", "anno_archivio", "data_incidenza", "data_intervento", "sede_icdo3", "morfologia_icdo3"],
                        converters={col: (lambda d, f=format: convert_date(d, f)) for col, format in dates_format.items()},
                        decimal=",")

print("merging data")
dataset = df_hists.merge(df_neopls, how='inner', on=['id_tot', 'id_neopl']).dropna(subset=['data_incidenza'])

print("cleaning data")
dataset['anno'] = dataset["data_incidenza"].apply(lambda x: x.year)
dataset = dataset.drop(columns=['anno_archivio', 'data_incidenza', 'data_intervento'])
dataset = dataset.drop(dataset[dataset.anno < 1980].index)
input_cols = ['diagnosi', 'macroscopia', 'notizie']
replace_nulls(dataset, {col: "" for col in input_cols})
for col in input_cols:
    dataset[col] = dataset[col].apply(lambda text: text.strip())
dataset.drop(dataset[(dataset["diagnosi"] == "") & (dataset["macroscopia"] == "") & (dataset["notizie"] == "")].index, inplace=True)

print("splitting dataset into train, validation, test and unsupervised")
dfTrain, dfTest = train_test_split(dataset, "anno", test_from_year=2009)
dfTrain, dfVal = train_test_split(dfTrain, "anno", test_from_year=2008)
# train:  70780 samples
# val:    10952 samples
# test:   12719 samples

dfUnsup = df_hists[(~df_hists.id_neopl.isin(set(dataset.id_neopl.unique()))) & (~df_hists.id_tot.isin(set(dataset.id_tot.unique())))]      # 1372421 samples
dfUnsup = dfUnsup.drop(columns=["id_neopl"])
# TODO: add an index to unsupervised csv

print("saving cleaned csv")
if not os.path.exists(args.dataset_dir): os.makedirs(args.dataset_dir)
dfTrain.to_csv(os.path.join(args.dataset_dir, TRAINING_SET), index=False)
dfVal.to_csv(os.path.join(args.dataset_dir, VALIDATION_SET), index=False)
dfTest.to_csv(os.path.join(args.dataset_dir, TEST_SET), index=False)
dfUnsup.to_csv(os.path.join(args.dataset_dir, UNSUPERVISED_SET), index=False)


print("generating tokenizers")
p = Preprocessor.get_default()
texts = p.preprocess_batch(merge_and_extract(dfTrain, input_cols))
tknzrs = []
for n in range(1,4):
    print(f"creating tokenizer with {n}-gram codec")
    t = Tokenizer(n_grams=n).create_codec(texts, min_occurrences=10).save(os.path.join(args.dataset_dir, f"tokenizer-{n}gram.json"))
    tknzrs.append(t)
