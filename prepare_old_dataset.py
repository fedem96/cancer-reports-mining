import argparse
import os

import pandas as pd

from utils.constants import *
from utils.utilities import *

# csv = [filename,
#         types,
#         converters,
#         dates_format,
#         handle_nans]

parser = argparse.ArgumentParser(description='Clean and prepare the old dataset (the one with all cancer types)')
parser.add_argument("-d", "--dataset-dir", help="directory for the cleaned dataset", default=os.path.join(DATASETS_DIR, OLD_DATASET), type=str)
parser.add_argument("-i", "--histologies", help="histologies file", default=OLD_HISTOLOGIES_FILE, type=str)
parser.add_argument("-n", "--neoplasms", help="neoplasms file", default=OLD_NEOPLASMS_FILE, type=str)
args = parser.parse_args()

csv1 = [args.histologies,
        "ISO-8859-1",
        {"id_neopl": 'Int64', "notizie": str, "diagnosi": str, "macroscopia": str}]


csv2 = [args.neoplasms,
        "ISO-8859-1",
        {"id_neopl": 'Int64', "stadio": 'Int64', "dimensioni": float, "sede_icdo3": str, "morfologia_icdo3": str},
        {},
        {"anno_archivio": "%y", "data_incidenza": "%Y-%m-%d", "data_intervento": "%Y-%m-%d"},
        {},
        ','
        ]


df1 = read_csv(*csv1)

df2 = read_csv(*csv2)
df2["anno"] = df2["anno_archivio"]
df2.loc[df2["anno"].isnull(), "anno"] = df2.loc[df2["anno"].isnull(), "data_incidenza"]
df2.loc[df2["anno"].isnull(), "anno"] = df2.loc[df2["anno"].isnull(), "data_intervento"]

df2 = df2.drop(columns=['anno_archivio', 'data_incidenza', 'data_intervento'])
df2 = df2.drop(df2.index[df2["anno"].isnull()])
df2["anno"] = pd.DatetimeIndex(df2['anno']).year
df2 = df2.drop(df2.index[df2["anno"] < 1980])

df2 = df2.merge(df1, how='inner')

dfTrain, dfTest = train_test_split(df2, "anno", 2008)
# train:  70836 samples
# test:   23671 samples


dfUnsup = df1[df1["id_neopl"].isnull()]             # 1496896 samples
dfUnsup = dfUnsup.drop(columns=["id_neopl"])
# TODO: add an index to unsupervised csv

if not os.path.exists(args.dataset_dir): os.makedirs(args.dataset_dir)

dfTrain.to_csv(os.path.join(args.dataset_dir, TRAINING_SET), index=False)
dfTest.to_csv(os.path.join(args.dataset_dir, TEST_SET), index=False)
dfUnsup.to_csv(os.path.join(args.dataset_dir, UNSUPERVISED_SET), index=False)


#df2["anno"].value_counts().sort_index().plot(kind="bar")
#plt.show()