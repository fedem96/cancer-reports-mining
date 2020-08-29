import os

import pandas as pd

from utils.constants import *
from utils.utilities import *

# csv = [filename,
#         types,
#         converters,
#         dates_format,
#         handle_nans]

csv1 = [os.path.join("train_data", "ISTOLOGIE_corr.csv"),
        "ISO-8859-1",
        {"id_neopl": 'Int64', "notizie": str, "diagnosi": str, "macroscopia": str}]


csv2 = [os.path.join("train_data", "RTRT_NEOPLASI_corr.csv"),
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

dfTrain.to_csv(TRAINING_SET_FILE, index=False)
dfTest.to_csv(TEST_SET_FILE, index=False)
dfUnsup.to_csv(UNSUPERVISED_SET_FILE, index=False)


#df2["anno"].value_counts().sort_index().plot(kind="bar")
#plt.show()