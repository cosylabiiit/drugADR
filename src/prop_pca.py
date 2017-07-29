import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import datetime
mlb = MultiLabelBinarizer()

def stitch_to_pubchem(_id):
    assert _id.startswith("CID")
    return int(int(_id[3:]) - 1e8)

columns = [
    'stitch_id_flat',
    'stitch_id_sterio',
    'umls_cui_from_label',
    'meddra_type',
    'umls_cui_from_meddra',
    'side_effect_name',
]
df = pd.read_table('../data/meddra_all_se.tsv', names=columns)
df.drop(df[df.meddra_type == "LLT"].index, inplace=True)
print (df.info())

df = df.groupby('stitch_id_flat').side_effect_name.apply(list).reset_index()
df['pubchem_id'] = df.stitch_id_flat.map(stitch_to_pubchem)
print (df.head())

d2 = pd.read_excel("../data/2d_prop.xlsx")
d3 = pd.read_excel("../data/3d_prop.xlsx")
print (d2.shape, d3.shape)

d2 = d2.select_dtypes(include=['int64','float64'])
d3 = d3.select_dtypes(include=['float64'])
y=mlb.fit_transform(sedf['side_effect_name'])

print (y.shape)

for ix in range(y.shape[0]):
    q = (roc_auc_score(y[ix], y[ix]))
    if q == 0:
        print ("True", ix)
df = pd.concat([sedf, d2, d3], axis=1)

df.drop("stitch_id_flat", inplace=True, axis=1)
X = df.drop("side_effect_name", axis=1)
Y = df[df.columns[0]]

X.fillna(X.mean(), inplace=True)
X = StandardScaler().fit_transform(X)
mlb = MultiLabelBinarizer()
y=mlb.fit_transform(Y)
pca = PCA(n_components=300)# adjust yourself
X_pca = pca.fit_transform(X)
plt.figure(figsize=(15,5))
plt.plot(pca.explained_variance_ratio_.cumsum()*100)
plt.ylabel("(%)variance retained")
plt.xlabel("Number of Principal Components")
plt.autoscale(False)
plt.show()
plt.savefig('../data/PCA.png', bbox_inches='tight')
print (pca.explained_variance_ratio_.cumsum()[100])
