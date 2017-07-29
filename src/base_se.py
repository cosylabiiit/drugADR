import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import auc, roc_curve, roc_auc_score
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

d2 = pd.read_excel("2d_prop.xlsx")
d3 = pd.read_excel("3d_prop.xlsx")
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
pca = PCA(n_components=100)# adjust yourself

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
pca.fit(X)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

all_clfs = [
    LogisticRegression(C=20, penalty='l2'),
    RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, min_samples_split=10),
    SVC(probability=True, kernel='rbf'),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=7)
]

def get_scores(clf, X_t_train, y_train, X_t_test, y_test):
    clf.fit(X_t_train, y_train)
    app = dict()
    score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average=None)
    avg_sample_score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average='samples')
    avg_prec = average_precision_score(y_test, clf.predict(X_t_test))
    metrics = [score, avg_sample_score, roc_auc_score(y_test, clf.predict_proba(X_t_test))]
    app['Classwise Scores'] = ([(mlb.classes_[l], score[l]) for l in score.argsort()[::-1]])
    app['F2 Score'] = avg_sample_score
    app['ROC_AUC'] = roc_auc_score(y_test, clf.predict_proba(X_t_test))
    app['Precision Score Avg (PR Curve)'] = avg_prec
    return app

data = []
class_scores = []
metrics = []
for clfx in all_clfs:
    print (clfx)
    start = datetime.datetime.now()
    classifier = OneVsRestClassifier(clfx)
    data.append(get_scores(classifier, X_t_train, y_train, X_t_test, y_test))
    print (datetime.datetime.now() - start)
    print ('-'*80)

pickle.dump(data, open("../data/all_se_clf_data.sav","wb"))
pd.DataFrame(data)
print (data)
