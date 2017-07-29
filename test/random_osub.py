import pickle
import pandas as pd
import numpy as np
import xlsxwriter
import csv
import random
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import datetime
from sklearn.metrics import auc, roc_curve, roc_auc_score

mlb = MultiLabelBinarizer()

misc = list()
df = pd.read_excel("../data/misc.xlsx")
for i in df.MISCELLANEOUS:
    misc.append(i)
all_se = list()
df = pd.read_csv("../data/unique_SE.csv")
for i in df.side_effect_name:
    all_se.append(i)
s1 = set(all_se)
s2 = set(misc)
s3 = list(s1 - s2)

print ("Total:",len(s1))
print ("Misc:",len(s2))
print ("Total excluding misc:",len(s3))

l1 = s3

columns = [
    'stitch_id_flat',
    'stitch_id_sterio',
    'umls_cui_from_label',
    'meddra_type',
    'umls_cui_from_meddra',
    'side_effect_name',
]
sedf = pd.read_table('../data/meddra_all_se.tsv', names=columns)
sedf.drop(sedf[sedf.meddra_type == "LLT"].index, inplace=True)

sedf = sedf.groupby('stitch_id_flat').side_effect_name.apply(list).reset_index()
sedf['labels'] = None
d2 = pd.read_excel("../data/2d_prop.xlsx")
d3 = pd.read_excel("../data/3d_prop.xlsx")
print (d2.shape, d3.shape)

d2 = d2.select_dtypes(include=['int64','float64'])
d3 = d3.select_dtypes(include=['float64'])

def check_availability(x1, x2):
    if float(len(set(x1).intersection(set(x2))))/len(x1) > 0:
        return True, float(len(set(x1).intersection(set(x2))))
    return False, float(len(set(x1).intersection(set(x2))))

def get_scores(clf, X_t_train, y_train, X_t_test, y_test):
    clf.fit(X_t_train, y_train)
    y_score = clf.predict_proba(X_t_test)
    app = dict()
    score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average=None)
    #auc_score = roc_auc_score(y_test, clf.predict(X_t_test), average='samples')
    avg_sample_score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average='samples')
    prec_score = precision_score(y_test, clf.predict(X_t_test), average='micro')
    rec_score = recall_score(y_test, clf.predict(X_t_test), average='micro')
    avg_prec = average_precision_score(y_test, clf.predict(X_t_test))
    metrics = [score, avg_sample_score, roc_auc_score(y_test, clf.predict_proba(X_t_test))]
    #app['Classwise Scores'] = ([(mlb.classes_[l], score[l]) for l in score.argsort()[::-1]])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(list(enumerate(mlb.classes_)))):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[mlb.classes_[i]] = auc(fpr[i], tpr[i])

    app['F2 Score'] = avg_sample_score
    app['ROC_AUC'] = roc_auc_score(y_test, clf.predict_proba(X_t_test))
    app['Classwise F2 Scores'] = ([(mlb.classes_[l], score[l]) for l in score.argsort()[::-1]])
    app['P_AUPR'] = avg_prec
    app['Precision'] = prec_score
    app['Recall'] = rec_score
    app['ROC_AUC_samples'] = roc_auc
    return app


li = list()

for r in range(5):
    d = dict()
    random.shuffle(l1)
    d['o1'] = l1[:65]
    d['o2'] = l1[65:107]
    d['o3'] = l1[107:144]
    d['o4'] = l1[144:238]
    d['o5'] = l1[238:436]
    d['o6'] = l1[436:818]
    d['o7'] = l1[818:955]
    d['o8'] = l1[955:1044]
    d['o9'] = l1[1044:1137]
    d['o10'] = l1[1137:1373]
    d['o11'] = l1[1373:1470]
    d['o12'] = l1[1470:1492]
    d['o13'] = l1[1492:1553]
    d['o14'] = l1[1553:1612]
    d['o15'] = l1[1612:1779]
    d['o16'] = l1[1779:1903]
    d['o17'] = l1[1903:2055]
    d['o18'] = l1[2055:2115]
    d['o19'] = l1[2115:2399]
    d['o20'] = l1[2399:2442]
    d['o21'] = l1[2442:2466]
    d['o22'] = l1[2466:2507]
    d['o23'] = l1[2507:2711]
    d['o24'] = l1[2711:2963]
    d['o25'] = l1[2963:3198]
    d['o26'] = l1[3198:3287]
    d['o27'] = l1[3287:3325]
    d['o28'] = l1[3325:3638]
    d['o29'] = l1[3638:3653]
    d['o30'] = l1[3653:3675]
    df=pd.DataFrame.from_dict(d,orient='index').transpose()

    se = df
    test_cols = list(se.columns)
    test_cols_update = test_cols[:]

    df = se[test_cols_update]

    labels = {}
    for cx in df.columns:
        data = {
            'label': list(df.columns).index(cx),
            'name': cx
        }
        labels[tuple(list(df[cx].dropna().values))] = data

    z = []

    for ix in range(sedf.side_effect_name.shape[0]):
        chk = sedf.side_effect_name[ix]
        var = []
        for jx in labels.keys():
            chk = check_availability(sedf.side_effect_name[ix], jx)
            z.append(chk[1])
            if chk[0]:
                var.append(labels[jx]['name'])
        sedf['labels'][ix] = var

    y=mlb.fit_transform(sedf['labels'])
    label_w = list(labels.values())
    label_names = list()
    for i in label_w:
        label_names.append(i['name'])

    df = pd.concat([sedf, d2, d3], axis=1)
    df.drop('side_effect_name', axis=1, inplace=True)
    df.drop("stitch_id_flat", inplace=True, axis=1)

    X = df.drop("labels", axis=1)
    Y = df[df.columns[0]]
    X.fillna(X.mean(), inplace=True)
    X = StandardScaler().fit_transform(X)
    mlb = MultiLabelBinarizer()
    y=mlb.fit_transform(Y)
    pca = PCA(n_components=100)# adjust yourself

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)

    all_clfs = [
    LogisticRegression(C=20, penalty='l2'),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42, min_samples_split=5),
    SVC(probability=True, kernel='rbf'),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=7)
    ]

    data = []
    class_scores = []
    metrics = []
    for clfx in all_clfs:
        print (clfx)
        start = datetime.datetime.now()
        classifier = OneVsRestClassifier(clfx)
        data.append(get_scores(classifier, X_t_train, y_train, X_t_test,y_test))
        print (datetime.datetime.now() - start)
        print ('-'*80)

    data[0]['Classifier'] = 'Logistic Regression'
    data[1]['Classifier'] = 'Random Forest'
    data[2]['Classifier'] = 'SVM'
    data[3]['Classifier'] = 'GaussianNB'
    data[4]['Classifier'] = 'kNN'

    li.append(data)

pickle.dump(li, open("../data/list_res_Sub_Sys.sav",'wb'))
