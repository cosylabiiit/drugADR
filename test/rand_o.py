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
df = pd.read_excel("misc.xlsx")
for i in df.MISCELLANEOUS:
    misc.append(i)
all_se = list()
df = pd.read_csv("unique_SE.csv")
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
sedf = pd.read_table('../meddra_all_se.tsv', names=columns)
sedf.drop(sedf[sedf.meddra_type == "LLT"].index, inplace=True)

sedf = sedf.groupby('stitch_id_flat').side_effect_name.apply(list).reset_index()
sedf['labels'] = None
d2 = pd.read_excel("2d_prop.xlsx")
d3 = pd.read_excel("3d_prop.xlsx")
print (d2.shape, d3.shape)

d2 = d2.select_dtypes(include=['int64','float64'])
d3 = d3.select_dtypes(include=['float64'])

def check_availability(x1, x2):
    if float(len(set(x1).intersection(set(x2))))/len(x1) > 0:
        return True, float(len(set(x1).intersection(set(x2))))
    return False, float(len(set(x1).intersection(set(x2))))

def get_scores(clf, X_t_train, y_train, X_t_test, y_test):
    clf.fit(X_t_train, y_train)
    app = dict()
    score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average=None)
    avg_sample_score = fbeta_score(y_test, clf.predict(X_t_test), beta=2, average='samples')
    prec_score = precision_score(y_test, clf.predict(X_t_test), average='micro')
    rec_score = recall_score(y_test, clf.predict(X_t_test), average='micro')
    avg_prec = average_precision_score(y_test, clf.predict(X_t_test))
    metrics = [score, avg_sample_score, roc_auc_score(y_test, clf.predict_proba(X_t_test))]
    #app['Classwise Scores'] = ([(mlb.classes_[l], score[l]) for l in score.argsort()[::-1]])
    app['F2 Score'] = avg_sample_score
    app['ROC_AUC'] = roc_auc_score(y_test, clf.predict_proba(X_t_test))
    app['P_AUPR'] = avg_prec
    app['Precision'] = prec_score
    app['Recall'] = rec_score
    return app

li = list()

for r in range(100):
    d = dict()
    random.shuffle(l1)
    d['o1'] = l1[:65]
    d['o2'] = l1[65:107]
    d['o3'] = l1[107:144]
    d['o4'] = l1[144:228]
    d['o5'] = l1[228:238]
    d['o6'] = l1[238:436]
    d['o7'] = l1[436:818]
    d['o8'] = l1[818:938]
    d['o9'] = l1[938:955]
    d['o10'] = l1[955:962]
    d['o11'] = l1[962:988]
    d['o12'] = l1[988:998]
    d['o13'] = l1[998:1004]
    d['o14'] = l1[1004:1044]
    d['o15'] = l1[1044:1087]
    d['o16'] = l1[1087:1101]
    d['o17'] = l1[1101:1137]
    d['o18'] = l1[1137:1169]
    d['o19'] = l1[1169:1253]
    d['o20'] = l1[1253:1296]
    d['o21'] = l1[1296:1316]
    d['o22'] = l1[1316:1413]
    d['o23'] = l1[1413:1435]
    d['o24'] = l1[1435:1471]
    d['o25'] = l1[1471:1496]
    d['o26'] = l1[1496:1553]
    d['o27'] = l1[1553:1574]
    d['o28'] = l1[1574:1590]
    d['o29'] = l1[1590:1604]
    d['o30'] = l1[1604:1612]
    d['o31'] = l1[1612:1727]
    d['o32'] = l1[1727:1779]
    d['o33'] = l1[1779:1888]
    d['o34'] = l1[1888:1903]
    d['o35'] = l1[1903:2055]
    d['o36'] = l1[2055:2115]
    d['o37'] = l1[2115:2399]
    d['o38'] = l1[2399:2442]
    d['o39'] = l1[2442:2466]
    d['o40'] = l1[2466:2507]
    d['o41'] = l1[2507:2615]
    d['o42'] = l1[2615:2644]
    d['o43'] = l1[2644:2711]
    d['o44'] = l1[2711:2815]
    d['o45'] = l1[2815:2863]
    d['o46'] = l1[2863:2887]
    d['o47'] = l1[2887:2903]
    d['o48'] = l1[2903:2913]
    d['o49'] = l1[2913:2963]
    d['o50'] = l1[2963:3012]
    d['o51'] = l1[3012:3093]
    d['o52'] = l1[3093:3116]
    d['o53'] = l1[3116:3154]
    d['o54'] = l1[3154:3213]
    d['o55'] = l1[3213:3236]
    d['o56'] = l1[3236:3277]
    d['o57'] = l1[3277:3302]
    d['o58'] = l1[3302:3598]
    d['o59'] = l1[3598:3613]
    d['o60'] = l1[3613:3635]
    d['o61'] = l1[3635:3652]
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

pickle.dump(li, open("../data/list_res_organ.sav",'wb'))
