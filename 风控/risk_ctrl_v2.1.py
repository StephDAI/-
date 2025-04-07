import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("fk_cs.csv")
df = df.drop(columns=df.columns[0])
x = df.drop(columns=["act_idn_sky","flag_60","cdzbxyfg","product_type_new2","wkibusty_num","agr_mortgage","agr_subsidy"])  #特征变量
y = df["flag_60"]

#把Y和N转化为1和0
x.replace('Y',1, inplace=True)
x.replace('N',0,inplace=True)

#填补空值
#分类问题
classify = ["cred24yq3qk","cred24yq6qk","cd24ljyq6q","cd24lxyq3q","peredulv","perconty","clientchr","hvhsid","hvlivepraveflg","ctirecord_bad_if","ctiself_if","ctiself_num","wcicartty","car_model_id","posttitle","factaid"]
for trait in classify:
    x[trait].fillna(x[trait].mode().iloc[0],inplace=True)

#日期问题
def parse_date(date, mode):
    refer_year = 2010
    refer_month = 1
    if pd.isnull(date): return False
    if mode==1:
        year = int(str(date).split('.')[0])
        month = int(str(date).split('.')[1].lstrip('0'))
    elif mode == 2:
        year = date//100
        month = date%100       
    if year >= refer_year:
        gap = month-1+(year-refer_year)*12
    elif year < refer_year:
        gap = (-1)*((refer_year-1-year)*12+13-month)
    return gap

from dateutil import parser
refer = parser.parse("20190101")
date_col = ["cdzbzzrq","cdzbzzkh"]
for i in range(len(x)): 
    parsed_date = parser.parse(str(int(x.loc[i]["applydate"])))
    num = (parsed_date-refer).days
    x.loc[i,"applydate"] = num
    for col in date_col:
        try:
            if type(parse_date(x.loc[i][col],1))==int :
                x.loc[i,col] = parse_date(x.loc[i][col],1)
        except:
            if type(parse_date(x.loc[i][col],2)==int) :
                x.loc[i,col] = parse_date(x.loc[i][col],2)


#剩下
remaining = []
for col in x.columns.to_list():
    for data in x[col]:
        if pd.isnull(data):
            remaining.append(col)
            break
for trait in remaining:
    x[trait].fillna(x[trait].mean(),inplace=True)

# import numpy as np
# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp.fit(x)
# X=imp.transform(x)

#归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

#SMOTE
from sklearn.neighbors import NearestNeighbors
def SMOTE(X,Y,N,k=5):
    x_minor = X[Y==1]
    N_per_sample = N//len(x_minor)
    k = min(k,len(x_minor)-1)
    synthetic_samples = []
    synthetic_labels = []
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(x_minor)
    for sample in x_minor:
        _ , indices = knn.kneighbors(sample.reshape(1,-1),k)
        for _ in range(N_per_sample):
            index = np.random.choice(indices[0])
            neighbor = x_minor[index]
            difference = neighbor - sample
            co = np.random.random()
            synthetic_samples.append(sample + co * difference)
            synthetic_labels.append(1)
    x_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)
    x_balanced = np.concatenate([X,x_synthetic],axis=0)
    y_balanced = np.concatenate([Y,y_synthetic],axis=0)
    return x_balanced,y_balanced





#划分训练集与测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
x_bal,y_bal = SMOTE(x_train.to_numpy(),y_train.to_numpy(),40000)
#随机森林建模
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(max_features=23)
# params = {'max_features':[23,24,25]}
# model = GridSearchCV(model,params,scoring='f1',verbose=2)
model.fit(x_bal,y_bal)

from sklearn import metrics

y_pred = model.predict(x_test)

l = list(y_test)
#print(model.best_params_)
print("F1:",metrics.f1_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

# 计算AUC
auc = metrics.auc(fpr, tpr)
print("AUC:",auc)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
# importance = model.feature_importances_.astype(float)
# compare = [*zip(x.columns,importance)]
# compare = sorted(compare,key=lambda x:abs(x[1]),reverse=True)
# for i in range(15):
#     print(compare[i][0])
import matplotlib.pyplot as plt
def plot_AUC(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
 
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
plot_AUC(model,x_test,y_test)
