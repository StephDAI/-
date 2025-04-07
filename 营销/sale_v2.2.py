import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("yx_cs.csv")
x = df.drop(columns=["cusno","y"])
y = df["y"]

#字符串数据处理
def string_processing(series):
    dict = {}
    series_cp = series.copy()
    for i in range(len(series)):
        series_cp[i] = dict.setdefault(series[i],len(dict))
    return series_cp

str_col = ["ci03_gov_fund_ind","ci03_nationality_cd","ci07_dd_occu_code2","indv_monthly_income_curr_cd","nation_cd","idcus_info_qly_code_rtg"]
for col in str_col:
    x[col] = string_processing(x[col]).astype('int')


#日期处理
x["ci07_dd_empl_from"].replace(99991231,np.nan,inplace=True)
from dateutil import parser
refer = parser.parse("20240901")
for i in range(len(x)):
    parsed_date = parser.parse(str(int(x.loc[i]["lastest_txn_dt"])))
    num = (parsed_date-refer).days
    x.loc[i,"lastest_txn_dt"] = num
refer = parser.parse("20190101")
for i in range(len(x)):
    parsed_date = parser.parse(str(int(x.loc[i]["open_acct_dt"])))
    num = (parsed_date-refer).days
    x.loc[i,"open_acct_dt"] = num
refer = parser.parse("20100101")
for i in range(len(x)):
    if np.isnan(x.loc[i]["ci07_dd_empl_from"]):continue
    parsed_date = parser.parse(str(int(x.loc[i]["ci07_dd_empl_from"])))
    num = (parsed_date-refer).days
    x.loc[i,"ci07_dd_empl_from"] = num

#数据填充
classifier = ["ci03_gov_fund_ind",
            "ci03_locker_holder",
            "ci03_nationality_cd",
            "ci03_resident_status",
            "ci03_tfn_ind",
            "ci07_dd_boc_cust_year",
            "is_pub_plcmt_fin",
            "ci07_dd_bus_rshp_boc",
            "ci07_dd_empr_inds",
            "ci07_dd_hgst_degree",
            "ci07_dd_law_rec",
            "ci07_dd_marital_status",
            "ci07_dd_nation_code",
            "ci07_dd_occu_code2",
            "ci07_dd_pro_title",
            "ci07_dd_pstn",
            "ci07_dd_rshp_id_pl",
            "ci07_dd_sex_code",
            "ci07_dd_src_cust",
            "medicare_card",
            "prpfx",
            "soldier_card",
            "advance",
            "base",
            "is_china_bank_secu",
            "is_crdt_card_effect_cust",
            "is_ebank",
            "is_indv_bond",
            "is_indv_dpsit",
            "is_indv_fund",
            "is_indv_insure",
            "is_indv_struc_dpsit",
            "is_indv_tpcc",
            "is_mbank",
            "is_mbank_month_txn",
            "buy_twoset_above_hos_flag",
            "cust_crdt_cd",
            "cust_rela_setup_chnl",
            "cust_status_cd",
            "exist_invold_case_crdt_card_flag",
            "indv_monthly_income_curr_cd",
            "indv_monthly_income_range_cd",
            "nation_cd",
            "nation_std_distr_cd",
            "payoff_salary_flag",
            "pr_hoshld_type_cd",
            "resdnt_cotx_flag",
            "resdnt_flag",
            "idcus_info_qly_code_rtg",
            "is_debit_card",
            "is_indv_loan"]
for trait in classifier:
    x[trait].fillna(x[trait].mode().iloc[0],inplace=True)

remaining = []
for col in x.columns.to_list():
    for data in x[col]:
        if pd.isnull(data):
            remaining.append(col)
            break
for trait in remaining:
    x[trait].fillna(x[trait].mean(),inplace=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x.values,x.columns.get_loc(i)) for i in x.columns]

import openpyxl
wb = openpyxl.load_workbook("yx_cs.xlsx")
ws = wb.active
ws.append(vif)
wb.save("yx_cs.xlsx")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#lightGBM
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

model = LGBMClassifier(learning_rate=0.05,n_estimators=500,max_depth=5,num_leaves=31,colsample_bytree=1.0,subsample=0.8)
# params = {'n_estimators':[100,500,750]} 
# model = GridSearchCV(model,params,scoring='f1',verbose=2)
model.fit(x_train,y_train)

from sklearn import metrics

y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)

#print(model.best_params_)
print("F1 for test:",metrics.f1_score(y_test, y_pred))
print("F1 for train:",metrics.f1_score(y_train, y_pred_train))

# 计算AUC

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

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
plot_AUC(model,x_train,y_train)