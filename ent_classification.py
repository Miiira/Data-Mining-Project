import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report, auc

from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sns.set(style='white', context='notebook', palette='deep')


# load data from purchase history form
colnames = ['client_no', 'prd_code', 'qrje', 'zfy', 'gmcs', 'trade_date', 'pro_name', 'pro_type']
gmls = pd.read_table("~/Desktop/POCData/20160731/g_customer_gmls_20171109.txt", 
                     header = None, names = colnames, dtype={"client_no": np.str},)

gmls.head()

gmls.isnull().sum()/len(gmls)

# load data from enterprise information form
ent_cols = ['cust_no', 'industry_type', 'rcurrency', 'register_capital', 'scope_new', 'employer_num', 'newtech_corpornot',
           'listing_corpornot', 'other_credit_level', 'mainprod_percent', 'tax_paying', 'std_cert_no_1', 'std_cert_no_2', 
           'core_cust_no', 'etl_date', 'register_capital_rmb']
ent_info = pd.read_table("~/Desktop/POCData/20160731/g_customer_ent_info_20171108.txt", header = None, 
                         names = ent_cols,dtype = {'cust_no': np.str, 'employer_num':np.int,'core_cust_no': np.str})

ent_info.head()

ent_info.isnull().sum() / len(ent_info)
# "mainprod_percent" feature contains unique value 0
ent_info.mainprod_percent.unique()
# "etl_date" feature contains unique value 2016-09-30
ent_info.etl_date.unique()
# drop low quality features
ent_info = ent_info.drop(labels = ['tax_paying', 'register_capital', 'mainprod_percent', 'etl_date'], axis = 1)

# load data from enterprise asset form
ent_asset_cols = ['customerid',
'guaranteesum',
'assuresum',
'bailratio',
'eachsum',
'guarantyrate',
'evaluatevalue',
'allvalue',
'excessasset',
'excessprofit',
'guarantysum']
ent_asset = pd.read_table("~/Desktop/POCData/20160731/g_customer_ent_assets_info_20171108.txt", 
                          header = None, names = ent_asset_cols)

ent_asset.head()

ent_asset.isnull().sum() / len(ent_asset)
# some features contains a large percentage of value 0
sum(ent_asset.eachsum.notnull() & ent_asset.eachsum != 0)
sum(ent_asset.evaluatevalue.notnull() & ent_asset.evaluatevalue != 0)
sum(ent_asset.allvalue.notnull() & ent_asset.allvalue != 0)
sum(ent_asset.excessasset.notnull() & ent_asset.excessasset != 0)
sum(ent_asset.excessprofit.notnull() & ent_asset.excessprofit != 0)
# drop low quality features
ent_asset = ent_asset.drop(['guaranteesum', 'assuresum', 'bailratio', 'eachsum', 'evaluatevalue', 'allvalue', 'excessasset', 'excessprofit'], 1)

# load data from enterprise loan form
ent_loan_act_cols = ['dk60btno',
'dk60date',
'cust_no',
'dk60amntykAvg30',
'dk60amntykMin30',
'dk60amntykMax30',
'dk60amntdkAvg30',
'dk60amntdkMin30',
'dk60amntdkMax30',
'dk60amtdfkAvg30',
'dk60amtdfkMin30',
'dk60amtdfkMax30',
'dk60amntbkAvg30',
'dk60amntbkMin30',
'dk60amntbkMax30',
'dk60amntykAvg15',
'dk60amntykMin15',
'dk60amntykMax15',
'dk60amntdkAvg15',
'dk60amntdkMin15',
'dk60amntdkMax15',
'dk60amtdfkAvg15',
'dk60amtdfkMin15',
'dk60amtdfkMax15',
'dk60amntbkAvg15',
'dk60amntbkMin15',
'dk60amntbkMax15',
'dk60amntykAvg5',
'dk60amntykMin5',
'dk60amntykMax5',
'dk60amntdkAvg5',
'dk60amntdkMin5',
'dk60amntdkMax5',
'dk60amtdfkAvg5',
'dk60amtdfkMin5',
'dk60amtdfkMax5',
'dk60amntbkAvg5',
'dk60amntbkMin5',
'dk60amntbkMax5']
ent_loan_act = pd.read_table("~/Desktop/POCData/20160731/g_customer_ent_loan_act_20171108.txt", 
                             header = None, names = ent_loan_act_cols, dtype = {'cust_no': np.str})

ent_loan_act.head()

nt_loan_act.isnull().sum() / len(ent_loan_act)

# identity some duplicated features
sum(ent_loan_act.dk60amntykMin30 == ent_loan_act.dk60amntykMax30) / len(ent_loan_act)
sum(ent_loan_act.dk60amntdkMin30 == ent_loan_act.dk60amntdkMax30) / len(ent_loan_act)
sum(ent_loan_act.dk60amtdfkMin30 == ent_loan_act.dk60amtdfkMax30) / len(ent_loan_act)
sum(ent_loan_act.dk60amntbkMin30 == ent_loan_act.dk60amntbkMax30) / len(ent_loan_act)

# drop low quality features
ent_loan_act = ent_loan_act.drop(labels = ['dk60amntykAvg5','dk60amntykMin5','dk60amntykMax5','dk60amntdkAvg5','dk60amntdkMin5','dk60amntdkMax5','dk60amtdfkAvg5','dk60amtdfkMin5','dk60amtdfkMax5','dk60amntbkAvg5','dk60amntbkMin5','dk60amntbkMax5',
'dk60amntykMin30','dk60amntdkMin30','dk60amtdfkMin30','dk60amntbkMin30','dk60amntykMin15','dk60amntdkMin15','dk60amtdfkMin15','dk60amntbkMin15'], axis = 1)

# join forms
indexed_asset = ent_asset.set_index('customerid')
asset_info = pd.merge(ent_info, indexed_asset, left_on='cust_no', right_index=True)

indexed_gmls = gmls.set_index('client_no')
ent_asset_gmls = pd.merge(asset_info, indexed_gmls, how = 'left', left_on='core_cust_no', right_index=True)

ent_summary = pd.merge(ent_asset_gmls, ent_loan_act, how = 'left', on = 'cust_no')
ent_summary.head()

len(ent_summary)

# construct features
ent_summary['purchase'] = ent_summary['pro_type'].map(lambda s: 0 if pd.isnull(s) else 1)
ent_summary['has_loan'] = ent_summary['dk60date'].map(lambda s: 0 if pd.isnull(s) else 1)

# drop unrelated features
ent_summary = ent_summary.drop(['pro_type','dk60btno','dk60date','prd_code', 'qrje','zfy', 'gmcs', 'trade_date', 'pro_name'], axis=1)

ent_summary.isnull().sum() / len(ent_summary)

ent_summary.shape

ent_summary.industry_type.value_counts().sort_values().describe()

# delete outliers
numeric_features = ['dk60amntykAvg30',
'dk60amntykMax30',
'dk60amntdkAvg30',
'dk60amntdkMax30',
'dk60amtdfkAvg30',
'dk60amtdfkMax30',
'dk60amntbkAvg30',
'dk60amntbkMax30',
'dk60amntykAvg15',
'dk60amntykMax15',
'dk60amntdkAvg15',
'dk60amntdkMax15',
'dk60amtdfkAvg15',
'dk60amtdfkMax15',
'dk60amntbkAvg15',
'dk60amntbkMax15',
'employer_num',
'register_capital_rmb',
'guarantysum',
'guarantyrate']

for col in numeric_features:
    percent99 = np.percentile(ent_summary[col][ent_summary[col].notnull()], 99)
    indices_to_drop = ent_summary[ent_summary[col] > percent99].index
    ent_summary = ent_summary.drop(indices_to_drop, axis = 0).reset_index(drop=True)

ent_summary.shape

ent_summary.isnull().sum()

# Analyze features

# has_credit
ent_summary['has_credit'] = ent_summary['other_credit_level'].map(lambda s: 0 if pd.isnull(s) else 1)

g = sns.factorplot(x="has_credit",y="purchase",data=ent_summary,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("purchase")

# guarantysum
g = sns.kdeplot(ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())], color="Red", shade = True)

g = sns.distplot(ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())].skew()))
g = g.legend(loc="best")

# do log transformation
ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())] = ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["guarantysum"][(ent_summary["guarantysum"].notnull())].skew()))
g = g.legend(loc="best")

g = sns.kdeplot(ent_summary["guarantysum"][(ent_summary["purchase"] == 0) & (ent_summary["guarantysum"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["guarantysum"][(ent_summary["purchase"] == 1) & (ent_summary["guarantysum"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("guarantysum")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["guarantysum"][(ent_summary["purchase"] == 0) & (ent_summary["guarantysum"].notnull())].describe()

ent_summary["guarantysum"][(ent_summary["purchase"] == 1) & (ent_summary["guarantysum"].notnull())].describe()

# guarantyrate
g = sns.kdeplot(ent_summary["guarantyrate"][(ent_summary["guarantyrate"].notnull())])

g = sns.distplot(ent_summary["guarantyrate"][(ent_summary["guarantyrate"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["guarantyrate"][(ent_summary["guarantyrate"].notnull())].skew()))
g = g.legend(loc="best")

g = sns.kdeplot(ent_summary["guarantyrate"][(ent_summary["purchase"] == 0) & (ent_summary["guarantyrate"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["guarantyrate"][(ent_summary["purchase"] == 1) & (ent_summary["guarantyrate"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("guarantyrate")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["guarantyrate"][(ent_summary["purchase"] == 0) & (ent_summary["guarantyrate"].notnull())].describe()
ent_summary["guarantyrate"][(ent_summary["purchase"] == 1) & (ent_summary["guarantyrate"].notnull())].describe()

# employer_num
g = sns.kdeplot(ent_summary["employer_num"][(ent_summary["employer_num"].notnull())], color="Red", shade = True)

g = sns.distplot(ent_summary["employer_num"][(ent_summary["employer_num"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["employer_num"][(ent_summary["employer_num"].notnull())].skew()))
g = g.legend(loc="best")

# do log transformation
ent_summary["employer_num"][(ent_summary["employer_num"].notnull())] = ent_summary["employer_num"][(ent_summary["employer_num"].notnull())].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(ent_summary["employer_num"][(ent_summary["employer_num"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["employer_num"][(ent_summary["employer_num"].notnull())].skew()))
g = g.legend(loc="best")

g = sns.kdeplot(ent_summary["employer_num"][(ent_summary["purchase"] == 0) & (ent_summary["employer_num"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["employer_num"][(ent_summary["purchase"] == 1) & (ent_summary["employer_num"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("employer_num")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["employer_num"][(ent_summary["purchase"] == 0) & (ent_summary["employer_num"].notnull())].describe()

ent_summary["employer_num"][(ent_summary["purchase"] == 1) & (ent_summary["employer_num"].notnull())].describe()

# register_capital_rmb
g = sns.kdeplot(ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())], color="Red", shade = True)

g = sns.distplot(ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())].skew()))
g = g.legend(loc="best")

# do log transformation
ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())] = ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())], color="m", label="Skewness : %.2f"%(ent_summary["register_capital_rmb"][(ent_summary["register_capital_rmb"].notnull())].skew()))
g = g.legend(loc="best")

g = sns.kdeplot(ent_summary["register_capital_rmb"][(ent_summary["purchase"] == 0) & (ent_summary["register_capital_rmb"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["register_capital_rmb"][(ent_summary["purchase"] == 1) & (ent_summary["register_capital_rmb"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("register_capital_rmb")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["register_capital_rmb"][(ent_summary["purchase"] == 0) & (ent_summary["register_capital_rmb"].notnull())].describe()

ent_summary["register_capital_rmb"][(ent_summary["purchase"] == 1) & (ent_summary["register_capital_rmb"].notnull())].describe()

# dk60amntykMax30
g = sns.kdeplot(ent_summary["dk60amntykMax30"][(ent_summary["dk60amntykMax30"].notnull())], color="Red", shade = True)

sum(ent_summary["dk60amntykMax30"].notnull())

sum(ent_summary["dk60amtdfkMax30"].notnull() & ent_summary["dk60amtdfkMax30"]!=0)

g = sns.kdeplot(ent_summary["dk60amtdfkMax30"][(ent_summary["purchase"] == 0) & (ent_summary["dk60amtdfkMax30"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["dk60amtdfkMax30"][(ent_summary["purchase"] == 1) & (ent_summary["dk60amtdfkMax30"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("dk60amtdfkMax30")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["dk60amtdfkMax30"][(ent_summary["purchase"] == 0) & (ent_summary["dk60amtdfkMax30"].notnull())].describe()

ent_summary["dk60amtdfkMax30"][(ent_summary["purchase"] == 1) & (ent_summary["dk60amtdfkMax30"].notnull())].describe()

# confirm that the distribution of some features are skewed
# do log transformation
to_log = ['dk60amntykAvg30', 'dk60amntykMax30', 'dk60amtdfkAvg30', 'dk60amtdfkMax30', 'dk60amntdkAvg30', 'dk60amntdkMax30',
         'dk60amntbkAvg30', 'dk60amntbkMax30', 'dk60amntykAvg15', 'dk60amntykMax15', 'dk60amtdfkAvg15', 'dk60amtdfkMax15', 
          'dk60amntdkAvg15', 'dk60amntdkMax15','dk60amntbkAvg15', 'dk60amntbkMax15']
for col in to_log:
    ent_summary[col][(ent_summary[col].notnull())] = ent_summary[col][(ent_summary[col].notnull())].map(lambda i: np.log(i) if i > 0 else 0)

# drop features that do not have obvious disparity in two classes
ent_summary = ent_summary.drop(['has_loan', 'newtech_corpornot'], 1)

# construct new feature
ent_summary.loc[ent_summary['other_credit_level'] == 'C', 'other_credit_level'] = 0
ent_summary.loc[ent_summary['other_credit_level'] == 'CC', 'other_credit_level'] = 1
ent_summary.loc[ent_summary['other_credit_level'] == 'CCC', 'other_credit_level'] = 2
ent_summary.loc[ent_summary['other_credit_level'] == 'B', 'other_credit_level'] = 3
ent_summary.loc[ent_summary['other_credit_level'] == 'BB', 'other_credit_level'] = 4
ent_summary.loc[ent_summary['other_credit_level'] == 'BBB', 'other_credit_level'] = 5
ent_summary.loc[ent_summary['other_credit_level'] == 'A', 'other_credit_level'] = 6
ent_summary.loc[ent_summary['other_credit_level'] == 'AA', 'other_credit_level'] = 7
ent_summary.loc[ent_summary['other_credit_level'] == 'AAA', 'other_credit_level'] = 8
ent_summary['A_credit'] = ent_summary['other_credit_level'].map(lambda s: 1 if (5 < s) else 0)

g = sns.factorplot(x="A_credit",y="purchase",data=ent_summary,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("purchase")

# feature transformation
ent_summary.scope_new = ent_summary.scope_new - 3

ent_summary.loc[ent_summary['rcurrency'] != 1, 'rcurrency'] = 0

ent_summary.loc[ent_summary['listing_corpornot'] > 1, 'listing_corpornot'] = 1

# rcurrency
g = sns.factorplot(x="rcurrency",y="purchase",data=ent_summary,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("purchase")

# listing_corpornot
g = sns.factorplot(x="listing_corpornot",y="purchase",data=ent_summary,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("purchase")

# scope
g = sns.factorplot(x="scope_new",y="purchase",data=ent_summary,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("purchase")

ent_summary.isnull().sum()

# fill missing values
ent_summary['listing_corpornot'] = ent_summary['listing_corpornot'].fillna(ent_summary['listing_corpornot'].mode()[0])

ent_summary['guarantyrate'] = ent_summary['guarantyrate'].fillna(ent_summary['guarantyrate'].median())

ent_summary['guarantysum'] = ent_summary['guarantysum'].fillna(ent_summary['guarantysum'].median())

for col in to_log:
    ent_summary[col] = ent_summary[col].fillna(0)

# age
ent_summary["age"] = pd.Series([i[6:10] for i in ent_summary["std_cert_no_1"] ])
ent_summary["age"] = 2017 - pd.to_numeric(ent_summary['age'], errors='coerce')
# fill wrong age with NA
ent_summary['age'] = [np.nan if (i < 0) | (i > 100) else i for i in ent_summary['age']]
# fill missing value with median
ent_summary['age'] = ent_summary['age'].fillna(ent_summary['age'].median())

g = sns.kdeplot(ent_summary["age"][ent_summary.age.notnull()], color="Red", shade = True)

g = sns.kdeplot(ent_summary["age"][(ent_summary["purchase"] == 0) & (ent_summary["age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(ent_summary["age"][(ent_summary["purchase"] == 1) & (ent_summary["age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("age")
g.set_ylabel("Frequency")
g = g.legend(["Not purchase","purchase"])

ent_summary["age"][(ent_summary["purchase"] == 0) & (ent_summary["age"].notnull())].describe()

ent_summary["age"][(ent_summary["purchase"] == 1) & (ent_summary["age"].notnull())].describe()

# regularization
def zscore(col):
    return [(x - col.mean()) / col.std() for x in col]

to_zscore = ['employer_num',
'register_capital_rmb',
'guarantysum',
'guarantyrate',
'age',
'dk60amntykAvg30',           
'dk60amntykMax30',           
'dk60amtdfkAvg30',           
'dk60amtdfkMax30',           
'dk60amntdkAvg30',           
'dk60amntdkMax30',
'dk60amntykAvg15',             
'dk60amntykMax15',             
'dk60amntdkAvg15',             
'dk60amntdkMax15',             
'dk60amtdfkAvg15',             
'dk60amtdfkMax15']  

for col in to_zscore:
    ent_summary[col] = zscore(ent_summary[col])

# drop columns that contain unique value 0
ent_summary = ent_summary.drop(['dk60amntbkMax30','dk60amntbkMax15','dk60amntbkAvg30', 'dk60amntbkAvg15'], 1)

ent_summary.isnull().sum()

ent_summary.describe()

# summary of data

# drop unrelated or processed features
ent_summary = ent_summary.drop(['other_credit_level', 'cust_no', 'core_cust_no', 'std_cert_no_1', 'std_cert_no_2','industry_type'], 1)

ent_summary.shape

# review class imbalance problem
print(ent_summary.groupby('purchase').size())

sns.countplot(x="purchase", data=ent_summary)

# confirm data types
ent_summary.dtypes

#Correlation map to see how features are correlated
corrmat = ent_summary.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

ent_summary.head()

# feature selection with Random Forest
y = ent_summary["purchase"]
X = ent_summary.drop(labels = ["purchase"],axis = 1)

rf = RandomForestClassifier(random_state=2).fit(X_train, y_train)

cols = ent_summary.drop(['purchase'], 1).columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'feature importances': rf.feature_importances_
    })

feature_dataframe = feature_dataframe.sort_values(by=['feature importances'], ascending=False)

g = sns.factorplot(x="feature importances",y="features",data=feature_dataframe,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)

feature_dataframe

# drop unselected features
ent_summary = ent_summary.drop(['dk60amntykAvg15','dk60amntykMax30','dk60amntykMax15','dk60amntykAvg30'],1)


# Train
y = ent_summary["purchase"]
X = ent_summary.drop(labels = ["purchase"],axis = 1)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(fbeta_score(true_value, pred, beta=1)))

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.20)

def evaluatemodel(classifier, name):
    # build model with SMOTE imblearn
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=6), \
                                   classifier)

    smote_model = smote_pipeline.fit(X_train, y_train)
    smote_prediction = smote_model.predict(X_test)

    print("normal data distribution: {}".format(Counter(y)))
    X_smote, y_smote = SMOTE().fit_sample(X, y)
    print("SMOTE data distribution: {}".format(Counter(y_smote)))
    
    print("Confusion Matrix: ")
    #print(confusion_matrix(y_test, smote_prediction))
    plot_confusion_matrix(confusion_matrix(y_test, smote_prediction))

    print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

    print_results("\nSMOTE + " + name + " classification", y_test, smote_prediction)


def ROC(classifer, name):
    # build model with SMOTE imblearn
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=6), \
                                   classifier)

    smote_model = smote_pipeline.fit(X_train, y_train)
    smote_prediction = smote_model.predict(X_test) 
    X_smote, y_smote = SMOTE().fit_sample(X, y)
    print_results("\nSMOTE + " + name + " classification")

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = smote_pipeline.predict_proba(X_test)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    print('AUC:', auc(fpr, tpr))

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# Random Forest
evaluatemodel(RandomForestClassifier(random_state=2), "RandomForest")

# AdaBoost
evaluatemodel(AdaBoostClassifier(DecisionTreeClassifier(random_state=2),random_state=2,learning_rate=0.1), "AdaBoost")

# Extra Trees
evaluatemodel(ExtraTreesClassifier(random_state=2), "ExtraTrees")

# Gradient Boost
evaluatemodel(GradientBoostingClassifier(random_state=2), "GradientBoost")

# Logistic Regression
evaluatemodel(LogisticRegression(random_state = 2), "LogisticRegression")

# SVM
evaluatemodel(SVC(random_state=2), "SVM")

# Decision Tree
evaluatemodel(DecisionTreeClassifier(random_state=2), "DecisionTree")

# Neural Network
evaluatemodel(MLPClassifier(random_state=2), "MLP")

# Random Forest
def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f2: {}".format(fbeta_score(true_value, pred, beta=2)))

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.20)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=6), RandomForestClassifier(random_state=2))

smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

print("normal data distribution: {}".format(Counter(y)))
X_smote, y_smote = SMOTE().fit_sample(X, y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))
    
print("Confusion Matrix: ")
print(confusion_matrix(y_test, smote_prediction))
#plot_confusion_matrix(confusion_matrix(y_test, smote_prediction))

print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

print_results("\nSMOTE + RandomForest classification", y_test, smote_prediction)
    
# Compute predicted probabilities: y_pred_prob
y_pred_prob = smote_pipeline.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print('AUC:', auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Cross Validation
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# Hyperparameter Tuning
# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[100,200],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="f1", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,y_train)

print(gsadaDTC.best_params_)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_

#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [0.2, 0.5],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ['gini', 'entropy']}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="f1", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,y_train)

ExtC_best = gsExtC.best_estimator_

print(gsExtC.best_params_)

# Best score
gsExtC.best_score_

# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [0.2, 0.5],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ['gini', 'entropy']}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="f1", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_

print(gsRFC.best_params_)

# Best score
gsRFC.best_score_


# Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves", X_train,y_train,(0.7, 1.01),cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,y_train,(0.7, 1.01),cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,y_train,(0.7, 1.01),cv=kfold)


# Ensemble Modeling
random_state = 2
classifiers = []
gbc = GradientBoostingClassifier(random_state=random_state)
classifiers.append(gbc)
lr = LogisticRegression(random_state = random_state)
classifiers.append(lr)
svc = SVC(random_state=random_state)
classifiers.append(svc)
dt = DecisionTreeClassifier(random_state=random_state)
classifiers.append(dt)
mlp = MLPClassifier(random_state=random_state)
classifiers.append(mlp)
for classifier in classifiers:
    classifier.fit(X_train, y_train)


test_purchased_GBV = pd.Series(gbc.predict(X_test), name="GBC")
test_purchased_LR = pd.Series(lr.predict(X_test), name="LR")
test_purchased_SVC = pd.Series(svc.predict(X_test), name="SVC")
test_purchased_DT = pd.Series(dt.predict(X_test), name="DT")
test_purchased_MLP = pd.Series(mlp.predict(X_test), name="MLP")

test_purchased_RFC = pd.Series(RFC_best.predict(X_test), name="RFC")
test_purchased_ExtC = pd.Series(ExtC_best.predict(X_test), name="ExtC")
test_purchased_AdaC = pd.Series(ada_best.predict(X_test), name="Ada")

# Concatenate all classifier results
ensemble_results = pd.concat([test_purchased_RFC,test_purchased_ExtC,test_purchased_AdaC, test_purchased_GBV, test_purchased_LR, test_purchased_SVC, test_purchased_DT, test_purchased_MLP],axis=1)

# correlation heatmap of all basic classifiers
g= sns.heatmap(ensemble_results.corr(),annot=True)


# Stacking
train = pd.concat([X_train, y_train], 1)
test = pd.concat([X_test, y_test], 1)

from sklearn.cross_validation import KFold

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, model_name, seed=0, params=None):
        if seed != None:
            params['random_state'] = seed
        self.clf = clf(**params)
        self.name = model_name

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def score(self, x, y):
        return self.clf.score(x, y)
    
    def fit(self,x,y):
        try:
            return self.clf.fit(x,y)
        except AttributeError:
            return self.clf.train(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
    def model_name(self):
        return self.name


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Another version of get_oof
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    train_accuracy = 0
    test_accuracy = 0

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:, 0]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 0]
        train_accuracy += clf.score(x_tr, y_tr)
        test_accuracy += clf.score(x_te, y_te)
    
    train_accuracy = train_accuracy/len(kf)
    test_accuracy = test_accuracy/len(kf)
    print('train accuracy: '%(clf.model_name(), train_accuracy))
    print('test accuracy: '%(clf.model_name(), test_accuracy))
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# best parameters
rf_params = {
    'bootstrap': False, 
    'criterion': 'gini',
    'max_depth': None, 
    'max_features': 3, 
    'min_samples_leaf': 1, 
    'min_samples_split': 3, 
    'n_estimators': 100,
    'n_jobs': -1,
    'warm_start': True, 
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'bootstrap': False, 
    'criterion': 'gini', 
    'max_depth': None, 
    'max_features': 1, 
    'min_samples_leaf': 1, 
    'min_samples_split': 3,
    'n_estimators': 300,
    'n_jobs': -1,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'algorithm': 'SAMME.R', 
    'learning_rate': 0.3, 
    'n_estimators': 2,
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 3,
    'subsample':0.5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'poly',
    'C' : 0.025 ,
    'probability' : True
    }

gbm_params = {
    'learning_rate' : 0.4,
    'n_estimators' : 500,
    'max_depth' : 4,
    'min_child_weight': 2,
    #gamma=1,
    'gamma':0.9,                        
    'subsample':0.5,
    'colsample_bytree':0.8,
    'objective': 'binary:logistic',
    'reg_lambda':5,
    'nthread':-1,
    'scale_pos_weight':1
}

rf = SklearnHelper(RandomForestClassifier, 'RandomForest', seed=SEED, params=rf_params) # 
et = SklearnHelper(ExtraTreesClassifier, 'ExtraTrees',seed=SEED, params=et_params)
ada = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada_params)
gb = SklearnHelper(GradientBoostingClassifier, 'GradientBoosting', seed=SEED, params=gb_params)
svc = SklearnHelper(SVC, 'SVM',seed=SEED, params=svc_params)
gbm = SklearnHelper(xgb.XGBClassifier, 'XGB', seed=SEED, params=gbm_params)

try:
    y_train = train['purchase'].ravel()
    train = train.drop(['purchase'], axis=1)
    y_test = test['purchase'].ravel()
    test = test.drop(['purchase'], axis=1)
except KeyError:
    print('no need')
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
print("Training is complete")

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
print("Training is complete")

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
print("Training is complete")

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
print("Training is complete")

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
print("Training is complete")

gbm_oof_train, gbm_oof_test = get_oof(gbm, x_train, y_train, x_test) #XGBoost
print("Training is complete")

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
     'XGBoost': gbm_oof_train.ravel()
    })
base_predictions_train.head()

# concatenate dataframes
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, gbm_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, gbm_oof_test), axis=1)

# stacking with XGBoost
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
#predictions = gbm.predict(x_test)

# Cross validation score
cross_val_score(gbm, x_train, y = y_train, scoring = "f1", cv = kfold, n_jobs=4).mean()
predictions = gbm.predict(x_test)
plot_confusion_matrix(confusion_matrix(y_test, predictions))
print_results("\nSMOTE + Second Level XGBoost classification", y_test, predictions)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=5), gbm)

smote_model = smote_pipeline.fit(x_train, y_train)
smote_prediction = smote_model.predict(x_test)

print("normal data distribution: {}".format(Counter(y)))
X_smote, y_smote = SMOTE().fit_sample(X, y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))
    
print("Confusion Matrix: ")
#print(confusion_matrix(y_test, smote_prediction))
plot_confusion_matrix(confusion_matrix(y_test, smote_prediction))

print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(x_test, y_test)))

print_results("\nSMOTE + Second Level XGBoost classification", y_test, smote_prediction)
    
# Compute predicted probabilities: y_pred_prob
y_pred_prob = smote_pipeline.predict_proba(x_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print('AUC:', auc(fpr, tpr))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Voting
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best), ('ada', ada_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)

cvs = cross_val_score(votingC, X_train, y = y_train, scoring = "f1", cv = kfold, n_jobs=4)
cvs.mean()


# Feature Importance

# best parameter of RF
rfc = RandomForestClassifier(random_state=2, bootstrap=False, criterion='gini', max_depth=None, max_features=0.2, 
                             min_samples_leaf=1, min_samples_split=3, n_estimators=300).fit(X_train, y_train)

cols = ent_summary.drop(['purchase'], 1).columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'feature importances': rfc.feature_importances_
    })

feature_dataframe = feature_dataframe.sort_values(by=['feature importances'], ascending=False)

g = sns.factorplot(x="feature importances",y="features",data=feature_dataframe,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)

feature_dataframe
