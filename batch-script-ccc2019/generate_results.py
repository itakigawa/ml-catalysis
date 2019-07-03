
# coding: utf-8

# # batch script for reproducibility

# jupyter nbconvert --to script generate_results.ipynb

# In[1]:


import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap as lsc
from matplotlib import pyplot as plt


# In[2]:


import pandas as pd
import numpy as np
import scipy
import sklearn 
import xgboost

import math
import random 
import re
import itertools

from collections import Counter 

from scipy.stats import binom_test
from scipy.stats import norm

from functools import partial

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge 
from xgboost import XGBRegressor

from ksuzuki_pylib import *


# In[3]:


print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)


# In[4]:


np.random.seed(0)
random.seed(0)


# In[5]:


# xgb ranges
range_depth = [6, 7, 8]
range_subsample = [0.8, 0.9, 1]
range_colsample = [0.8, 0.9, 1]
range_lr = [0.1, 0.05]


# In[6]:


# ranges
logparams1 = [1e-2, 1e-1, 1e0, 1e1, 1e2]
logparams2 = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
logparams3 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
logparams4 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

range_lasso = logparams1
range_ridge = logparams1
range_krr_alpha = logparams3
range_krr_gamma = logparams3
range_svr_C = logparams2
range_svr_gamma = logparams4
range_svr_epsilon = logparams1


# In[7]:


# for debugging
debug_run = False
#debug_run = True
if debug_run:
    # xgb ranges
    range_depth = [3,4]
    range_subsample = [0.9, 1]
    range_colsample = [0.9, 1]
    range_lr = [0.1, 0.05]
    
    # ranges
    logparams1 = [1e-2, 1e-1]
    logparams2 = [1.0, 1e1]
    logparams3 = [1.0, 1e-1]
    logparams4 = [1.0, 1e-1]
    
    range_lasso = logparams1
    range_ridge = logparams1
    range_krr_alpha = logparams3
    range_krr_gamma = logparams3
    range_svr_C = logparams2
    range_svr_gamma = logparams4
    range_svr_epsilon = logparams1


# In[8]:


# OCM
ntree = 500
if debug_run:
    ntree = 10
    
# Lasso
cv_lasso = GridSearchCV(
    Lasso(random_state=929), 
    param_grid={"alpha": range_lasso})

# Ridge
cv_ridge = GridSearchCV(
    Ridge(random_state=929), 
    param_grid={"alpha": range_ridge})

# KRR
cv_krr = GridSearchCV(
    KernelRidge(kernel='rbf'), 
    param_grid={"alpha": range_krr_alpha, 
                "gamma": range_krr_gamma})

# SVR
cv_svr = GridSearchCV(
    SVR(kernel='rbf'), 
    param_grid={"C": range_svr_C, 
                "gamma": range_svr_gamma, 
                "epsilon": range_svr_epsilon})

# XGB
cv_xgb = GridSearchCV(
    XGBRegressor(random_state=929), 
    param_grid={"max_depth": range_depth, 
                "subsample": range_subsample, 
                "colsample_bytree": range_colsample,
                "learning_rate": range_lr})


# ## OCM data

# In[9]:


ocm = pd.read_csv("input/OCM_matrix.csv").drop(['Unnamed: 0'], axis=1)

ocm_desc = pd.read_csv("input/OCM_matrix_desc.csv")
ind = ocm_desc['Unnamed: 0']
ocm_desc.index = ind
ocm_desc = ocm_desc.drop(['Unnamed: 0'], axis=1)

#exclude cat which has 'Th' in its compser
ind = ocm['Th'] == 0
ocm = ocm[ind]

#exclude data which has over 5 metals in his component
ind = (ocm.loc[:,:'Zr']>0).sum(axis=1)
ind = ind < 5
ocm = ocm.loc[ind]

ind = (ocm_desc.loc[:,:'Zr']>0).sum(axis=1)
ind = ind < 5
ocm_desc = ocm_desc.loc[ind]

ocm = ocm.drop(['Support_Co'],axis=1)
ocm_desc = ocm_desc.drop(['Support_Co'],axis=1)

# rename columns
sup = ocm.loc[:,'Support_Zr':'Support_Si'].columns
prom = ocm.loc[:,'Promotor_B':'Promotor_S'].columns
comp = ocm.loc[:,:'Zr'].columns
cond = ocm.loc[:,'Temperature, K':'P total, bar'].columns
prep = ocm.loc[:,'Impregnation':'Therm.decomp.'].columns

#exclude experimental condition feature
nocond = list(comp) + list(sup)
ocm_nocond = ocm[nocond]
ocm_nocond = pd.concat([ocm_nocond,ocm.iloc[:,-1]], axis=1)


# In[10]:


desc = pd.read_csv('input/Descriptors.csv',skiprows = [0],index_col='symbol').drop(['Unnamed: 0',
                                                                               'name',
                                                                               'ionic radius',
                                                                               'covalent radius',
                                                                               'VdW radius',
                                                                               'crystal radius',
                                                                               'a x 106 ',
                                                                               'Heat capacity ',
                                                                               'l',
                                                                               'electron affinity ',
                                                                               'VE',
                                                                               'Surface energy '],axis=1)
desc = desc.loc[ocm.loc[:,:'Zr'].columns,]
desc = desc.fillna(desc.mean())


# In[11]:


# num of elements in each multicomponent catalyst
print(Counter((ocm.loc[:,:'Zr']>0).sum(axis=1)))


# ## Fig.5

# In[12]:


def scatter(pred,exp,xlabel='Yield_ex(%)',ylabel='Yield_pr(%)',Title='',col='blue',wt_value=True,save=False):
    range = [ocm.iloc[:,-1].min() ,ocm.iloc[:,-1].max() ]
    val0 = (pred - exp)**2
    str0 = 'RMSE: %1.3f' % math.sqrt(val0.mean())
    plt.figure(figsize=(3, 3), dpi=100)
    plt.plot(range, range, color='0.5')
    plt.scatter(exp,pred, s=10,
                facecolors=col, edgecolors=col, alpha=0.4)
    plt.xlim(range[0], range[1])
    plt.ylim(range[0], range[1])
    if wt_value:
        plt.text(range[1] - 0.97 * (range[1] - range[0]),
                 range[1] - 0.05 * (range[1] - range[0]), str0, fontsize=8.5)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(Title, fontsize=15)
    #plt.legend(loc='lower right')

    if save :
        plt.savefig(Title.replace(' ','_') + 'one_shot' + '.png', format='png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(Title.replace(' ','_') + 'one_shot' + '.pdf', format='pdf',
                    bbox_inches='tight', dpi=300)
        plt.close()


# ### ocm_nocond

# In[13]:


cv_xgb.fit(ocm_nocond.iloc[:,:-1], ocm_nocond.iloc[:,-1])

cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']

print(cvdepth, cvsubsample, cvcolsample, cvlr)


# In[14]:


estimator = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)
print(estimator)

best_xgb_ocmnocond = estimator


# In[15]:


train_test = shuffle(ocm_nocond, random_state=10)


# In[16]:


cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10)
cv.cross_validation()


# In[17]:


cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      titlesize = 11,                       filename='output/fig5_onlycomposition_',
                      close=True)


# ### ocm

# In[18]:


cv_xgb.fit(ocm.iloc[:,:-1], ocm.iloc[:,-1])

cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']

print(cvdepth, cvsubsample, cvcolsample, cvlr)


# In[19]:


estimator = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)
print(estimator)

best_xgb_ocm = estimator


# In[20]:


train_test = shuffle(ocm, random_state=10)


# In[21]:


cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10)
cv.cross_validation()


# In[22]:


cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      titlesize = 11,                       filename='output/fig5_composition&condition_',
                      close=True)


# ### ocm_desc

# In[23]:


cv_xgb.fit(ocm_desc.iloc[:,:-1],ocm_desc.iloc[:,-1])

cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']

print(cvdepth, cvsubsample, cvcolsample, cvlr)


# In[24]:


estimator = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)
print(estimator)

best_xgb_ocmdesc = estimator


# In[25]:


train_test = shuffle(ocm_desc, random_state=10)


# In[26]:


cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()


# In[27]:


xgb_cvbest = cv


# In[28]:


cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4-fig5_gb_',                      titlesize=11,
                      close=True)


# ## Fig.4

# In[29]:


train_test = shuffle(ocm_desc, random_state=10)


# In[30]:


# LASSO
cv_lasso.fit(ocm_desc.iloc[:,:-1],ocm_desc.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
estimator = Lasso(random_state=929, alpha=cvalpha)

best_lasso_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4_lasso_',                       titlesize=11,
                      close=True)


# In[31]:


# Ridge
cv_ridge.fit(ocm_desc.iloc[:,:-1],ocm_desc.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
estimator = Ridge(random_state=929, alpha=cvalpha)

best_ridge_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4_ridge_',                      titlesize=11,
                      close=True)


# In[32]:


# KRR
cv_krr.fit(ocm_desc.iloc[:,:-1],ocm_desc.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
estimator = KernelRidge(kernel='rbf',alpha=cvalpha, gamma=cvgamma)

best_krr_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                       os_title='',                      filename='output/fig4_krr_',                      titlesize=11,
                      close=True)


# In[33]:


# SVR
cv_svr.fit(ocm_desc.iloc[:,:-1], ocm_desc.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
estimator = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

best_svr_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4_svr_',                      titlesize=11,
                      close=True)


# In[34]:


# RFR
estimator = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

best_rfr_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4_rfr_',                      titlesize=11,
                      close=True)


# In[35]:


# ETR
estimator = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

best_etr_ocmdesc = estimator
print(estimator)

cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()
cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/fig4_etr_',                      titlesize=11,
                      close=True)


# ## Fig.6 (A)

# In[36]:


plt.figure(figsize=(6, 4), dpi=300)
xgb_cvbest.importance.mean(axis=1).sort_values(ascending=False).iloc[:20][::-1].plot(kind='barh',color='b')
plt.savefig('output/fig6_ocm.png', format='png',bbox_inches='tight', dpi=300)
plt.savefig('output/fig6_ocm.pdf', format='pdf',bbox_inches='tight', dpi=300)
plt.close()


# ## Fig.2 (A)

# In[37]:


common = []
tmp = ocm.loc[:,:'Zr']
count = 0
for key, row in tmp.iterrows():
    for i, v in tmp.iloc[count + 1:].iterrows():
        temp = (np.array(row) > 1e-5)
        temp2 = (np.array(v) > 1e-5)
        val = (temp * temp2).sum()
        common.append(val)  
    count = count + 1

freq = dict(Counter(common))
plt.figure(figsize=(2.5, 3))
plt.bar(freq.keys(), freq.values())
plt.xlabel('# Common Elements')
plt.ylabel('Frequency (log)')
plt.yscale('log')
plt.savefig('output/fig2_ocm.png',format='png', bbox_inches='tight',dpi=300)
plt.savefig('output/fig2_ocm.pdf',format='pdf', bbox_inches='tight',dpi=300)
plt.close()


# In[38]:


ocm_orig = pd.read_csv("input/OCM_matrix.csv").drop(['Unnamed: 0'], axis=1)
val = (ocm_orig.loc[:,:'Zr'] > 0).sum(axis=1).mean()
print('%1.2f' % val)


# In[39]:


dat = (ocm_orig.loc[:,:'Zr'] > 0).sum(axis=0).sort_values()
ind = dat < 10
print(dat[ind])


# ## Fig.3 & Table 3

# In[40]:


def crossvalid(xx,yy,model,cv,txt):
    cv = Cv_Pred_Expected(estimator=model,X=xx,y=yy,cv=10,redc = True)
    cv.cross_validation()
    tes_rmse = cv.rmse
    tes_std = cv.std
    trn_rmse = cv.rmse_train
    trn_std = cv.std_train
    print()
    print("[%s] RMSE %1.3f (STD: %1.3f) ... test" % (txt,tes_rmse,tes_std))
    print("[%s] RMSE %1.3f (STD: %1.3f) ... train" % (txt,trn_rmse,trn_std))
    ret_obj = {}
    ret_obj['tes_mean'] = tes_rmse
    ret_obj['tes_sd']   = tes_std 
    ret_obj['trn_mean'] = trn_rmse
    ret_obj['trn_sd']   = trn_std 
    return ret_obj


# ### ocm_nocond

# In[41]:


rmse_ocm_nocond = {}
data = shuffle(ocm_nocond, random_state=10)

target_data = ocm_nocond


# In[42]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)


# In[43]:


model = best_lasso
print(model)
rmse_ocm_nocond['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_ocm_nocond['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10,  'Ridge')

model = best_krr
print(model)
rmse_ocm_nocond['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_ocm_nocond['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_ocm_nocond['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_ocm_nocond['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb_ocmnocond
print(model)
rmse_ocm_nocond['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### ocm

# In[44]:


rmse_ocm = {}
data = shuffle(ocm, random_state=10)
target_data = ocm


# In[45]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)


# In[46]:


model = best_lasso
print(model)
rmse_ocm['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_ocm['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr
print(model)
rmse_ocm['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_ocm['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_ocm['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_ocm['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb_ocm
print(model)
rmse_ocm['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### ocm_desc

# In[47]:


rmse_ocm_desc = {}
data = shuffle(ocm_desc, random_state=929)
target_data = ocm_desc


# In[48]:


model = best_lasso_ocmdesc
print(model)
rmse_ocm_desc['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge_ocmdesc
print(model)
rmse_ocm_desc['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr_ocmdesc
print(model)
rmse_ocm_desc['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr_ocmdesc
print(model)
rmse_ocm_desc['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr_ocmdesc
print(model)
rmse_ocm_desc['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr_ocmdesc
print(model)
rmse_ocm_desc['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb_ocmdesc
print(model)
rmse_ocm_desc['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### Fig.3

# In[49]:


methods = ['Lasso','Ridge','KRR','SVR','RFR','ETR','XGB']

tes_mean = [rmse_ocm_desc[m]['tes_mean'] for m in methods]
tes_sd   = [rmse_ocm_desc[m]['tes_sd']   for m in methods]
trn_mean = [rmse_ocm_desc[m]['trn_mean'] for m in methods]
trn_sd   = [rmse_ocm_desc[m]['trn_sd']   for m in methods]

# 2*sigma = 95% CI
tes_ci = 2 * np.array(tes_sd)
trn_ci = 2 * np.array(trn_sd)

plt.figure(figsize=(6,3))

ind = np.arange(len(methods))
width = 0.4
plt.bar(ind, trn_mean, width, color='0.9', align='center', yerr=trn_ci)
plt.bar(ind+width, tes_mean, width, color='0.6', align='center', yerr=tes_ci)

plt.ylabel('RMSE (%)')
plt.xticks(ind+width/2,methods,rotation=45, fontsize=12)
plt.legend(('Training Error',
            'Test Error'), loc ="upper right", prop={'size':12})
plt.ylim(0,np.max(tes_mean)+5)

for x,y in zip(ind, trn_mean):
    plt.text(x, y+1.5, '%.2f' % y, ha='center', va= 'bottom', rotation='90', fontsize=10)

for x,y in zip(ind+width, tes_mean):
    plt.text(x, y+1.5, '%.2f' % y, ha='center', va= 'bottom', rotation='90', fontsize=10)

plt.savefig('output/fig3.png',format='png', bbox_inches='tight',dpi=300)
plt.savefig('output/fig3.pdf',format='pdf', bbox_inches='tight',dpi=300)
plt.close()


# ### Table 3

# In[50]:


pd.options.display.precision = 2

print(pd.DataFrame(rmse_ocm_nocond))

print(pd.DataFrame(rmse_ocm))

print(pd.DataFrame(rmse_ocm_desc))


# ## WGS data

# In[51]:


desc = pd.read_csv('input/Descriptors.csv',skiprows = [0],index_col='symbol').drop(['Unnamed: 0',
                                                                               'name',
                                                                               'ionic radius',
                                                                               'covalent radius',
                                                                               'VdW radius',
                                                                               'crystal radius',
                                                                               'a x 106 ',
                                                                               'Heat capacity ',
                                                                               'l',
                                                                               'electron affinity ',
                                                                               'VE',
                                                                               'Surface energy '],axis=1)
desc = desc.iloc[:83, :]


# In[52]:


wgs = pd.read_csv("input/wgs.csv")
wgs.index = list(wgs.iloc[:,0])
wgs = wgs.drop([wgs.columns[0]], axis=1)

wgs_desc = pd.read_csv("input/wgs_desc.csv")
wgs_desc.index = list(wgs_desc.iloc[:,0])
wgs_desc = wgs_desc.drop([wgs_desc.columns[0]], axis=1)

atom = list(wgs.loc[:,:'Pd'].columns) + list(wgs.loc[:,'Li':'Sr'].columns)
desc = desc.loc[atom]

desc=desc.fillna(desc.mean())


# In[53]:


# WGS
ntree = 1500
if debug_run:
    ntree = 10

# Lasso
cv_lasso = GridSearchCV(
    Lasso(random_state=929), 
    param_grid={"alpha": range_lasso})

# Ridge
cv_ridge = GridSearchCV(
    Ridge(random_state=929), 
    param_grid={"alpha": range_ridge})

# KRR
cv_krr = GridSearchCV(
    KernelRidge(kernel='rbf'), 
    param_grid={"alpha": range_krr_alpha, 
                "gamma": range_krr_gamma})

# SVR
cv_svr = GridSearchCV(
    SVR(kernel='rbf'), 
    param_grid={"C": range_svr_C, 
                "gamma": range_svr_gamma, 
                "epsilon": range_svr_epsilon})

# XGB
cv_xgb = GridSearchCV(
    XGBRegressor(random_state=929), 
    param_grid={"max_depth": range_depth, 
                "subsample": range_subsample, 
                "colsample_bytree": range_colsample,
                "learning_rate": range_lr})


# ## Fig.6 (B)

# In[54]:


cv_xgb.fit(wgs_desc.iloc[:,:-1], wgs_desc.iloc[:,-1])

cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']

print(cvdepth, cvsubsample, cvcolsample, cvlr)


# In[55]:


estimator = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)
print(estimator)

best_xgb_wgsdesc = estimator


# In[56]:


train_test = shuffle(wgs_desc, random_state=929)


# In[57]:


cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()


# In[58]:


xgb_cvbest = cv


# In[59]:


cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/wgs_gb_',                      titlesize=11,
                      close=True)


# In[60]:


plt.figure(figsize=(6, 4), dpi=300)
xgb_cvbest.importance.mean(axis=1).sort_values(ascending=False).iloc[:20][::-1].plot(kind='barh',color='b')
plt.savefig('output/fig6_wgs.png', format='png',bbox_inches='tight', dpi=300)
plt.savefig('output/fig6_wgs.pdf', format='pdf',bbox_inches='tight', dpi=300)
plt.close()


# ## Fig.2 (B)

# In[61]:


common = []
hoge = wgs.ix[:,'Pt':'Pd']
count = 0
for key, row in hoge.iterrows():
    for i, v in hoge.iloc[count + 1:].iterrows():
        temp = (np.array(row) > 1e-5)
        temp2 = (np.array(v) > 1e-5)
        val = (temp * temp2).sum()
        common.append(val)     
    count = count + 1

freq = dict(Counter(common))
plt.figure(figsize=(1.5, 3))
plt.bar(freq.keys(), freq.values())
plt.xlabel('# Common Elements')
plt.ylabel('Frequency (log)')
plt.yscale('log')
plt.savefig('output/fig2_wgs.png',format='png', bbox_inches='tight',dpi=300)
plt.savefig('output/fig2_wgs.pdf',format='pdf', bbox_inches='tight',dpi=300)
plt.close()


# ## Table 4 (wgs)

# ### wgs_nocond

# In[62]:


comppro = list(wgs.loc[:, :'Pd'].columns)
wgs_nocond = wgs.loc[:, comppro + ['CO Conversion']]


# In[63]:


rmse_wgs_nocond = {}
data = shuffle(wgs_nocond, random_state=10)
target_data = wgs_nocond


# In[64]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# XGB
cv_xgb.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']
best_xgb = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)


# In[65]:


model = best_lasso
print(model)
rmse_wgs_nocond['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_wgs_nocond['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10,  'Ridge')

model = best_krr
print(model)
rmse_wgs_nocond['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_wgs_nocond['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_wgs_nocond['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_wgs_nocond['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb
print(model)
rmse_wgs_nocond['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### wgs

# In[66]:


rmse_wgs = {}
data = shuffle(wgs, random_state=10)
target_data = wgs


# In[67]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# XGB
cv_xgb.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']
best_xgb = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)


# In[68]:


model = best_lasso
print(model)
rmse_wgs['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_wgs['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr
print(model)
rmse_wgs['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_wgs['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_wgs['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_wgs['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb
print(model)
rmse_wgs['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### wgs_desc

# In[69]:


rmse_wgs_desc = {}
data = shuffle(wgs_desc, random_state=929)
target_data = wgs_desc


# In[70]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)


# In[71]:


model = best_lasso
print(model)
rmse_wgs_desc['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_wgs_desc['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr
print(model)
rmse_wgs_desc['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_wgs_desc['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_wgs_desc['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_wgs_desc['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb_wgsdesc
print(model)
rmse_wgs_desc['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### Table 4

# In[72]:


pd.options.display.precision = 2

print(pd.DataFrame(rmse_wgs_nocond))

print(pd.DataFrame(rmse_wgs))

print(pd.DataFrame(rmse_wgs_desc))


# ## CO oxidation data

# In[73]:


co = pd.read_csv('input/co.csv')
ind = co['Data No']
co.index = ind
co= co.drop(['Data No'], axis=1)

co_desc = pd.read_csv('input/co_desc.csv')
ind = co_desc['Data No']
co_desc.index = ind
co_desc= co_desc.drop(['Data No'], axis=1)


# In[74]:


# devine to each groups
baseatom = co.loc[:,:'Pd'].columns
supatom = co.loc[:,'Al_s':'Co_s'].columns
pro = co.loc[:,'Ce_p':'Cu_p']
proatom = [re.findall(r'[A-Z][a-z]?',x) for x in co.loc[:,'Ce_p':'Cu_p']]
proatom = [y for x in proatom for y in x ]
env = co.drop(list(baseatom)+list(pro),axis=1).iloc[:,:-1].columns

conv = co.iloc[:,-1]


# In[75]:


desc = pd.read_csv('input/Descriptors.csv',skiprows = [0],index_col='symbol').drop(['Unnamed: 0',
                                                                               'name',
                                                                               'ionic radius',
                                                                               'covalent radius',
                                                                               'VdW radius',
                                                                               'crystal radius',
                                                                               'a x 106 ',
                                                                               'Heat capacity ',
                                                                               'l',
                                                                               'electron affinity ',
                                                                               'VE',
                                                                               'Surface energy '],axis=1)
desc=desc.loc[list(baseatom)+list(proatom),:]
desc = desc.fillna(desc.mean())


# In[76]:


# CO oxdation
ntree = 1500
if debug_run:
    ntree = 10

# Lasso
cv_lasso = GridSearchCV(
    Lasso(random_state=929), 
    param_grid={"alpha": range_lasso})

# Ridge
cv_ridge = GridSearchCV(
    Ridge(random_state=929), 
    param_grid={"alpha": range_ridge})

# KRR
cv_krr = GridSearchCV(
    KernelRidge(kernel='rbf'), 
    param_grid={"alpha": range_krr_alpha, 
                "gamma": range_krr_gamma})

# SVR
cv_svr = GridSearchCV(
    SVR(kernel='rbf'), 
    param_grid={"C": range_svr_C, 
                "gamma": range_svr_gamma, 
                "epsilon": range_svr_epsilon})

# XGB
cv_xgb = GridSearchCV(
    XGBRegressor(random_state=929), 
    param_grid={"max_depth": range_depth, 
                "subsample": range_subsample, 
                "colsample_bytree": range_colsample,
                "learning_rate": range_lr})


# ## Fig.6 (C)

# In[77]:


cv_xgb.fit(co_desc.iloc[:,:-1], co_desc.iloc[:,-1])

cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']

print(cvdepth, cvsubsample, cvcolsample, cvlr)


# In[78]:


estimator = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)
print(estimator)

best_xgb_codesc = estimator


# In[79]:


train_test = shuffle(co_desc, random_state=929)


# In[80]:


cv = Cv_Pred_Expected(estimator=estimator,                      X=train_test.iloc[:,:-1],                      y=train_test.iloc[:,-1],                      cv=10, redc=True)
cv.cross_validation()


# In[81]:


xgb_cvbest = cv


# In[82]:


cv.plot_pred_expected(save=True,                      title='',                      os_title='',                      filename='output/co_gb_',                      titlesize=11,
                      close=True)


# In[ ]:


plt.figure(figsize=(6, 4), dpi=300)
xgb_cvbest.importance.mean(axis=1).sort_values(ascending=False).iloc[:20][::-1].plot(kind='barh',color='b')
plt.savefig('output/fig6_co.png', format='png',bbox_inches='tight', dpi=300)
plt.savefig('output/fig6_co.pdf', format='pdf',bbox_inches='tight', dpi=300)
plt.close()


# ## Fig.2 (C)

# In[ ]:


common = []
hoge = co.loc[:,'Pt':'Pd']
count = 0
for key, row in hoge.iterrows():
    for i, v in hoge.iloc[count + 1:].iterrows():
        temp = (np.array(row) > 1e-5)
        temp2 = (np.array(v) > 1e-5)
        val = (temp * temp2).sum()
        common.append(val)    
    count = count + 1

freq = dict(Counter(common))
plt.figure(figsize=(1.5, 3))
plt.bar(freq.keys(), freq.values())
plt.xlabel('# Common Elements')
plt.ylabel('Frequency (log)')
plt.yscale('log')
plt.savefig('output/fig2_co.png',format='png', bbox_inches='tight',dpi=300)
plt.savefig('output/fig2_co.pdf',format='pdf', bbox_inches='tight',dpi=300)
plt.close()


# ## Table 4 (co)

# ### co_nocond

# In[ ]:


co_nocond = pd.concat([co.loc[:,:'Pd'], pro, co.iloc[:,-1]],axis=1)


# In[ ]:


rmse_co_nocond = {}
data = shuffle(co_nocond, random_state=10)
target_data = co_nocond


# In[ ]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# XGB
cv_xgb.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']
best_xgb = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)


# In[ ]:


model = best_lasso
print(model)
rmse_co_nocond['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_co_nocond['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10,  'Ridge')

model = best_krr
print(model)
rmse_co_nocond['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_co_nocond['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_co_nocond['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_co_nocond['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb
print(model)
rmse_co_nocond['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### co

# In[ ]:


rmse_co = {}
data = shuffle(co, random_state=10)
target_data = co


# In[ ]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# XGB
cv_xgb.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvdepth = cv_xgb.best_params_['max_depth']
cvsubsample = cv_xgb.best_params_['subsample']
cvcolsample = cv_xgb.best_params_['colsample_bytree']
cvlr = cv_xgb.best_params_['learning_rate']
best_xgb = XGBRegressor(n_estimators=ntree,                       max_depth=cvdepth,                       subsample=cvsubsample,                       colsample_bytree=cvcolsample,                       learning_rate=cvlr,                       random_state=929, seed=929, n_jobs=-1)


# In[ ]:


model = best_lasso
print(model)
rmse_co['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_co['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr
print(model)
rmse_co['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_co['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_co['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_co['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb
print(model)
rmse_co['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### co_desc

# In[ ]:


rmse_co_desc = {}
data = shuffle(co_desc, random_state=929)
target_data = co_desc


# In[ ]:


# Lasso
cv_lasso.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_lasso.best_params_['alpha']
best_lasso = Lasso(random_state=929, alpha=cvalpha)

# Ridge
cv_ridge.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_ridge.best_params_['alpha']
best_ridge = Ridge(random_state=929, alpha=cvalpha)

# KRR
cv_krr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvalpha = cv_krr.best_params_['alpha']
cvgamma = cv_krr.best_params_['gamma']
best_krr = KernelRidge(kernel='rbf', alpha=cvalpha, gamma=cvgamma)

# SVR
cv_svr.fit(target_data.iloc[:,:-1], target_data.iloc[:,-1])
cvC = cv_svr.best_params_['C']
cvgamma = cv_svr.best_params_['gamma']
cvepsilon = cv_svr.best_params_['epsilon']
best_svr = SVR(kernel='rbf', epsilon=cvepsilon, C=cvC, gamma=cvgamma)

# RFR
best_rfr = RandomForestRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)

# ETR
best_etr = ExtraTreesRegressor(n_estimators=ntree, random_state=929, n_jobs=-1)


# In[ ]:


model = best_lasso
print(model)
rmse_co_desc['Lasso'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Lasso')

model = best_ridge
print(model)
rmse_co_desc['Ridge'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'Ridge')

model = best_krr
print(model)
rmse_co_desc['KRR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'KRR')

model = best_svr
print(model)
rmse_co_desc['SVR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'SVR')

model = best_rfr
print(model)
rmse_co_desc['RFR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'RFR')

model = best_etr
print(model)
rmse_co_desc['ETR'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'ETR')

model = best_xgb_codesc
print(model)
rmse_co_desc['XGB'] = crossvalid(data.iloc[:,:-1], data.iloc[:,-1], model, 10, 'XGB')


# ### Table 4

# In[ ]:


pd.options.display.precision = 2

print(pd.DataFrame(rmse_co_nocond))

print(pd.DataFrame(rmse_co))

print(pd.DataFrame(rmse_co_desc))

