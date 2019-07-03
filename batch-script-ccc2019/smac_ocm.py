
# coding: utf-8

# jupyter nbconvert --to script smac_ocm.ipynb

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


# for SMAC?
from matplotlib import gridspec 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

import skopt
from sklearn.base import BaseEstimator, RegressorMixin
from skopt import gp_minimize 
from skopt import Optimizer
from skopt.acquisition import gaussian_pi
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.learning import GaussianProcessRegressor
from skopt.learning import ExtraTreesRegressor as opt_ET
from skopt.learning import RandomForestRegressor as opt_RF

import sklearn.gaussian_process as gp


# In[4]:


print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)

print('skopt:', skopt.__version__)


# In[5]:


np.random.seed(0)
random.seed(0)


# In[6]:


debug = False
#debug = True
num_search = 400
num_averaging = 10
num_init = 10
num_tree = 300
num_tree_smac = 500
if debug:
    num_search = 4
    num_averaging = 2
    num_init = 10
    num_tree = 10
    num_tree_smac = 20


# ## OCM data

# In[7]:


ocm = pd.read_csv("input/OCM_matrix.csv").drop(['Unnamed: 0'], axis=1)
ocm_desc = pd.read_csv("input/OCM_matrix_desc.csv").drop(['Unnamed: 0'], axis=1)

#drop which has 'Th' in its composer
ocm = ocm.ix[ocm_desc.index]

#exclude data which has over 5 metals in his component
ind = (ocm.ix[:,:'Zr']>0).sum(axis=1)
ind = ind < 5
ocm = ocm.ix[ind]

ind = (ocm_desc.ix[:,:'Zr']>0).sum(axis=1)
ind = ind < 5
ocm_desc = ocm_desc.ix[ind]

#drop 'Support_Co'(becaues all value are zero)
ocm = ocm.drop(['Support_Co'], axis=1)
ocm_desc = ocm_desc.drop(['Support_Co'], axis=1)

ocm_atom = ocm.ix[:,:'Zr'].columns


# In[8]:


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


# ## code

# In[9]:


def posterior(x, p_x, p_y, model):
    if len(p_x.shape) == 1:
        model.fit(p_x.reshape(-1, 1), p_y)
        mu, sigma = model.predict(x.reshape(-1,1), return_std=True)
    else:
        model.fit(p_x, p_y)
        mu, sigma = model.predict(x, return_std=True)
    ind = np.where(sigma == 0.)
    sigma[ind] = 1e-5
    return mu, sigma

def EI(mu, sigma, cur_max):
    Z = (mu - cur_max) / sigma
    ei = (mu - cur_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


# In[10]:


def rand_search(ninit, x, y, random_state = 929):
    random.seed(random_state)
    ninit = ninit # number of first point
    niter = len(x) # number of iteration
    true_max = np.max(y)
    order = list(range(len(x)))
    random.shuffle(order)

    y_list = []
    z_list = []
    for i in range(ninit):
        ridx = order[i]
        y_list.append(y.iloc[ridx])
        
    cur_max = np.array(y_list).max()
    for j in range(num_search):
        ridx = order[j + ninit]
        y_list.append(y.iloc[ridx])
        yp = np.array(y_list)
        cur_max = np.max(yp)
        z_list.append(cur_max)
        
        if cur_max >= true_max:
            print('max found', j)
            
        print('iter:{0}, current_max:{1}'.format(j,cur_max))
     
    return z_list


# In[11]:


print('Random Search')
rand = []
for i in range(num_averaging):
    rand.append(rand_search(num_init,ocm_desc.ix[:,:-1],ocm_desc.ix[:,-1],random_state=i*10))


# In[12]:


def exploration(ninit, model, x, y, random_state=929):
    random.seed(random_state)
    true_max = np.max(y)
    order = list(range(len(x)))
    random.shuffle(order)

    x_list = []
    y_list = []
    z_list = []

    used = set()

    for i in range(ninit):
        ridx = order[i]
        x_list.append(x.iloc[ridx, :])
        y_list.append(y.iloc[ridx])
        used.add(ridx)
        
    print(y_list)
    for j in range(num_search):
        xp = np.array(x_list)
        yp = np.array(y_list)
        cur_max = np.max(yp)
        # fit surrogate model
        model.fit(xp, yp)
        _mu = model.predict(np.array(x))
        mu = _mu.reshape(-1)
        idlist = np.argsort(mu)[::-1]
        p = 0
        max_idx = idlist[p]
        while max_idx in used:
            p += 1
            max_idx = idlist[p]
            
        used.add(max_idx)
        x_list.append(x.iloc[max_idx, :])
        y_list.append(y.iloc[max_idx])
        z_list.append(cur_max)
        
        print('iter:{0}, current_max:{1}'.format(j + 1,cur_max))
        
    return x_list, y_list, z_list


# In[13]:


X, y = ocm_desc.iloc[:,:-1], ocm_desc.iloc[:,-1]


# In[14]:


#Hyper ParametorはGrid searchにて決定
estimator=XGBRegressor(n_estimators=num_tree,max_depth=8,subsample=0.9,colsample_bytree=0.8,learning_rate=0.05,random_state=929,n_jobs=-1)

print('exploitation w/ XGB')
res_model_pred_xgb = []
for i in range(num_averaging):
    res_model_pred_xgb.append(exploration(num_init,estimator,X,y,random_state=i*10))


# In[15]:


estimator=RandomForestRegressor(n_estimators=num_tree,random_state=929,n_jobs=-1)

print('exploitation w/ RF')
res_model_pred_rf = []
for i in range(num_averaging):
    res_model_pred_rf.append(exploration(num_init,estimator,X,y,random_state=i*10))


# In[16]:


estimator=ExtraTreesRegressor(n_estimators=num_tree,random_state=929,n_jobs=-1)

print('exploitation w/ ET')
res_model_pred_et = []
for i in range(num_averaging):
    res_model_pred_et.append(exploration(num_init,estimator,X,y,random_state=i*10))


# In[17]:


kernel = gp.kernels.Matern(nu = 2.5)
estimator = gp.GaussianProcessRegressor(kernel=kernel,
                                    alpha=1e-2,
                                    n_restarts_optimizer=10,
                                    normalize_y=True,
                                    random_state=929)

print('exploitation w/ GP')
res_model_pred_gp = []
for i in range(num_averaging):
    res_model_pred_gp.append(exploration(num_init,estimator,X,y,random_state=i*10))


# In[18]:


def bo(ninit, model, x, y, random_state = 929):
    random.seed(random_state)
    ninit = ninit # number of first point
    niter = len(x) # number of iteration
    true_max = np.max(y)
    order = list(range(len(x)))
    random.shuffle(order)

    x_list = []
    y_list = []
    z_list = []
    used = set()
    
    for i in range(ninit):
        ridx = order[i]
        x_list.append(x.iloc[ridx, :])
        y_list.append(y.iloc[ridx])
        used.add(ridx)

    for j in range(num_search):
        xp = np.array(x_list)
        yp = np.array(y_list)
        cur_max = np.max(yp)
        # fit surrogate model
        model.fit(xp, yp)
        _mu, sigma = model.predict(x, return_std=True)
        mu = _mu.reshape(-1)
        ind = np.where(sigma == 0.)
        sigma[ind] = 1e-5
        # compute EI
        Z = (mu - cur_max) / sigma
        ei = (mu - cur_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] == 0.0
        idlist = np.argsort(ei)[::-1]
        p = 0
        max_idx = idlist[p]
        while max_idx in used:
            p += 1
            max_idx = idlist[p]
        
        used.add(max_idx)
        x_list.append(x.iloc[max_idx, :])
        y_list.append(y.iloc[max_idx])
        z_list.append(cur_max)
            
        print('iter:{0}, current_max:{1}'.format(j,cur_max))
        
    return x_list, y_list, z_list


# In[19]:


kernel = gp.kernels.Matern(nu = 2.5)
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    alpha=1e-2,
                                    n_restarts_optimizer=10,
                                    normalize_y=True,
                                    random_state=929)


# In[20]:


print('BO w/ GP')
res_model_bo_gp = []
for i in range(num_averaging):
    res_model_bo_gp.append(bo(num_init,model,X,y,random_state=i*10))


# In[21]:


print('BO w/ RF')
model = opt_RF(n_estimators=num_tree,n_jobs=-1, random_state=929)
res_model_bo_rf = []
for i in range(num_averaging):
    res_model_bo_rf.append(bo(num_init,model,X,y,random_state=i*10))


# In[22]:


print('BO w/ ET')
model = opt_ET(n_estimators=num_tree, n_jobs=-1,random_state=929)
res_model_bo_et = []
for i in range(num_averaging):
    res_model_bo_et.append(bo(num_init,model,X,y,random_state=i*10))


# In[23]:


def random_cand_ocm(x, num,random_state=929 ):
    rand_list = []
    for n in range(num):
        random.seed(random_state + n)
        np.random.seed(random_state + n)
        #make random vector(belong to metal composition)
        temp = np.random.uniform(low=0, high=100,size=len(comp_f)) 
        comp_rand = pd.Series(temp, index=list(comp_f))
        sys = np.random.randint(low=1, high=4 + 1)
        ind = comp_rand.sort_values().iloc[sys:].index
        comp_rand[ind] = 0
        comp_rand = comp_rand[list(comp_f)]
        
        temp = np.random.uniform(low=0, high=100,size=len(sup_f)) 
        comp_rand_sup = pd.Series(temp, index=list(sup_f))
        sys = np.random.randint(low=0, high=2)
        ind = comp_rand_sup.sort_values().iloc[sys:].index
        comp_rand_sup[ind] = 0
        comp_rand_sup = comp_rand_sup[list(sup_f)]
        
        comp_rand = pd.concat([comp_rand, comp_rand_sup])
        comp_rand = comp_rand * 100 / comp_rand.sum()
        comp_rand = comp_rand[list(comp_f) + list(sup_f)]
        #display(comp_rand)

        #make random vector(belong to Promotor)
        f = random.choice(list(prom_f) + ['nan'])
        if f != 'nan':
            temp = ocm.ix[0, prom_f]
            temp[f] = 1
            temp[set(prom_f) - {f}] = 0
        else:
            temp = ocm.ix[0, prom_f]
            temp[prom_f] = 0
        
        prom_rand = temp
        #display(prom_rand)

        #make random vector(belong to condition)
        vec = np.array(ocm.ix[:,cond_f] )
        a = np.mean(vec, axis=0) - vec.min(axis=0)
        b = vec.max(axis=0) - vec.min(axis=0)
        v = a / b
        p = np.array([-1])
        while any((p < 0) | (p > 1)):
            mu ,sigma = v, np.diag([0.2 for i in range(len(cond_f))])
            p = np.random.multivariate_normal(mu, sigma)

        p = p * b + vec.min(axis=0)
        p = pd.Series(p, index=cond_f)
        p_ch4_o2 = {'p(CH4)/p(O2)', 'p(CH4), bar', 'p(O2), bar'}
        if (p['p(O2), bar'] > 0):
            p['p(CH4)/p(O2)'] = p['p(CH4), bar'] / p['p(O2), bar']
        unique = list((Counter(ocm['Contact time, s'])))
        contact = random.choice(unique)
        p['Contact time, s'] = contact
        cond_rand = p
        #display(cond_rand)

        #make random vector(belong to Preparation)
        f = random.choice(list(pre_f))
        temp = ocm.ix[0, pre_f]
        temp[f] = 1
        temp[set(pre_f) - {f}] = 0
        pre_rand = temp
        
        cat_rand = pd.concat([comp_rand, prom_rand, cond_rand, pre_rand])
        rand_list.append(cat_rand)
    
    comp_rand = pd.DataFrame(rand_list)
    feature=comp_times_base(comp_rand.ix[:,:'Zr'],desc.ix[ocm_atom].T,sort=True,times=True)
    feature=pd.DataFrame(feature,index=comp_rand.index)
    feature = feature.ix[:,:desc.shape[1]*4-1]
    ind = set_rownames(desc,feature,name='Metal')
    feature.columns = ind
    ind = feature.max() > 0
    feature = feature.ix[:,ind]
    comp_rand = pd.concat([feature, comp_rand], axis=1)
    comp_rand = comp_rand[x.columns]
    
    return comp_rand 


# In[24]:


def roen(x, y, model, random_state=929):
    x_ch = x.ix[:,:-2].copy()
    neighbor = []
    
    for key, row in x_ch.iterrows():
        count = 0
        nei_4 = []
        #print('ind={0}'.format(key))
        if x.ix[key, 'make_nei'] == True:
            for count in range(4):
                seed = (key + 1) * (count + 1) * random_state
                random.seed(seed)
                np.random.seed(seed)
                f = list(ocm.ix[:,:-1].columns)
                f = f + ['Promotor_nan']  
                f.remove('Th')
                change_f = random.choice(f)
                row_ch = row.copy()

                if change_f in comp_sup_at:
                    a = np.array(row_ch[change_f]) - x_ch[change_f].min()
                    b = x_ch[change_f].max() - x_ch[change_f].min()
                    v = a.astype(float) / b.astype(float)
                    p = np.array([-1])
                    while (p < 0) | (p > 1):
                        p = random.normalvariate(v, 0.2)

                    p = p * b + x_ch[change_f].min()
                    row_ch[change_f] = p

                    if sum(row_ch[comp_f] > 0) > 4:
                        ind = row_ch[comp_f] > 0
                        atom = row_ch[comp_f][ind].idxmin()
                        row_ch[atom] = 0
                        
                    row_ch[comp_sup_at] = (row_ch[comp_sup_at] * 100) / row_ch[comp_sup_at].sum()

                elif change_f in (prom_f + ['Promotor_nan']):
                    if change_f != 'Promotor_nan':
                        row_ch[change_f] = 1
                        ind = set(prom_f) - {change_f}
                        row_ch[ind] = 0
                    else:
                        row_ch[prom_f] = 0

                elif change_f in cond_f:
                    a = np.array(row_ch[change_f]) - x_ch[change_f].min()
                    b = x_ch[change_f].max() - x_ch[change_f].min()
                    v = a.astype(float) / b.astype(float)
                    p = -1
                    while (p < 0) | (p > 1):
                        p = random.normalvariate(v, 0.2)

                    p = p * b + x_ch[change_f].min()
                    row_ch[change_f] = p
                    p_ch4_o2 = {'p(CH4)/p(O2)', 'p(CH4), bar', 'p(O2), bar'}
                    if row_ch['p(O2), bar'] > 0: 
                        row_ch['p(CH4)/p(O2)'] = row_ch['p(CH4), bar'] / row_ch['p(O2), bar']

                elif change_f in pre_f:
                    row_ch[change_f] = 1
                    ind = set(pre_f) - {change_f}
                    row_ch[ind] = 0
                    #display(row_ch)

                nei_4.append(row_ch)

            nei_4 = pd.DataFrame(nei_4, index=[0,1,2,3])

            feat = comp_times_base(nei_4.ix[:,:'Zr'],desc.ix[ocm_atom].T,sort=True,times=True)
            feat = pd.DataFrame(feat, index=nei_4.index)
            feat = feat.ix[:,:desc.shape[1]*4-1]
            ind = set_rownames(desc,feat,name='Metal')
            nei_4[ind] = feat 

            mu, sigma = model.predict(np.array(nei_4), return_std=True)
            ind = y.values.argmax()
            cur_max = y[ind]
            ei = EI(mu, sigma, cur_max)
            ind = np.argmax(ei)
            cand = nei_4.iloc[ind].copy()
            cand['ei'] = ei[ind]

            if x.ix[key, 'ei'] < cand['ei']:
                cand['make_nei'] = True
                neighbor.append(cand)
            else:
                x.ix[key, 'make_nei'] = False
                neighbor.append(x.ix[key])
        else:
            neighbor.append(x.ix[key])
    
    
    res = pd.DataFrame(neighbor, index=x.index)
    #display(res)
    return res


# In[25]:


def smac(model, init_x, init_y, roen_func, random_cand, rand=False, random_state = 929):
    cur_max = init_y.max()
    model.fit(np.array(init_x), np.array(init_y))
    mu, sigma = posterior(np.array(init_x), np.array(init_x), init_y, model)
    ei = EI(mu, sigma, init_y.max())
    ei = pd.Series(ei, index=init_x.index, name='ei')
    make_nei = pd.Series(True, index=init_x.index, name='make_nei')
    next_x = pd.concat([init_x, ei, make_nei], axis = 1)
    while next_x['make_nei'].sum() != 0:
        next_x = roen_func(next_x, init_y, model)
        next_x = pd.DataFrame(next_x)
        print(next_x['make_nei'].sum())
        
    if rand == True:
        cand_model = next_x.sort_values(by='ei', ascending=False).ix[:,:-1]
        cand_rand= random_cand(init_x, 10) 
        mu, sigma = model.predict(np.array(cand_rand), return_std=True)
        ei = EI(mu, sigma, cur_max)
        ei = pd.Series(ei, cand_rand.index, name='ei')
        cand_rand = pd.concat([cand_rand, ei], axis=1)
        cand = pd.concat([cand_model, cand_rand])
    else:
        cand = next_x
        
    
    return cand


# In[26]:


comp_f = ocm_desc.ix[:,'Ag':'Zr'].columns
sup_f = ocm_desc.ix[:,'Support_Si':'Support_Zr'].columns
comp_sup_at = list(comp_f) + list(sup_f)
prom_f = ocm_desc.ix[:,'Promotor_B':'Promotor_S'].columns
pre_f = ocm_desc.ix[:,'Impregnation':'Therm.decomp.'].columns
cond_f = ocm_desc.ix[:,'Temperature, K':'Contact time, s'].columns

categ = {}
categ['comp_f'] = list(comp_f)
categ['sup_f'] = list(sup_f)
categ['prom_f'] = list(prom_f)
categ['pre_f'] = list(pre_f)
categ['cond_f'] = list(cond_f)


# In[27]:


model = opt_ET(n_estimators=num_tree_smac, random_state=929)


# In[28]:


aho = smac(model=model, init_x=ocm_desc.ix[:,:-1], init_y=ocm_desc.ix[:,-1], roen_func=roen, random_cand=random_cand_ocm, rand=True)


# In[29]:


k = KMeans(n_clusters=421, random_state=929)
cluster = k.fit_predict(aho.ix[:,:-1])
cluster = pd.Series(cluster, index=aho.index, name='cluster')
aho = pd.concat([aho,cluster], axis=1)


# In[30]:


model.fit(ocm_desc.ix[:,:-1],ocm_desc.ix[:,-1])
pred_y = model.predict(aho.ix[:,:-2])
pred_y = pd.Series(pred_y,index=aho.index, name='pred_y')
cand = pd.concat([aho,pred_y], axis = 1)


# In[31]:


#choose point which has most better ei value in each cluster
clus_high = cand.sort_values(by=['cluster','ei']).drop_duplicates(subset=['cluster'],keep='last')
clus_high = clus_high.sort_values(by='ei', ascending=False)

main = clus_high.loc[:, :'Zr']
supp = clus_high.loc[:, 'Support_Si':'Support_Zr']
targ = pd.concat([main, supp], axis=1)

hogege = []
for key,row in targ.iterrows():
    temp = [str(i)+':'+str(round(v,1)) for i,v in row[row>0].sort_values(ascending=False).iteritems()]
    hogege.append(temp)
    
hogege = [' '.join(x) for x in hogege]


# In[32]:


w = 0.4
hoge = clus_high.iloc[:20]
x = np.arange(hoge.shape[0])
pred_y = list(clus_high['pred_y'])

extra = []
for y in x:
    extra.append(y)
    
ytick = []
for n in range(20):
    ytick.append(hogege[n])
    
plt.figure(figsize=(6,4), dpi=100)
plt.barh(x,hoge['ei'][::-1],label='EI')
for n,i in enumerate(x[::-1]):
    plt.text(clus_high['ei'].iloc[n],i-0.4,str(round(clus_high['ei'].iloc[n],1)),fontsize=9)
    
    
plt.xlim([0,25])
plt.yticks(x[::-1],ytick)
plt.savefig('output/fig11.png',format='png',dpi=100,bbox_inches='tight')
plt.savefig('output/fig11.pdf',format='pdf',dpi=100,bbox_inches='tight')
plt.close()


# In[33]:


rand_pred = np.array(rand).mean(axis=0)
gp_mean_pred= np.array([x[2] for x in res_model_pred_gp]).mean(axis=0)
rf_mean_pred= np.array([x[2] for x in res_model_pred_rf]).mean(axis=0)
et_mean_pred= np.array([x[2] for x in res_model_pred_et]).mean(axis=0)
xgb_mean_pred= np.array([x[2] for x in res_model_pred_xgb]).mean(axis=0)


# In[34]:


#random initinal
plt.figure(figsize=(6,3), dpi=100)
plt.plot(rand_pred, label='Random')
plt.plot(gp_mean_pred, label='GPR')
plt.plot(rf_mean_pred, label='RFR')
plt.plot(et_mean_pred, label='ETR')
plt.plot(xgb_mean_pred, label='XGB')

plt.xlabel('iteration')
plt.ylabel('C2 Yield(%)')
plt.legend()
plt.savefig('output/fig9.png',format='png',dpi=100,bbox_inches='tight')
plt.savefig('output/fig9.pdf',format='pdf',dpi=100,bbox_inches='tight')
plt.close()


# In[35]:


gp_mean_bo = np.array([x[2] for x in res_model_bo_gp]).mean(axis=0)
rf_mean_bo = np.array([x[2] for x in res_model_bo_rf]).mean(axis=0)
et_mean_bo = np.array([x[2] for x in res_model_bo_et]).mean(axis=0)


# In[36]:


# random initial
plt.figure(figsize=(6,3), dpi=100)
plt.plot(rand_pred, label='Random')
plt.plot(gp_mean_bo, label='BO(GPR)')
plt.plot(rf_mean_bo, label='SMAC(RFR)')
plt.plot(et_mean_bo, label='SMAC(ETR)')

plt.xlabel('iteration')
plt.ylabel('C2 Yield(%)')
plt.legend(loc='lower right')
plt.savefig('output/fig10.png',format='png',dpi=100,bbox_inches='tight')
plt.savefig('output/fig10.pdf',format='pdf',dpi=100,bbox_inches='tight')
plt.close()

