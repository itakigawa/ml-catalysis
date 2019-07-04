  # 各種パッケージ、モジュールのロード
import pandas as pd
import math
import numpy as np
import random
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def rmse_calc(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def scatter_plot(model, X_train, y_train, X_test, y_test, xlabel='Experimental yiled (%)', ylabel='Predicted yield (%)', Title='title', fontsize=14, Filename='hoge', save=False, wt_value=True, close=False):
    print(X_train.shape, y_train.shape)
    model.fit(X_train, y_train)
    range = [min(y_train.min(), y_test.min()) ,
             max(y_train.max(), y_test.max()) ]
    val0 = (model.predict(X_train) - y_train)**2
    str0 = 'RMSE(Train): %1.3f' % math.sqrt(val0.mean())
    val1 = (model.predict(X_test) - y_test)**2
    str1 = 'RMSE(Test): %1.3f' % math.sqrt(val1.mean())
    print(str0)
    print(str1)
    #str2 = 'R2(Train): %1.3f' % model.score(X_train, y_train)
    #str3 = 'R2(Test): %1.3f' % model.score(X_test, y_test)
    plt.figure(figsize=(3, 3), dpi=100)
    plt.plot(range, range, color='0.5')
    plt.scatter(y_train, model.predict(X_train), s=5,
                facecolors='blue', edgecolors='blue', label='Training', alpha=0.4)
    plt.scatter(y_test, model.predict(X_test), s=5,
                facecolors='red', edgecolors='red', label='Test', alpha=0.4)
    plt.xlim(range[0], range[1])
    plt.ylim(range[0], range[1])
    if wt_value:
        plt.text(range[1] - 0.97 * (range[1] - range[0]),
                 range[1] - 0.05 * (range[1] - range[0]), str0, fontsize=8.5)
        plt.text(range[1] - 0.97 * (range[1] - range[0]),
                 range[1] - 0.10 * (range[1] - range[0]), str1, fontsize=8.5)
    #plt.text(0.5, 25.5, str2, fontsize=8.5)
    #plt.text(0.5, 24, str3, fontsize=8.5)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(Title, fontsize=fontsize)
    #plt.legend(loc='lower right')

    if save :
        plt.savefig(Filename + 'one_shot' + '.png', format='png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(Filename + 'one_shot' + '.pdf', format='pdf',
                    bbox_inches='tight', dpi=300)
        if close:
            plt.close()
        # pd.concat([pd.DataFrame(model.predict(X_train),
        #                        index=y_train.index), y_train], axis=1).to_csv(trainname)
        # pd.concat([pd.DataFrame(model.predict(X_test),
        #                        index=y_test.index), y_test], axis=1).to_csv(testname)


class Cv_Pred_Expected:

    def __init__(self, cv, X, y, estimator, redc=False, random_state=929):
        self.cv = cv
        self.X = X
        self.y = y
        self.estimator = estimator
        self.random_state = random_state
        self.redc_model = redc

    def _stratified_cv(self, shuffle=shuffle, train_pred=False):
        n = self.X.shape[0]
        num = int(n / self.cv)
        divi = [x + 1 for x in range(num) if (num) % (x + 1) == 0]
        cent = divi[-1]
        y_bin = [int(x / (n / cent)) for x in range(n)]
        #stk = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=shuffle)
        
        stk = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=shuffle)

        pred = []
        train_pred = []
        rmse = []
        rmse_train = []
        rmse_train_sum = []
        ind = []
        importance = []
        sortind = self.y.sort_values().index
        for train, test in stk.split(self.X.ix[sortind], y_bin):
            realind_train = self.X.iloc[train, :].index
            realind_test = self.X.iloc[test, :].index
            self.estimator.fit(self.X.ix[realind_train, :], self.y[realind_train])
            prediction = self.estimator.predict(self.X.ix[realind_test])
            prediction_train = self.estimator.predict(self.X.ix[realind_train])
            
            rmse.append(rmse_calc(prediction, self.y[realind_test]))
            rmse_train.append(rmse_calc(prediction_train, self.y[realind_train]))
            train_sum = (np.array(prediction_train) - self.y[realind_train])**2
            rmse_train_sum.extend(list(train_sum))
            pred.extend(list(prediction))
            train_pred.extend(list(prediction_train))
            ind.append(list(realind_test))
            if hasattr(self.estimator, 'feature_importances_'):
                feature_df = pd.DataFrame(self.estimator.feature_importances_, index=self.X.columns)
                importance.append(feature_df)

        ind_flat = [y for x in ind for y in x]
        pred = pd.Series(pred, index = ind_flat)
        pred_obs = pd.concat([pred, self.y], axis=1)
        r2 = r2_score(pred_obs.ix[:, -1], pred_obs.ix[:, -2])
        

        self.pred = pred
        self.ind = ind_flat
        self.each_ind = ind
        self.each_rmse = rmse
        self.each_rmse_train = rmse_train
        self.rmse = rmse_calc(pred_obs.ix[:,0], pred_obs.ix[:,1])
        self.rmse_train = math.sqrt(np.sum(rmse_train_sum) / len(rmse_train_sum))
        self.std = np.std(rmse)
        self.std_train = np.std(rmse_train)
        self.r2 = r2
        
        each_rmse_min_rmse = np.absolute(self.rmse - self.each_rmse)
        self._oneshot_ind = np.argmin(each_rmse_min_rmse)

        if hasattr(self.estimator, 'feature_importances_'):
            for n, i in enumerate(importance):
                if n == 0:
                    res = importance[0]
                else:
                    res = pd.concat([res, importance[n]], axis=1)

            self.importance = res

    def cross_validation(self, shuffle=True):
        self._stratified_cv(shuffle=shuffle)
        self.str1 = 'RMSE_test: %1.3f' % self.rmse
        self.str2 = 'STD_test: %1.3f' % self.std
        self.str3 = 'R2: %1.3f' % self.r2
        self.str4 = 'RMSE_train: %1.3f' % self.rmse_train
        self.str5 = 'STD_train: %1.3f' % self.std_train
        # self.shape = 'Data size: %1.3f' % self.X.shape
        print(self.str1, self.str2)
        print(self.str4, self.str5)
        print('Data size:' + str(self.X.shape))

        # 触媒を誤差の大きい順に並べる
        if self.redc_model == False:
            y_pred = pd.Series(self.pred, name='y_pred', index=self.ind)
            rmse_temp = pd.Series(
                abs(self.pred - self.y[self.ind]), name='abs_error', index=self.ind)
            X_temp = pd.concat([self.X, self.y], axis=1)
            X_temp = pd.concat([X_temp, y_pred], axis=1)
            self.X_sort_pred = pd.concat([X_temp, rmse_temp], axis=1).sort_values(
                by='abs_error', ascending=False)
            self.atom = [{i: '%1.3f' % v for i, v in row.iteritems() if v > 0}
                         for key, row in self.X_sort_pred.ix[:, :'Zr'].iterrows()]
            # self.atom=sorted(atom,key=)
        else:
            y_pred = pd.Series(self.pred, name='y_pred', index=self.ind)
            rmse_temp = pd.Series(
                abs(self.pred - self.y[self.ind]), name='abs_error', index=self.ind)
            X_temp = pd.concat([self.X, self.y], axis=1)
            X_temp = pd.concat([X_temp, y_pred], axis=1)
            self.X_sort_pred = pd.concat([X_temp, rmse_temp], axis=1).sort_values(
                by='abs_error', ascending=False)

    def plot_pred_expected(self, title='',os_title='', filename='', xlab='Experimental yield (%)', ylab='Predicted yield (%)', titlesize=20, save=False, range=[0, 35], one_shot=True, wt_value=False, close=False):
        #str1 = 'STD test: %1.3f' % self.std
        str0 = 'RMSE(Train): %1.3f' % self.rmse_train
        str1 = 'RMSE(Test): %1.3f' % self.rmse
        plt.figure(figsize=(4, 4), dpi=100)
        plt.plot(range, range, color='0.5')
        plt.scatter(self.y.sort_index(), self.pred.sort_index(
        ), s=10, facecolors='none', c='black', edgecolors='black', label='test', alpha=0.3)
        plt.xlim(range[0], range[1])
        plt.ylim(range[0], range[1])
        if wt_value == True:
            plt.text(range[1] - 0.97 * (range[1] - range[0]),
                     range[1] - 0.05 * (range[1] - range[0]), str0, fontsize=8.5)
            # plt.text(range[1] - 0.97 * (range[1] - range[0]),
            #         range[1] - 0.10 * (range[1] - range[0]), str1, fontsize=8.5)
            plt.text(range[1] - 0.97 * (range[1] - range[0]),
                     range[1] - 0.10 * (range[1] - range[0]), str1, fontsize=8.5)
            
        plt.xlabel(xlab, fontsize=14)
        plt.ylabel(ylab, fontsize=14)
        plt.title(title, fontsize=titlesize)
        if close:
            plt.close()
        #if save:
        #    plt.savefig(filename + '.pdf', format='pdf',bbox_inches='tight', dpi=300)
        #    plt.savefig(filename + '.png', format='png',bbox_inches='tight', dpi=300)
            
        if one_shot == True:
            train_ind = sorted(set(self.ind) - set(self.each_ind[self._oneshot_ind]))
            test_ind = sorted(self.each_ind[self._oneshot_ind])
            X_train, y_train= self.X.loc[train_ind], self.y[train_ind]
            X_test, y_test= self.X.loc[test_ind], self.y[test_ind]
            scatter_plot(self.estimator, X_train, y_train, X_test, y_test, xlabel=xlab, ylabel=ylab, Title=os_title, fontsize=titlesize, Filename=filename, save=save, wt_value=wt_value, close=close)

def comp_times_base(comp, base, sort=False, times=True, attention=False):
    count = 0
    for key, rows in comp.iterrows():
        stack = np.vstack((rows, base))
        if times == True:
            time = np.array(base) * np.array(rows)
            stack = np.vstack((rows, time))
    
        if sort == True:
            stack = pd.DataFrame(stack).sort_values(
                                                    [0], ascending=False, axis=1)
        
        stack = pd.DataFrame(stack).ix[1:, :]
        stack = np.array(stack)
        
        if count == 0:
            if attention:
                res = np.sum(stack, axis=1)
            else:
                res = np.array(stack.T.flatten())
            
            count += 1
        else:
            if attention:
                res = np.vstack((res, np.sum(stack, axis=1)))
            else:
                res = np.vstack((res, np.array(stack.T.flatten())))
            
            count += 1
    return res

def set_rownames(desc, data, name=''):
    ind = [divmod(int(key), desc.shape[1])[1] for key, col in data.iteritems()]
    
    count = 1
    i = 0
    col = []
    for x in range(int(data.shape[1] / desc.shape[1])):
        i = 0
        while i < desc.shape[1]:
            col.append(name + '_' + str(x + 1) + ' ' + desc.ix[:, i].name)
            i = i + 1

    return col

