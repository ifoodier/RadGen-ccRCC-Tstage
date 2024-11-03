# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso, lasso_path, LassoCV
import os


def move_lowvariance(X, y, path):
    fs_path = os.path.join(path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)

    sel = VarianceThreshold(threshold=0.8)
    result = sel.get_support()
    features = X.columns
    features_split = []
    sel_features = []
    sel_features_split = []
    first = []
    sel_first = []
    for index, item in enumerate(features):
        item_split = item.split('_')
        features_split.append(item_split)
        first.append(item_split[0])
        if result[index]:
            sel_features.append(item)
            sel_features_split.append(item_split)
            sel_first.append(item_split[0])
    print("move_lowvariance features reduced from {0} to {1}".format(len(features), len(sel_features)))
    X = X[sel_features]
    # X.to_csv(os.path.join(fs_path, 'low_variance_feature.csv'))
    y = y
    features = sel_features

    return X, y, features


def select_KBest(X, y, out_path):
    fs_path = os.path.join(out_path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)

    # sel = SelectKBest(f_classif, k=10).fit(X, np.ravel(y))
    sel = SelectKBest(f_classif, k='all').fit(X, np.ravel(y))
    scores = sel.scores_
    pvalue = sel.pvalues_
    features = X.columns
    result = sel.get_support()
    sort_features = []
    for index, item in enumerate(features):
        if result[index]:
            sort_features.append(item)
    # 将特征按分数 从大到小 排序
    named_scores = zip(features, scores, pvalue)
    sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)
    sorted_pvalue = [each[2] for each in sorted_named_scores if each[2] < 0.05]
    sorted_scores = [each[1] for each in sorted_named_scores]
    sorted_names = [each[0] for each in sorted_named_scores]
    print(len(sorted_pvalue))
    num = len(sorted_pvalue)
    # if num > 200:
    # 	num = num//2
    # num = 180
    sel_features = sorted_names[0:num]
    X = X[sel_features]
    y = y
    print("select_KBest features reduced from {0} to {1}".format(len(features), len(sel_features)))

    return X, y, sel_features


def lasso_filter(X, y, cv, out_path):
    fs_path = os.path.join(out_path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)
    # model = LassoLarsCV().fit(X, y)
    model = LassoCV(cv=cv, max_iter=1000, n_alphas=100, normalize=True).fit(X, y)
    m_log_alphas = -np.log10(model.alphas_)
    features = X.columns
    coef = pd.Series(model.coef_, index=features)
    sel_coef = coef[coef != 0]
    # sel_coef = sel_coef[14:16]
    print(sel_coef)
    sel_features = sel_coef.index
    print(sel_features)
    lasso_result = X[sel_features]
    print(" lasso_filter features reduced from {0} to {1}".format(len(features), len(sel_features)))
    best_lambda = model.alpha_  # 最佳lambda
    intercept = model.intercept_  # 最佳截距
    #
    int_lam = os.path.join(fs_path, 'int_lam.txt')
    with open(int_lam, 'w') as f:  #
        f.write('best_lambda: ' + str(best_lambda) + '   intercept: ' + str(intercept))
    log_alpha = os.path.join(fs_path, '-log_alpha.txt')
    with open(log_alpha, 'w') as f:  #
        f.write(str(m_log_alphas))
    with open(log_alpha, 'a') as f:  #
        f.write(str(m_log_alphas))
    alpha = os.path.join(fs_path, 'alpha.txt')
    with open(alpha, 'w') as f:  #
        f.write(str(model.alphas_))
    with open(alpha, 'a') as f:  #
        f.write(str(model.alphas_))

    y = pd.DataFrame(y, index=X.index)

    lasso_result.to_csv(os.path.join(fs_path, 'x_train_sel.csv'), encoding='utf_8_sig')  # 需要可导出

    print(len(sel_coef))
    return lasso_result, y, sel_features, sel_coef


def rad_score(feature, coef, path_in, file_name):
    feature_name = feature.columns
    coefficient = coef[feature_name]

    radscore = np.dot(feature, coefficient)
    radscore = pd.DataFrame(radscore, index=feature.index)
    radscore.columns = ['rad_score']

    radscore.to_csv(os.path.join(path_in, file_name + 'radscore.csv'))
    return radscore
