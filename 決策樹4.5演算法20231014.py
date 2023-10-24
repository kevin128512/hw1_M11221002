# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:38:32 2023

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class C45Tree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.tree = None
        self.features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.features = list(X.columns)
        self.tree = self.c45_tree(X.values, y.values, self.features)

    def predict(self, X):
        X = X.values
        predictions = []
        for sample in X:
            predictions.append(self.predict_sample(sample, self.tree))
        return predictions

    def predict_sample(self, sample, tree):
        for feature, subtree in tree.items():
            feature_index = self.features.index(feature)
            sample_value = sample[feature_index]
            if sample_value in subtree:
                if isinstance(subtree[sample_value], dict):
                    return self.predict_sample(sample, subtree[sample_value])
                else:
                    return subtree[sample_value]
        return 0  # 返回預設的類別

    def best_feature(self, X, y):
        gain_ratios = []
        for i in range(X.shape[1]):
            gain_ratios.append(self.gain_ratio(X[:, i], y))
        return np.argmax(gain_ratios)

    def gain_ratio(self, X_col, y):
        # Information needed (entropy of the current set)
        info_D = self.entropy(y)

        # Splitting data on attribute values
        values, counts = np.unique(X_col, return_counts=True)
        info_after_split = 0
        split_info = 0
        for value, count in zip(values, counts):
            subset_y = y[X_col == value]
            weight = count / len(y)
            info_after_split += weight * self.entropy(subset_y)
            split_info -= weight * np.log2(weight)
        gain = info_D - info_after_split
        if split_info == 0:
            return 0
        return gain / split_info

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def c45_tree(self, X, y, features, current_depth=0):
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return unique_classes[0]
        if len(features) == 0 or (self.max_depth and current_depth >= self.max_depth) or len(X) < self.min_samples_split:
            return np.bincount(y).argmax()

        feature_index_in_sub_X = self.best_feature(X, y)
        best_feat_name = features[feature_index_in_sub_X]
        tree = {best_feat_name: {}}

        # Get the real index in the original dataset
        feature_index = self.features.index(best_feat_name)

        for value in np.unique(X[:, feature_index_in_sub_X]):
            sub_X = X[X[:, feature_index_in_sub_X] == value]
            sub_y = y[X[:, feature_index_in_sub_X] == value]
            if len(sub_X) == 0:
                tree[best_feat_name][value] = np.bincount(y).argmax()
            else:
                tree[best_feat_name][value] = self.c45_tree(sub_X, sub_y, features, current_depth + 1)
        return tree


# 讀取UCI Adult資料集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
data = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')

# 數據預處理
X = data.drop("income", axis=1)
y = data["income"].map({'<=50K': 0, '>50K': 1}).astype(int)
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用C4.5決策樹訓練模型
clf = C45Tree(max_depth=10, min_samples_split=50)
clf.fit(X_train, y_train)

# 預測訓練資料和測試資料的結果
train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

# 計算訓練和測試的分類正確率
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
# 將測試資料結果輸出到Excel檔案
output = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds})
output.to_excel('decision_tree_results.xlsx', index=False, engine='openpyxl')
#output.to_excel('C:\\Users\\User\\Documents\\test\\decision_tree_results.xlsx', index=False, engine='openpyxl')