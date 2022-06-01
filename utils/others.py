import numpy as np
import pickle
import copy
import types
import pandas as pd


# 保存对象
def save(obj, file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(obj, f)
    return None


# 读取对象
def load(file_name):
    with open(file_name, 'rb+') as f:
        obj = pickle.load(f)
    return obj


# 保存当前工作区
ignore_names = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__builtins__']


def save_space(locals, file_name):
    save_dict = {}
    for name, val in locals.items():
        if callable(val):
            continue
        if name in ignore_names:
            continue
        if type(val) == types.ModuleType:
            continue
        print('saving', name, type(val))
        save_dict[name] = copy.copy(val)
    # 保存
    save(save_dict, file_name)
    return None


# 恢复工作区
def load_space(locals, file_name):
    save_dict = load(file_name)
    for name, val in save_dict.items():
        locals[name] = val


# 写xls
def numpy2xlsx(data, file_name):
    if not file_name.endswith('.xlsx'):
        file_name += '.xlsx'
    df = pd.DataFrame(data, columns=None, index=None)
    df.to_excel(file_name, index=False, header=False)
    return


# 读xls
def xlsx2numpy(file_name):
    if not file_name.endswith('.xlsx'):
        file_name += '.xlsx'
    data = pd.read_excel(file_name, header=None)
    data = np.array(data)
    return data

def spec_cluster(A, n_cluster=3):
    from sklearn.cluster import KMeans
    # 计算L
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    sqD = np.diag(1.0 / (D ** (0.5)))
    L = np.dot(np.dot(sqD, L), sqD)
    # 特征值
    lam, H = np.linalg.eig(L)
    order = np.argsort(lam)
    order = order[:n_cluster]
    V = H[:, order]
    # 聚类
    res = KMeans(n_clusters=n_cluster).fit(V)
    lbs = res.labels_
    return lbs


# 读txt
def read_txt(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return lines


# 写txt
def write_txt(file_name, lines):
    lines=[line+'\r\n' for line in lines]
    with open(file_name, 'w') as file:
        file.writelines(lines)
    return None
