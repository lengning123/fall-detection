import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from lstm import lstm_model
from scipy.signal import butter, filtfilt

file_label=['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT',
              'CHU', 'CSI', 'CSO', 'LYI', 'FOL', 'FKL', 'BSC', 'SDL']
label_dict={'STD':0,'WAL':1,'JOG':2,'JUM':3,'STU':4,'STN':5,'SCH':6,'SIT':7,
            'CHU':8,'CSI':9,'CSO':10,'LYI':11,'FOL':12,'FKL':13,'BSC':14,'SDL':15}
num_label=2
# 设置 GPU 内存动态分配
#gpus = tf.config.list_physical_devices('GPU')


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    使用Butterworth低通滤波器对时序数据进行滤波。

    参数：
    - data: 输入时序数据 (1D 数组)
    - cutoff: 截止频率 (Hz)
    - fs: 采样频率 (Hz)
    - order: 滤波器阶数 (默认4阶)

    返回：
    - 滤波后的数据
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 滤波器系数
    filtered_data = filtfilt(b, a, data)  # 零相位滤波
    return filtered_data

def apply_filter_to_dataset(dataset, cutoff, fs, order=1):
    """
    对二维时序数据集的每个特征列进行Butterworth低通滤波。

    参数：
    - dataset: 二维数组 (shape: [samples, features])
    - cutoff: 截止频率
    - fs: 采样频率
    - order: 滤波器阶数

    返回：
    - 滤波后的数据集
    """
    filtered_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[1]):  # 遍历每个特征列
        filtered_dataset[:, i] = butter_lowpass_filter(dataset[:, i], cutoff, fs, order)
    return filtered_dataset


def collect_path(base_dir, num_types):
    """
    从所有类型文件夹中收集样本及其对应的标签。
    :param base_dir: 数据主目录路径（如 data/）。
    :param num_types: 数据类型总数（如 11）。
    :param label_dict: 标签字典，用于编码标签。
    :return: 特征数组和标签数组。
    """

    all_file = []
    for type_label in range(num_types):
        type_dir = os.path.join(base_dir, file_label[type_label])
        if not os.path.exists(type_dir):
            continue

        # 遍历当前类型文件夹下的所有 CSV 文件
        for file_name in os.listdir(type_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(type_dir, file_name)
                all_file.append(file_path)
    return np.asarray(all_file)

# 标准化数据
def normalize_data(all_file):
    for file_path in all_file:
        df = pd.read_csv(file_path)
        # 提取特征和标签
        features = df.iloc[:, 2:8].values.astype('float32')  # 前 n-1 列是特征
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    max = np.max(features,axis=0)
    min = np.min(features,axis=0)
    df=pd.DataFrame(data=np.asarray([mean,std,max,min]))
    writer_into = pd.ExcelWriter('normalize.xlsx')
    df.to_excel(writer_into)
    writer_into.close()

# 数据增强
def oversample_with_augmentation(features, labels,  minority_class=1):
    minority_indices = np.where(labels == minority_class)[0]
    majority_indices = np.where(labels != minority_class)[0]

    num_majority = len(majority_indices)
    print(num_majority)
    num_minority = len(minority_indices)
    num_to_add = int((num_majority - num_minority)/2/num_minority)

    # 为少数类增加样本
    minority_augmented = features[minority_indices]
    label_min = labels[labels == 1]
    # label_min=labels[minority_indices].copy()
    label_min_tihuan = label_min.copy()
    sample = minority_augmented.copy()

    for _ in range(num_to_add):
        noise = np.random.normal(loc=0.0, scale=0.001, size=sample.shape)
        sample=sample + noise
        minority_augmented=np.concatenate([minority_augmented,sample],axis=0)
        label_min=np.concatenate([label_min,label_min_tihuan],axis=0)

    # 对多数类加入随机扰动
    majority_augmented = []
    for idx in majority_indices:
        sample = features[idx]
        noise = np.random.normal(loc=0.0, scale=0.001, size=sample.shape)
        majority_augmented.append(sample + noise)

    majority_augmented = np.array(majority_augmented)

    balanced_features = np.concatenate([majority_augmented, minority_augmented], axis=0)
    balanced_labels = np.concatenate([labels[majority_indices], label_min], axis=0)

    return balanced_features, balanced_labels

def collect_samples_and_labels(all_file,label_dict,time_step,p,augment=True):
    nor=pd.read_excel('normalize.xlsx').values
    all_features = []
    all_labels = []
    for file_path in all_file:
        df = pd.read_csv(file_path)
        # 提取特征和标签
        features = df.iloc[:, 2:8].values.astype('float32')  # 前 n-1 列是特征
        labels = df.iloc[:, -1].apply(lambda x: label_dict.get(x, -1)).values  # 最后一列是标签
        if -1 in labels:
            raise ValueError(f"在文件 {file_path} 中发现未定义的标签")
        #滤波
        features = apply_filter_to_dataset(features,cutoff=60,fs=1/0.005,order=1)
        bcd_time=time_step-int(time_step*p)
        n = (len(features)-int(time_step*p))//bcd_time
        #窗口滑动
        for _ in range(n):
            start=_*bcd_time
            all_features.append(features[start:start+time_step])
            ll=labels[start:start+time_step]
            if len(ll[ll>=12])/len(ll) >0.4:
                all_labels.append(1)
            else:
                all_labels.append(0)


        #all_features.append(features)
        #all_labels.append(labels)

    # 合并所有文件中的数据
    all_features = np.vstack(all_features)
    all_features = (all_features-nor[0,:])/nor[1,:]
    all_features = (nor[2,:]-all_features)/(nor[2,:]-nor[3,:])
    all_labels = np.hstack(all_labels)

    if augment:
        all_features, all_labels = oversample_with_augmentation(all_features, all_labels)

    w = []
    for i in range(num_label):
        if len(all_labels[all_labels==i])!=0:
            w.append(len(all_labels)/num_types/len(all_labels[all_labels==i]))
            print(len(all_labels[all_labels==i])/len(all_labels))
        else:
            w.append(0.1)
    return w,all_features, all_labels



# 数据主目录和类别数
base_dir = "data/MobiAct_Dataset_v2.0/Annotated Data"
num_types = len(file_label)
time_step = int(2/0.005)  # 时间步长度
num_features = 6    # 特征数量
batch_size = 50

# 收集文件路径和标签
file_paths=collect_path(base_dir,num_types)

#标准化，归一化 index=['mean','std','max','min']
#normalize_data(file_paths)

# 按 8:2 比例划分训练集和测试集
train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)
#p是重叠窗口/整个窗口
train_w,train_features, train_labels = collect_samples_and_labels(train_files,label_dict,time_step=time_step,p=0.5)
train_w=tf.constant(train_w, dtype=tf.float32)
train_labels = tf.one_hot(train_labels,depth=num_label)
test_w,test_features, test_labels = collect_samples_and_labels(val_files,label_dict,time_step=time_step,p=0.5)
test_labels = tf.one_hot(test_labels,depth=num_label)

def build_tf_dataset(features, labels, batch_size):
    """
    构建 tf.data.Dataset 数据管道。
    :param features: 特征数组。
    :param labels: 标签数组。
    :param batch_size: 批次大小。
    :param shuffle: 是否打乱数据。
    :return: 数据管道。
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def process_batch(features, labels, time_step):
    # 确保总元素数量可以被分为 (time_step, features.shape[1])
    num_samples = features.shape[0]
    num_features = features.shape[1]

    # 如果样本数量不是 time_step 的整数倍，进行裁剪
    if num_samples % time_step != 0:
        remainder = num_samples % time_step
        features = features[:-remainder]  # 丢弃多余部分
        labels = labels[:-remainder]

    # 调整形状
    features = tf.reshape(features, (-1, time_step, num_features))
    #将时间框的最后个label作为label
    return features, labels



# 构建训练集和测试集数据管道

train_features, train_labels = process_batch(train_features, train_labels, time_step)
test_features, test_labels = process_batch(test_features, test_labels, time_step)


train_dataset = build_tf_dataset(train_features, train_labels, batch_size)
test_dataset = build_tf_dataset(test_features, test_labels, batch_size)


for X_batch, y_batch in train_dataset.take(1):
    print("X_batch shape:", X_batch.shape)  # (batch_size, num_features)
    print("y_batch shape:", y_batch.shape)  # (batch_size,)
    print("First label:", y_batch[0].numpy())


model = lstm_model(time_step,num_types,train_w)
model.fit(train_dataset, epochs=80, validation_data=test_dataset)
