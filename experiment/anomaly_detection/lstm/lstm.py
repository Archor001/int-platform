import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 数据预处理函数
def preprocess_data(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 保存原始特征数据
    original_features = df[
        ["Hop Latency (ms)", "e2e Delay (ms)", "Queue Occupancy", "IAT (us)", "Hop Jitter (us)", "e2e Jitter (us)"]]

    # 归一化特征
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(original_features)

    # 提取标签
    anomaly_labels = df["Is Anomaly"].values

    # 将数据转换为时序序列
    sequence_length = 50  # 使用前50个时间步预测当前状态
    X, y_anomaly, original_X = [], [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])
        y_anomaly.append(anomaly_labels[i])
        original_X.append(original_features.iloc[i - sequence_length:i].values)  # 保存原始特征数据

    X = np.array(X)
    y_anomaly = np.array(y_anomaly)
    original_X = np.array(original_X)  # 转换为NumPy数组

    # 划分训练集和测试集
    X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test = train_test_split(
        X, y_anomaly, original_X, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test

# 训练二分类模型
def train_anomaly_detection_model(X_train, y_anomaly_train, X_test, y_anomaly_test):
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=False)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    anomaly_output = Dense(1, activation='sigmoid', name='anomaly_output')(lstm_out)
    model = Model(inputs, anomaly_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(
        X_train, y_anomaly_train,
        validation_data=(X_test, y_anomaly_test),
        epochs=10,
        batch_size=64
    )
    return model, history

# 保存二分类结果
def save_anomaly_results(model, X, y_anomaly, original_X):
    anomaly_pred = model.predict(X)
    anomaly_pred_labels = (anomaly_pred > 0.5).astype(int).flatten()

    # 将结果保存到CSV文件
    results = pd.DataFrame({
        'True Anomaly': y_anomaly,
        'Predicted Anomaly': anomaly_pred_labels,
        **{f"Feature_{i}": original_X[:, -1, i] for i in range(original_X.shape[2])}  # 添加原始特征数据
    })
    results.to_csv('anomaly_detection_results.csv', index=False)
    print("Anomaly detection results saved to anomaly_detection_results.csv")

    return anomaly_pred_labels  # 返回预测标签

# 主函数
def main():
    # 数据文件路径
    # 上两级目录：os.pardir 表示上一级目录，重复两次表示上两级目录
    filepath = os.path.join(os.pardir, os.pardir, "int-dataset", "int_data_bmv2_all.csv")

    # 数据预处理
    X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test = preprocess_data(filepath)

    # 训练二分类模型
    anomaly_model, anomaly_history = train_anomaly_detection_model(X_train, y_anomaly_train, X_test, y_anomaly_test)

    # 保存二分类结果
    save_anomaly_results(anomaly_model, X_test, y_anomaly_test, original_X_test)

# 运行主函数
if __name__ == "__main__":
    main()