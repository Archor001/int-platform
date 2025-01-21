import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# 数据预处理函数
def preprocess_data(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 提取特征和目标列
    features = df[["e2e Delay (ms)", "Queue Occupancy", "Hop Latency (ms)"]]  # 特征
    target = df["Modal Type"]  # 目标变量

    # 将模态类型编码为数值
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # 将目标值转换为 one-hot 编码
    target_one_hot = to_categorical(target_encoded)

    # 归一化特征数据
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # 将数据转换为时序序列
    sequence_length = 50  # 使用前50个时间步预测当前状态
    X, y, original_X, true_modal_types = [], [], [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])  # 特征序列
        y.append(target_one_hot[i])  # 目标值（one-hot 编码）
        original_X.append(features.iloc[i - sequence_length:i].values)  # 保存原始特征数据
        true_modal_types.append(target[i])  # 保存真实模态类型

    X = np.array(X)
    y = np.array(y)
    original_X = np.array(original_X)  # 转换为NumPy数组
    true_modal_types = np.array(true_modal_types)  # 转换为NumPy数组

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, original_X_train, original_X_test, true_modal_types_train, true_modal_types_test = train_test_split(
        X, y, original_X, true_modal_types, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, original_X_train, original_X_test, true_modal_types_test, label_encoder

# 训练模态分类模型
def train_modal_classification_model(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(32, return_sequences=False)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    dense_out = Dense(32, activation='relu')(lstm_out)
    modal_output = Dense(y_train.shape[1], activation='softmax', name='modal_output')(dense_out)
    model = Model(inputs, modal_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=64
    )
    return model, history

# 保存模态分类结果
def save_modal_results(model, X, true_modal_types, original_X, label_encoder):
    modal_pred = model.predict(X)
    modal_pred_labels = np.argmax(modal_pred, axis=1)  # 将 one-hot 编码转换为类别标签

    # 将数值标签转换回模态类型
    modal_pred_types = label_encoder.inverse_transform(modal_pred_labels)

    # 提取原始特征数据（最后一个时间步的数据）
    original_features_last_step = original_X[:, -1, :]  # 取每个序列的最后一个时间步

    # 将结果保存到CSV文件
    results = pd.DataFrame({
        'True Modal Type': true_modal_types,  # 直接从原始数据中提取的真实模态类型
        'Predicted Modal Type': modal_pred_types,
        'e2e Delay (ms)': original_features_last_step[:, 0],  # 原始 e2e Delay
        'Queue Occupancy': original_features_last_step[:, 1],  # 原始 Queue Occupancy
        'Hop Latency (ms)': original_features_last_step[:, 2]  # 原始 Hop Latency
    })
    results.to_csv('modal_classification_results.csv', index=False)
    print("Modal classification results saved to modal_classification_results.csv")

    # 输出分类报告
    print("Classification Report:")
    print(classification_report(true_modal_types, modal_pred_types))

    # 输出准确率
    accuracy = accuracy_score(true_modal_types, modal_pred_types)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return modal_pred_types  # 返回预测标签

# 主函数
def main():
    # 数据文件路径
    filepath = os.path.join(os.pardir, os.pardir, "int-dataset", "int_data_bmv2_normal.csv")

    # 数据预处理
    X_train, X_test, y_train, y_test, original_X_train, original_X_test, true_modal_types_test, label_encoder = preprocess_data(filepath)

    # 训练模态分类模型
    modal_model, modal_history = train_modal_classification_model(X_train, y_train, X_test, y_test)

    # 保存模态分类结果
    save_modal_results(modal_model, X_test, true_modal_types_test, original_X_test, label_encoder)

# 运行主函数
if __name__ == "__main__":
    main()