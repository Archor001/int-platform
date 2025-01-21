import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Transformer 编码器层
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # 多头自注意力机制
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)  # 残差连接和层归一化

    # 前馈神经网络
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)  # 残差连接和层归一化

    return ff_output

# 构建 Transformer 模型
def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    # 堆叠多个 Transformer 编码器层
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # 全局平均池化
    x = GlobalAveragePooling1D()(x)

    # 输出层
    outputs = Dense(1, activation="sigmoid")(x)  # 二分类任务，使用 sigmoid 激活函数

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

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
    filepath = os.path.join(os.pardir, os.pardir, "int-dataset", "int_data_bmv2_all.csv")

    # 数据预处理
    X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test = preprocess_data(filepath)

    # 构建 Transformer 模型
    input_shape = (X_train.shape[1], X_train.shape[2])  # 输入形状 (序列长度, 特征数)
    head_size = 64  # 自注意力头的大小
    num_heads = 4  # 多头注意力的头数
    ff_dim = 128  # 前馈神经网络的隐藏层维度
    num_layers = 2  # Transformer 编码器层数
    dropout = 0.1  # Dropout 比例

    transformer_model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout)

    # 训练模型
    history = transformer_model.fit(
        X_train, y_anomaly_train,
        validation_data=(X_test, y_anomaly_test),
        epochs=20,
        batch_size=64
    )

    # 保存二分类结果
    save_anomaly_results(transformer_model, X_test, y_anomaly_test, original_X_test)

# 运行主函数
if __name__ == "__main__":
    main()