import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. 数据预处理函数
def preprocess_data(data):
    """
    数据预处理：将 Modal Type 转换为数值标签，标准化特征数据，并转换为 Transformer 所需的格式。
    """
    # 将 Modal Type 转换为数值标签
    label_encoder = LabelEncoder()
    data['Modal Type'] = label_encoder.fit_transform(data['Modal Type'])

    # 选择特征和目标变量
    features = data[['Hop Latency (ms)', 'e2e Delay (ms)', 'Queue Occupancy']]
    target = data['Modal Type']

    # 标准化特征数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 将目标变量转换为 one-hot 编码
    target_onehot = to_categorical(target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_onehot, test_size=0.2, random_state=42)

    # 将数据转换为 Transformer 所需的 3D 格式 [samples, timesteps, features]
    X_train_transformer = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_transformer = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train_transformer, X_test_transformer, y_train, y_test, label_encoder, scaler, features, target

# 2. Transformer 编码器层
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformer 编码器层。
    """
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)  # 残差连接 + LayerNorm

    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)  # 残差连接 + LayerNorm

    return ff_output

# 3. 构建 Transformer 模型函数
def build_transformer_model(input_shape, num_classes, head_size=32, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
    """
    构建 Transformer 模型。
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # 堆叠多个 Transformer 编码器层
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # 全局平均池化
    x = GlobalAveragePooling1D()(x)

    # 输出层
    outputs = Dense(num_classes, activation="softmax")(x)

    # 构建模型
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. 训练模型函数
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    训练 Transformer 模型，并返回训练历史记录。
    """
    # 使用 EarlyStopping 防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# 5. 评估模型并保存结果函数
def evaluate_and_save_results(model, X_test, y_test, label_encoder, features_test, target_test, output_file='modal_classification_results.csv'):
    """
    评估模型，并将结果保存到 CSV 文件中。
    """
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))  # 将预测结果转换为类别标签
    y_true_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))  # 将真实标签转换为类别标签

    features_test = features_test.to_numpy()
    # 保存结果到 CSV 文件
    results = pd.DataFrame({
        'Hop Latency (ms)': features_test[:, 0],  # 原始特征数据
        'e2e Delay (ms)': features_test[:, 1],
        'Queue Occupancy': features_test[:, 2],
        'True Modal Type': y_true_labels,
        'Predicted Modal Type': y_pred_labels
    })
    results.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")

# 6. 主函数
def main():
    # 读取数据
    filename = os.path.join(os.pardir, os.pardir, "int-dataset", "int_data_bmv2_normal.csv")
    data = pd.read_csv(filename)

    # 数据预处理
    X_train_transformer, X_test_transformer, y_train, y_test, label_encoder, scaler, features, target = preprocess_data(data)

    # 获取测试集的原始特征数据
    _, X_test_original, _, y_test_original = train_test_split(features, target, test_size=0.2, random_state=42)

    # 构建 Transformer 模型
    input_shape = (X_train_transformer.shape[1], X_train_transformer.shape[2])  # [timesteps, features]
    num_classes = y_train.shape[1]
    model = build_transformer_model(input_shape, num_classes)

    # 训练模型
    train_model(model, X_train_transformer, y_train, X_test_transformer, y_test)

    # 评估模型并保存结果
    evaluate_and_save_results(model, X_test_transformer, y_test, label_encoder, X_test_original, y_test_original)

    # 保存模型
    model.save('transformer_model.h5')
    print("Transformer 模型已保存为 transformer_model.h5")

# 运行主函数
if __name__ == "__main__":
    main()