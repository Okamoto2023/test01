import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#git
# add comment from test_v
# CSVファイルを読み込み、Datetimeをインデックスとして設定する
df = pd.read_csv('data.csv', index_col='Datetime', parse_dates=['Datetime'])

# 学習データの範囲を指定する
start_date = '2022-01-01'
end_date = '2022-02-01'
train_data = df.loc[start_date:end_date]

# テストデータの範囲を指定する
start_date = '2022-02-02'
end_date = '2022-02-03'
test_data = df.loc[start_date:end_date]

# 学習データを正規化する
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

# 学習データを生成する
def generate_train_data(train_data, lookback):
    X_train = []
    y_train = []
    for i in range(lookback, len(train_data)):
        X_train.append(train_data[i-lookback:i])
        y_train.append(train_data[i][3])  # Closeの列を取得
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

# LSTMモデルをトレーニングする
lookback = 24  # 過去24時間分のデータを使用する
X_train, y_train = generate_train_data(train_data, lookback)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# テストデータを予測する
def predict_close_price(test_data, target_time, model, lookback):
    X_test = []
    target_index = test_data.index.get_loc(target_time)
    for i in range(target_index - lookback, target_index):
        X_test.append(test_data.iloc[i:i+1])
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test.reshape(1, lookback, X_test.shape[2]))
    return scaler.inverse_transform(y_pred)[0][0]

# 任意の時刻から1時間後のCloseを予測する
target_time = '2022-02-02 01:00:00'
predicted_close_price = predict_close_price(test_data, target_time, model, lookback)
print(f"Predicted Close Price at {target_time}: {predicted_close_price}")
