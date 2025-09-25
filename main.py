import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import seaborn as sns

data = pd.read_csv('/content/ipl_data.csv')



df = data.copy()
matches_per_venue = df[['mid','venue']].drop_duplicates()
matches_cnt = matches_per_venue['venue'].value_counts()



runs_by_batsmen = df.groupby('batsman')['runs'].sum().sort_values(ascending=False).head(10)

wickets_by_bowlers = df.groupby('bowler')['wickets'].max().sort_values(ascending=False).head(10)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_cols = ['bat_team','bowl_team','venue','batsman','bowler']
label_encoder = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col])
  label_encoder[col] = le

feature_cols = ['bat_team','bowl_team','venue','batsman','bowler','runs','wickets','overs','striker']
target_col = 'total'
x = df[feature_cols]
y = df[target_col]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Input(shape=(int(x_train.shape[1]),)),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(216,activation='relu'),
    keras.layers.Dense(1,activation='linear')
])
huberloss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam',loss=huberloss)

model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))



y_pred = model.predict(x_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
