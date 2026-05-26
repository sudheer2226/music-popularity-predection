import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("compressed_data (1).csv")
df = df.dropna()

df.drop(['artist_name', 'song_name', 'lyrics'], axis=1, inplace=True)

median = df['new_artist_popularity'].median()
df['popular'] = (df['new_artist_popularity'] >= median).astype(int)
df.drop('new_artist_popularity', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['language', 'genres'], drop_first=True)

X = df.drop('popular', axis=1)
y = df['popular']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingClassifier()

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
