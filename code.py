import kagglehub
path = kagglehub.dataset_download("shreyanshverma27/online-salesdataset-popular-marketplace-data")
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
df = pd.read_csv(os.path.join(path, 'Online Sales Data.csv'))
df = shuffle(df)
df
df = df.drop_duplicates()
for col in df.columns:
 if df[col].dtype == "object":
 df[col] = df[col].fillna(df[col].mode()[0])
 else:
 df[col] = df[col].fillna(df[col].mean())
df
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].bar(df['Product Category'], df['Units Sold'])
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_title(f'Frequency of Product Category Sold')
axes[0, 0].set_xlabel('Product Category')
axes[0, 0].set_ylabel('Units Sold')
axes[0, 1].hist(df['Units Sold'])
axes[0, 1].set_title(f'Frequency of Units Sold')
axes[0, 1].set_xlabel('Units Sold')
axes[0, 1].set_ylabel('Frequency')
axes[1, 0].hist(df['Unit Price'])
axes[1, 0].set_title(f'Frequency of Unit Price')
axes[1, 0].set_xlabel('Unit Price')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].hist(df['Total Revenue'])
axes[1, 1].set_title(f'Frequency of Total Revenue')
axes[1, 1].set_xlabel('Total Revenue')
axes[1, 1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
 df[col] = le.fit_transform(df[col])
scaler = StandardScaler()
X = df.drop('Product Category', axis=1)
y = df['Product Category']
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
 X_scaled, y, test_size=0.2, random_state=3327
)
model = SVC()
model.fit(X_train, y_train)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, test_predict)
print(f"Training Accuracy: {train_accuracy*100}%")
print(f"Test Accuracy: {test_accuracy*100}%")
