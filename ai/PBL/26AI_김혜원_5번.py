import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 데이터 로드
data = pd.read_csv('customer_data_balanced.csv')

# 2. 전처리
data = pd.get_dummies(data, columns=['ContractType'])
target = 'IsChurn'
features = data.columns.drop(target)

X = data[features]
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 클래스 불균형 가중치 설정
class_weights = {0: 1.0, 1: 2.0}

# 4. 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. 학습
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=2
)

# 6. 예측 (threshold 0.5 기본)
y_pred_prob = model.predict(X_test).flatten()
threshold = 0.5
y_pred = (y_pred_prob >= threshold).astype(int)

# 7. 평가 지표 출력
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Threshold: {threshold}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}\n")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)

# TP (True Positive) — 참 긍정
# 모델이 ‘긍정(1, 예: 이탈)’이라고 예측했고, 실제도 ‘긍정’인 경우
# 실제 이탈한 고객을 정확히 맞춘 것

# FP (False Positive) — 거짓 긍정
# 모델이 ‘긍정’이라고 예측했지만, 실제는 ‘부정(0, 예: 이탈 아님)’인 경우
# 이탈하지 않은 고객을 잘못 이탈자로 판단한 것 → ‘오탐’이라고도 함

# FN (False Negative) — 거짓 부정
# 모델이 ‘부정’이라고 예측했지만, 실제는 ‘긍정’인 경우
# 실제 이탈한 고객을 놓친 것 → ‘누락’ 또는 ‘미탐’이라고도 함

# TN (True Negative) — 참 부정
# 모델이 ‘부정’이라고 예측했고, 실제도 ‘부정’인 경우
# 이탈하지 않은 고객을 정확히 맞춘 것

print("\n[Confusion Matrix 해석]")
print(f"TP(참 긍정): {cm[1,1]}, FP(거짓 긍정): {cm[0,1]}")
print(f"FN(거짓 부정): {cm[1,0]}, TN(참 부정): {cm[0,0]}")


