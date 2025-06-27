import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. 데이터 불러오기
data = pd.read_csv('web_server_logs_2.csv')

# 2. 파생 변수 생성
# timestamp에서 hour 추출
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour

# status_code가 400 이상이면 오류로 간주
data['is_error'] = data['status_code'] >= 400

# label 컬럼 생성 (이진 분류용)
data['label'] = data['is_error'].astype(int)

# 3. 특성 선택
features = ['hour', 'status_code']
target = 'label'

# 4. 학습/테스트 데이터 분리 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# 5. 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습 (Logistic Regression 사용)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. 예측 및 평가
y_pred = model.predict(X_test_scaled)

# 평가 지표 출력: Accuracy, Precision, Recall, F1-Score
report = classification_report(y_test, y_pred, target_names=['Non-Malicious', 'Malicious'])
print(report)

# 분류 결과 해석 요약
# - precision (정밀도): 예측한 악성 요청 중 100%가 실제 악성 (오탐 없음)
# - recall (재현율): 실제 악성 요청을 100% 탐지 (누락 없음)
# - f1-score: 정밀도와 재현율이 완벽해 f1도 1.00
# - support: 평가에 사용된 샘플 수 (비악성 201건, 악성 99건)
# - accuracy: 전체 300건 중 100% 정확히 예측 (정확도 1.00)
# → 모델 성능이 매우 우수하나, 과적합 여부도 검토 필요