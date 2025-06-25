import joblib
from sklearn.datasets import make_regression

X, x = make_regression(n_samples=100, n_features=1, noise=0.1)

loaded_model = joblib.load('linear_model.pkl')
print('load ok!')

y_pred = loaded_model.predict(X)
print(y_pred)