
from sklearn.linear_model import LogisticRegression


def train_model():
    X = [[1], [2], [3], [4], [5], [6]]
    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    return model


def predict_result(hours):
    model = train_model()
    return int(model.predict([[hours]])[0])


if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully.")
    print("Prediction for 5 hours:", int(model.predict([[5]])[0]))
