
from ml_model import train_model, predict_result


def test_model_training():
    model = train_model()
    assert model is not None


def test_prediction():
    result = predict_result(5)
    assert result in [0, 1]


def test_prediction_type():
    result = predict_result(5)
    assert isinstance(result, int)
