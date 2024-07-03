if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5000")

@custom
def train_model(train_data, *args, **kwargs):

    train_dicts = train_data['split'][0]['train_dicts']
    y_train = train_data['split'][0]['y_train']

    dv = DictVectorizer()
    x_train = dv.fit_transform(train_dicts)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    print(lr.intercept_)

    with mlflow.start_run():
        with open("dv.bin", "wb") as f:
            pickle.dump(dv, f)

        mlflow.log_artifact("dv.bin")
        mlflow.sklearn.log_model(lr, "model")

    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
