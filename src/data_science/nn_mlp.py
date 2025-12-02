# nn_mlp.py
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def mlp_classifier(hidden=(64, 32), alpha=1e-4, max_iter=300):
    clf = MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter, random_state=42
    )
    pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("mlp", clf)])
    return pipe


def mlp_regressor(hidden=(128, 64), alpha=1e-4, max_iter=400):
    reg = MLPRegressor(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter, random_state=42
    )
    pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("mlp", reg)])
    return pipe
