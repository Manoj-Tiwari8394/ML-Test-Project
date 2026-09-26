from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_data
from src.preprocess import split_features_target, split_train_test


def test_split_features_target_separates_measurements_and_species():
    data = load_iris_data()

    features, target = split_features_target(data)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target) == len(data)


def test_train_test_split_is_reproducible_and_stratified():
    data = load_iris_data()

    first = split_train_test(data, test_size=0.2, random_state=7)
    second = split_train_test(data, test_size=0.2, random_state=7)
    X_train, X_test, y_train, y_test = first

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert set(y_train) == set(y_test) == set(data[TARGET_COLUMN])
    assert X_train.equals(second[0])
    assert X_test.equals(second[1])
    assert y_train.equals(second[2])
    assert y_test.equals(second[3])
