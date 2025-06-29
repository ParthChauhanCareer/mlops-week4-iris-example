import pandas as pd

def test_data_load():
    df = pd.read_csv("iris.csv")
    assert not df.empty, "Dataframe is empty"

def test_column_names():
    df = pd.read_csv("iris.csv")
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert set(df.columns) == expected_columns, "Columns do not match expected"

def test_no_nulls():
    df = pd.read_csv("iris.csv")
    assert df.isnull().sum().sum() == 0, "Data contains nulls"
