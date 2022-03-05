import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data(request):

    df = pd.read_csv("../preprocessed_data.csv")

    return df