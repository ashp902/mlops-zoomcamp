if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def split(data, *args, **kwargs):
    
    train_dicts = data[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    y_train = y_train = data['duration'].values

    return {'train_dicts': train_dicts, 'y_train': y_train}



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
