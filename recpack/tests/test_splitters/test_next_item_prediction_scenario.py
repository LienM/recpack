import pytest
import pandas as pd
from recpack.data.matrix import InteractionMatrix
from recpack.splitters.scenarios import NextItemPrediction


class TestNextItemPredictionScenario():

    @pytest.fixture()
    def data_m_small(self):
        input_dict = {
            InteractionMatrix.USER_IX: [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            InteractionMatrix.ITEM_IX: [0, 1, 2, 0, 1, 2, 3, 3, 5, 6],
            InteractionMatrix.TIMESTAMP_IX: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

        df = pd.DataFrame.from_dict(input_dict)
        df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)
        data = InteractionMatrix(
            df, InteractionMatrix.ITEM_IX, InteractionMatrix.USER_IX, timestamp_ix=InteractionMatrix.TIMESTAMP_IX
        )
        return data

    def test_next_item_prediction_split(self, data_m_small):
        im = data_m_small
        scenario = NextItemPrediction(validation=False)
        scenario._split(im)

        train = scenario.train_X
        test_data_in = scenario.test_data_in
        test_data_out = scenario.test_data_out
        # Intersection asserts: every user should appear in data_out and data_in
        train_users = set(train.indices[0])
        test_data_out_users = set(test_data_out.indices[0])
        test_data_in_users = set(test_data_in.indices[0])
        assert train_users.intersection(test_data_in_users)
        assert train_users.intersection(test_data_out_users)
        # Length asserts: every user should have exactly one data_out and data_in entry
        assert test_data_out.shape[0] == im._df["uid"].nunique()
        assert test_data_in.shape[0] == im._df["uid"].nunique()

    def test_next_item_prediction_split_w_validation(self, data_m_small):
        im = data_m_small
        scenario = NextItemPrediction(validation=True)
        scenario._split(im)

        train = scenario.train_X
        test_data_in = scenario.test_data_in
        test_data_out = scenario.test_data_out
        test_val_in = scenario._validation_data_in
        test_val_out = scenario._validation_data_out
        # Intersection asserts: every user should appear in _out and _in
        train_users = set(train.indices[0])
        test_data_out_users = set(test_data_out.indices[0])
        test_data_in_users = set(test_data_in.indices[0])
        test_val_out_users = set(test_val_out.indices[0])
        test_val_in_users = set(test_val_in.indices[0])
        assert train_users.intersection(test_data_in_users)
        assert train_users.intersection(test_data_out_users)
        assert train_users.intersection(test_val_in_users)
        assert train_users.intersection(test_val_out_users)
        # Length asserts: every user should have exactly one _out and _in entry
        assert test_data_out.shape[0] == im._df["uid"].nunique()
        assert test_data_in.shape[0] == im._df["uid"].nunique()
        assert test_val_out.shape[0] == im._df["uid"].nunique()
        assert test_val_in.shape[0] == im._df["uid"].nunique()
