

class Scenario:

    def __init__(self):
        self.training_data = None
        self.validation_data_in = None
        self.validation_data_out = None
        self.test_data_in = None
        self.test_data_out = None

    def split(self, data, data_2):
        pass

    @property
    def validation_data(self):
        return (self.validation_data_in, self.validation_data_out)

    @property
    def test_data(self):
        return (self.test_data_in, self.test_data_out)
