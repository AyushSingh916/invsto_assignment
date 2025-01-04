import unittest
import pandas as pd
from datetime import datetime

data = pd.read_csv('stock_data.csv')

class TestDataValidation(unittest.TestCase):

    def test_datetime_column(self):
        for value in data['datetime']:
            self.assertIsInstance(pd.to_datetime(value), pd.Timestamp)
    
    def test_numeric_columns(self):
        for col in ['close', 'high', 'low', 'open']:
            for value in data[col]:
                self.assertIsInstance(value, (float, int))
    
    def test_integer_column(self):
        for value in data['volume']:
            self.assertIsInstance(value, int)
    
    def test_instrument_column(self):
        for value in data['instrument']:
            self.assertIsInstance(value, str)

if __name__ == '__main__':
    unittest.main()
