"""
This is a unit test for testing model inference.

The test verifies that an obviously true inference
example is predicted as true by the model.
"""


import os
import unittest
import yaml
import joblib
import pandas as pd


class ModelUnitTest(unittest.TestCase):
    """
    This class specifies the model inference unit test.
    """

    # Static method for loading config
    @staticmethod
    def load_model_pipeline():
        # Load config
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        with open(os.path.join(parent_dir, 'config.yaml'), 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)  # Load config file
        # Load model pipeline using config
        return joblib.load(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), '..', '..', config['output']['model_path']
                )
            )
        )

    # Perform unit test
    def test_inference(self):
        # Create an inference dictionary that should give us a result of 1
        inference_df = pd.DataFrame(
            [
                {
                    "age": 1,
                    "job": "student",
                    "marital": "single",
                    "education": "tertiary",
                    "default": "yes",
                    "balance": 40000,
                    "housing": "no",
                    "loan": "no",
                    "contact": "cellular",
                    "day_of_week": 31,
                    "month": "mar",
                    "duration": 2000,
                    "campaign": 20,
                    "pdays": 400,
                    "previous": 50,
                    "poutcome": "success"
                }
            ]
        )
        # Load model
        model = self.load_model_pipeline()
        # Perform inference
        result = model.predict(inference_df)
        # Define the expected output
        expected_output = 1
        # Assert that the result matches the expected output
        self.assertEqual(
            result,
            expected_output,
            "Model inference result does not match expected output."
        )


if __name__ == '__main__':
    unittest.main()
