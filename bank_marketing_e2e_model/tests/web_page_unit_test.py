"""
This is a unit test for testing the functionality of the streamlit web page.

The test selects an option from an inference data point and
verifies that entry is as expected.
"""


import time

import unittest

from selenium import webdriver
from selenium.webdriver.common.by import By


class WebPageDropUnitTest(unittest.TestCase):
    """
    A class to perform a unit test on the web page.

    Methods:
        setUp: Load chrome driver
        test_web_page: Tests web page by selecting a feature category
        tearDown: Close chrome driver
    """

    # Set up
    def setUp(self):
        """Load chrome driver"""
        # Set driver
        self.driver = webdriver.Chrome()
        # Set local url
        self.local_url = 'http://localhost:8501'

    # Perform unit test
    def test_web_page(self):
        """Tests web page by selecting a feature category"""
        # Start the web driver
        driver = self.driver
        # Open the Streamlit app
        driver.get(self.local_url)
        time.sleep(10)  # zzzzzzzzzz
        # Select first drop-down box
        driver.find_element(
            By.CSS_SELECTOR,
            '[data-testid="stSelectbox"]'
        ).click()
        time.sleep(3)  # zzz
        # Select an option from drop-down
        driver.find_element(
            By.XPATH,
            '/html/body/div/div[2]/div/div/div/div/div/div/ul/div/div/li[2]/div/div/div'
        ).click()
        time.sleep(3)  # zzz
        # Get result
        result = driver.find_element(
            By.XPATH,
            '/html/body/div/div[1]/div[1]/div/div/div/section/'
            'div[1]/div/div/div[14]/div/div/div/div[1]/div[1]'
        ).text
        # Define the expected output
        expected_output = 'management'
        self.assertEqual(result, expected_output, "Selected item does not match expected output.")

    # Close web page
    def tearDown(self):
        """Close chrome driver"""
        self.driver.quit()


if __name__ == '__main__':
    unittest.main()
