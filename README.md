# bank-marketing-e2e-model

This repo creates a basic end-to-end inference pipeline.

We import data from UCI, transform and save this data, use it to train and evaluate a model, and allow users to make inferences using this model.

<div align="center">
<img width="378" alt="image" src="https://github.com/Haydn-Martin/bank-marketing-e2e-model/assets/76120434/5b2b5b33-8e60-4cae-8253-b4250a185c95">
<p>Streamlit webpage UI screenshot</p>
</div>

## Description

In this project we train a logistic regression model to predict if a customer will subscribe to a term deposit. The model achieves an AUC ROC score of >65% with no hyperparameter optimisation.

The data comes from UCI. From the description:

"The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y)."

You can find it [here](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The data was prepared using pandas and the model was trained and evaluated using sklearn. A simple web page was created using streamlit that allows users to input inference examples for the model to use to make a prediction. Users are able to perform all the key steps in the process via the web page (get data, train, evaluate, and make a prediction).

## Motivation

The point of this project is to illustrate the fact that data scientists can create a (semi-robust) (but basic) end-to-end inference pipeline without the need for other teams, fancy tools, or months of development.

It is faaaar from comprehensive or complete. Shortcuts have been taken. Things have been missed. Patterns and packages have been used to _illustrate_ how one might use these rather than always implemented to the full extent they should be in a production application.

## Usage

1. Clone and navigate to the repo
2. Run `docker compose up` in your terminal at the root directory
3. Naviagte to [localhost:8501](http://localhost:8501/) in your browser

## Main Tools

- `poetry`
- `config.yaml`
- `numpy` + `pandas`
- `sklearn`
- `unittest`
- `selenium`
- `streamlit`
- `mypy` + `pylint`
- `docker`

## Potential Improvements
- Abstract away UCI-specific data source to allow inputs from other sources
- Expand the config to allow other models, training options, validation techniques, etc.
- Experiment with data prep, models, hyperparameters to improve model performance
- Move away from streamlit to allow UI customisability (REST API, HTML, CSS, JS)
- Improve testing (add more unit tests, validate input data, etc.)
- Specific exceptions
- Etc.

## License

[MIT](https://choosealicense.com/licenses/mit/)
