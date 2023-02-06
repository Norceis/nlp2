import csv
from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import PredictionOutput

from classes import MakeTorchData


def create_standardized_data(feature_number: Union[None, int] = None) -> None:
    """
    This function goes through each line in the dataset via "next" generator and depending
    on the sample rewrites it, so the data is standardized (explained in detail in readme.md)
    :param feature_number: optional, specifies number of columns, if the dataset does not contain header
    :return: None, writes file directly to .csv
    """

    rows = []
    with open('../data/recruitment_data.csv', 'r', encoding='utf-8') as opened_file:
        reader = csv.reader(opened_file)
        while True:
            try:
                line = next(reader)
                if len(line) == 1:
                    line = line[0]     # because some samples were enclosed in additional quote we need to
                    rows.append(line)  # get it out of the one-element list it was put into by csv module
                else:
                    rows.append(line)  # rest of the samples were not modified
            except StopIteration:
                break

    if not feature_number:
        feature_number = len(rows[0])  # number of features present in header of csv file

    with open('../data/recruitment_data_modified_python.csv', 'w', encoding='utf-8') as writing_file:

        for sample in rows:
            if len(sample) > feature_number:
                try:
                    writing_file.write(sample + '\n')  # outlier samples (n_features > 13) caused problems
                except TypeError:  # here, this solution works correctly
                    writing_file.write(','.join(sample) + '\n')
            else:
                writing_file.write(','.join(sample) + '\n')


def preprocess_data(data: pd.DataFrame,
                    columns_to_drop: Union[None, list] = None,
                    price_clip: int = 1000) -> pd.DataFrame:
    """
    This function summarizes the exploratory work done in data_preprocessing.ipynb.
    Due to the nature of the problem posed in this project it was suited for this particular
    dataset, but it can be improved in more general form.
    :param data: DataFrame with the raw data
    :param columns_to_drop: optional, which columns to drop in order to have more compact DataFrame
    :param price_clip: optional, used to cut off samples under this treshold in 'Price'
    :return: processed DataFrame with 'Price', concatenated 'Days_passed', 'Name', 'Description' columns
    """

    # dropping nans in price
    data_dropped_price_nans = data.dropna(subset=['Price']).reset_index(drop=True)

    # adding column with date only
    data_dropped_price_nans['Date'] = pd.to_datetime(data_dropped_price_nans['Added_at']).dt.date

    # clipping samples for our needs
    data_reduced_dims = data_dropped_price_nans[data_dropped_price_nans.Condition == 'Używane']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Type == 'Sprawny']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Brand == 'iPhone']

    # dropping insignificant columns
    if columns_to_drop is None:
        columns_to_drop = ['Voivodeship', 'Scrap_time', 'Views',
                           'User_since', 'Added_at', 'URL', 'Brand', 'Condition',
                           'Offer_from', 'Type']

    data_reduced_dims = data_reduced_dims.drop(columns=columns_to_drop)

    # filtering phone cases, offered services, other phones based on price
    data_filtered = data_reduced_dims[data_reduced_dims.Price > price_clip]

    # applying lowercase to all letters in Name and Description
    data_filtered.loc[:, 'Name'] = data_filtered.loc[:, 'Name'].str.lower()
    data_filtered.loc[:, 'Description'] = data_filtered.loc[:, 'Description'].str.lower()

    # adding column with days passed since first date
    data_filtered['First_date'] = data_filtered['Date'].min()
    data_filtered['Date'] = pd.to_datetime(data_filtered.Date, format='%Y-%m-%d')
    data_filtered['First_date'] = pd.to_datetime(data_filtered.First_date, format='%Y-%m-%d')
    data_filtered['Days_passed'] = data_filtered['Date'] - data_filtered['First_date']
    data_filtered['Days_passed_name'] = data_filtered['Days_passed'].astype(str) + ' ' + data_filtered['Name']
    data_filtered['Days_passed_name_desc'] = data_filtered['Days_passed'].astype(str) + ' ' + \
                                             data_filtered['Name'] + ' ' + data_filtered['Description']

    # dropping insignificant columns
    data_concatenated = data_filtered.drop(columns=['Name', 'Description',
                                                    'First_date', 'Date', 'Days_passed']).reset_index(drop=True)

    return data_concatenated


def compute_metrics(eval_pred: PredictionOutput) -> Dict:
    """
    This functions returns a dictionary of metrics, that is used in torch models for checkpoints, but also can be used
    manually to test data predicted by the model.
    :param eval_pred: Data prediction outputted by torch model
    :return: Dictionary of metrics
    """
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def foresee(model: Trainer,
            scaler: StandardScaler,
            tokenizer: AutoTokenizer,
            text: str = 'iphone 11',
            days: tuple = (1, 90)) -> np.ndarray:
    """
    This function returns 'Price' predictions of a given text in a time period
    :param model: Trainer object, trained on the dataset
    :param scaler: Scaler, fitted to the dataset, it's needed for inversion of data scaling
    :param tokenizer: Tokenizer object, used to tokenize the dataset in order to be predicted by model
    :param text: Text to be processed; to the beginning of this string is added number of days specified in :param days
    :param days: Range of days (equal to number of samples in this prediction dataset) that will be added to the text
    :return: Numpy array with 'Price' predictions (rising number of days equals to time progression)
    """
    samples_to_predict = []
    placeholder_prices = []

    # result of this loop are 2 lists:
    # samples are in form for example: '10 days iphone 11'
    # all placeholder_prices are equal to 0

    for days_passed in range(*days):
        samples_to_predict.append(str(days_passed) + ' days ' + text)
        placeholder_prices.append(0)

    tokens = tokenizer(samples_to_predict, truncation=True, padding=True, max_length=50)
    dataset = MakeTorchData(tokens, np.asarray(placeholder_prices).ravel())
    predictions = model.predict(dataset)
    prices = scaler.inverse_transform(np.asarray(predictions[0]).reshape(-1, 1))

    return prices
