import pandas as pd
from typing import Union

def preprocess_data(data: pd.DataFrame,
                    columns_to_drop: Union[None, list] = None,
                    price_clip: int = 1000) -> pd.DataFrame:
    # dropping nans in price
    data_dropped_price_nans = data.dropna(subset=['Price']).reset_index(drop=True)

    # adding important data type and column with date only
    # data_dropped_price_nans['Added_at'] = pd.to_datetime(data_dropped_price_nans['Added_at'])
    data_dropped_price_nans['Date'] = pd.to_datetime(data_dropped_price_nans['Added_at']).dt.date

    # clipping samples for our needs
    data_reduced_dims = data_dropped_price_nans[data_dropped_price_nans.Condition == 'Używane']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Type == 'Sprawny']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Brand == 'iPhone']

    if columns_to_drop is None:
        columns_to_drop = ['Voivodeship', 'Scrap_time', 'Views',
                           'User_since', 'Added_at', 'URL', 'Brand', 'Condition',
                           'Offer_from', 'Type']

    data_reduced_dims = data_reduced_dims.drop(columns=columns_to_drop)

    # filtering phone cases, offered services, other phones
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

    # removing insignificant columns
    data_concatenated = data_filtered.drop(columns=['Name', 'Description',
                                                    'First_date', 'Date', 'Days_passed']).reset_index(drop=True)

    return data_concatenated
