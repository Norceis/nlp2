import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # dropping nans in price
    data_dropped_price_nans = data.dropna(subset=['Price']).reset_index(drop=True)

    # adding important data type and column with date only
    data_dropped_price_nans['Added_at'] = pd.to_datetime(data_dropped_price_nans['Added_at'])
    data_dropped_price_nans['Date'] = pd.to_datetime(data_dropped_price_nans['Added_at']).dt.date

    # clipping samples for our needs
    data_reduced_dims = data_dropped_price_nans[data_dropped_price_nans.Condition == 'Używane']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Type == 'Sprawny']
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Brand == 'iPhone']
    data_reduced_dims = data_reduced_dims.drop(
        columns=['Voivodeship', 'Scrap_time', 'Views', 'User_since', 'Added_at', 'URL', 'Brand', 'Condition',
                 'Offer_from', 'Type'])

    # filtering phone cases, offered services, other phones
    data_filtered = data_reduced_dims[data_reduced_dims.Price > 1000]

    # applying lowercase to all letters in Name and Description and also concatenating them
    data_filtered.loc[:, 'Name'] = data_filtered.loc[:, 'Name'].str.lower()
    data_filtered.loc[:, 'Description'] = data_filtered.loc[:, 'Description'].str.lower()
    data_filtered['Concatenated_description'] = data_filtered['Name'] + ' ' + data_filtered['Description']

    # removing insignificant columns
    data_concatenated = data_filtered.drop(columns=['Name', 'Description']).reset_index(drop=True)

    return data_concatenated