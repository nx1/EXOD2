import pandas
import pandas as pd

from exod.utils.logger import logger


def save_info(dictionary, savepath):
    logger.info(f'Saving to {savepath}')
    series = pd.Series(dictionary)
    series.to_csv(savepath)


def load_info(loadpath):
    logger.info(f'Loading info from {loadpath}')
    df = pd.read_csv(loadpath, index_col=0).T
    dict_list = df.to_dict(orient='records')[0]
    return dict_list


def save_df(df, savepath):
    if df is None:
        logger.info(f'No df found, not saving to {savepath}')
        return None
    if df.empty:
        logger.info(f'df is empty, not saving to {savepath}')
        return None
    logger.info(f'Saving df to: {savepath}')
    logger.info(f'\n{df}')
    df.to_csv(savepath, index=False)


def load_df(loadpath):
    logger.info(f'Loading df from: {loadpath}')
    df = pd.read_csv(loadpath)
    return df


def save_result(key, value, runid, savedir):
    """
    Save a key/value pair to a .csv file.
    """
    if isinstance(value, pd.DataFrame):
        # Append the runid to the dataframe if it is not there
        if 'runid' not in value.columns:
            value['runid'] = runid
        save_df(df=value, savepath=savedir / f'{key}.csv')
    elif isinstance(value, dict):
        # append the runid to the dictionary if it is not there
        value['runid'] = runid
        save_info(dictionary=value, savepath=savedir / f'{key}.csv')
    else:
        logger.warning(f'{key} {value} is not a dict or df!!')


def get_unique_xy(x, y):
    """Get the unique pairs of two lists."""
    unique_xy = set()  # Set to store unique pairs
    for x, y in zip(x, y):
        unique_xy.add((x, y))
    return unique_xy


