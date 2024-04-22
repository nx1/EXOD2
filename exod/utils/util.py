import pandas as pd

from exod.utils.logger import logger


def save_info(dictionary, savepath):
    logger.info(f'Saving to {savepath}')
    series = pd.Series(dictionary)
    series.to_csv(savepath)


def save_df(df, savepath):
    if df is None:
        logger.info(f'No df found, not saving to {savepath}')
        return None
    logger.info(f'Saving df to: {savepath}')
    logger.info(f'\n{df}')
    df.to_csv(savepath, index=False)


def get_unique_xy(x, y):
    """Get the unique pairs of two lists."""
    unique_xy = set()  # Set to store unique pairs
    for x, y in zip(x, y):
        unique_xy.add((x, y))
    return unique_xy


