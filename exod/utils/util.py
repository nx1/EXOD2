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
