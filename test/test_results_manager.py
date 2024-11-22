import pytest
from exod.post_processing.results_manager import ResultsManager


@pytest.fixture
def rm():
    return ResultsManager()

def test_region_identifier_order(rm):
    df_reg = rm.df_regions
    df_lcf = rm.df_lc_features

    df_reg['region_identifier'] = df_reg.apply(lambda row: f"('{row['runid']}', '{row['label']}')", axis=1)
    df_reg = df_reg.set_index(df_reg['region_identifier'])
    assert df_reg.index.equals(rm.df_lc_idx.index)
    df_lcf['region_identifier'] = df_lcf.apply(lambda row: f"('{row['runid']}', '{row['label']}')", axis=1)
    df_lcf = df_lcf.set_index(df_lcf['region_identifier'])
    assert df_lcf.index.equals(rm.df_lc_idx.index)
