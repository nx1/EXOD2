
import pytest
from exod.pre_processing.event_filtering import filter_PN_file

def test_unsupported_obs():
    obsids = ['0820880501', # EPN BURST
              '0153950401'  # EPN Timing Mode
             ]

    for obs in obsids:
        with pytest.raises(Exception):
            filter_PN_file(obs)

def test_PN_PrimeFullWindow():
    obsid = '0882110401'  # EPN PrimeFullWindow
def test_PN_PrimeFullWindowExtended():
    obsid = '0201900201' # EPN PrimeFullWindowExtended
def test_PN_PrimeLargeWindow():
    obsid = '0510010701' # EPN PrimeLargeWindow
def test_PN_PrimeSmallWindow():
    obsid = '0161960201'
def test_MOS_PrimeFullWindow():
    obsid = '0863401101' # EMOS1 PrimeFullWindow
def test_MOS_PrimePartialRFS():
    obsid = '0153951201'
def test_MOS_PrimePartialW2():
    obsid = '0306870101' # EMOS1 PrimePartialW2
def test_MOS_PrimePartialW3():
    obsid = '0810811801' # EMOS1 PrimePartialW3
def test_MOS_PrimePartialW4():
    obsid = '0109070401'
def test_MOS_PrimePartialW5():
    obsid = '0111290401'
def test_MOS_PrimePartialW6():
    obsid = '0116700301'