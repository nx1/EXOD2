from re import X
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.wcs import FITSFixedWarning, WCS
import warnings

warnings.filterwarnings('ignore', category=FITSFixedWarning)

def wcs_from_eventlist_file(filename):
    f  = fits.open(filename)
    h  = f[0].header

    ximagebinsize = 80
    yimagebinsize = 80

    ctype_x = h['REFXCTYP']  # 'RA---TAN'
    ctype_y = h['REFYCTYP']  # 'DEC--TAN'
    crval_x = h['REFXCRVL']  # 266.433791666667
    crval_y = h['REFYCRVL']  # -29.03225
    crpix_x = h['REFXCRPX'] / ximagebinsize  # 25921 / 80 = 324.0125
    crpix_y = h['REFYCRPX'] / yimagebinsize  # 25921 / 80 = 324.0125
    cdelt_x = h['REFXCDLT'] * ximagebinsize  # -0.0001111111111111112
    cdelt_y = h['REFYCDLT'] * yimagebinsize  #  0.0001111111111111112
    naxis_x = h['REFXLMAX'] / ximagebinsize  # 51840 / 80 = 648
    naxis_y = h['REFYLMAX'] / yimagebinsize  # 51840 / 80 = 648

    header = fits.Header()
    header['NAXIS']  = 2
    header['NAXIS1'] = naxis_x  # 648
    header['NAXIS2'] = naxis_y  # 648
    header['CTYPE1'] = ctype_x  # 'RA---TAN'
    header['CTYPE2'] = ctype_y  # 'DEC--TAN'
    header['CRVAL1'] = crval_x  # 266.433791666667
    header['CRVAL2'] = crval_y  # -29.03225
    header['CRPIX1'] = crpix_x  # 324.0125
    header['CRPIX2'] = crpix_y  # 324.0125
    header['CDELT1'] = cdelt_x  # -0.0001111111111111112
    header['CDELT2'] = cdelt_y  #  0.0001111111111111112

    header['DATE-OBS'] = h['DATE-OBS']  
    header['DATE-END'] = h['DATE-END']
    header['EQUINOX']  = h['EQUINOX']

    # header['MJD-END']  = h1['MJD-END']
    # header['MJD-OBS']  = h1['MJD-OBS']
    
    return WCS(header)


if __name__ == "__main__":
    i1 = '../../data/processed/0724210501/pps/P0724210501PNS003IMAGE_1000.FTZ'
    i2 = '../../data/processed/0724210501/pps/P0724210501M1S001IMAGE_1000.FTZ'
    i3 = '../../data/processed/0724210501/pps/P0724210501M2S002IMAGE_1000.FTZ'
    images = [i1,i2,i3]
    
    f1 = '../../data/raw/0724210501/P0724210501PNS003PIEVLI0000.FTZ'
    f3 = '../../data/raw/0724210501/P0724210501M2S002MIEVLI0000.FTZ'
    f2 = '../../data/raw/0724210501/P0724210501M1S001MIEVLI0000.FTZ'
    event_lists = [f1,f2,f3]
    
    for img, evt in zip(images, event_lists):
        print(f'Image:      {img}')
        print(f'='*80+'\n')
        f = fits.open(img)
        h = f[0].header
        # print(repr(h))
        img_wcs = WCS(h)
        print(repr(img_wcs.to_header()))
        print(f'\nimg wcs:\n{img_wcs}\n========================')
    
        print(f'Event List: {evt}')
        print(f'='*80+'\n')
        wcs = wcs_from_eventlist_file(evt)
        print(repr(wcs.to_header()))
        print(f'\nwcs:\n{wcs}\n=--------------------')
          
        assert np.array_equal(img_wcs.wcs.ctype, wcs.wcs.ctype)
        assert np.allclose(img_wcs.wcs.crval,    wcs.wcs.crval, atol=1e-6)
        assert np.allclose(img_wcs.wcs.crpix,    wcs.wcs.crpix, atol=0.5)
        assert np.allclose(img_wcs.wcs.cdelt,    wcs.wcs.cdelt, atol=1e-6)
        assert np.allclose(img_wcs.wcs.get_pc(), wcs.wcs.get_pc())
        assert img_wcs.naxis == wcs.naxis 
