#TODO: Match the candidate transients with Simbad and archival X-ray catalogs (from STONKS?)
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
Simbad.add_votable_fields("otype")
from exod.utils.classes_Simbad import simbad_classifier


def simbad_lookup(tab_ra, tab_dec):
    tab_names=[]
    tab_types=[]
    for (ra, dec) in zip(tab_ra, tab_dec):
        result_table = Simbad.query_region(SkyCoord(ra, dec,
                                                    unit=(u.deg, u.deg), frame='icrs'),
                                           radius=10 * u.arcsec)
        print(result_table)
        if result_table is not None:
            result = result_table[0]
            tab_names.append(result["MAIN_ID"])
            tab_types.append(simbad_classifier[result["OTYPE"]])
        else:
            tab_names.append("")
            tab_types.append("")
    return tab_names, tab_types

if __name__=='__main__':
    tab_ra, tab_dec = [19.786252, 15], [-34.1923,0]
    print(simbad_lookup(tab_ra, tab_dec))
