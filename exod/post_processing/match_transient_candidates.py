#TODO: Match the candidate transients with archival X-ray catalogs (from STONKS?)
from astropy.coordinates import SkyCoord, Angle, angular_separation
import astropy.units as u
from astroquery.simbad import Simbad
Simbad.add_votable_fields("otype")
from astroquery.vizier import Vizier
from exod.utils.classes_Simbad import simbad_classifier


def simbad_lookup(tab_ra, tab_dec):
    tab_names=[]
    tab_precise_type=[]
    tab_types=[]
    tab_sep=[]
    for (ra, dec) in zip(tab_ra, tab_dec):
        result_table = Simbad.query_region(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
                                           radius=10 * u.arcsec)
        if result_table is not None:
            result = result_table[0]
            tab_names.append(result["MAIN_ID"])
            tab_precise_type.append(result["OTYPE"])
            if result["OTYPE"] in simbad_classifier.keys():
                tab_types.append(simbad_classifier[result["OTYPE"]])
            else:
                tab_types.append('Unknown')
            tab_sep.append(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs').separation(
                SkyCoord(ra=result["RA"], dec=result["DEC"], unit=(u.hourangle, u.deg), frame='icrs')
                                              ).to(u.arcsec).value)
        else:
            tab_names.append("")
            tab_types.append("")
            tab_precise_type.append("")
            tab_sep.append("")
    return tab_names,tab_precise_type, tab_types, tab_sep

def xmm_lookup(tab_ra, tab_dec):
    tab_names=[]
    tab_varflag=[]
    tab_sep=[]
    for (ra, dec) in zip(tab_ra, tab_dec):
        result_table = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'),
                                           radius=10 * u.arcsec, catalog='IX/69/xmm4d13s')
        if len(result_table)>0:
            result = result_table[0]
            tab_names.append("4XMM "+result["_4XMM"][0])
            tab_varflag.append(result["V"][0])
            tab_sep.append(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs').separation(
                SkyCoord(ra=result["RA_ICRS"][0], dec=result["DE_ICRS"][0], unit=(u.deg, u.deg), frame='icrs')
                                              ).to(u.arcsec).value)
        else:
            tab_names.append("")
            tab_varflag.append("")
            tab_sep.append("")
    return tab_names, tab_varflag, tab_sep

def GLADE_lookup(tab_ra, tab_dec):
    tab_GLADE_ID=[]
    tab_dist=[]
    tab_mass=[]
    for (ra, dec) in zip(tab_ra, tab_dec):
        result_table = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'),
                                           radius=10 * u.arcsec, catalog='GLADE+')
        if len(result_table)>0:
            result = result_table[0]
            tab_GLADE_ID.append(result["GLADE_"][0])
            tab_dist.append(result["dL"][0])
            tab_mass.append(result["M_"][0])
        else:
            tab_GLADE_ID.append("")
            tab_dist.append("")
            tab_mass.append("")
    return tab_GLADE_ID, tab_dist, tab_mass

def Gaia_lookup(tab_ra, tab_dec):
    tab_Gaia_name=[]
    tab_dist=[]
    tab_sep=[]
    for (ra, dec) in zip(tab_ra, tab_dec):
        result_table = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'),
                                           radius=10 * u.arcsec, catalog='I/355/gaiadr3')
        if len(result_table)>0:
            result = result_table[0]
            tab_Gaia_name.append(list(result["Source"])[0])
            tab_dist.append(list(result["Dist"].value)[0])
            tab_sep.append(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs').separation(
                SkyCoord(ra=result["RA_ICRS"][0], dec=result["DE_ICRS"][0], unit=(u.deg, u.deg), frame='icrs')
                                              ).to(u.arcsec).value)
        else:
            tab_Gaia_name.append("")
            tab_dist.append("")
            tab_sep.append("")
    return tab_Gaia_name, tab_dist,tab_sep

if __name__=='__main__':
    tab_ra, tab_dec = [19.786252, 15, 283.3269], [-34.1923, 0, +33.04672]
    print(simbad_lookup(tab_ra, tab_dec))
    print(GLADE_lookup(tab_ra, tab_dec))
    print(Gaia_lookup(tab_ra, tab_dec))
    print(xmm_lookup(tab_ra, tab_dec))
