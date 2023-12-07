import os
import shlex
import subprocess
from exod.utils.path import data_raw, data_processed

def filter_events_file(obs, min_energy=0.2, max_energy=10.):
    """Filtering scripts for all EPIC data. Requires having pre-set 'setsas' and 'export CCFPATH=' in the terminal.
    Takes the energy range in keV and observation name as arguments"""

    print(f'Filtering observation {obs}...')
    os.makedirs(os.path.join(data_processed, obs), exist_ok=True)

    min_PI = int(min_energy*1000)
    max_PI = int(max_energy*1000)

    raw_PN_file = os.path.join(data_raw,obs,f'P{obs}PNS001PIEVLI.FTZ')
    clean_PN_file = os.path.join(data_processed, obs, f'PN_pattern_clean.fits')
    cmd=(f'evselect table={raw_PN_file} withfilteredset=Y filteredset={clean_PN_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EP && (PATTERN<=4) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)

    raw_M1_file = os.path.join(data_raw,obs,f'P{obs}M1S002MIEVLI.FTZ')
    clean_M1_file = os.path.join(data_processed, obs, f'M1_pattern_clean.fits')
    cmd=(f'evselect table={raw_M1_file} withfilteredset=Y filteredset={clean_M1_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)

    raw_M2_file = os.path.join(data_raw,obs,f'P{obs}M2S003MIEVLI.FTZ')
    clean_M2_file = os.path.join(data_processed, obs, f'M2_pattern_clean.fits')
    cmd=(f'evselect table={raw_M2_file} withfilteredset=Y filteredset={clean_M2_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)


def filter_events_file_gti_only(obs, min_energy=0.2, max_energy=10.):
    """Filtering scripts for all EPIC data, keeping only GTIs.
     Requires having pre-set 'setsas' and 'export CCFPATH=' in the terminal.
    Takes the energy range in keV and observation name as arguments"""

    print(f'Filtering observation {obs}...')
    os.makedirs(os.path.join(data_processed, obs), exist_ok=True)

    min_PI = int(min_energy*1000)
    max_PI = int(max_energy*1000)

    raw_PN_file = os.path.join(data_raw,obs,f'P{obs}PNS001PIEVLI.FTZ')
    PN_highE_rates = os.path.join(data_processed,obs,f'PN_highE_rates.fits')
    PN_gti_file = os.path.join(data_processed,obs,f'PN_gti.fits')
    clean_PN_file = os.path.join(data_processed, obs, f'PN_clean.fits')
    imagePN = os.path.join(data_processed, obs, f'PN_image.fits')
    #Extracting high energy rates
    cmd=(f'evselect table={raw_PN_file} withrateset=Y rateset={PN_highE_rates} maketimecolumn=Y timebinsize=100 '
         f'makeratecolumn=Y expression="#XMMEA_EP && (PI in [10000:12000]) && (PATTERN==0)" -V 0')
    os.system(cmd)
    #Computing GTIs
    cmd=(f'tabgtigen table={PN_highE_rates} expression="RATE<=0.5" gtiset={PN_gti_file} -V 0')
    os.system(cmd)
    #Cleaning events list
    cmd=(f'evselect table={raw_PN_file} withfilteredset=Y filteredset={clean_PN_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EP && gti({PN_gti_file},TIME) && (PATTERN<=4) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)
    cmd=(f'evselect table={clean_PN_file} imagebinning=binSize imageset={imagePN} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    os.system(cmd)

    raw_M1_file = os.path.join(data_raw,obs,f'P{obs}M1S002MIEVLI.FTZ')
    M1_highE_rates = os.path.join(data_processed,obs,f'M1_highE_rates.fits')
    M1_gti_file = os.path.join(data_processed,obs,f'M1_gti.fits')
    clean_M1_file = os.path.join(data_processed, obs, f'M1_clean.fits')
    imageM1 = os.path.join(data_processed, obs, f'M1_image.fits')
    #Extracting high energy rates
    cmd=(f'evselect table={raw_M1_file} withrateset=Y rateset={M1_highE_rates} maketimecolumn=Y timebinsize=100 '
         f'makeratecolumn=Y expression="#XMMEA_EM && (PI>10000) && (PATTERN==0)" -V 0')
    os.system(cmd)
    # Computing GTIs
    cmd=(f'tabgtigen table={M1_highE_rates} expression="RATE<=0.4" gtiset={M1_gti_file} -V 0')
    os.system(cmd)
    # Cleaning events list
    cmd=(f'evselect table={raw_M1_file} withfilteredset=Y filteredset={clean_M1_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)
    cmd=(f'evselect table={clean_M1_file} imagebinning=binSize imageset={imageM1} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    os.system(cmd)


    raw_M2_file = os.path.join(data_raw,obs,f'P{obs}M2S003MIEVLI.FTZ')
    M2_highE_rates = os.path.join(data_processed,obs,f'M2_highE_rates.fits')
    M2_gti_file = os.path.join(data_processed,obs,f'M2_gti.fits')
    clean_M2_file = os.path.join(data_processed, obs, f'M2_clean.fits')
    imageM2 = os.path.join(data_processed, obs, f'M2_image.fits')
    #Extracting high energy rates
    cmd=(f'evselect table={raw_M2_file} withrateset=Y rateset={M2_highE_rates} maketimecolumn=Y timebinsize=100 '
         f'makeratecolumn=Y expression="#XMMEA_EM && (PI>10000) && (PATTERN==0)" -V 0')
    os.system(cmd)
    # Computing GTIs
    cmd=(f'tabgtigen table={M2_highE_rates} expression="RATE<=0.4" gtiset={M2_gti_file} -V 0')
    os.system(cmd)
    # Cleaning events list
    cmd=(f'evselect table={raw_M2_file} withfilteredset=Y filteredset={clean_M2_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    os.system(cmd)
    cmd=(f'evselect table={clean_M2_file} imagebinning=binSize imageset={imageM2} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    os.system(cmd)


#filter_events_file('0831790701', min_energy=0.2, max_energy=10.)
filter_events_file_gti_only('0831790701', min_energy=0.2, max_energy=10.)