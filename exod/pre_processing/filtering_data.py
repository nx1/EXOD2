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

filter_events_file('0831790701', min_energy=0.2, max_energy=10.)