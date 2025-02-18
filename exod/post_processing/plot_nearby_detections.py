# Display detection.

# Create a figure with one square subplot in the top left corner.
# Another subplot is shown in the top left showing an aitoff projection of the detections.
# A zoom is in shown in the middle subplot of showing the nearby detcetions.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u

from exod.post_processing.cluster_regions import ClusterRegions
from exod.utils.path import savepaths_combined

df_regions = (pd.read_csv(savepaths_combined['regions']))
cr = ClusterRegions(df_regions)
df_regions_unique = cr.df_regions_unique
df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'])
df_cmatch_gaia = pd.read_csv(savepaths_combined['cmatch_gaia'])
df_cmatch_om = pd.read_csv(savepaths_combined['cmatch_om'])
print(df_cmatch_om)

src = df_regions.loc[np.random.randint(0, len(df_regions))]
show_within = 100 * u.arcsec


def get_neaby_regions(df, ra, dec, within=40*u.arcsec):
    within = within.to(u.deg).value
    ra_hi, ra_lo = ra + within, ra - within
    dec_hi, dec_lo = dec + within, dec - within
    mask = (df['ra_deg'] > (ra - within)) & (df['ra_deg'] < (ra + within)) & (df['dec_deg'] > (dec - within)) & (df['dec_deg'] < (dec + within))
    return df[mask]

df_regions_nearby = get_neaby_regions(df=df_regions, ra=src['ra_deg'], dec=src['dec_deg'], within=show_within)


fig, ax = plt.subplots()
ax.set_title(f'Nearby detections <{show_within}" ({len(df_regions_nearby)})')
ax.scatter(src['ra_deg'], src['dec_deg'], s=50.0, color='red', marker='o', label='Source')
ax.scatter(df_regions_nearby['ra_deg'], df_regions_nearby['dec_deg'], s=1.0, alpha=1.0, color='black', marker='.')
for i, row in df_regions_nearby.iterrows():
    ax.text(row['ra_deg'], row['dec_deg'], row['runid'], fontsize=8)
extent = show_within.to(u.deg).value
ax.set_xlim(src['ra_deg'] - extent, src['ra_deg'] + extent)
ax.set_ylim(src['dec_deg'] - extent, src['dec_deg'] + extent)
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
plt.legend()

plt.show()
