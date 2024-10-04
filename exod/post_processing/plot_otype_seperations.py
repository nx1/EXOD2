import pandas as pd
import matplotlib.pyplot as plt

from exod.utils.path import savepaths_combined, data_plots

# Show all pandas Rows
pd.set_option('display.max_rows', None)

df = pd.read_csv(savepaths_combined['cmatch_simbad'])
df = df[df['SEP_ARCSEC'] != 9999.00]

all_res = []
for otype in df['main_type'].unique():
    sub = df[df['main_type'] == otype]
    n = len(sub)
    if n < 10:
        continue
    sep_mean = sub['SEP_ARCSEC'].mean()
    sep_std = sub['SEP_ARCSEC'].std()
    sep_86 = sub['SEP_ARCSEC'].quantile(0.86)
    sep_14 = sub['SEP_ARCSEC'].quantile(0.14)
    res = {'otype'   : otype,
           'n': n,
           'sep_mean': sep_mean.round(2),
           'sep_std' : sep_std.round(2),
           'sep_86'  : sep_86.round(2),
           'sep_14'  : sep_14.round(2)}
    all_res.append(res)

df_res = pd.DataFrame(all_res)
df_res = df_res.sort_values('sep_mean', ascending=False).reset_index(drop=True)
print(df_res)


fig = plt.subplots(figsize=(8, 20))
plt.errorbar(df_res['sep_mean'], df_res.index, xerr=df_res['sep_std'], fmt='o', color='black', lw=1.0)
plt.yticks(df_res.index, df_res['otype'])
plt.xlabel('Mean Separation (arcsec)')
plt.tight_layout()
plt.savefig(data_plots / 'otype_seperations.png')
plt.savefig(data_plots / 'otype_seperations.pdf')
plt.show()
