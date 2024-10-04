"""
View images from an XMM observation in DS9.
"""
from exod.xmm.observation import Observation
import subprocess

def view_obs_images(obsid, ra=None, dec=None):
    obs = Observation(obsid)
    obs.get_images()
    if obs.images:
        cmds = [f'ds9 -log {img.path}' for img in obs.images]
        if ra and dec:
            cmds = [f'{cmd} -crosshair {ra} {dec} wcs fk5' for cmd in cmds]

        for cmd in cmds:
            print(cmd)
            subprocess.run(f'{cmd} &', shell=True)
    else:
        print(f'No images found for observation {obsid}')

if __name__ == '__main__':
    ra = 49.57958313490073
    dec = -66.48425455995473
    view_obs_images('0803990201', ra, dec)
