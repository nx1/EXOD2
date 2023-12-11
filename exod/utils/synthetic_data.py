import numpy as np

def create_fake_burst(cubeshape, time_interval, time_peak_fraction, position, width_time, amplitude, size_arcsec):
    time = np.arange(0, cubeshape[-1])
    peak_data = np.zeros(cubeshape, dtype=int)
    time_peak = time_peak_fraction*len(time)
    width_bins = width_time / time_interval

    #Poissonian PSF
    sigma_2d = (6.6 / size_arcsec)# * 2.355  # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
    for x in range(int(position[0]-10*sigma_2d), int(position[0]+10*sigma_2d)):
        for y in range(int(position[1]-10*sigma_2d), int(position[1]+10*sigma_2d)):
            sqdist = (x-position[0])**2+(y-position[1])**2
            psf = (1/(2*np.pi*np.sqrt(sigma_2d)))*np.exp(-(sqdist)/(2*(sigma_2d**2)))
            peak_data[x,y]+=np.random.poisson(psf*(amplitude*time_interval)*np.exp(-(time-time_peak)**2/(2*(width_bins**2))))
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return peak_data