# Ulas Kamaci - 2019-9-22
# script for simulation of a starshade + lens system.
# object plane is P0, starshade plane is P1, lens plane is P2, image plane is P3.
# P0-->P1 is Fraunhofer approximation, P1-->P2 and P2-->P3 are Fresnel approximation
import numpy as np
import matplotlib.pyplot as plt

# the units are in meters

distance_object = 3e17 # distance from the object to the lens
distance_starshade = 30e6 # distance from the starshade to the lens
focal_length = 30 # focal length of the lens
diameter_lens = 2.4 # diameter of the lens (used in pupil func.)
diameter_starshade = 50 # diameter of the starshade
wavelength = 633e-9
psf_width = 1001
nyquist_interval = wavelength * focal_length / diameter_lens
sampling_interval = nyquist_interval / 40
pixel_fov = sampling_interval * distance_object / focal_length
print('field of view per pixel is: {} M km'.format(pixel_fov/1e9))

def pupil(diameter, x, y):
    return x**2 + y**2 <= (diameter/2)**2 # returns 1 if the point is in the circle

def starshade_circular(diameter, x, y):
    return x**2 + y**2 > (diameter/2)**2 # returns 1 if the point is out of the circle

fxx = np.arange(
    -(psf_width - 1) / 2,
    (psf_width - 1) / 2 + 1
) / (psf_width * sampling_interval)

xx = np.arange(
    -(psf_width - 1) / 2,
    (psf_width - 1) / 2 + 1
) * (sampling_interval)

fx, fy = np.meshgrid(fxx, fxx)
x, y = np.meshgrid(xx, xx)

starshade_modified = starshade_circular(
    diameter_starshade,
    wavelength*focal_length*fx,
    wavelength*focal_length*fy
) * np.e**(
    1j * np.pi * wavelength / distance_object *
    focal_length**2 * (fx**2 + fy**2)
)

starshade_modified_f = np.fft.fftshift(np.fft.fft2(starshade_modified))
otf_multiplier = np.fft.ifft2(
    np.e**(
    -1j * np.pi / wavelength * distance_starshade /
    focal_length**2 * (x**2 + y**2)
) * starshade_modified_f
)

coherent_otf = (wavelength**7 * focal_length**7 * distance_object /
    distance_starshade * pupil(
    diameter_lens,
    wavelength*focal_length*fx,
    wavelength*focal_length*fy
) * otf_multiplier
)

coherent_psf = np.fft.ifft2(coherent_otf)
incoherent_psf = np.abs(coherent_psf)**2
plt.figure()
plt.imshow(incoherent_psf)
plt.show()
plt.colorbar()
