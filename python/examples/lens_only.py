# Ulas Kamaci - 2019-9-22
# script for simulation of a single lens imaging without starshade (or,
# out of region of influence of the starshade).
# object plane is P0, lens plane is P1, image plane is P2.
# P0-->P1 is Fraunhofer approximation, P1-->P2 is Fresnel approximation
import numpy as np
import matplotlib.pyplot as plt

# the units are in meters

distance_object = 3e17 # distance from the object to the lens
focal_length = 30 # focal length of the lens
diameter_lens = 2.4 # diameter of the lens (used in pupil func.)
wavelength = 633e-9
psf_width = 1001
nyquist_interval = wavelength * focal_length / diameter_lens
sampling_interval = nyquist_interval / 20
pixel_fov = sampling_interval * distance_object / focal_length
print('field of view per pixel is: {} M km'.format(pixel_fov/1e9))

def pupil(diameter, x, y):
    return x**2 + y**2 <= (diameter/2)**2 # returns 1 if the point is in the circle

# generate the Fourier domain grid
fxx = np.arange(
    -(psf_width - 1) / 2,
    (psf_width - 1) / 2 + 1
) / (psf_width * sampling_interval)
fyy = np.arange(
    -(psf_width - 1) / 2,
    (psf_width - 1) / 2 + 1
) / (psf_width * sampling_interval)

fx, fy = np.meshgrid(fxx, fyy)

# first compute the psf in the Fourier domain, where it's called otf. this
# way we get rid of convolution, and just perform multiplication. we'll then
# take inverse Fourier transform to get the psf.
coherent_otf = wavelength**4 * focal_length**3 * distance_object * pupil(
    diameter_lens,
    wavelength*focal_length*fx,
    wavelength*focal_length*fy
) * np.e**(
    1j * np.pi * wavelength / distance_object *
    focal_length**2 * (fx**2 + fy**2)
)

coherent_psf = np.fft.fftshift(np.fft.ifft2(coherent_otf))
incoherent_psf = np.abs(coherent_psf)**2
plt.figure()
plt.imshow(incoherent_psf)
plt.show()
plt.colorbar()
