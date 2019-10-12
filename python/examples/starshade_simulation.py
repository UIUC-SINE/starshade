import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy.ma as ma
from scipy.signal import convolve2d

dim = 100
pc_to_meter = 3.08567782e16
au_to_meter = 149597870700.
focal_length_lens = 30
diameter_lens = 2.4 
dist_to_ss = 10*pc_to_meter
LAMBDA = 633e-9
roi = 120.*au_to_meter
angres = (4*633e-9)/(25)
print (angres*10*pc_to_meter) # spatial resolution at 10 pc
telescope_diameter = 1.1
exo_pos = [0, au_to_meter, 0]
dist_ss_t = 63942090. #distance for hypergaussian function t_MH

# Inner working angle is the angle which the starshade subtends, it is given by R/z and for a 25m starshade at 50 000 km, this angle is 100 mas. this is the angle beyond which can detect an exo, require 100mas for earth sun sep
# the fov can be 360 deg, depends on the detector
# the angular resolution goes as 4*lambd/25m ~ 20 mas

## define a meshgrid
def gen_grid(nx, ny, field_size):
    x=np.linspace(0, field_size, nx)
    y=np.linspace(0, field_size, ny)
    xv, yv = np.meshgrid(x,y)
    return np.stack((xv,yv), axis=2)

#hypergaussian apodization profile
def t_MH(r, theta):
    a=b=12.5
    n=6.
    return np.abs((r<a)-1)*(1.-np.exp(-((r-a)/b)**n))

def cart_to_pol(x,y):
    r = (x**2 + y**2)**.5
    theta = (np.logical_and(r!=0, y>=0)*2 - 1) * np.arccos(x/r)
    return r, theta

def spher_to_cart(r, theta, phi):
    return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

def cart_to_spher(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan(y/x)
    theta = np.arccos(z/r)
    return np.array([r, theta, phi])

def plane_wave(A, r, k):
    return A*np.exp(1j* np.inner(k, r))
    #note can save a computation if z is fixed and reuse it over all x,y.

def field_after_ss(x,y):
    field_before_ss = plane_wave(np.array([x,y, 10*pc_to_meter]), k_exo)
    (r_, th) = cart_to_pol(x,y)
    tA = t_MH(r_, th)
    return tA*field_before_ss

def pupil_func(x, y, r=3.):
    return ((x**2 + y**2)**.5 <= r)

def fresnel(p_x,p_y,image,propagation_distance=dist_ss_t,wavelength=633e-9):
    # convolving with the Fresnel impulse response via FFT multiplication
    fft = np.fft.fft2(image)

    pixelsize = p_x[1] - p_x[0]
    npixels = p_x.size
    freq_nyquist = 0.5/pixelsize
    freq_x = np.linspace(-1.0,1.0,npixels)*freq_nyquist

    pixelsize = p_y[1] - p_y[0]
    npixels = p_y.size
    freq_nyquist = 0.5/pixelsize
    freq_y = np.linspace(-1.0,1.0,npixels)*freq_nyquist

    freq_xy = np.array(np.meshgrid(freq_y,freq_x)) #create frequency grid
    
    k = 2 * np.pi / wavelength
    fft *= np.exp(1j * propagation_distance * k) * np.exp((-1j) * np.pi * wavelength * propagation_distance * np.fft.fftshift(freq_xy[0]**2 + freq_xy[1]**2) ) #p88 goodman

    ifft = np.fft.ifft2(fft)

    return ifft

def huygens_fresnel(p_x,p_y,image,propagation_distance=dist_ss_t,wavelength=633e-9):
    # convolving with the Fresnel impulse response via FFT multiplication
    fft = np.fft.fft2(image)

    pixelsize = p_x[1] - p_x[0]
    npixels = p_x.size
    freq_nyquist = 0.5/pixelsize
    freq_x = np.linspace(-1.0,1.0,npixels)*freq_nyquist

    pixelsize = p_y[1] - p_y[0]
    npixels = p_y.size
    freq_nyquist = 0.5/pixelsize
    freq_y = np.linspace(-1.0,1.0,npixels)*freq_nyquist

    freq_xy = np.array(np.meshgrid(freq_y,freq_x)) #create frequency grid
    
    k = 2 * np.pi / wavelength 
    fft *= np.exp(1j * propagation_distance * k * np.sqrt(1 - np.fft.fftshift((wavelength*freq_xy[0])**2 + (wavelength*freq_xy[1])**2)) )

    masked_fft = ma.masked_where((wavelength*np.sqrt(np.fft.fftshift( freq_xy[0]**2 + freq_xy[1]**2 )) >= 1.), fft)
    ifft = np.fft.ifft2(masked_fft)

    return ifft


def fraunhofer(x_coord,y_coord, amplitude,wavelength=633e-9):
    # coordinates in meters
    # returns propogated angle_x [rad], angle_y, complex_amplitude
    # fourier transform of aperture 
    # if aperture + lense, then gets evaluated at x*lambda*f where x is the input grid of the original amplitude

    F1 = np.fft.fft2(amplitude)
    F2 = np.fft.fftshift( F1 )

    # frequency for axis 1
    freq_nyquist = 0.5/(x_coord[1] - x_coord[0])
    freq_n = np.linspace(-1.0,1.0,len(x_coord))
    freq_x = freq_n * freq_nyquist
    freq_x *= wavelength

    # frequency for axis 2
    freq_nyquist = 0.5/(y_coord[1] - y_coord[0])
    freq_n = np.linspace(-1.0,1.0,len(y_coord))
    freq_y = freq_n * freq_nyquist
    freq_y *= wavelength 
    return freq_x, freq_y, F2

# plane wave at dist 10pc + 500000km
# plane wave at complement of starshade

xvals = np.linspace(-120,120,1001)
yvals = np.linspace(-120,120,1001)
xx, yy = np.meshgrid(xvals, yvals)

# first define the plane waves representing sources (each plane wave is defined by an amplitude and angle or k_vector)
# use babinets principle -> 1) propagate as if there were no starshade, 2) propagate as if starshade complement, subtract 1 from 2

pupil_aperture = pupil_func(xx, yy, r=40)
(r_, th) = cart_to_pol(xx,yy)
tA_complement = 1-t_MH(r_, th) #complement of starshade aperture

# for this test exo, we place it at 200mas separation from its host star, this corresponds to twice the earth sun distance. 
k_exo = np.array([0, -np.sin((2*au_to_meter/(10*pc_to_meter))), np.cos((2*au_to_meter/(10*pc_to_meter)))])*2*np.pi/633e-9 #k vector of incoming plane wave
k_star = np.array([0,0,1])*2*np.pi/633e-9

#we treat the starshade as positioned at z=0 plane, first we evaluate the field immediately after the starshade complement
field_after_ss = (plane_wave(1, np.array([xx,yy, 0]), k_star) +  plane_wave(.1, np.array([xx,yy,0]), k_exo) ) * tA_complement 
test_d_f = fresnel(xvals, yvals, field_after_ss)
test_d_hf = huygens_fresnel(xvals, yvals, field_after_ss)

field_free_prop = plane_wave(1, np.array([xx,yy, dist_ss_t]), k_star) +  plane_wave(.1, np.array([xx,yy, dist_ss_t]), k_exo) #this is the field as if there was no ss, propagated to before the telescope

#the field prior to the telescope, evaluated using babinets principle
field_at_telescope = field_free_prop - test_d_hf

#we propagate the field at the telescope focused using a lense
fx, fy, image_field_compl = fraunhofer(xvals, yvals, pupil_func(xx, yy, r=2.4) * test_d_hf)
fx, fy, image_field_free = fraunhofer(xvals, yvals, pupil_func(xx, yy, r=2.4) * field_free_prop)

plt.figure()
#plt.imshow(np.abs(field_at_telescope)**2)
plt.imshow( np.abs( image_field_free- image_field_compl )**2 )
#plt.imshow(np.abs(test4 - test3)**2)
plt.show()


#=======================================
# another method to compute the lens propagation, first multiple by a phase factor and then propagate using fresnel
def phase_factor(x, y):
    return np.exp(-1j* np.pi * (x**2 + y**2 )/ (30000000 * 633e-9))

telescope = pupil_func(xx, yy, r=2.4)*phase_factor(xx, yy)
test3 = huygens_fresnel(xvals, yvals, test_d_hf * telescope, propagation_distance=30000000)
test4 = huygens_fresnel(xvals, yvals, field_free_prop * telescope, propagation_distance=30000000)
#=======================================

plt.figure()
#plt.imshow(np.abs(test_ss_d), cmap='Blues', extent=[-120,120,-120,120])
#plt.imshow(np.abs(test), cmap='Blues', extent=[-120,120,-120,120])
plt.imshow(np.abs(test_noss_d - test_ss_d), cmap='Blues', extent=[-120,120,-120,120])
plt.colorbar()
plt.title('Resultant Field at d with circ starshade and (exo+star)')
plt.xticks(np.linspace(-120,120, 6))
plt.yticks(np.linspace(-120,120, 6))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
print(cke)
#zeropadding of size 2n-1
#k = 2pi/lambda
