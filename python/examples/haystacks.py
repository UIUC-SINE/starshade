import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy.ma as ma
from matplotlib.colors import LogNorm
from astropy.io import fits

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

def k_vector(x, y,  distance, wl):
    #returns k vector for a given pixel
    norm = (x**2 + y**2 + distance**2)**.5
    k = 2*np.pi/wl
    return np.array([-x*k/norm, -y*k/norm, distance*k/norm])

# does the plane wave propagation make sense or from the object scene to ss should I use fresnel prop? did this before and it was below my computers fp accuracy
# exact quanitification and understanding of resolution requirements, read up on sampling theorem
# if the plane wave model makes sense, why not precompute?
# selection of apodization profile

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


def plane_field(amplitude_stack, r, k_stack):
    size_x = int(len(amplitude_stack))
    size_y = int(len(amplitude_stack[0]))
    stack = np.zeros((1001, 1001), dtype='complex128')
    l = 0
    for i in range(size_x):
        for j in range(size_y):
            stack += plane_wave(amplitude_stack[i][j], r, k_stack[:,i,j])
            print (l)
            l += 1
    return np.sum(stack)

haystacks_file = 'modern_cube_zodi1inc0dist10_0.70-0.87um.fits'
haystacks = fits.open(haystacks_file)

cubehdr = haystacks[0].header
N_EXT = cubehdr['N_EXT'] 
dx = cubehdr['PIXSCALE'] #units are AU

first_slice = haystacks[1].data
first_wavel = haystacks[1].header['WAVEL'] # wavelength in micrometers
first_slice = first_slice[int((len(first_slice)-1)/2) - 20:int((len(first_slice)-1)/2) + 20, int((len(first_slice)-1)/2) - 20:int((len(first_slice)-1)/2) + 20] #cropping the image

dimx = len(first_slice)
dimy = len(first_slice[0])

xvals = np.arange(0, int(dimx),1) * dx * au_to_meter
xvals -=  xvals[int((dimx)/2)]
yvals = np.arange(0, int(dimy),1) * dx * au_to_meter
yvals -=  yvals[int((dimy)/2)]

xx, yy = np.meshgrid(xvals, yvals)

k_stack_1 = k_vector(xx, yy, 10*pc_to_meter, first_wavel*1e-6)

xvals = np.linspace(-100,100,1001)
yvals = np.linspace(-100,100,1001)
xx, yy = np.meshgrid(xvals, yvals)

(r_, th) = cart_to_pol(xx,yy)
tA_complement = 1-t_MH(r_, th) #complement of starshade aperture

field_after_ss = plane_field(first_slice, np.array([xx, yy, 0]), k_stack_1) * tA_complement

field_compl = fresnel(xvals, yvals, field_after_ss)
field_free_prop =  plane_field(first_slice, np.array([xx, yy, dist_ss_t]), k_stack_1)

fx, fy, image_field_compl = fraunhofer(xvals, yvals, pupil_func(xx, yy, r=2.4) * field_compl)
fx, fy, image_field_free = fraunhofer(xvals, yvals, pupil_func(xx, yy, r=2.4) * field_free_prop)

plt.figure()
plt.imshow( np.abs( image_field_free- image_field_compl )**2) #, norm=LogNorm(vmin=0.001,vmax=300)
plt.show()



