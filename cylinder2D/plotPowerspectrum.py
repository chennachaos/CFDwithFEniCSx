import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *

from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
#from scipy import fft, arange
import scipy.fftpack



def plotSpectrum(y,Fs):
 """
 Plots a Single-Sided Amplitude Spectrum of y(t)
 """
 n = len(y) # length of the signal
 k = arange(n)
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = frq[range(int(n/2))] # one side frequency range
# print max(frq)

 Y = fft(y)/n # fft computing and normalization
 Y = Y[range(int(n/2))]
 #print Y
 ind = np.argmax(Y)
 #print np.max(Y), ind
 print(" Strouhal frequency =  %.4f" % frq[ind], " Hz")
 plot(frq, np.abs(Y),'r') # plotting the spectrum
 #plot(frq, Y,'r') # plotting the spectrum
 xlabel('Freq (Hz)')
 ylabel('|Y(freq)|')
 xlim(0,1.0)
 grid('on')


params = {'axes.labelsize': 12,
     'legend.fontsize': 12,
     'xtick.labelsize': 14,
     'ytick.labelsize': 14}
     
rcParams.update(params)     


inpfile=sys.argv[1]


x=np.loadtxt(inpfile)


ind  = int(sys.argv[2])
fact = float(sys.argv[3])

nn=0
if(len(sys.argv) > 4):
  nn = int(sys.argv[4])

x[:,ind]=fact*x[:,ind]

sampling=1.0/(x[nn+1,0]-x[nn,0])

#print sampling

N=size(x[:,0])

#plt.figure(1,[8,8])

#plt.axes().set_aspect(0.5)

t=x[nn:N,0]
val=x[nn:N,ind]

max = max(val)
min = min(val)
avg = 0.5*(min + max)


Nt=size(t)

rms=0.0
for jj in range(Nt):
   rms = rms + val[jj]*val[jj]

rms = np.sqrt(rms/Nt)


print(" Maximum   = %.4f" % max)
print(" Minimum   = %.4f" % min)
print(" Average   = %.4f" % avg)
print(" Amplitude = %.4f" % (max-avg))
print(" RMS       = %.4f" % rms)

subplot(2,1,1)
#plt.axes().set_aspect(5)
plot(t, val, 'k-', linewidth=1.0, markersize=8.0)
#ylim(0,2.0)
#axis([0, 10.0, 0.0, 1.0])
xlabel('Time')
ylabel('Amplitude')
grid('on')
subplot(2,1,2)
plotSpectrum(val, sampling)
show()

plt.show()
plt.savefig("ff.png", dpi=500)

