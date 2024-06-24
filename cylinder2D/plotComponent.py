import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *

params = {'axes.labelsize': 12,
     'legend.fontsize': 12,
     'xtick.labelsize': 14,
     'ytick.labelsize': 14}
     
rcParams.update(params)     


ff   =  sys.argv[1]

ind1  = int(sys.argv[2])
ind2  = int(sys.argv[3])
fact1 = float(sys.argv[4])
fact2 = float(sys.argv[5])

inpfile = ff

x=np.loadtxt(inpfile)

x[:,ind1]=fact1*x[:,ind1]
x[:,ind2]=fact2*x[:,ind2]

x[:,ind2] = x[:,ind2] + 0.0

col='k'

if(len(sys.argv) > 6):
   col=sys.argv[6]


plt.plot(x[:,ind1],x[:,ind2], col, linewidth=2.0, markersize=8.0)
#plt.xlim(0.0,1.5)
#plt.xlim(0.0,40.0)
#plt.ylim(430.0,450.0)
#plt.ylim(-0.5,0.1)
#plt.ylim(0.0,4.0)

plt.grid('on')

#plt.xlabel('time',fontsize=14)
#plt.ylabel('Lift force',fontsize=14)

plt.show()
outfile = ff + '.eps'
plt.savefig(outfile, dpi=200)

