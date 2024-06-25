import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *

params = {'axes.labelsize': 12,
     'legend.fontsize': 12,
     'xtick.labelsize': 14,
     'ytick.labelsize': 14}
     
rcParams.update(params)     

x=np.loadtxt("Cylinder2D-Re100-forces.dat")

x[:,1]=2.0*x[:,1]
x[:,2]=2.0*x[:,2]


plt.plot(x[:,0], x[:,2], 'k', linewidth=2.0, markersize=8.0)
#plt.xlim(0.0,1.5)
#plt.xlim(0.0,40.0)
#plt.ylim(430.0,450.0)
#plt.ylim(-0.5,0.1)
#plt.ylim(0.0,4.0)

plt.grid('on')

plt.xlabel('Time',fontsize=14)
plt.ylabel('Lift coefficient',fontsize=14)

plt.tight_layout()

plt.show()

plt.savefig('Cylinder2D-Re100-CL.png', dpi=500)

