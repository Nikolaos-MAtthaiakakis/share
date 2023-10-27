import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math

theta_in = np.arange(0,90,5)
kxs = np.empty((100,theta_in.size))
thetas = np.empty((100,theta_in.size))
Rmeep = np.empty((100,theta_in.size))
Ez = np.empty((100,theta_in.size))

for j in range(theta_in.size):
  f = np.genfromtxt("tmp/flux_t{}.dat".format(theta_in[j]), delimiter=",")
  kxs[:,j] = f[:,0]
  thetas[:,j] = f[:,2]
  Rmeep[:,j] = f[:,3]
  Ez[:,j] = f[:,4]

wvl = f[:,1]
# create a 2d matrix for the wavelength by repeating the column vector for each angle
wvls = np.matlib.repmat(np.reshape(wvl, (wvl.size,1)),1,theta_in.size)

plt.figure()
plt.subplot(2, 1, 1)
plt.pcolormesh(wvls, kxs,  Rmeep, cmap='hot', shading='gouraud', vmin=0, vmax=Rmeep.max())
#plt.axis([kxs[0,0], kxs[0,-1], wvl[-1], wvl[0]])
#plt.yticks([t for t in np.arange(0.4,0.9,0.1)])
#plt.clim(0,0.8)
plt.ylabel("$k_x/2p$")
plt.xlabel("energy (eV)")
plt.title("reflectance (meep)")
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0,0.4,0.1)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in np.arange(0,0.4,0.1)])

plt.subplot(2, 1, 2)
plt.pcolormesh(wvls, thetas, Rmeep, cmap='hot', shading='gouraud', vmin=0, vmax=Rmeep.max())
#plt.axis([thetas.min(), thetas.max(), wvl[-1], wvl[0]])
#plt.xticks([t for t in range(0,100,20)])
#plt.yticks([t for t in np.arange(0.4,0.9,0.1)])
#plt.clim(0,0.8)
plt.ylabel("? (degrees)")
plt.xlabel("energy (eV)")
plt.title("reflectance (meep)")
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0,0.4,0.1)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in np.arange(0,0.4,0.1)])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("media/theta.png")


plt.figure()
plt.subplot(2, 1, 1)
plt.pcolormesh(wvls, kxs,  Ez, cmap='hot', shading='gouraud', vmin=0, vmax=Ez.max())
#plt.axis([kxs[0,0], kxs[0,-1], wvl[-1], wvl[0]])
#plt.yticks([t for t in np.arange(0.4,0.9,0.1)])
#plt.clim(0,0.8)
plt.ylabel("$k_x/2p$")
plt.xlabel("energy (eV)")
plt.title("Ez (meep)")
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0,0.4,0.1)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in np.arange(0,0.4,0.1)])

plt.subplot(2, 1, 2)
plt.pcolormesh(wvls, thetas, Ez, cmap='hot', shading='gouraud', vmin=0, vmax=Ez.max())
#plt.axis([thetas.min(), thetas.max(), wvl[-1], wvl[0]])
#plt.xticks([t for t in range(0,100,20)])
#plt.yticks([t for t in np.arange(0.4,0.9,0.1)])
#plt.clim(0,0.8)
plt.ylabel("Theta (degrees)")
plt.xlabel("energy (eV)")
plt.title("Ez (meep)")
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0,0.4,0.1)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in np.arange(0,0.4,0.1)])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("media/thetaEz.png")

np.savetxt('thetas.txt', thetas, delimiter=',')
np.savetxt('Rmeep.txt', Rmeep, delimiter=',')
np.savetxt('kxs.txt', kxs, delimiter=',')
np.savetxt('Ez.txt', Ez, delimiter=',')
np.savetxt('wvls.txt', wvls, delimiter=',')
