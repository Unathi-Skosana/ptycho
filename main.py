import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from zernike import RZern

# TeX typesetting
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Aesthetics
plt.close('all')
plt.style.use('seaborn-pastel')

cart = RZern(10)
L, K = 250, 250
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)

fig,ax = plt.subplots()
c = np.zeros(cart.nk)
c[11] = 1.0
Phi = cart.eval_grid(c, matrix=True)
ax.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1), cmap='RdBu')
plt.show()
