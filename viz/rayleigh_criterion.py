from scipy.special import j1,jn_zeros
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.style.use('mint')

    x = np.linspace(-4 * np.pi, 4 * np.pi, 10000)
    y1 = (2 * j1(x + .80 * np.pi ) / (x + .80 * np.pi))**2
    y2 = (2 * j1(x - .80 * np.pi) / (x - .80 * np.pi))**2


    golden_mean = (np.sqrt(5) - 1) / 2  # Aesthetic ratio
    fig_width_pt = 400  # column width
    inches_per_pt = 1 / 72.27  # Convert pt to inches
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width_pt * inches_per_pt * golden_mean # height in inches
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel(r'$x$')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-4*np.pi, 4*np.pi)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.annotate("",
            xy=(-.8 * np.pi, 1.005), xycoords='data',
            xytext=(.8 * np.pi, 1.005), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            lw=1.0,
                            connectionstyle="arc",
                            )
            )

    ax.annotate(r"$\Delta x$",
            (0, 1.07),
            ha='center', va='center')

    ax.annotate("",
            xy=((1.22 - .8)* np.pi, .55), xycoords='data',
            xytext=(.8 * np.pi, .55), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            lw=1.0,
                            connectionstyle="arc",
                            )
            )

    ax.annotate(r"$\Delta x'$",
            (.93 * np.pi, .47),
            ha='center', va='center')

    ax.set_xticks([-4*np.pi, -3*np.pi ,-2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels([r'$-4\pi$',r'$-3\pi$',r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$',r'$2\pi$',r'$3\pi$',r'$4\pi$'])

    ax.axvline(x=(1.22 - .8)*np.pi,
            linestyle='--',
            lw=2,
            color="C01",
            ymin=0,
            ymax=0.5)

    ax.axvline(x=.8*np.pi,
            linestyle='--',
            lw=2,
            color="C02",
            ymin=0.5,
            ymax=0.9)

    ax.plot(x, y1, lw=2, zorder=10, color="C01")
    ax.plot(x, y2, lw=2, zorder=10, color="C02")

    fig.savefig("rayleigh.png",
                bbox_inches='tight',
                pad_inches=0,
                transparent=False)
    plt.show()
