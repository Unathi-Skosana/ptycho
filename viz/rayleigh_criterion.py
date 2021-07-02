from scipy.special import j1,jn_zeros
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.style.use('mint')

    x = np.linspace(-4 * np.pi, 4 * np.pi, 10000)
    y1 = (2 * j1(x + 0.75 * np.pi ) / (x + 0.75 * np.pi))**2
    y2 = (2 * j1(x- 0.55 * np.pi) / (x - 0.55 * np.pi))**2


    golden_mean = (np.sqrt(5) - 1) / 2  # Aesthetic ratio
    fig_width_pt = 500  # column width
    inches_per_pt = 1 / 72.27  # Convert pt to inches
    fig_width = fig_width_pt * inches_per_pt * golden_mean
    fig_height = fig_width_pt * inches_per_pt  # height in inches
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
            xy=(-0.75 * np.pi, 1.005), xycoords='data',
            xytext=(0.55 * np.pi, 1.005), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            lw=1.0,
                            connectionstyle="arc"),
            )

    ax.annotate(r"$\Delta x$",
            (-0.10 * np.pi, 1.04),
            ha='center', va='center')


    ax.plot(x, y1, lw=2, zorder=10, color="C01")
    ax.plot(x, y2, lw=2, zorder=10, color="C02")
    ax.plot(x, y1+y2, lw=2, zorder=10, color="C03", ls='--')

    fig.savefig("rayleigh.png",
                bbox_inches='tight',
                pad_inches=0,
                transparent=False)
    plt.show()
