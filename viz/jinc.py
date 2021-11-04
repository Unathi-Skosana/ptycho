from scipy.special import j1,jn_zeros
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.style.use('mint')

    x = np.linspace(0, 4 * np.pi, 10000)
    y = (2 * j1(x) / x)**2


    golden_mean = (np.sqrt(5) - 1) / 2  # Aesthetic ratio
    fig_width_pt = 400  # column width
    inches_per_pt = 1 / 72.27  # Convert pt to inches
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width_pt * inches_per_pt * golden_mean # height in inches
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel(r'$x$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 4*np.pi)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])

    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(['', '', r'$0.5$', '', r'$1.0$'])

    ax.annotate(r"$\left(2\frac{J_1(x)}{x} \right)^2$",
            xytext=(1.5 * np.pi, .65), xycoords='data',
            xy=(0.7 * np.pi, (2 * j1(0.7 * np.pi) / (0.7 * np.pi))**2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            lw=1,
                            connectionstyle="arc3, rad=0.3"),
            )

    for i,z in enumerate(jn_zeros(1, 3)):
        ax.axvline(x=z, linestyle='--', lw=2, color="C0{0}".format(i+2), ymin=0, ymax=0.20,
                label=r'$({0:.2f}\pi, 0)$'.format(z / np.pi))
    ax.plot(x, y, lw=2, zorder=10, color="C01")
    ax.legend()

    fig.savefig("jinc.png",
                bbox_inches='tight',
                pad_inches=0,
                transparent=False)
    plt.show()
