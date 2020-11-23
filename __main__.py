
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    #As a path collection

    from matplotlib.collections import LineCollection

    # In polar coordinates
    fig, ax = plt.subplots(subplot_kw=dict(polar=True, facecolor='none'))
    r = np.arange(0, 3.4, 0.01)
    theta = 2*np.pi*r
    ax.set_axis_off()
    ax.grid(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.plot(theta, r, linewidth=2, color='k');

    plt.show()
