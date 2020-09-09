if __name__ == "__main__":
    import numpy as np
    import csv
    import matplotlib.font_manager
    import matplotlib.pyplot as plt
    from matplotlib import rc

    matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    # Aesthetics
    rc('text', usetex=True)
    plt.style.use('seaborn-white')

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Roboto Condensed'
    plt.rcParams['font.sans-serif'] = 'Roboto Condensed'
    plt.rcParams['font.monospace'] = 'Roboto Mono'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'light'
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 8

    # Set an aspect ratio
    width, height = plt.figaspect(1.68)
    fig = plt.figure(figsize=(width,height), dpi=168)

    ax = plt.subplot(111)

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./convergence/beta={}_sse.dat'.format(v), delimiter=" ")
        x, y = np.split(a,[-1],axis=1)
        ax.plot(x, y, alpha=0.5, lw=1)
        ax.set_title(r'Sum of squares error for ePIE')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
    ax.legend(list(map(lambda x : r'$\beta={}$'.format(x), [0.4, 0.7, 1.0])))

    plt.show()

