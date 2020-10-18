if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc

    # set style
    plt.style.use('mint')

    # Set an aspect ratio
    width, height = plt.figaspect(1.21803399)
    fig = plt.figure(figsize=(width,height), dpi=96)

    ax = plt.subplot(111)

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./convergence_checkboard/beta={}_Eo.dat'.format(v),
                delimiter=",")
        x1, y1 = np.split(a,[-1],axis=1)
        ax.plot(x1, y1, alpha=0.5, lw=1)
        ax.set_title(r'Sum of squares error for ePIE')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
    ax.legend(list(map(lambda x : r'$\alpha={}$'.format(x), [0.4, 0.7, 1.0])))

    plt.show()

