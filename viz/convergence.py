if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc

    # set style
    plt.style.use('mint')

    # changing alpha
    fig1, ax1 = plt.subplots()

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./data/lena&cameraman/alpha={}_rms.dat'.format(v),
        delimiter=",")
        x1, y1 = np.split(a,[-1],axis=1)
        ax1.plot(x1, y1, lw=1)
        ax1.set_title(r'Root mean square error')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Error')
    ax1.legend(list(map(lambda x : r'$\alpha={}$'.format(x), [0.4, 0.7, 1.0])))
    fig1.savefig('rms_alpha_epie.png', bbox_inches='tight',
                pad_inches=0, transparent=False)


    # changing beta
    fig2, ax2 = plt.subplots()

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./data/lena&cameraman/beta={}_rms.dat'.format(v),
                delimiter=",")
        x1, y1 = np.split(a,[-1],axis=1)
        ax2.plot(x1, y1, lw=1)
        ax2.set_title(r'Root mean square error')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Error')
    ax2.legend(list(map(lambda x : r'$\beta={}$'.format(x), [0.4, 0.7, 1.0])))
    fig2.savefig('rms_beta_epie.png', bbox_inches='tight',
                pad_inches=0, transparent=False)

    # changing alpha
    fig3, ax3 = plt.subplots()

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./data/lena&cameraman/alpha={}_sse.dat'.format(v),
                delimiter=",")
        x1, y1 = np.split(a,[-1],axis=1)
        ax3.plot(x1, y1, lw=1)
        ax3.set_title(r'Sum of squares error')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Error')
    ax3.legend(list(map(lambda x : r'$\alpha={}$'.format(x), [0.4, 0.7, 1.0])))
    fig3.savefig('sse_alpha_epie.png', bbox_inches='tight',
                pad_inches=0, transparent=False)


    # changing alpha
    fig4, ax4 = plt.subplots()

    for v in [0.4, 0.7, 1.0]:
        a = np.loadtxt('./data/lena&cameraman/beta={}_sse.dat'.format(v),
                delimiter=",")
        x1, y1 = np.split(a,[-1],axis=1)
        ax4.plot(x1, y1, lw=1)
        ax4.set_title(r'Sum of squares error')
        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Error')
    ax4.legend(list(map(lambda x : r'$\beta={}$'.format(x), [0.4, 0.7, 1.0])))
    fig4.savefig('sse_beta_epie.png', bbox_inches='tight',
                pad_inches=0, transparent=False)

    plt.show()
