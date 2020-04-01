# Christopher Iliffe Sprague
# sprague@kth.se
# Useful functions

import matplotlib.pyplot as plt, numpy as np, pptk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.markers import MarkerStyle

def plot_canonical_data():

    # create figure
    fig = plt.figure()

    # baltic plot
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    # extract data
    sm, _, _ = zip(*np.load('baltic.npy', allow_pickle=True))

    # stack submaps into one set
    sm = np.vstack(sm)
    
    # plot the points
    ax.plot(sm[:,0], sm[:,1], sm[:,2], c=sm[:,2], cmap='jet', marker=',')#, alpha=0.1)

    # remove background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #ax.set_axis_off()
    ax.set_aspect('equal')

    # save memory
    del sm
    del _

    # shetland plot
    ax = fig.add_subplot(1, 3, 2, projection='3d')

    # extract data
    sm0, _, _ = zip(*np.load('shetland_a.npy', allow_pickle=True))
    sm1, _, _ = zip(*np.load('shetland_b.npy', allow_pickle=True))

    # stack submaps into one set
    sm = np.vstack((np.vstack(sm0), np.vstack(sm1)))
    sm = np.vstack(sm)
    
    # plot the points
    ax.scatter(sm[:,0], sm[:,1], sm[:,2], c=sm[:,2], cmap='jet', marker=',', s=0.001, alpha=0.1)

    # remove background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_axis_off()
    ax.set_aspect('equal')

    # save memory
    del sm0
    del sm1
    del sm
    del _

    # antarctica plot
    ax = fig.add_subplot(1, 3, 3, projection='3d')

    # extract data
    sm, _, _ = zip(*np.load('antarctica.npy', allow_pickle=True))

    # stack submaps into one set
    sm = np.vstack(sm)
    
    # plot the points
    ax.scatter(sm[:,0], sm[:,1], sm[:,2], c=sm[:,2], cmap='jet', marker=',', s=0.001)#, alpha=0.1)

    # remove background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_axis_off()
    ax.set_aspect('equal')

    fig.savefig('plot.png', bbox_inches='tight', dpi=5000)

if __name__ == "__main__":

    plot_canonical_data()