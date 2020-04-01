# Christopher Iliffe Sprague
# sprague@kth.se
# Test our net

from pointnetkl import *
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from mpl_toolkits import mplot3d
import pptk

def load_all_datasets():

    # datasets
    paths = [
        ('../data/antarctica/antarctica_7/', 'antarctica', False),
        ('../data/baltic/', 'baltic', False),
        ('../data/borno/overnight/', 'borno', False),
        ('../data/shetland/', 'shetland', False),
        ('../data/antarctica/antarctica_11_sqr/', 'antarctica_slam', True),
        ('../data/borno/slam_8/', 'borno_slam', True),
        ('../data/antarctica/antarctica_11_sqr/', 'antarctica_val', False),
        ('../data/borno/slam_8/', 'borno_val', False),
    ]

    # load and save each dataset
    for p in paths:

        # feedback
        print('Loading {}'.format(p[0]))

        # saving destination
        name = '../data/consolidated/{}'.format(p[1])

        # load the dataset
        Data.get_dataset(p[0], fname=name, voxel_size=0.05, discarded=p[2])

def train_network(lf=Model.kl_loss):

    # clear GPU cache
    torch.cuda.empty_cache()

    # training sets
    d = Data([
        '../data/consolidated/{}.npy'.format(name) for name in 
        ['antarctica', 'baltic', 'borno', 'shetland']
    ], -1)

    # training and testing set
    n = int(0.2*len(d))
    d0, d1 = torch.utils.data.random_split(d, [len(d)-n, n])

    # training set
    #d0 = d
    d0 = torch.utils.data.DataLoader(
        d0,
        batch_size=40,
        num_workers=torch.multiprocessing.cpu_count(),
        pin_memory=True,
        drop_last=True,
        sampler=torch.utils.data.RandomSampler(d0, replacement=True, num_samples=500)
    )
    d0.dataset.name = 'Training'

    # testing set
    #d1 = Data(['../data/consolidated/antarctica.npy'])
    d1 = torch.utils.data.DataLoader(
        d1,
        batch_size=40,
        num_workers=torch.multiprocessing.cpu_count(),
        pin_memory=True,
        drop_last=True,
        sampler=torch.utils.data.RandomSampler(d1, replacement=True, num_samples=500)
    )
    d1.dataset.name = 'Validation'

    # instantiate pointnetkl
    shape = [1000]*4
    net = CovarianceNet(
        shape=shape,
        dp=0.4,
        name='{}_{}'.format(lf.__name__, shape)
    )

    # train pointnetkl
    net.optimise([d0, d1], lf=lf, gpu=True, lr=1e-4, wd=1e-4, epo=50000)

def plot_data():

    # datasets
    names = [
        'baltic',
        'borno',
        'shetland',
        'antarctica'
    ]

    # for each dataset
    for e in names:

        # feedback
        print('Plotting {}'.format(e))

        # load data
        data = np.load('../data/consolidated/{}.npy'.format(e), allow_pickle=True)

        # extract submaps
        sm, _, _, rhos = zip(*data)

        # concatenate all submaps
        sm = np.vstack(sm)

        # voxelise the data
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(sm)
        # pc = o3d.geometry.voxel_down_sample(pc, voxel_size=0.02)
        # sm = np.asarray(pc.points)

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot points
        ax.scatter(sm[:,0], sm[:,1], sm[:,2], c=np.linalg.norm(sm, axis=1), cmap='jet', marker='.', s=1, linewidths=0, alpha=0.1)

        # plot style
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plot labels
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        plt.tight_layout()

        # save plot
        fig.savefig('../figs/{}_data.png'.format(e), dpi=500)

        # release memory
        del data, sm 

def plot_variance_distribution():

    # datasets
    names = [
        'antarctica',
        'antarctica_7',
        'antarctica_11',
        'baltic',
        'borno',
        'shetland_a',
        'shetland_b',
        'spain'
    ]

    names = ['antarctica']

    # for each environment
    for e in names:

        # feedback
        print('Plotting {}'.format(e))

        # load data
        data = np.load('../data/consolidated/{}.npy'.format(e), allow_pickle=True)

        # extract submaps
        sm, _, _ = zip(*data)

        # compute the z-variance for each submap
        variances = np.arary([a[:,2].var() for a in sm])

def plot_mean_submap():

    # datasets
    names = [
        'antarctica',
        'antarctica_7',
        'antarctica_11',
        'baltic',
        'borno',
        'shetland_a',
        'shetland_b',
        'spain',
        'MMT'
    ]

    # for each environment
    for e in names:

        # feedback
        print('Plotting {}'.format(e))

        # load data
        data = np.load('../data/consolidated/{}.npy'.format(e), allow_pickle=True)

        # extract submaps
        sm, _, _ = zip(*data)

        # compute the z-variance for each submap
        var = np.array([a[:,2].var() for a in sm])

        # compute the average variance
        avar = var.mean()

        # get index with variance closest to average
        i = np.argmin(abs(var - avar))

        # select mean submap
        sm = sm[i]

        print(sm.shape)

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot points
        ax.scatter(sm[:,0], sm[:,1], sm[:,2], c=sm[:,2], cmap='jet', marker='.', s=3, linewidths=0)

        # plot style
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plot labels
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        plt.tight_layout()

        # save plot
        fig.savefig('../figs/{}_data.png'.format(e), dpi=500, bbox_inches='tight')



if __name__ == "__main__":

    train_network()
    #load_all_datasets()
    #plot_data()
    #plot_mean_submap()

    # # load data
    # data = np.load('../data/consolidated/antarctica.npy', allow_pickle=True)

    # # extract submaps
    # sm, _, _ = zip(*data)

    # variances = np.array([a[:,2].var() for a in sm])

    # print(np.average([s.shape[0] for s in sm]))

    # i = np.argmax(variances)

    # pptk.viewer(sm[i], sm[i][:,2])

    # net = torch.load('../models/kl_loss_[1000, 1000, 1000, 1000].net')
    # net.plot_loss()
    # plt.show()

    # # load dataset
    # d = Data('../data/consolidated/baltic.npy')
    # print(max(d.rhos))
    # d = Data('../data/consolidated/shetland_a.npy')
    # print(max(d.rhos))
    # d = Data('../data/consolidated/borno.npy')
    # print(max(d.rhos))
    # d = Data('../data/consolidated/borno.npy')
    # print(max(d.rhos))

    # names = [
    #     'antarctica',
    #     'baltic',
    #     'borno',
    #     'shetland_a',
    #     'shetland_b',
    #     'spain'
    # ]

    # for name in names:
    #     print(name)
    #     print(max(Data('../data/consolidated/{}.npy'.format(name)).rhos))
