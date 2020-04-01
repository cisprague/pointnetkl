# Christopher Iliffe Sprague
# sprague@kth.se

import sys, open3d as o3d, glob, numpy as np, tqdm, matplotlib.pyplot as plt, pptk, os
from sklearn import preprocessing
from pointnet.model import PointNetfeat
import torch, multiprocessing as mp
torch.manual_seed(0)

class Data(torch.utils.data.Dataset):

    '''
    Dataset from which to train.
    '''
    
    def __init__(self, fpaths, n=None, randomise=True):
        
        # submaps
        self.submaps, self.priors, self.posteriors, self.rhos, self.smfn, self.cvfn = Data.load_data(fpaths, n, randomise=randomise)
        
        # become dataset
        torch.utils.data.Dataset.__init__(self)

        
    @staticmethod
    def get_paths(source, discarded=False):

        '''
        Returns list of submap-covariance pair filenames.
        '''
        
        # list of submap-covariance pair file names
        ml = list()
        
        # sorting function
        key = lambda x: int(x.split('_')[-1].split('.')[0])
        
        # get paths
        for dpath, dname, fnames in os.walk(source):
            
            # get submaps and covariances
            sm = glob.glob('{}/submap_*.xyz'.format(dpath))
            cv = glob.glob('{}/submap_*.txt'.format(dpath))
            
            # if there's data
            if len(sm) > 0 and len(cv) > 0 and ('Discarded' not in dpath if not discarded else True):
                
                # sort the data
                sm = sorted(sm, key=key)
                cv = sorted(cv, key=key)
                
                # match quantities
                n = min(len(sm), len(cv))
                sm, cv = sm[:n], cv[:n]
                
                # for each submap-covariance pair
                for a in zip(sm, cv):
                    
                    # ensure file numbers match
                    assert(key(a[0]) == key(a[1]))
                    
                    # add to the master list
                    ml.append(a)
                    
        return ml
    
    @staticmethod
    def get_dataset(path, fname=None, voxel_size=0.01, euc_norm=False, discarded=False):

        # get file destinations
        paths = Data.get_paths(path, discarded=discarded)

        # create a pool of parallel workers
        p = mp.Pool(mp.cpu_count())
        
        # retreive and process data in parallel
        d = p.starmap(Data.get_sample, [(*fn, voxel_size) for fn in paths])

        # save if desired
        if fname is not None:
            np.save(fname, d)

        return d
        
    @staticmethod
    def get_sample(submap_fname, covariance_fname, voxel_size):

        # load the submap
        sm = np.loadtxt(submap_fname)
        cv = np.loadtxt(covariance_fname).reshape(-1, 2, 2)
        prior, posterior = cv

        # ensure off-diagonals of priors are zero
        prior[0, 1] = 0
        prior[1, 0] = 0

        # zero-mean the submap
        sm = sm - sm.mean(axis=0)

        # normalise the submap to a unit-sphere
        rho = max(np.linalg.norm(sm, axis=1))
        sm /= rho

        # voxelise the normalised submap
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sm)
        pc = o3d.geometry.voxel_down_sample(pc, voxel_size=voxel_size)
        sm = np.asarray(pc.points)

        return sm, prior, posterior, rho, submap_fname, covariance_fname
    
    @staticmethod
    def load_data(fpaths, ns, randomise=True):

        '''
        fpaths = string if just loading one dataset
        fpaths = list of strings for combining datasets
        '''

        # if only given one filename
        if not isinstance(fpaths, list):
            fpaths = [fpaths]

        # lists of submaps, priors, posteriors, and scale-factors
        sml, pril, posl, sfl, smfnl, cvfnl = list(), list(), list(), list(), list(), list()
        
        # smallest number of data points - not efficient, but done once
        n = min([len(np.load(fpath, allow_pickle=True)) for fpath in fpaths])

        # specify number of data
        if ns is None:
            ns = n
        elif ns == -1:
            pass
        else:
            ns = min(ns, n)
        
        # for each path
        for fpath in fpaths:

            print("Loading {}".format(fpath))

            # load submaps, priors, posteriors, and scale-factors
            sm = np.load(fpath, allow_pickle=True)
            
            # shuffle the data
            if randomise: np.random.shuffle(sm)
            
            # subsample the data to match smallest number of data points
            sm = sm[:ns]
            
            # extract the data
            sm, priors, posteriors, rhos, smfn, cvfn = zip(*sm)
            
            # add data to overall list
            sml  += list(sm)
            pril += list(priors)
            posl += list(posteriors)
            sfl  += list(rhos)
            smfnl += list(smfn)
            cvfnl += list(cvfn)

        # number of points in the largest pointcloud
        n = max([a.shape[0] for a in sml])

        # duplicate random points so each pointcloud has same number
        sml = [np.vstack((a, a[np.random.choice(a.shape[0], n-a.shape[0])])) for a in sml]

        # format priors, posteriors, and scale-factors as arrays also
        pril = np.stack(pril)
        posl = np.stack(posl)
        sfl = np.stack(sfl)
        smfnl = np.stack(smfnl)
        cvfnl = np.stack(cvfnl)

        return sml, pril, posl, sfl, smfnl, cvfnl
    
    def __len__(self):
        return len(self.submaps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.submaps[idx], dtype=torch.float32), 
            torch.tensor(self.priors[idx], dtype=torch.float32), 
            torch.tensor(self.posteriors[idx], dtype=torch.float32),
            torch.tensor(self.rhos[idx], dtype=torch.float32),
            self.smfn[idx],
            self.cvfn[idx]
        )

class Model(object):
    
    def __init__(self, name='covnet'):
        
        # name - used for saving model
        self.name = name

        # loss record
        self.loss = dict()
    
    def optimise(self, data, lf=torch.nn.MSELoss(), gpu=True, lr=1e-4, wd=1e-4, epo=5000):
        
        # put network on GPU
        self.cuda() if gpu else self.cpu()
            
        # optimisation algorithm
        oa = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
        
        # initialise loss records
        [self.loss.update({d.dataset.name: list()}) for d in data]
        
        # progress bar
        pb = tqdm.tqdm(range(epo))

        # early stopping
        es = EarlyStopping(patience=20, verbose=True)
        
        # training episode loop
        for i, _ in enumerate(pb):

            # save the model every 100 iterations
            if i%10 == 0: torch.save(self, '../models/{}_running.net'.format(self.name))
            
            # data set loop
            for j, d in enumerate(data):

                # zero gradients
                oa.zero_grad()
                
                # episode loss for this dataset
                loss = list()

                # toggle prediction mode
                if j == 0:
                    self.train()
                else:
                    self.eval()
            
                # batch loop
                for b in d:

                    # clear GPU memory
                    torch.cuda.empty_cache()

                    # extract submap and posterior
                    submaps, priors, posteriors, rhos, smfn, cvfn = b

                    # allocate data
                    if gpu:
                        submaps, priors, posteriors, rhos = (
                            submaps.cuda(), 
                            priors.cuda(), 
                            posteriors.cuda(), 
                            rhos.cuda()
                        )
                    else:
                        submaps, priors, posteriors, rhos = (
                            submaps.cpu(), 
                            priors.cpu(), 
                            posteriors.cpu(), 
                            rhos.cpu()
                        )

                    # compute loss
                    l = lf(self(submaps, priors, rhos), posteriors, reduce=False)

                    # record loss
                    [loss.append(float(a.item())) for a in l]

                    # reduce loss and optimise
                    if j == 0:
                        l = sum(l)
                        l.backward()

                # optimisation step
                if j == 0:
                    oa.step()

                # average the losses
                loss = sum(loss)/len(loss)
            
                # record average loss
                self.loss[d.dataset.name].append(loss)
            
            # set progress bar description
            losses = ["{0} {1:.4f} ".format(k, self.loss[k][-1]) for k in self.loss.keys()]
            pb.set_description(str(losses))

            # check for early stopping
            es(self.loss['Validation'][-1], self)

        # put the net on cpu
        self.cpu()
              
    @staticmethod
    def eigen_loss(y, yp):
        
        # compute eigen values
        y = torch.stack([a.symeig(eigenvectors=True).eigenvalues for a in y])
        yp = torch.stack([a.symeig(eigenvectors=True).eigenvalues for a in yp])
        
        # l2 loss
        l = (yp - y)**2
        l = l.mean()
        return l

    @staticmethod
    def kl_loss(sigma0, sigma1, reduce=False):

        # compute zero-mean divergences
        d = [(torch.trace(b.inverse().mm(a)) - 2 + torch.log(b.det()/a.det()))/2 for a,b in zip(sigma0, sigma1)]

        # average the losses
        return sum(d)/len(d) if reduce else d

    def plot_loss(self, ax=None):
        
        # create plot if not given
        if ax is None:
            fig, ax = plt.subplots(1)
            
        # labels
        labels = self.loss.keys()
        
        # line colours in grayscale
        colours = np.linspace(0, 0.5, len(labels))
            
        # plot losses for each dataset
        for k, c in zip(labels, colours):
            ax.plot(self.loss[k], c=str(c), label=k)
            
        # legend
        ax.legend()
        
        return ax

class CovarianceNet(Model, torch.nn.Module):
    
    def __init__(self, shape=[512, 256, 128], dp=0.5, name='covnet'):
        
        # become model and torch model
        Model.__init__(self, name=name)
        torch.nn.Module.__init__(self)
        
        # vanilla pointnet - gives 1024 global features
        self.pointnet = PointNetfeat(global_feat=True, feature_transform=True)
        
        # dropout rate
        self.dp = dp
        
        # mlp - 3 outputs for positive definite covariance matrix
        s = [1024 + 1] + shape
        self.mlp = torch.nn.Sequential(*[
            op for i in range(len(s) - 1) for op in [
                torch.nn.Linear(s[i], s[i+1]),
                torch.nn.BatchNorm1d(s[i+1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dp)
            ]
        ], torch.nn.Linear(s[-1], 3))
        
    def forward(self, submaps, priors, rhos):
        
        # format incoming pointcloud
        x = submaps.transpose(1, 2)
        
        # compute submap features - 1024
        x = self.pointnet(x)[0]

        # # scale priors with submap scale
        # priors = torch.stack([r**-2*p for r, p in zip(rhos, priors)])

        # # get 3 unique parameters from cholesky decomposition of priors
        # priors = priors.cholesky().reshape(-1, 4)
        # priors = torch.cat((priors[:,:2], priors[:,3:]), dim=1)

        # # concatenate submap features and prior cholesky parameters
        #x = torch.cat((x, priors), dim=1)

        # concatenate submap features and scale factor
        x = torch.cat((x, rhos.reshape(-1,1)), dim=1)
        
        # compute values
        x = self.mlp(x)

        # lower triangle values
        li = x[:,0]

        # diagnonal values - strictly positive to enfore uniqueness
        di = torch.exp(x[:,1:])

        # construct diagonal elements
        L = torch.stack([a.diagflat() for a in di])

        # add lower triangular elements
        L = torch.stack([b + a.expand(2,2).tril() - a.expand(2).diagflat() for a,b in zip(li, L)])

        # compose postitive definite matrix
        A = torch.stack([a.mm(a.t()) for a in L])

        # # scale predicted posterior with submap scale
        #A = torch.stack([r**2*a for r, a in zip(rhos, A)])

        return A

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model, '../models/{}_checkpoint.net'.format(model.name))
        self.val_loss_min = val_loss

if __name__ == '__main__':

    # data source (destination is the same)
    source = '/home/cisprague/dev/ipp/data/antarctica/antarctica_11_rect/'
    print(Data.get_paths(source))