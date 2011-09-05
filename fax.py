
import numpy as np

import multiprocessing as multiproc
import glob
import os
import itertools

class Trajectory(object):
    def __init__(self, run, clone):
        self.run       = run
        self.clone     = clone
        self.data      = dict()
        self.coalesced = None

    def add_generation(self, gen, data):
        if gen in self.data:
            raise ValueError, 'Generation %s already exists in trajectory R%dC%d' % (gen, self.run, self.clone)

        if self.coalesced is not None:
            self.coalesced = None

        self.data[gen] = data

    def coalesce(self,keeplast=False):
        """
        Merge the generation data into a single numpy array.

        @param keeplast=False
            Keep the last frame of each generation, otherwise (default) ignore it.
        """

        if self.coalesced is None:
            vals = list()
            for k in sorted(self.data.iterkeys()):
                data = self.data[k]
                n = data.size
                for i, v in enumerate(data):
                    if i < n - 1:
                        vals.append(v)
                    elif not keeplast:
                        vals.append(v)
            A = np.array(vals)
            self.coalesced = A


    def get_trajectory_data(self, keeplast=False):
        """
        returns a numpy.array of the coalesced trajectory
        """

        if self.coalesced is None:
            self.coalesce(keeplast=keeplast)
        return self.coalesced



class Project(object):
    def __init__(self):
        self.projdata = dict()
    
    def add_generation(self, run, clone, gen, data):
        if run not in self.projdata:
            self.projdata[run] = dict()

        if clone not in self.projdata[run]:
            self.projdata[run] = Trajectory(run, clone)

        self.projdata[run][clone].add_generation(gen, traj)


    def get_trajectory(self, run, clone, coalesce=False):
        traj = self.projdata[run][clone]
        if coalesce:
            traj.coalesce()
        return traj

    def get_trajectories(self):
        for rundata in self.projdata.itervalues():
            for traj in rundata.itervalues():
                yield traj

    def coalesce(self, keeplast=False):
        for traj in self.get_trajectories():
            traj.coalesce(keeplast=keeplast)

    def merge(self, proj):
        """
        Add the data in *proj* to the current project
        """
        merge_projects(self, proj)


def merge_projects(proj1, proj2):
    """
    Update proj1 with the data in proj2
    WARNING: this modifes the state proj1
    """

    for run,rundata in proj2.data.iteritems():
        for clone, clonedata in rundata.iteritems():
            for gen, gendata in clonedata.data.iteritems():
                proj1.add_generation(run, clone, gen, gendata)
    return proj1



def start_pool(nprocs=None):
    if nprocs is None or 0:
        nprocs = multiproc.cpu_count()

    pool = multiproc.Pool(nprocs)
    return pool

def terminate_pool(pool):
    pool.join()
    pool.close()
    pool.terminate()


def load_project(root, nprocs=None, pool=None, coalesce=False, verbose=True):
    """
    Reads the data into a Project object.

    @param root:
        The root to the analysis directory. For example, given a file analysis/rmsd/C-alpha/RUN1234/CLONE4567/GEN4242.dat, root would be 'analysis/rmsd/C-alpha'
    @param nprocs:
        The number of processors to use (default = number of processors on the machine). If this is 1, itertools.imap is used to apply *fn*, otherwise a multiprocessor.Pool is used.
    @param pool:
        The pool of processors to use. By default a new one is created using multiprocessor.Pool(nprocs) and destroyed on completion, unless one is provided. 

    @return project:
        The coalesced project.

    """


    data_itr = glob.iglob(os.path.join(root,'RUN*','CLONE*','GEN*.dat'))

    def processor(path):
        if verbose:
            print 'Processing', path

        data  = np.loadtxt(path, delimiter=',', unpack=True)
        run   = data[0,0]
        clone = data[1,0]
        gen   = data[2,0]
        proj  = Project()
        proj.add_geneartion(run, clone, gen, data[-1])
        return proj

    if nprocs == 1:
        return itertools.imap(processor, data_iter)


    if pool is None:
        mypool = start_pool(nprocs)
    else:
        mypool = pool

    projects = mypool.map(process, data_itr)
    project  = reduce(merge_projects, projects, Project())

    if coalesce:
        project.coalesce()


    if pool is None:
        terminate_pool(pool)


    return project


def process_trajectories(proj, fn, nprocs=None, pool=None, verbose=True):


    if pool is None:
        mypool = start_pool(nprocs)
    else:
        mypool = pool

    def traj_processor(traj):
        if verbose:
            print 'Processing R %d C %d' % (traj.run, traj.clone)
        return fn(traj)

    results = mypool.map(processor, proj.get_trajectories())

    if pool is None:
        terminate_pool(mypool)

    return results
