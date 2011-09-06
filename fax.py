
import numpy as np

import multiprocessing as multiproc
import glob
import os
import itertools
import functools

class Pool(object):
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None):
        self.nprocs = processes or 1

        if not self.nprocs == 1:
            self.pool = multiproc.Pool(processes=processes, initializer=initializer, initargs=initargs, maxtasksperchild=maxtasksperchild)
        else:
            self.pool = None

    def map(self, func, iterable, chunksize=1):
        if self.nprocs == 1:
            return itertools.imap(func, iterable)
        else:
            return self.pool.map(func, iterable, chunksize)

    def finish(self):
        if not self.pool is None:
            self.pool.close()
            self.pool.join()
            self.pool.terminate()



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


    def get_generations(self):
        for gen in sorted(self.data.keys()):
            yield gen

    def get_generation_data(self, gen):
        return self.data[gen]


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
            self.projdata[run][clone] = Trajectory(run, clone)

        self.projdata[run][clone].add_generation(gen, data)


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

    def write(self, root, pool=Pool(processes=1)):

        for run, rundata in self.projdata.iteritems():
            for clone, traj in rundata.iteritems():
                dirname = os.path.join(root, 'RUN%04d' % run, 'CLONE%04d' % clone)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                # force evaluation
                list(pool.map(functools.partial(_save_gen, dirname, traj), traj.get_generations()))


def _save_gen(dirname, traj, gen):
    path = os.path.join(dirname, 'GEN%04d.dat' % gen)
    run = traj.run
    clone = traj.clone
    data = traj.get_generation_data(gen)
    print 'Saving', path
    with open(path, 'w') as fd:
        for ix, value in enumerate(data):
            fd.write('%(run)d,%(clone)d,%(gen)d,%(frame)d,%(data)s\n' % {
                    'run':run, 'clone':clone, 'gen':gen, 'frame':ix, 'data':data[ix]})




def merge_projects(proj1, proj2):
    """
    Update proj1 with the data in proj2
    WARNING: Statefull: this modifes the state of proj1
    """

    for run,rundata in proj2.projdata.iteritems():
        for clone, clonedata in rundata.iteritems():
            for gen, gendata in clonedata.data.iteritems():
                proj1.add_generation(run, clone, gen, gendata)
    return proj1




def load_project_processor(path):
    print 'Loading', path

    data  = np.loadtxt(path, delimiter=',', unpack=True)
    run   = data[0,0].astype(int)
    clone = data[1,0].astype(int)
    gen   = data[2,0].astype(int)
    proj  = Project()
    proj.add_generation(run, clone, gen, data[-1])
    return proj



def load_project(root, pool=None, coalesce=False, chunksize=None):
    """
    Reads the data into a Project object.

    @param root:
        The root to the analysis directory. For example, given a file analysis/rmsd/C-alpha/RUN1234/CLONE4567/GEN4242.dat, root would be 'analysis/rmsd/C-alpha'
    @param pool:
        The pool of processors to use. By default a new *Pool* is created and destroyed on completion, unless one is provided.
    @param coalesce:
        Coalesce the project trajectories.

    @return project:
        The *Project* instances.

    """


    data_itr = glob.iglob(os.path.join(root,'RUN*','CLONE*','GEN*.dat'))


    if pool is None:
        mypool = Pool()
    else:
        mypool = pool

    projects = mypool.map(load_project_processor, data_itr)
    project  = reduce(merge_projects, projects, Project())

    if coalesce:
        project.coalesce()


    if pool is None:
        pool.finish()


    return project


def process_trajectories_processor(fn, traj):
    print 'Processing R %d C %d' % (traj.run, traj.clone)
    return fn(traj)



def process_trajectories(proj, fn, nprocs=None, pool=None, verbose=True, chunksize=1):

    func = functools.partial(process_trajectories_processor, fn)

    processor = None

    if pool is None:
        mypool = Pool()
    else:
        mypool = pool

    results = mypool.map(func, proj.get_trajectories(), chunksize=chunksize)

    if pool is None:
        mypool.finish()

    return results
