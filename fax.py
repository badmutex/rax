
import numpy as np

import multiprocessing as multiproc
import glob
import os
import itertools
import functools
import logging

################################################################################
#                           Setup logging abilities                            #
################################################################################

def _setup_logger():
    logformat = '%(asctime)s %(module)s %(levelname)s %(message)s'
    logger    = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler   = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(logformat)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

_LOGGER = _setup_logger()




def set_logging_level(lvl):
    """
    Set the logging level.

    @param lvl (logging.LEVEL): a logging level from the logging module

    Example:
        import fax
        import logging
        fax.set_logging_level(logging.DEBUG)
    """
    global _LOGGER
    _LOGGER.setLevel(lvl)


def _log_message(lvl, msg, *args, **kws):
    global _LOGGER
    _LOGGER.log(lvl, msg, *args, **kws)

def log_debug(*args, **kws):
    _log_message(logging.DEBUG, *args, **kws)

def log_info(*args, **kws):
    _log_message(logging.INFO, *args, **kws)

def log_warning(*args, **kws):
    _log_message(logging.WARNING, *args, **kws)

def log_error(*args, **kws):
    _log_message(logging.ERROR, *args, **kws)

def log_critical(*args, **kws):
    _log_message(logging.CRITICAL, *args, **kws)


################################################################################
#                               Process pool                                   #
################################################################################


class Pool(object):
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None):
        self.nprocs = processes or 1

        if not self.nprocs == 1:
            self.pool = multiproc.Pool(processes=processes, initializer=initializer, initargs=initargs, maxtasksperchild=maxtasksperchild)
        else:
            self.pool = None

    def map(self, func, iterable, chunksize=1, force=False):
        """
        Map a function over the iterable.

        @param func (function): the function to apply
        @param iterable (anything implementing __iter__()): the sequence of values
        @param chunksize=1 (int): the chunksize when using multiple processors
        @param force=False (boolean): force evaluation of the result when using single processor, otherwise (default) use lazy paradigm

        @return (list or generator): a list (nprocs > 1 or force == True) or a generator (nprocs == 1) of values from applying the function to the iterable
        """
        if self.nprocs == 1:
            log_info('Pool: mapping using single processor')
            generator = itertools.imap(func, iterable)
            if force:
                return list(generator)
            else:
                return generator
        else:
            log_info('Pool: mapping using %d processors' % self.nprocs)
            return self.pool.map(func, iterable, chunksize)

    def finish(self):
        """
        Clear up the pool if using multiple processors
        """
        if not self.pool is None:
            self.pool.close()
            self.pool.join()
            self.pool.terminate()



DEFAULT_POOL = Pool(processes=1)

def setup_pool(processes):
    """
    Setup the default module-level process pool.

    @param processes (int): the number of processors to use.

    Example:
     import fax
     fax.setup_pool(42)
     ...
     project = fax.load_project(root)
    """

    global DEFAULT_POOL
    DEFAULT_POOL = Pool(processes=processes)


def get_pool(pool):
    """
    Chooses between the module default Pool or the provided one if it is of type Pool

    @param pool (Pool or anything): if type(pool) is Pool return pool else return DEFAULT_POOL
    """

    if type(pool) is Pool:
        mypool = pool
        log_debug('Using provided Pool(%s)' % mypool.nprocs)
        return mypool
    else:
        global DEFAULT_POOL
        mypool = DEFAULT_POOL
        log_debug('Using default Pool(%s)' % mypool.nprocs)
        return mypool



################################################################################
#                        Abstractions over the data                            #
################################################################################


class Trajectory(object):
    def __init__(self, run, clone):
        self.run       = run
        self.clone     = clone
        self.data      = dict()
        self.coalesced = None
        self._coalesced_keeplast = False

    def add_generation(self, gen, data):
        """
        Add data for a generation

        @param gen (int): The generation for which the data corresponds to.
        @param data (one dimensional numpy.array). The values for this generation
        """

        if gen in self.data:
            raise ValueError, 'Generation %s already exists in trajectory R%dC%d' % (gen, self.run, self.clone)

        if self.coalesced is not None:
            self.coalesced = None

        self.data[gen] = data


    def numframes(self, keeplast=False):
        """
        @param keeplast=False (boolean): keep the last frame between trajectories
        @return (int): the number of frames in the Trajectory
        """

        self.coalesce(keeplast=keeplast)
        return self.coalesced.shape[0]

    def length(self, output_freq, keeplast=False):
        """
        Get the time elapsed in the trajectory.
        Units are arbitrary: the *user* is responsible for correctness

        @param output_freq (float): the time between frames
        @param keeplast=False (boolean): keep the last frame between trajectories

        @return (float): the elapsed time in arbitrary units of time
        """

        length = self.numframes(keeplast=keeplast)
        return length * output_freq


    def is_coalesced(self, keeplast=False):
        """
        @param keeplast=False (boolean): keep the last frame between trajectories
        @return (boolean): is the Trajectory coalesced?

        """
        return self.coalesced is not None and self._coalesced_keeplast and keeplast


    def coalesce(self,keeplast=False):
        """
        Merge the generation data into a single numpy array.

        @param keeplast=False (boolean): Keep the last frame of each generation, otherwise (default) ignore it.
        """

        self._coalesced_keeplast = keeplast

        if not self.is_coalesced(keeplast=keeplast):
            vals = list()
            N = len(self.data)
            for L, k in enumerate(sorted(self.data.iterkeys())):
                log_debug('Coalescing R%dC%dG%d' % (self.run, self.clone, k))
                data = self.data[k]
                n = data.size
                for i, v in enumerate(data):
                    if i < n - 1 or L == N - 1:
                        vals.append(v)
                    elif not keeplast:
                        vals.append(v)
            A = np.array(vals)
            self.coalesced = A


    def get_generations(self):
        """
        Iterate over the generations.

        @return (generator): a generator over the trajectories in the project
        """
        for gen in sorted(self.data.keys()):
            yield gen

    def count_generations(self):
        """
        Get the number of generations in a trajectory

        @return (int): the number of generations in this trajectory
        """

        return len(self.data.keys())

    def get_generation_data(self, gen):
        return self.data[gen]


    def get_trajectory_data(self, keeplast=False):
        """
        Get the numpy.array of the coalesced trajectory

        @return (numpy.array): The (optionally coalesced) data for the trajectory.
        """

        if self.coalesced is None:
            self.coalesce(keeplast=keeplast)
        return self.coalesced

    def __iter__(self):
        data = self.get_trajectory_data()
        return iter(data)


class Project(object):
    def __init__(self, outputfreq=None):
        """
        @param outputfreq=None (float): time between frames
        """

        log_debug('Project.__init__(outputfreq=%s)' % outputfreq)

        self.projdata = dict()
        self.outputfreq = outputfreq
    
    def get_trajectory_lengths(self, keeplast=False, pool=DEFAULT_POOL):
        """
        @param keeplast=False (boolean): keep the frame between generations
        @param pool=DEFAULT_POOL (Pool)
        """
        log_debug('Project.get_trajectory_lengths: self.outputfreq = %s' % self.outputfreq)

        if type(self.outputfreq) is not float or self.outputfreq <= 0:
            raise ValueError, 'I need to know the output frequency'

        myfn   = functools.partial(_get_traj_lengths, self.outputfreq, keeplast=keeplast)
        result = pool.map(myfn, self.get_trajectories())
        return result

    def set_outputfrequency(self, outputfreq):
        """
        @param outputfreq (float): the time between frames
        """

        self.outputfreq = outputfreq

    def add_generation(self, run, clone, gen, data):
        """
        Add data for a generation. Create the Trajectory if needed.

        @param run (int)
        @param clone (int)
        @param gen (int)
        @param data (one-dimensional numpy.array)
        """

        if run not in self.projdata:
            self.projdata[run] = dict()

        if clone not in self.projdata[run]:
            self.projdata[run][clone] = Trajectory(run, clone)

        self.projdata[run][clone].add_generation(gen, data)


    def get_trajectory(self, run, clone, coalesce=False, keeplast=False):
        """
        Ge the (optionally coalesced) Trajectory

        @param run (int)
        @param clone (int)
        @param coalesce=False (boolean)

        @return (Trajectory)
        """
        traj = self.projdata[run][clone]
        if coalesce:
            traj.coalesce(keeplast=keeplast)
        return traj

    def get_trajectories(self):
        """
        Get the sequence of the Trajectories in the Project

        @return (generator over Trajectories)
        """

        for rundata in self.projdata.itervalues():
            for traj in rundata.itervalues():
                yield traj

    def coalesce(self, keeplast=False):
        """
        Coalesce the project

        @param keeplast=False (boolean): keep the last frame of the generations
        """
        for traj in self.get_trajectories():
            traj.coalesce(keeplast=keeplast)

    def merge(self, proj):
        """
        Add the data in *proj* to the current project

        @param proj (Project): merge the Trajectories in *proj* into this Project
        """
        _merge_projects(self, proj)

    def write(self, root, pool=DEFAULT_POOL):
        """
        Write the project out to a root directory.
        This creates the root/RUNXXXX/CLONEYYYY/GENZZZZ.dat files.

        @param root (string): the root under which the RUN/CLONE/GEN files will be created
        @param pool=DEFAULT_POOL (Pool): The *Pool* to used (default with 1 processor)
        """

        log_info('Saving project under %s' % root)

        for run, rundata in self.projdata.iteritems():
            for clone, traj in rundata.iteritems():
                dirname = os.path.join(root, rcg_path_name('RUN',run), rcg_path_name('CLONE',clone))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                # force evaluation
                list(pool.map(functools.partial(_save_gen, dirname, traj), traj.get_generations()))

    def savetxt(self, path, run, clone, keeplast=False, **kws):
        """
        Save a generation to a text file using numpy.savetxt

        @param path (string): the path to save to
        @param run (int)
        @param clone (int)
        @param keeplast=False (boolean): keep the frames between two generations when coalescing
        @param kws: keywords to be passed to numpy.savetxt
        """

        log_info('Saving trajectory to %s' % path)

        traj = self.get_trajectory(run, clone)
        data = traj.get_trajectory_data(keeplast=keeplast)
        np.savetxt(path, data.transpose(), **kws)



def _save_gen(dirname, traj, gen):
    path  = os.path.join(dirname, rcg_path_name('GEN', gen) +'.dat')
    run   = traj.run
    clone = traj.clone
    data  = traj.get_generation_data(gen)
    log_debug('Saving %s' % path)
    with open(path, 'w') as fd:
        for ix, value in enumerate(data):
            fd.write('%(run)d,%(clone)d,%(gen)d,%(frame)d,%(data)s\n' % {
                    'run':run, 'clone':clone, 'gen':gen, 'frame':ix, 'data':data[ix]})




def _merge_projects(proj1, proj2):
    """
    Update proj2 with the data in proj1
    WARNING: Statefull: this modifes the state of proj2
    """

    for run,rundata in proj1.projdata.iteritems():
        for clone, clonedata in rundata.iteritems():
            for gen, gendata in clonedata.data.iteritems():
                proj2.add_generation(run, clone, gen, gendata)

    log_debug('_merge_projects: proj.outputfreq = %s' % proj2.outputfreq)
    return proj2

def _get_traj_lengths(outputfreq, traj, keeplast=False):
    length = traj.length(outputfreq, keeplast=keeplast)
    r,c = traj.run, traj.clone
    return r, c, length



################################################################################
#                Load and process Projects and Trajectories                    #
################################################################################


def _load_project_processor(path, **initprojkws):
    log_debug('_load_project_processor: Loading %s' % path)
    log_debug('_load_project_processor: initprojkws=%s' % initprojkws)

    data  = np.loadtxt(path, delimiter=',', unpack=True)
    run   = data[0,0].astype(int)
    clone = data[1,0].astype(int)
    gen   = data[2,0].astype(int)
    proj  = Project(**initprojkws)
    proj.add_generation(run, clone, gen, data[-1])
    return proj


def rcg_path_name(name, value):
    return '%s%04d' % (name, value)



def load_project(root, runs=None, clones=None, gens=None, pool=DEFAULT_POOL, coalesce=False, chunksize=None, **initprojkws):
    """
    Reads the data into a Project object.

    @param root (string): The root to the analysis directory. For example, given a file analysis/rmsd/C-alpha/RUN1234/CLONE4567/GEN4242.dat, root would be 'analysis/rmsd/C-alpha'
    @param pool=DEFAULT_POOL (Pool): The pool of processors to use. By default a new *Pool* is created and destroyed on completion, unless one is provided.
    @param coalesce=False (boolean): Coalesce the project trajectories.
    @param runs=None (list of ints): list of runs to load
    @param clones=None (list of ints):  a list of clones to load
    @param gens=None (list of ints):  a list of generations to load

    @param **initprojkws: parameters to pass to the Project constructor

    @return (Project)

    """

    log_debug('load_project: initprojkws=%s' % initprojkws)

    def filter_rcg(paths, runs, clones, gens):

        runs   = runs   or []
        clones = clones or []
        gens   = gens   or []

        runsp   = map(lambda v: rcg_path_name('RUN', v)  , runs)
        clonesp = map(lambda v: rcg_path_name('CLONE', v), clones)
        gensp   = map(lambda v: rcg_path_name('GEN', v)  , gens)


        for p in paths:

            oks = [len(runs) < 1, len(clones) < 1, len(gens) < 1]
            for pat in runsp:
                if pat in p:
                    oks[0] = True
                    break

            for pat in clonesp:
                if pat in p:
                    oks[1] = True
                    break

            for pat in gensp:
                if pat in p:
                    oks[2] = True
                    break

            if all(oks): yield p


    log_info('Searching for data in %s' % root)

    myglob   = os.path.join(root, 'RUN*', 'CLONE*', 'GEN*.dat')
    data_itr = glob.iglob(myglob)
    data_itr = filter_rcg(data_itr, runs, clones, gens)


    log_info('Loading data')
    myfn = functools.partial(_load_project_processor, **initprojkws)
    log_debug('load_project: loadfn: %s' % myfn)
    projects = pool.map(myfn, data_itr)

    log_info('Accumulating project data')
    project  = reduce(_merge_projects, projects, Project())

    if coalesce:
        log_info('Coalescing project')
        project.coalesce()


    return project


def _process_trajectories_processor(fn, traj):
    log_debug('Processing R %d C %d' % (traj.run, traj.clone))
    return fn(traj)



def process_trajectories(proj, fn, pool=DEFAULT_POOL):
    """
    Map a function over the *Trajectories* in a *Project*

    @param proj (Project)
    @param fn (Trajectory -> r: a function accepting a single argument of type *Trajectory*)
    @param pool=DEFAULT_POOL (Pool)

    @return (sequence of r)
    """

    func = functools.partial(_process_trajectories_processor, fn)

    log_info('Processing trajectories')
    results = pool.map(func, proj.get_trajectories())


    return results
