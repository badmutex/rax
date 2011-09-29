
import numpy as np

import ezlog
import ezpool

import glob
import os
import itertools
import functools
import shutil


################################################################################
#                               setup logging                                  #
################################################################################

_logger = ezlog.setup(__name__)


################################################################################
#                               setup pool                                     #
################################################################################

def setup_pool(ncores):
    ezpool.setup_pool(ncores)


################################################################################
#                        Abstractions over the data                            #
################################################################################


class Trajectory(object):
    def __init__(self, run, clone, gens=None):
        """
        @param run (int)
        @param clone (int)
        @param gens=None (int): the total number of generations this trajectory could have
        """

        self.run                 = run
        self.clone               = clone
        self.gens                = gens
        self.data                = dict()
        self.coalesced           = None
        self._coalesced_keeplast = False

        ## keep track of generations as they come in
        self._known_gens         = set()
        if type(gens) is int and gens > 0:
            self._all_gens       = self._compute_all_generations_set()
        else:
            self._all_gens       = None

    def set_num_gens(self, gens):
        """
        @param gens (int)
        """
        self.gens = gens

    def _compute_all_generations_set(self):
        """
        Compute the set of all possible generations this trajectory could have
        """
        return set(xrange(self.gens))

    def missing_generations(self):
        """
        @return (set of ints): the set of generations that are missing in this trajectory
        """

        if self._all_gens is None:
            self._all_gens = self._compute_all_generations_set()

        return self._all_gens - self._known_gens

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

        self._known_gens.add(gen)


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
                _logger.debug('Coalescing R%dC%dG%d' % (self.run, self.clone, k))
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

        @return (list of int): a generator over the generation numbers in the Trajectory in sorted order
        """
        return list(self._known_gens)

    def count_generations(self):
        """
        Get the number of generations in a trajectory

        @return (int): the number of generations in this trajectory
        """

        return len(self._known_gens)

    def get_generation_data(self, gen):
        """
        @return (numpy.array): an array of the values for the specified generation
        """
        return self.data[gen]


    def get_trajectory_data(self, keeplast=False):
        """
        Get the numpy.array of the coalesced trajectory

        @return (numpy.array): The (optionally coalesced) data for the entire trajectory.
        """

        if self.coalesced is None:
            self.coalesce(keeplast=keeplast)
        return self.coalesced

    def map(self, fn):
        """
        Map a function over the values in the trajectory

        @param fn (a -> b)

        @return a new trajectory
        """

        if fn is None:
            raise ValueError, 'Cannot use function of None'

        t = Trajectory(self.run, self.clone)
        for g,data in self.data.iteritems():
            newdata = map(fn, data)
            newdata = np.array(newdata)
            t.add_generation(g, newdata)
        return t


    def __iter__(self):
        data = self.get_trajectory_data()
        return iter(data)


def _trajectory_map(traj, fn=None):
    """
    used in Project.map to ensure serialization out Pool works
    """
    return traj.map(fn)


class Project(object):
    def __init__(self, runs=0, clones=0, gens=0, outputfreq=None, description=None, extrafiles=set(), metadata=dict()):
        """
        @param runs=None (int): the number of runs in the project
        @param clones=None (int): the number of clones in the project
        @param gens=None (int): the number of generations in the project
        @param outputfreq=None (float): time between frames
        @param description=None (string)
        @param description=set() (set)
        """

        self.projdata    = dict()
        self.runs        = runs
        self.clones      = clones
        self.gens        = gens
        self.outputfreq  = outputfreq
        self.description = description
        self.extrafiles  = extrafiles
        self.metadata    = metadata

        self._descfile   = 'README'
        self._extradir   = 'extrafiles'
        self._metadatafile = 'metadata'
        self._rcg_file   = 'num_runs_clones_gens'


    def set_description(self, desc):
        """@param desc (string)"""
        self.description = desc

    def add_metadata(self, name, value):
        """
        Meta data is stored as string representations of the *value*s in a dictionary indexed by *name*
        @param name (str): the key
        @param value (a): the value; stored as metadata[name] = str(value)
        @raise ValueError: if the *name* is already associated with some metadata
        """

        if name in self.metadata:
            raise ValueError, 'Project metadata already contains a value %s for key %s' % (self.metadata[name], name)
        self.metadata[name] = str(value)

    def set_extrafiles(self, extrafiles):
        """@param extrafiles (set)"""
        self.extrafiles = set(extrafiles)

    def add_extrafiles(self, *paths):
        """
        @param paths (string): the paths to add the extra files
        @raise ValueError: if a path is already tracked
        """

        for p in paths:
            if p in self.extrafiles:
                raise ValueError, 'I already know about %s' % p
            self.extrafiles.add(p)


    def get_trajectory_lengths(self, keeplast=False, pool=None):
        """
        @param keeplast=False (boolean): keep the frame between generations
        @param pool=DEFAULT_POOL (Pool)
        """
        _logger.debug('Project.get_trajectory_lengths: self.outputfreq = %s' % self.outputfreq)

        pool   = ezpool.get_pool(pool)

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


    def contains_trajectory(self, run, clone):
        """
        Does this project already contain this Trajectory?
        @param run (int)
        @param clone (int)
        @returns (boolean)
        """

        return run in self.projdata and clone in self.projdata[run]

    def init_trajectory(self, run, clone):
        """
        Add an empty trajectory
        @param run (int)
        @param clone (int)
        @raise ValueError: if the trajectory is already present
        """

        if self.contains_trajectory(run, clone):
            raise ValueError, 'Project already contains (%s, %s)' % (run, clone)

        if not run in self.projdata:
            self.projdata[run] = dict()

        if not clone in self.projdata[run]:
            self.projdata[run][clone] = Trajectory(run, clone)


    def add_generation(self, run, clone, gen, data):
        """
        Add data for a generation. Create the Trajectory if needed.

        @param run (int)
        @param clone (int)
        @param gen (int)
        @param data (one-dimensional numpy.array)
        @raise ValueError if adding this generation would overwrite one already present
        """

        ## the exception handles 2 cases:
        #  1) create an empty trajectory for the given (run,clone)
        #     => no exception raised
        #  2) a trajectory exists, raising an exception
        #     => let the trajectory handle adding the generation data as normal
        try: self.init_trajectory(run, clone)
        except ValueError: pass
        self.projdata[run][clone].add_generation(gen, data)

    def add_trajectory(self, run, clone, traj):
        """
        Add a trajectory to the project

        @param run (int)
        @param clone (int)
        @param traj (Trajectory)

        @raise ValueError if adding this trajectory would overwrite one already present
        """
        self.init_trajectory(run, clone)
        self.projdata[run][clone] = traj

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


    def map(self, fn, pool=None):
        """
        Map a function over the values in the project
        @param fn (a -> b)
        @param pool=DEFAULT_POOL (Pool)

        @return a new transformed project
        """

        pool = ezpool.get_pool(pool)

        _logger.info('Applying function %s to project' % fn)

        p = Project(outputfreq=self.outputfreq, description=self.description, extrafiles=self.extrafiles)
        trajs = self.get_trajectories()
        mapper = functools.partial(_trajectory_map, fn=fn)

        for t2 in pool.map(mapper, trajs):
            p.add_trajectory(t2.run, t2.clone, t2)

        return p


    def write(self, root, pool=None):
        """
        Write the project out to a root directory.
        This creates the root/RUNXXXX/CLONEYYYY/GENZZZZ.dat files.

        @param root (string): the root under which the RUN/CLONE/GEN files will be created
        @param pool=DEFAULT_POOL (Pool): The *Pool* to used (default with 1 processor)

        Example:
          root = '/tmp/testroot'
          myproject.write(root)
          # results in /tmp/testroot/RUN1234/CLONE5678/GEN9012.dat, etc
        """

        _logger.info('Saving project under %s' % root)

        pool = ezpool.get_pool(pool)

        ## write the data
        for run, rundata in self.projdata.iteritems():
            for clone, traj in rundata.iteritems():
                dirname = os.path.join(root, rcg_path_name('RUN',run), rcg_path_name('CLONE',clone))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                # force evaluation
                list(pool.map(functools.partial(_save_gen, dirname, traj), traj.get_generations()))


        ## write the number of runs, clones, and generations
        rcgpath = os.path.join(root, self._rcg_file)
        with open(rcgpath, 'w') as fd:
            fd.write('%d %d %d' % (self.runs, self.clones, self.gens))
        _logger.info('Wrote the number of runs (%s), clones (%d), and gens (%d) to %s' % (self.runs, self.clones, self.gens, rcgpath))

        ## write the metadata
        mdpath = os.path.join(root, self._metadatafile)
        with open(mdpath, 'w') as fd:
            for k, v in self.metadata.iteritems():
                fd.write('%s = %s\n' % (k, v))
        _logger.info('Wrote metadata to %s' % mdpath)

        ## write the description
        with open(os.path.join(root, self._descfile), 'w') as fd:
            fd.write(self.description)
            _logger.info('Wrote description')

        ## copy the extra files
        if self.extrafiles and len(set(self.extrafiles)) == len(self.extrafiles):
            outdir = os.path.join(root, self._extradir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                _logger.info('Created %s' % outdir)

            for p in self.extrafiles:
                outname = os.path.basename(p)
                target = os.path.join(outdir, outname)

                if os.path.exists(target):
                    _logger.warning('Extrafile %s already exists: skipping' % target)
                    continue

                shutil.copy(p, outdir)
                _logger.info('Copied %s to %s' % (p, target))


    def savetxt(self, path, run, clone, keeplast=False, **kws):
        """
        Save a generation to a text file using numpy.savetxt

        @param path (string): the path to save to
        @param run (int)
        @param clone (int)
        @param keeplast=False (boolean): keep the frames between two generations when coalescing
        @param kws: keywords to be passed to numpy.savetxt
        """

        _logger.info('Saving trajectory to %s' % path)

        traj = self.get_trajectory(run, clone)
        data = traj.get_trajectory_data(keeplast=keeplast)
        np.savetxt(path, data.transpose(), **kws)


    def missing_data(self):
        """
        Iterate over the (run,clone,gen) 3-tuples that are missing.
        If the gen is None, then the trajectory given by (run, clone) is missing

        @return (generator over (int,int,int))
        """

        for r in xrange(self.runs):
            for c in xrange(self.clones):
                if self.contains_trajectory(r, c):
                    t = self.get_trajectory(r, c)
                    t.set_num_gens(self.gens)
                    for gen in t.missing_generations():
                        yield t.run, t.clone, gen
                else:
                    yield (r, c, None)




def _save_gen(dirname, traj, gen):
    path  = os.path.join(dirname, rcg_path_name('GEN', gen) +'.dat')
    run   = traj.run
    clone = traj.clone
    data  = traj.get_generation_data(gen)
    _logger.debug('Saving %s' % path)
    with open(path, 'w') as fd:
        for ix, value in enumerate(data):
            fd.write('%(run)d,%(clone)d,%(gen)d,%(frame)d,%(data)s\n' % {
                    'run':run, 'clone':clone, 'gen':gen, 'frame':ix, 'data':data[ix]})



def _merge_projects_seq(projs, **initprojkws):
    """
    Used in the fax.load_project function.
    @param projs (iterable of Projects)
    @param **initprojkws: key-word arguments to be passed to the Project constructor
    @return (Project)
    """

    _logger.debug('_merge_projects_seq')

    mainproj = Project(**initprojkws)
    for p in projs:
        for t in p.get_trajectories():
            for g in t.get_generations():
                mainproj.add_generation(t.run, t.clone, g, t.get_generation_data(g))
    return mainproj



def _get_traj_lengths(outputfreq, traj, keeplast=False):
    length = traj.length(outputfreq, keeplast=keeplast)
    r,c = traj.run, traj.clone
    return r, c, length



################################################################################
#                Load and process Projects and Trajectories                    #
################################################################################


def _load_project_processor(path):
    _logger.debug('_load_project_processor: Loading %s' % path)

    data  = np.loadtxt(path, delimiter=',', unpack=True)

    if len(data.shape) == 2:
        run    = data[0,0].astype(int)
        clone  = data[1,0].astype(int)
        gen    = data[2,0].astype(int)
        values = data[-1]
    elif len(data.shape) == 1:
        run    = data[0].astype(int)
        clone  = data[1].astype(int)
        gen    = data[2].astype(int)
        values = np.array([data[-1]])
    else:
        _logger.error('Cannot load datafile %s: irregular shape: %s' % (path, data.shape))

    proj  = Project()
    proj.add_generation(run, clone, gen, values)
    return proj


def rcg_path_name(name, value):
    return '%s%04d' % (name, value)



def load_project(root, runs=None, clones=None, gens=None, pool=None, coalesce=False, chunksize=None, **initprojkws):
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

    _logger.debug('load_project: initprojkws=%s' % initprojkws)

    pool = ezpool.get_pool(pool)

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


    _logger.info('Searching for data in %s' % root)

    myglob   = os.path.join(root, 'RUN*', 'CLONE*', 'GEN*.dat')
    data_itr = glob.iglob(myglob)
    data_itr = filter_rcg(data_itr, runs, clones, gens)


    ## load the project data
    _logger.info('Loading data')
    myfn = functools.partial(_load_project_processor)
    _logger.debug('load_project: loadfn: %s' % myfn)
    projects = pool.map(myfn, data_itr)

    ## reduce to a single Project instance
    _logger.info('Accumulating project data')
    project = _merge_projects_seq(projects, **initprojkws)

    ## load the number of runs/clones/gens
    _logger.info('Reading the number of runs/clones/gens')
    rcgpath = os.path.join(root, project._rcg_file)
    if os.path.exists(rcgpath):
        with open(rcgpath) as fd:
            rcgs           = fd.readline().strip().split()
            rs,cs,gs       = map(lambda s: int(s.strip()), rcgs)
            project.runs   = rs
            project.clones = cs
            project.gens   = gs
    else:
        _logger.warning('Cannot find number of runs/clones/gens at %s' % rcgpath)

    ## load the metadata
    _logger.info('Loading metadata')
    mdpath = os.path.join(root, project._metadatafile)
    if os.path.exists(mdpath):
        with open(mdpath) as fd:
            for line in itertools.imap(str.strip, fd):
                splitted = line.split('=')

                # the values may have '=' that would have be split
                k, v     = splitted[0], '='.join(splitted[1:])
                k, v     = map(str.strip, (k,v))

                project.add_metadata(k, v)
    else:
        _logger.warning('Cannot find metadata file %s' % mdpath)
    _logger.debug('_load_project: loaded metadata: %s' % project.metadata)

    ## load the description
    descfile = os.path.join(root, project._descfile)
    _logger.info('Loading description')
    if os.path.exists(descfile):
        with open(descfile) as fd:
            desc = fd.read()
            project.set_description(desc)
    else:
        _logger.warning('Cannot find description file %s' % descfile)

    ## load the extra files
    extrasdir = os.path.join(root, project._extradir)
    _logger.info('Loading extra files')
    if os.path.exists(extrasdir):
        files = os.listdir(extrasdir)
        project.set_extrafiles(files)
    else:
        _logger.warning('Cannot find extrafiles directory %s' % extrasdir)



    if coalesce:
        _logger.info('Coalescing project')
        project.coalesce()


    return project


def _process_trajectories_processor(fn, traj):
    _logger.debug('Processing R %d C %d' % (traj.run, traj.clone))
    return fn(traj)



def process_trajectories(proj, fn, pool=None):
    """
    Map a function over the *Trajectories* in a *Project*

    @param proj (Project)
    @param fn (Trajectory -> r: a function accepting a single argument of type *Trajectory*)
    @param pool=DEFAULT_POOL (Pool)

    @return (sequence of r)
    """

    pool = ezpool.get_pool(pool)

    func = functools.partial(_process_trajectories_processor, fn)

    _logger.info('Processing trajectories')
    results = pool.map(func, proj.get_trajectories())


    return results
