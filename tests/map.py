
import fax
import ezlog

import sys
import os
import functools

log = ezlog.setup()


log.info('defining assign')
def assign(v, boundary=0.3):
    if v <= boundary:
        s = 0
    else:
        s = 1

    return s


log.info('Setting up vars')
mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
nprocs    = 2

ezlog.set_level(ezlog.INFO)
fax.setup_pool(nprocs)


log.info('Loading project')
project     = fax.load_project(inroot, outputfreq=0.5)
find_states = functools.partial(assign, boundary=0.9)
p2          = project.map(find_states)

log.info('Print results')
for t in p2.get_trajectories():
    print t.run, t.clone, t.get_trajectory_data()
