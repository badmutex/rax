
import fax

import sys
import os
import functools


def assign(v, boundary=0.3):
    if v <= boundary:
        s = 0
    else:
        s = 1

    return s


mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
nprocs    = 1

fax.set_logging_level(fax.logging.INFO)
fax.setup_pool(nprocs)


project     = fax.load_project(inroot, outputfreq=0.5)
find_states = functools.partial(assign, boundary=0.9)
p2          = project.map(find_states)

for t in p2.get_trajectories():
    print t.run, t.clone, t.get_trajectory_data()
