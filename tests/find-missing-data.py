
import fax
import numpy as np

import sys
import os

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
outroot   = os.path.join(mydirname, 'test-write-project')
nprocs    = 1

traj = fax.Trajectory(0,1)
traj.set_num_gens(19)

generations = set(np.random.random_integers(19, size=(42,)))
for gen in generations:
    data = np.random.random_sample(4)
    print 'Adding Gen', gen, 'data', data
    traj.add_generation(gen, data)

missing = traj.missing_generations()
print 'Trajectory missing generations', missing
