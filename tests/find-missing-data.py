
import fax
import numpy as np

import sys
import os

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')

traj = fax.Trajectory(0,1)
traj.set_num_gens(19)

generations = set(np.random.random_integers(19, size=(42,)))
for gen in generations:
    data = np.random.random_sample(4)
    print 'Adding Gen', gen, 'data', data
    traj.add_generation(gen, data)

missing = traj.missing_generations()
print 'Trajectory missing generations', missing


project        = fax.load_project(inroot)

# project.runs   = 3
# project.clones = 2
# project.gens   = 30

print 'Project missing:', list(project.missing_data())
