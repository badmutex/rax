
import fax

import sys
import os

fax.set_logging_level(fax.logging.INFO)

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
nprocs    = 2
pool      = fax.Pool(nprocs)

fax.setup_pool(nprocs)


print 'Loading project from', inroot
project = fax.load_project(inroot, outputfreq=0.5)
# project.set_outputfrequency(0.5) # ns
lengths = project.get_trajectory_lengths()

for r,c,l in lengths:
    print 'RUN', r, 'CLONE', c, l, 'ns'
