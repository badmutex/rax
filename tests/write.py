
import fax

import sys
import os

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
outroot   = os.path.join(mydirname, 'test-write-project')
nprocs    = 1

# fax.set_logging_level(fax.logging.DEBUG)


print 'Loading project from', inroot
project = fax.load_project(inroot, gens=[0])
project.set_description('Hello World')

print 'Description:',  project.description
print 'Extra files:', project.extrafiles


print 'Writing project to', outroot
project.write(outroot)
