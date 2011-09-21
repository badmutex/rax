
import fax

import sys
import os

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
outroot   = os.path.join(mydirname, 'test-write-project')
nprocs    = 1

# fax.set_logging_level(fax.logging.DEBUG)


project = fax.load_project(inroot, gens=[0])

project.set_description('Hello World')

print 'Metadata:', project.metadata

project.add_metadata('the answer', 42)

project.add_extrafiles(sys.argv[0])


project.write(outroot)
