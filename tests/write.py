
import fax
import ezlog

import sys
import os

mydirname = os.path.dirname(sys.argv[0])
inroot    = os.path.join(mydirname, 'relaxation-data')
outroot   = os.path.join(mydirname, 'test-write-project')
nprocs    = 1

log = ezlog.setup(__name__)
ezlog.set_level(ezlog.INFO, name=fax.__name__)
ezlog.set_level(ezlog.DEBUG, name=__name__)


project = fax.load_project(inroot, gens=[0])

project.set_description('Hello World')

log.info('Metadata:', project.metadata)

project.add_metadata('the answer', 42)

project.add_extrafiles(sys.argv[0])


log.debug('writing project %s' % outroot)
project.write(outroot)
