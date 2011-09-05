#!/usr/bin/env python

import fax

import numpy as np
import scipy.sparse as sps
import scipy.linalg as linalg
import scipy.io as sio

import glob
import os
import sys
import functools


def choose_state(v, ranges):
    for s, (min_exc, max_inc) in enumerate(ranges):
        if min_exc < v <= max_inc:
            return s

    raise ValueError, 'Unknown state for value %s given ranges %s' % (v, ranges)


def process_trajectory(ranges, traj):
    run   = traj.run
    clone = traj.clone
    data  = traj.get_trajectory_data()

    N = sps.lil_matrix((len(ranges), len(ranges)), dtype=np.uint)
    R = len(ranges)

    T = dict()
    for r in xrange(R):
        T[r] = list()
    life = 0

    for i in xrange(data.size - 1):
        j = i + 1
        v1, v2 = data[i], data[j]

        s, t = map(lambda v: choose_state(v, ranges), (v1,v2))
        N[s, t] += 1

        if s == t:
            life += 1
        elif life > 0:
            T[s].append(life)
            life = 0

    T[s].append(life)
    return T, N
        
    


if __name__ == '__main__':

    fahroot    = sys.argv[1]
    nprocs     = int(sys.argv[2])
    outfreq_ns = float(sys.argv[3])
    mtxfile    = sys.argv[4]
    groups     = sys.argv[5:]
    ranges     = map(lambda g: map(float, g.split(':')), groups)


    # transition count matrix
    N = sps.lil_matrix((len(ranges),len(ranges)), dtype=np.uint)

    # average lifetimes
    T = np.zeros(len(ranges))

    print 'Searching for datafiles'
    datafiles = glob.iglob(os.path.join(fahroot, 'RUN*', 'CLONE*', 'GEN*.dat'))
    # print 'Got', len(datafiles), 'files'


    # read the trajectories in and determine the trajectory lifetimes and transition counts
    processor = functools.partial(process_trajectory, ranges)
    pool      = fax.Pool(nprocs=nprocs)
    project   = fax.load_project(fahroot, pool=pool, coalesce=True)
    results   = fax.process_trajectories(project, processor, pool=pool, nprocs=nprocs)

    # compute lifetimes and update transition counts matrix
    Ts = np.zeros(len(ranges))
    for ts, M in results:
        N = N + M
        for s, lifes in ts.iteritems():
            T[s] += sum(lifes)
            Ts[s] += len(lifes)
    T = T / Ts

    print 'N'
    print N
    print 'T'
    print T

    # symetrize the transition count matrix
    N = N + N.transpose() - np.diag(N.diagonal())

    print 'N_sym'
    print N

    print 'eigenvalues'
    print linalg.eigvalsh(N)


    K = sps.dok_matrix((len(ranges),len(ranges)))
    N = sps.dok_matrix(N)

    for i, j in N.iterkeys():
        num = N[i,j]
        denom = T[i] * N[i,:].sum()
        k = num / denom

        K[i,j] = outfreq_ns / k

    print 'K:'
    print K

    print 'Exchanges'
    for i,j in sorted(K .iterkeys()):
        mi, Mi = ranges[i]
        mj, Mj = ranges[j]

        print '%5s (%4s,%4s] %5s %5s (%4s,%4s]: %f' % (i, mi, Mi, '->', j, mj, Mj, K[i,j])


    print 'Saving', mtxfile
    sio.mmwrite(mtxfile, K)

    pool.finish()

