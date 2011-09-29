#!/usr/bin/env python

import rax

import numpy as np
import scipy.sparse as sps
import scipy.linalg as linalg
import scipy.io as sio

import glob
import os
import sys
import functools
import itertools
import logging


def choose_state(ranges, v):
    for s, (min_exc, max_inc) in enumerate(ranges):
        if min_exc < v <= max_inc:
            return s

    raise ValueError, 'Unknown state for value %s given ranges %s' % (v, ranges)


def process_trajectory(ranges, traj):
    data = traj.get_trajectory_data()
    assignments = itertools.imap(functools.partial(choose_state, ranges), data)
    N = np.zeros((len(ranges),len(ranges)), dtype=int)

    assignments = np.array(map(functools.partial(choose_state, ranges), data), dtype=int)

    T = dict()
    for r in xrange(len(ranges)):
        T[r] = list()
    lifetime = 1

    for i in xrange(len(data) - 1):
        j = i + 1
        s,t = assignments[i], assignments[j]
        N[s,t] += 1

        if s == t:
            lifetime += 1
        else:
            T[s].append(lifetime)
            lifetime = 1

    return traj.run, traj.clone, T, N

def assignments(ranges, traj):
    run   = traj.run
    clone = traj.clone
    data  = traj.get_trajectory_data()

    return map(functools.partial(choose_state, ranges), data)


def mixs2d(M):
    a,b = M.shape
    for i in xrange(a):
        for j in xrange(b):
            yield i,j



if __name__ == '__main__':

    fahroot    = sys.argv[1]
    nprocs     = int(sys.argv[2])
    outfreq_ns = float(sys.argv[3])
    outfile    = sys.argv[4]
    groups     = sys.argv[5:]
    ranges     = map(lambda g: map(float, g.split(':')), groups)


    rax.set_logging_level(logging.INFO)


    # transition count matrix
    N = np.zeros((len(ranges),len(ranges)), dtype=int)

    # average lifetimes
    T = np.zeros(len(ranges))

    runs = None
    clones = None

    # read the trajectories in to determine the trajectory lifetimes and transition counts
    processor = functools.partial(process_trajectory, ranges)
    pool      = rax.Pool(nprocs)
    project   = rax.load_project(fahroot, runs=runs, clones=clones, pool=pool, coalesce=True)
    results   = rax.process_trajectories(project, processor, pool=pool)


    # # save the assignments
    # assignments = rax.process_trajectories(project, functools.partial(assignments, ranges), pool=pool)
    # np.savetxt(outfile, assignments[0], fmt='%d')
    # project.savetxt('test.dat', run, clone, fmt='%f')
    # sys.exit(1)

    # compute lifetimes and update transition counts matrix

    Ts = np.zeros(len(ranges))
    for run, clone, ts, M in results:
        N = N + M
        for s, lifes in ts.iteritems():
            lifes = filter(lambda x: x > 1, lifes)
            T[s] += sum(lifes)
            Ts[s] += len(lifes)

    for i in xrange(len(Ts)):
        T[i] = T[i] / Ts[i]

    print 'N'
    print N
    print 'T'
    print T

    # symetrize the transition count matrix
    # the diagonal should be 0
    N = N + N.T - 2*np.diag(N.diagonal())

    print 'N_sym'
    print N

    # compute the rates
    K = np.zeros((len(ranges),len(ranges)))

    for i, j in mixs2d(N):
        k = 0
        if not i == j:
            num = N[i,j]
            denom = T[i] * N[i, :].sum()
            k = num / denom
        else:
           k = -1/T[i]
        K[i,j] = k
    

    print 'K:'
    print K

    print 'eigenvalues(K)'
    eigvals = linalg.eigvals(K)
    print eigvals

    R = outfreq_ns / K

    print 'Exchanges'
    with open(outfile, 'w') as fd:
        fd.write('Eigenvalues = %s\n' % ', '.join(map(str, eigvals)))
        for i,j in mixs2d(R):
            if i <= j: continue

            fd.write('Exchange = %s %s %s ns\n' % (i, j, R[i,j]))

            mi, Mi = ranges[i]
            mj, Mj = ranges[j]

            print '%5s (%4s,%4s] %5s %5s (%4s,%4s]: %f' % (i, mi, Mi, '->', j, mj, Mj, R[i,j] / 10.**3)



    pool.finish()

