#!/usr/bin/python3

"""Darknet daemon that fetches non processed images and insert
into the DB. You must have a environment variable $MYSUPERDIR

"""

import argparse
import atexit
import logging
import math
import os
import psycopg2
import shlex
import signal
import socket
import subprocess
import sys
import myutilsgpu0
import rfcn
import time
import threading

###########################################################
def cleanup(pids, lockfile):
    """Clean up child processes and locks

    Args:
    pids(list of int): list of pids
    lockdir(str): path to store lockfiles
    lockfile(str): fullpath of lockfile
    """

    if os.path.isfile(lockfile):
        os.remove(lockfile)

    for pid in pids:
        os.kill(pid, signal.SIGTERM)

###########################################################
def check_params(tmpdir, indir):
    """Check if parameters are according to what expected

    Args:
    indir(str): input dir containing the images
    """

    for d in [tmpdir, indir]:
        if not os.path.isdir(indir):
            err = '  Error: "{}" does not exist.'.format(indir)
            raise NameError(err)

###########################################################
def check_another_instance(lockfile):
    """Check if there is another instance of the daemon running

    Args:
    lockfile(str): fullpath to lockfile
    Returns:
    bool: True if there is another instance runnin
    """

    if os.path.isfile(lockfile):
        errmsg = '''Error creating a lock.\n  Potentially another instance ''' \
                ''' is running. \n  If you are sure no other instance, run \n''' \
                '''  pkill -f "mydaemon.py|mymain.py"; rm {}'''.format(lockfile)
        raise SystemError(errmsg)

    f = open(lockfile, 'w')
    f.write(str(os.getpid()))

###########################################################
def run_rfcn(ids, hashes, rolls, in_dir, out_dir, caffemodel, prototxt, gpuid,
             methodid, dbjson, _buffer, rootdir, logfile, thresh, nms, _delete):
    """Run yolo method

    Args:
    darknetbin(str): path to darknet
    gpuid(int): id of the gpu. -1 if CPU.
    conffile(str): path to the configuration file
    archfile(str): path to the architecture file
    weightsfile(str): path to the weights file
    listfile(str): path to the list of images file
    indir(str): path to the input images
    outdir(str): path to the output dir
    methodid(int): Method id
    createimages(bool): true if want the images to be created

    Returns:
    subprocess.process: the process, containing pid, etc
    """

    cwd = os.getcwd()
    os.chdir(rootdir)

    if gpuid == -1:
        counter = rfcn.run(ids, hashes, rolls, in_dir, out_dir, caffemodel, prototxt, gpuid,
            methodid, dbjson, _buffer, logfile, thresh, nms, _delete)
    else:
        counter = rfcn.run(ids, hashes, rolls, in_dir, out_dir, caffemodel, prototxt, gpuid,
            methodid, dbjson, _buffer, logfile, thresh, nms, _delete)

    os.chdir(cwd)
    #return process

##########################################################
def get_chunks_ids(total, n, maxperchunk=10000):
    """Divide L in n evenly-distributed chunks

    Args:
    L(int): total number
    n(int): number of chunks

    Returns:
    list: each element contain the start of each chunk
    """

    L = total if total < maxperchunk*n else maxperchunk*n
    chunks = [0]

    if L == 1:
        chunks.append(1)
    else:
        chunksz = int(math.ceil(L/n))
        indices = list(range(0, L))
        chunks = list(range(0, L-1, chunksz))
        chunks.append(L-1)
    print("Chunks: {}".format(chunks))
    return chunks

##########################################################
def check_call(args, avoidedword):
    """Check if arguments contain avoided word

    Args:
    args(list): list of cmd line arguments

    Raises:
    Exception
    """

    for arg in args:    # Easilly kill later
        if avoidedword in os.path.basename(arg):
            err = 'Please run mydaemon.py instead.'
            raise Exception(err)

##########################################################
def load_params(jsonfile):
    """Load parameters file in Json format and from args.

    Args:
    jsonfile(str): fullpath to parameters file (json)

    Returns:
    dict: containing all the params

    Raises:
    Exception
    """

    p = myutilsgpu0.load_params(jsonfile)
    check_params(p['tmpdir'], p['indir'])
    return p

##########################################################
def get_run_filepaths(p, rootdir, logsuffix):
    """Return paths to yolo execution

    Args:
    p(dict): from the daemon json file
    logsuffix(str): suffix for the logfilename

    Returns:
    [datafile, archfile, modelfile, logfile]

    Raises:
    """
    datafile  = ''
    archfile  = os.path.join('models/pascal_voc/ResNet-101/rfcn_end2end/',
                             p['arch'])
    modelfile = os.path.join('data/rfcn_models/', p['model'])
    logfile   = os.path.join(p['tmpdir'], logsuffix + p['logfile'])
    return (datafile, archfile, modelfile, logfile)

##########################################################
def main():
    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
    daemonconf = os.path.join(rootdir, 'config/daemongpu0.json')

    p = load_params(daemonconf)
    #check_call(sys.argv, p['daemonfile'])
    mytime = myutilsgpu0.now(True)
    pids = []
    lockfile = os.path.join(p['tmpdir'], p['lockfile'])
    #check_another_instance(lockfile)
    atexit.register(cleanup, pids, lockfile)
    conn = myutilsgpu0.db_connect(p['dbconf'])
    methodid = 7
    ids,relpaths,rolls = myutilsgpu0.db_get_nonprocessed_images(conn,
                                                            methodid,
                                                            p['pklfile'],
                                                            p['ascending'],
                                                            p['szloop'])
    conn.close()
    threads = []

    chunksids = get_chunks_ids(len(ids), int (p['nprocs']))
    #chunksids = get_chunks_ids(len(ids), 2*int (p['nprocs']))
    (_, archfile, modelfile, logfile) = get_run_filepaths(p, rootdir, mytime)

    for i in range(0, len(chunksids)-1):
        _ids = ids[chunksids[i]:chunksids[i+1]]
        _relpaths = relpaths[chunksids[i]:chunksids[i+1]]
        _rolls = rolls[chunksids[i]:chunksids[i+1]]

        modelfile='data/rfcn_models/resnet101_rfcn_final.caffemodel'
        archfile='models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt'

        t = threading.Thread(target=run_rfcn, args=(_ids, _relpaths, _rolls,
                                                    p['indir'], p['outdir'],
                                                    modelfile, archfile,
                                                    int(p['gpuid']), methodid,
                                                    #0 if i%2==0 else 2, methodid,
                                                    p['dbconf'], p['buffer'], rootdir,
                                                    logfile, p['thresh'],
                                                    p['nms'], p['delete']))
        t.daemon = True
        t.start()
        threads.append(t)
        print('\n\n\n\nit:{}\n\n\n\n'.format(i))
        time.sleep(60)

    for tt in threads:
        tt.join()

    #return threads

##########################################################
if __name__ == "__main__":

    while True:
        main()

        print('########################################################## Waiting..,.')
        time.sleep(10)
