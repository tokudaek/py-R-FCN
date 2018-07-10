#!/usr/bin/env python3

from datetime import datetime
import json
import os
import hashlib
import random
import logging
import subprocess
import shutil
import dateutil.parser
import xml.etree.ElementTree as ET
import psycopg2
import pickle

##########################################################
def db_connect(json_file, autocommit=True):
    """Connect to db

    Args:
    json_file: fullpath to the credentials file to DB

    Returns:
    psycopg2.connection

    Raises:
    """
    dbparams = load_params(json_file)
    conn = psycopg2.connect('''dbname={} port={} host={} user={} ''' \
            ''' password={}'''.format(dbparams['dbname'], dbparams['port'],
                dbparams['host'], dbparams['user'], dbparams['password']))
    conn.autocommit = autocommit

    return conn

##########################################################
def db_insert_execution(conn, methodid, hostname, starttime, descr=''):
    """Insert entry into Executionmethod

    Args:
    conn(psycopg2.connection): open connection
    methoid(id): id of the method executed
    hostname(str): hostname of the execution
    starttime(str): time in isoformat

    Returns:
    bool: True if it was successful

    """

    cur = conn.cursor()


    query = '''INSERT INTO Execution ''' \
    ''' (methodid, hostname, starttime, descr) ''' \
    ''' VALUES ('{}', '{}', '{}', '{}'); SELECT currval( ''' \
    ''' 'execution_id_seq');'''.format(
            methodid, hostname, starttime, descr)

    cur.execute(query)
    return cur.fetchone()[0]

###########################################################
def db_insert_method(conn,  arch, weights, thresh, nms, commitid):
    """ Insert the method

    Args:
    conn(psycopg2.connection): open connection to the PG database
    framework(str): name of the framework (yolo, caffe, rcnn, ...)
    arch(str): architecture of the network (vgg16, alexnet, ...)
    weights(str): name of the weights file
    commitid(str): commit id of the method
    thresh(float): threshold of detections
    nms(float): non-maximal suppression parameter

    Returns:
    int: Method id
    """
    cur = conn.cursor()

    query = ''' INSERT INTO Method (framework, architecture, model, ''' \
            ''' thresh, nms, commitid) VALUES ('{}', '{}', '{}', ''' \
            ''' '{}','{}','{}'); SELECT currval('method_id_seq1') ''' \
            ''';'''.format('yolo', arch, weights, thresh, nms, commitid)
    cur.execute(query)
    return (cur.fetchone()[0])

###########################################################
def db_get_methodid(conn, arch, weights, commitid, thresh=-1.0,
        nms=-1.0):
    """ Get the method id with the parameters received. If non-existant,
    a new entry is created. The execution id is returned.

    Args:
    conn(psycopg2.connection): open connection to the PG database
    framework(str): name of the framework (yolo, caffe, rcnn, ...)
    arch(str): architecture of the network (vgg16, alexnet, ...)
    weights(str): name of the weights file
    commitid(str): id of the commit

    Returns:
    int: Method id
    """
    cur = conn.cursor()

    query = ''' SELECT id from Method WHERE lower(framework)='{}' AND ''' \
            ''' lower(architecture)='{}' AND lower(model)='{}' AND ''' \
            ''' lower(commitid)='{}'; '''.format('yolo',
                    arch.lower(), weights.lower(), commitid)
    cur.execute(query)
    result = cur.fetchone()

    if result:
        methodid = result[0]
    else:
        methodid = db_insert_method(conn, arch, weights, thresh,
                nms, commitid)

    return methodid

###########################################################
def db_get_nonprocessed_images(conn, methodid, pklfile=None,
                               ascend=True, limit=100000, sorted=False):
    """Get from table Image the ids of the non-corrupted images that were not
    processed yet by the method id

    Args:
    conn(psycopg2.connection): open connection to the PG database
    methodid(int): method id

    Returns:
    list: ids of non-processed images

    """

    if False and pklfile and os.path.exists(pklfile):
        with open(pklfile, 'rb') as fh:
            rows = pickle.load(fh)
            ids = [x[0] for x in rows]
            relpaths = [x[1] for x in rows]
            rolls = [x[2] for x in rows]
            logging.debug('Read {} non-processed images from pickle file.'.format(len(ids)))
        return ids, relpaths, rolls

    cur = conn.cursor()

    print('Getting non-processed images...')


    #INNER JOIN manhattanids ON Images.id=manhattanids.imageid
    #INNER JOIN localimages ON Images.id=localimages.imageid
    #INNER JOIN imagesodd ON Images.id=imagesodd.id
    #INNER JOIN westvillageids ON Images.id=westvillageids.id
    #INNER JOIN unionflatids ON Images.id=unionflatids.id
    #INNER JOIN imagesodd ON Images.id=imagesodd.id
    query = '''
    SELECT Images.id, Images.image, Images.roll from Images
    INNER JOIN localimages ON Images.id=localimages.imageid
    INNER JOIN tek.nonprocessed7 on Images.id=nonprocessed7.id
    ORDER BY Images.id {} limit {};
    '''.format('ASC' if ascend else  'DESC', limit)
    #'''.format(methodid, 'ASC' if ascend else  'DESC', limit)

    #print('\n\n\n')
    #print('\n\n\n')
    #print(query)
    #print('\n\n\n')
    #print('\n\n\n')
    cur.execute(query)
    rows = [x for x in cur.fetchall()]

    if pklfile and not os.path.exists(pklfile):
        with open(pklfile, 'wb') as fh:
            pickle.dump(rows, fh)

    ids = [x[0] for x in rows]
    relpaths = [x[1] for x in rows]
    rolls = [x[2] for x in rows]
    #print('\n\n\n')
    #print('\n\n\n')
    print('Got {} non-processed images.'.format(len(ids)))
    #print('\n\n\n')
    #print('\n\n\n')
    return ids, relpaths, rolls

##########################################################
def dump_vocxmls_into_db(dbconf, xmlsdir, methid):
    """Read a dir of xmls and dump into the db, using the
    credentials in conffile

    Args:
    conffile(str): file to the dbconf file
    xmlsdir(str): full path to the xmls directory
    """

    dbparams = load_params(dbconf)
    conn = psycopg2.connect('''dbname={} host={} user={} password={}'''.format(
        dbparams['dbname'], dbparams['host'], dbparams['user'],
        dbparams['password']))
    conn.autocommit = True
    structs = read_dir_xmls(xmlsdir)
    print('{} xmls sucessfully read.'.format(len(structs)))
    (nbboxes, nimageexecutions) = db_input_annotations(conn, structs, methid)
    print('{} bboxes sucessfully inserted into db.'.format(str(nbboxes)))
    print('{} imagemethods sucessfully inserted into db.'.format(str(nimageexecutions)))

##########################################################
def generate_csvs_from_xml(xmlsdir):
    """Input structures into db. It expects and input a list
    of 3-uples (width, height, [hash]), where [hash] is a list
    of hash with keys ['name', 'xmin', 'ymin', 'xmax', 'ymax']

    Args:
    filesbboxes(list): list of 3-uples (width, height, [hash]),
    where [hash] is a list of hashes, each with keys
    ['name', 'xmin', 'ymin', 'xmax', 'ymax']

    Return:
    nbboxes, nimages
    """
    filesbboxes = read_dir_xmls(xmlsdir)

    nbboxes = 0
    nimagemethods = 0

    for f in filesbboxes:
        id = f['id']

        for b in f['bboxes']:
            r = '{},{},{},{},{}'. \
                format(str(id).replace('.jpg', ''), int(b['xmin']), int(b['ymin']),
                       int(b['xmax']), int(b['ymax']))
            print(r)

def db_input_annotations(conn, filesbboxes, methodid):
    """Input structures into db. It expects and input a list
    of 3-uples (width, height, [hash]), where [hash] is a list
    of hash with keys ['name', 'xmin', 'ymin', 'xmax', 'ymax']

    Args:
    filesbboxes(list): list of 3-uples (width, height, [hash]),
    where [hash] is a list of hashes, each with keys
    ['name', 'xmin', 'ymin', 'xmax', 'ymax']

    Return:
    nbboxes, nimages
    """
    nbboxes = 0
    nimagemethods = 0

    cur = conn.cursor()

    for f in filesbboxes:
        id = f['id']

        for b in f['bboxes']:
            query =''' INSERT INTO Bbox (imageid, classid, x_min, y_min, x_max,
            y_max, prob, methodid) SELECT {}, Class.id, {}, {}, {}, {}, {},
            {} FROM Class WHERE Class.name='{}' ;
            '''.format( str(id), int(b['xmin']), int(b['ymin']),
                    int(b['xmax']), int(b['ymax']), 1, methodid, b['name'])
            try:
                cur.execute(query)
                conn.commit()
                nbboxes += 1
            except psycopg2.Error as e:
                print(e)
                break

        query =''' INSERT INTO ImageMethod (imageid, methodid)
         VALUES ({}, {}); '''.format(str(id), methodid)
        try:
            cur.execute(query)
            conn.commit()
            nimagemethods += 1
        except psycopg2.Error as e:
            print(e)
            continue
    return (nbboxes, nimagemethods)

##########################################################
def read_dir_xmls(xmlsdir):
    """Read all xmls in dir

    Args:
    xmlsdir(str): path to the directory containing xmls

    Returns:
    list of 3-uples: each 3-uple is given by read_vocxml,
    containing (width, height, has) where hash contains keys
    ['name', 'xmin', 'ymin', 'xmax', 'ymax']

    """
    files = []
    for xml in os.listdir(xmlsdir):
        if not xml.endswith('.xml'): continue
        fullfile = os.path.join(xmlsdir, xml)
        files.append(read_vocxml(fullfile))
    return files

##########################################################
def read_vocxml(xmlfile):
    """Read xml file in VOC format

    Args:
    xmlfile(str): xml input file in voc format

    Returns:
    dict: with keys ['id', 'width', 'height', 'bboxes'], and bboxes
    contains a list of hashes, each hash containing keys
    ['cls', 'xmin', 'ymin', 'xmax', 'ymax']
    """

    tree=ET.parse(xmlfile)
    root = tree.getroot()
    id = root.find('filename').text
    size = root.find('size')
    w = size.find('width').text
    h = size.find('height').text

    f = {}
    f['id'] = id
    f['width'] = int(w)
    f['height'] = int(h)

    bboxes = []
    for obj in root.iter('object'):
        aux = {}
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(difficult) == 1:
            continue

        xmlbox = obj.find('bndbox')
        aux['name'] = cls
        aux['xmin'] = int(xmlbox.find('xmin').text)
        aux['ymin'] = int(xmlbox.find('ymin').text)
        aux['xmax'] = int(xmlbox.find('xmax').text)
        aux['ymax'] = int(xmlbox.find('ymax').text)
        bboxes.append(aux)
    f['bboxes'] = bboxes
    return f

##########################################################
def get_corrupted_jpgs_in_dir(dirname):
    """Check integrity of all jpg files in dirname

    Args:
    dirname(str): path to jpg image

    Returns:
    list: list of corrupted jpgs 
    """

    ret = []
    for filename in os.listdir(dirname):
        if filename.lower().endswith('.jpg'):
            if not is_jpg_ok(os.path.join(dirname, filename)):
                print(filename)
                ret.append(filename)

##########################################################
def is_jpg_ok(fullfile):
    """Check if file is corrupted.

    Args:
    fullfile(str): fulle filename

    Returns:
    bool: False if file is corrupted.
    """

    process = subprocess.Popen(['identify', '-verbose', fullfile],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    err = stderr.decode('ascii')

    if err: return False
    return True

##########################################################
def get_random_partitions_from_dir(mydir, szs):
    """Get arbitrary number of partitions of sizes szs

    Args:
        dir(str) : directory to get the partitions
        szs(int[]): sizes of the partitions. If the last partition is -1,
        the size is thre remaining size

    Returns:
        `list` of `list`: returns a list of lists

    Example call:
        >> get_random_partitions('/home/foo', [3, 4, 2])

    """
    files = os.listdir(mydir)
    random.shuffle(files)
    partitions = []

    idx = 0

    for sz in szs:
        if sz == -1: idx_end = None
        else: idx_end = idx + sz

        #print (files[idx:idx+sz])
        partitions.append(files[idx:idx_end])
        idx = idx_end

    return partitions

##########################################################
def now(includetime=True):
    """Get current time in ISO 8601 format in string format

    Args:
        includetime(bool): includes current hour and minutes. Default is True.

    Returns:
        str: return the current date (and time) in string
    """
    if includetime:
        return (datetime.now().strftime("%Y-%m-%dT%H:%M"))
    else:
        return (datetime.now().strftime("%Y-%m-%d"))

##########################################################
def iso_to_datetime(timestr):
    """Get time in ISO 8601 format in string format

    Args:
        timestr(str): time in ISO8601 format

    Returns:
        datetime: return a datetime object represented by timestr in ISO format
    """
    return (dateutil.parser.parse(timestr))

##########################################################
def dt2str_underscores(dt):
    """Formats non-null datetime, containing Y,M,D,h,m,s to the
    underscores-format

    Args:
        dt(str): datetime object

    Returns:
        str: time in underscores format
    """

    return dt.strftime("%Y-%m-%d_%H-%M-%S")

##########################################################
def undersc_str2dt(undersc):
    """Converts the format with underscores to a datetime instance

    Args:
        undersc(str): time in underscores-format

    Returns:
        `datetime`: datetime instance
    """

    (mydate, mytime) = undersc.split("_")

    ymd = mydate.split("-")
    ymd = [int(i) for i in ymd]
    hms = mytime.split("-")
    hms = [int(i) for i in hms]

    if len(hms) == 3:
        return datetime(ymd[0], ymd[1], ymd[2], hms[0], hms[1], hms[2])
    else:
        return datetime(ymd[0], ymd[1], ymd[2], hms[0], hms[1], hms[2],
                                 hms[3])

##########################################################
def dt2str_iso(dt, show_ms=False):
    """Formats datetime to the iso format

    Args:
        dt(str): datetime object
        show_ms(bool): supress milisseconds

    Returns:
        str: time in iso format
    """

    if show_ms:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


##########################################################
def load_params(json_file):
    """Load file in format json of db

    Args:
        json_file(str): path to .json file

    Returns:
        str,str,str,str: containing respectively host, dbname, user, password

    """
    with open(json_file, 'r') as fh:
        params = json.load(fh)

    return params

##########################################################
def json_to_sql(j, sep='_', lower=False):
    """Tries to infer a SQL create statement based on the JSON file.
    Nested levels are inserted in the same level with underscores in the name

    Args:
        j(dict): dict potentially containing nested levels
        sep(str): separator. Defaults to '_'

    Returns:
        str: SQL create table statement

    """

    query = ''
    lindict = linearize_dict(j, sep, lower)

    for k in lindict.keys():
        query += k
        if isinstance(lindict[k], int):
            query += ' INTEGER, '
        elif isinstance(lindict[k], float):
            query += ' REAL, '
        elif isinstance(lindict[k], bool):
            query += ' BOOLEAN, '
        else:
            query += ' TEXT, '

    query = '''CREATE TABLE XXXX ({})'''.format(query[:-2])
    return query

##########################################################
def linearize_dict(indict, sep='_', lower=False):
    """Linearize levels of an input dict, using separator

    Args:
        indict(`dict`): input dict with nested levels
        sep(str): separator
        lower(bool): lower all itens

    Returns:
        `dict`: output dict with just one level

    """

    def linearize_level(level, sep, lower, suffix=''):

        outdict = {}
        for i in level.keys():
            new_suffix = suffix + i.lower() if lower else suffix + i

            if isinstance(level[i], dict):
                part = linearize_level(level[i], sep, lower, new_suffix + sep)
                outdict.update(part)
            else:
                outdict[new_suffix] = level[i]

        return outdict

    return linearize_level(indict, sep, lower)

##########################################################
def parse_m3u8(m3u8str):
    """Parse a m3u8 string and covnert to a dict

    Args:
        m3u8str(str): m3u in str format

    Returns:
        `dict`: returns a parsed dict
        

    """
    pattern_stream = '#EXT-X-STREAM-INF'
    d = m3u8str.split('\n')

    if not '#EXTM3U' in d[0]:
        print('Not a m3u8 file.')
        return

    d = d[1:]

    parsed = {}

    for i in range(len(d)):
        key_value = d[i].split(':')
        if len(key_value) < 2:
            if key_value[0]:
                parsed['uri'] = key_value[0]
        else:
            parsed[key_value[0]] = key_value[1].replace('"','')

    parsed_orig = dict(parsed)
    last_key = ''
    for k in parsed_orig.keys():
        if '=' in parsed_orig[k]:
            parsed[k] = {}
            attrs = parsed_orig[k].split(',')

            for a in range(len(attrs)):
                k_v = attrs[a].split('=')
                if len(k_v) == 1:
                    #parsed[k][last_key] += ',' + k_v[0]
                    parsed[last_key] += ',' + k_v[0]
                else:
                    #parsed[k][k_v[0]] = k_v[1]
                    parsed[k_v[0]] = k_v[1]
                    last_key = k_v[0]
    return parsed

##########################################################
def extract_frames(video, out_dir, fps=1, ext='.jpg', logfile='/tmp/extract_frames.log'):
    """Extract frames utilizing ffmpeg

    Args:
        video(str): input video file
        out_dir(str): output dir
        out_prefix(str): prefix for all out files
        fps(int): number of frames to be extracted per second
        ext(str): extension of the resulting frames
        logfile(str): log file

    Raises:
        FileNotFoundError: when file is not found
        CalledProcessError: when  there is a problem with the process execution
    """

    try:
        log = open(logfile, 'w')
    except Exception as e:
        raise

    logging.basicConfig(filename=logfile)

    prefix = video.split('/')[-2]
    prefix += '_' + os.path.split(video)[1][:-4]

    try:
        subprocess.check_call(["ffmpeg",
                               "-i", 
                               video,
                               "-r", str(fps),
                               os.path.join(out_dir) + '/' + prefix + '_%4d'+ ext],
                              stdout=log, stderr=log)
    except FileNotFoundError as e:
        logging.exception(e)
        raise exceptions.command_line
    except subprocess.CalledProcessError as e:
        logging.exception(e)
        raise exceptions.url_invalid
    finally:
        log.close()

##########################################################
def split_into_folders(abs_dirname, N=100):
    """Create subdirectories and move files into them.

    Args:
        abs_dirname(str): folder containing multiple files
        N(int): maximum number of files per folder

    Raises:
        FileNotFoundError: when file does not exist
    """

    if not os.path.exists(abs_dirname):
        raise FileNotFoundError('Directory does not exist ({0}).'.format(src_dir))

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    curr_subdir = None

    for f in sorted(files):
        # create new subdir if necessary
        if i % N == 0:
            subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(int(i/N) + 1))
            os.mkdir(subdir_name)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1

##########################################################
def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    """Snippet obtained from:
    http://stackoverflow.com/questions/1158076/implement-touch-using-python"""

    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)
##########################################################
def get_str_md5(string):
    """Get md5 from the string

    Args:
    string(str): input string

    Returns:
    str: hex md5 hash
    """
    x = hashlib.md5(string.encode())
    return x.hexdigest()


