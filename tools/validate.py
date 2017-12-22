#!/usr/bin/python3
"""Validate a method, based on DB stored data

Remaining
"""

import inspect
import utils
import pprint

class Bbox:
    def __init__(self, x=-1, y=-1, w=-1, h=-1, clsid=-1, prob=-1.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.clsid = clsid
        self.prob = prob

##########################################################
def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 < r2 else r2
    return right - left


##########################################################
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)

    if(w < 0 or h < 0): return 0

    area = w*h
    return area

##########################################################
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u

##########################################################
def box_iou(a, b):
    return float(box_intersection(a, b)) / box_union(a, b)

##########################################################
def compute_best_iou(gndtruthbbox, bboxes, epsilon=0.01):
    """short-description

    Args:
    gndtruthbbox(hash): hash containing keys ['clsid', 'x', 'y', 'w', 'h', 'prob'])
    bboxes(list of bboxes): list of hashes with the same strcture as above

    Returns:
    (float, int): best iou and index of the bbox with best match. (0,-1) if not found

    """

    g = gndtruthbbox
    numsameclass = 0
    best_iou = 0 
    best_idx = -1
    i = 0

    for bbox in bboxes:
        if bbox.clsid != gndtruthbbox.clsid:
            i += 1
            continue

        iou = box_iou(bbox, g)

        if iou > epsilon and iou > best_iou:
            best_iou = iou
            best_idx = i
        i += 1

    return best_iou, best_idx

##########################################################
def filter_bboxes_by_prob(proposals, probthresh):
    if probthresh == 0: return proposals

    new_proposals = []

    for p in proposals:
        if p.prob >= probthresh:
            new_proposals.append(p)
    return new_proposals

##########################################################
def filter_bboxes_by_cls(bboxes, clsid):
    newbboxes = []

    for b in bboxes:
        if b.clsid == clsid:
            newbboxes.append(b)
    return newbboxes

##########################################################
def compute_image_ious(gndtruths, proposals, cls, iouthresh=0.5):
    """ Compute method recall, computing for each image, and for each
    ground truth box the best match

    Args:
    groundtruths(list): set of ground truth bboxes
    proposals(list): set of bboxes detected by the method
    iouthresh(int): threshold of IoU

    Returns:
    ret

    """

    correct = 0
    acciou = 0
    indices = []

    i = 0
    for g in gndtruths:
        iou, idx = compute_best_iou(g, proposals)
        if iou >= iouthresh:
            correct += 1
            acciou += iou
        indices.append(idx)
    return correct, acciou, indices

##########################################################
def get_imageids_from_method(conn, gndtruthid):
    cur = conn.cursor()
    query = ''' SELECT imageid FROM ImageMethod WHERE methodid={} '''\
    ''' ORDER BY imageid;'''. format(gndtruthid)
    cur.execute(query)
    raw = cur.fetchall()
    ret = [x[0] for x in raw]
    return ret

##########################################################
def get_classes_from_method(conn, methodid):
    cur = conn.cursor()
    #query = ''' SELECT classid FROM MethodClass WHERE methodid={};'''. \ #TODO:FIX IT
    query = ''' SELECT classid FROM MethodClass WHERE methodid=1;'''. \
            format(methodid)
    cur.execute(query)
    raw = cur.fetchall()
    ret = [x[0] for x in raw]
    return ret

##########################################################
def create_dict_str(_dict, sortedkeys, header):
    line = header
    for k in sortedkeys:
        line += ','  +   str(_dict[k])
    line += '\n'
    return line

##########################################################
def get_bboxes_from_method(conn, methodid, imageid):
    cur = conn.cursor()
    query = ''' SELECT x_min, y_min, x_max, y_max, classid, prob FROM tek.Bbox ''' \
    ''' WHERE methodid={} AND imageid={};'''.format(methodid, imageid)
    cur.execute(query)
    raw = cur.fetchall()
    ret = []
    for r in raw:
        ret.append(Bbox(x=(r[2]+r[0])/2, w=r[2]-r[0], y=(r[3]+r[1])/2,
            h=r[3]-r[1], clsid=r[4], prob=r[5]))
    return ret

##########################################################
def get_formatted_output(nids, accgndtruths, accproposals,
        acccorrects, acciou, precision, recall):

    sortedkeys = sorted(accgndtruths.keys())
    retstr = ''
    for k in sortedkeys: retstr += ',' + str(k)
    retstr += '\n'

    retstr += create_dict_str(accgndtruths, sortedkeys,
            'Number of ground-truths')
    retstr += create_dict_str(accproposals, sortedkeys,
            'Number of bbox proposals')
    retstr += create_dict_str(acccorrects, sortedkeys,
            'Number of correct proposals')
    retstr += create_dict_str(acciou, sortedkeys,
            'Accumulated IoU')
    retstr += create_dict_str(precision, sortedkeys,
            'Precision')
    retstr += create_dict_str(recall, sortedkeys,
            'Recall')

    retstr += '\n'
    retstr += 'Num validations, {}\n'.format(str(nids))
    #retstr += 'Recall, {}'.format()
    return retstr

##########################################################
def compute_precision(acccorrects, accproposals):
    precision = {}
    for k in acccorrects.keys():
        if accproposals[k] != 0:
            precision[k] = float(acccorrects[k]) / accproposals[k]
        else:
            precision[k] = -1
    return precision

##########################################################
def compute_recall(acccorrects, accgndtruths):
    recall = {}
    for k in acccorrects.keys():
        if accproposals[k] != 0:
            recall[k] = float(acccorrects[k]) / accgndtruths[k]
        else:
            recall[k] = -1
    return recall

##########################################################
def validate_method(conn, methid, gndtruthid, outcsv, iouthresh=0.5,
        probthresh=0.0):
    """Validate method of method id methid against the results from the
    method gndtruthid

    Args:
    conn(psycopg2.connection): open connection
    methid(int): id of the method being validated
    grndtruthid(int): id of the method of manually labelling
    classid(int): class id
    iouthresh(float): threshold of IoU
    probthresh(float): threshold probability of each candidate
    """

    ids = get_imageids_from_method(conn, gndtruthid)
    classes = get_classes_from_method(conn, gndtruthid)

    accgndtruths = dict.fromkeys(classes, 0)
    accproposals = dict.fromkeys(classes, 0)
    acccorrects = dict.fromkeys(classes, 0)
    acciou = dict.fromkeys(classes, 0)
    accind = dict.fromkeys(classes, [])

    print('Validating ids:')
    for imageid in ids:
        print('{}'.format(imageid))
        gndtruths = get_bboxes_from_method(conn, gndtruthid, imageid)
        proposals = get_bboxes_from_method(conn, methid, imageid)

        for cls in classes:
            fproposals = filter_bboxes_by_prob(proposals, probthresh)
            fproposals = filter_bboxes_by_cls(fproposals, cls)
            fgndtruths = filter_bboxes_by_cls(gndtruths, cls)
            cor, iou, ind = compute_image_ious(fgndtruths, fproposals,
                    cls, iouthresh)

            accgndtruths[cls] += len(fgndtruths)
            accproposals[cls] += len(fproposals)
            acccorrects[cls] += cor
            acciou[cls] += iou
            accind[cls] += ind

    precision = compute_precision(acccorrects, accproposals)
    recall = compute_precision(acccorrects, accgndtruths)
    output = get_formatted_output(len(ids), accgndtruths, accproposals,
            acccorrects, acciou, precision, recall)
    with open(outcsv, 'w') as fh:
        fh.write(output)
    return output

##########################################################
def main():
    dbconfig = 'config/db.json'
    methid = 4
    gndtruthid = 2
    outcsv = '/tmp/validation'+str(methid)+'.csv'
    conn = utils.db_connect(dbconfig)

    validate_method(conn, methid, gndtruthid, outcsv)

if __name__ == "__main__":
    main()

