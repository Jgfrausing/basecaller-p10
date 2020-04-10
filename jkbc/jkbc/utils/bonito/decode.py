"""
Bonito utils
"""
import re
from collections import defaultdict

import parasail

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

def accuracy(ref: str, seq: str, return_alignment=False):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment, align_tuple = align(ref, seq)
    counts = defaultdict(int)
    cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    
    if return_alignment: return accuracy, align_tuple
    else: return accuracy


def align(ref, seq):
    """
    Get (and print) the alignment between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(ref, seq, 8, 4, parasail.dnafull)
    align_tuple = (alignment.traceback.query, alignment.traceback.comp, alignment.traceback.ref)

    return alignment, align_tuple


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return new_cigstr


