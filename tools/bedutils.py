#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

""" Utilities for processing BED files."""

import collections
import re
import sys

BedInterval = collections.namedtuple("BedInterval", ["chrom",
                                                     "start",
                                                     "stop"])

def get_intervals_from_bedfile(bedfile):
    """Generate a dict with list BedIntervals indexed by chromosome.

    Args:
        bedfile (str) : Path to bed file

    Returns:
        Dict[chrom] = list of BedIntervals for that chrom
    """
    intervals_by_chrom = collections.defaultdict(list)
    with open(bedfile, "r") as bf:
        for line in bf:
            columns = line.split('\t')
            # BED format from https://genome.ucsc.edu/FAQ/FAQformat#format1
            chrom = columns[0]
            start = int(columns[1])
            stop = int(columns[2])
            intervals_by_chrom[chrom].append(BedInterval(chrom, start, stop))
    return intervals_by_chrom

def intersect_interval(interval_1, interval_2):
    """Return intersection between two BedIntervals.

    Args:
        interval_1 (BedInterval) : First interval
        interval_2 (BedInterval) : Second interval

    Returns:
        A BedInterval with the intersection region, None otherwise
    """
    if (interval_1.chrom != interval_2.chrom):
        return None

    if (interval_1.start <= interval_2.start) and (interval_1.stop > interval_2.start) and (interval_1.stop < interval_2.stop):
        return BedInterval(interval_1.chrom, interval_2.start, interval_1.stop)
    elif (interval_2.start <= interval_1.start) and (interval_2.stop > interval_1.start) and (interval_2.stop < interval_1.stop):
        return BedInterval(interval_1.chrom, interval_1.start, interval_2.stop)
    elif (interval_1.start >= interval_2.start and interval_1.stop <= interval_2.stop):
        return interval_1
    elif (interval_2.start >= interval_1.start and interval_2.stop <= interval_1.stop):
        return interval_2
    else:
        return None

def intersect_intervals(intervals_1, intervals_2):
    """Returns intersection between sets of intervals.

    Args:
        intervals_1 (dict) - Dict of list of BedIntervals indexed by chrom
        intervals_2 (dict) - Dict of list of BedIntervals indexed by chrom

    Returns:
        Dict of list of intersecting BedIntervals indexed by chrom
    """
    intersections = collections.defaultdict(list)
    for chrom in intervals_1.keys():
        if chrom in intervals_2:
            for interval_1 in intervals_1[chrom]:
                for interval_2 in intervals_2[chrom]:
                    intersection = intersect_interval(interval_1, interval_2)
                    if intersection is not None:
                        intersections[chrom].append(intersection)
    return intersections
