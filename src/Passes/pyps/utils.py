# A small toolbox for various utility functions
import re
import os
import fileinput
import sys

guard_begin = "PIPS include guard begin:"
guard_begin_re = re.compile(r"^/\* %s (.+) \*/$" % guard_begin)
guard_end = "PIPS include guard end:"
include_re = re.compile(r"^\s*#\s*include\s*(\S+)\s*.*$")

def mkguard(guard, line):
    return "/* %s %s */\n" % (guard, line.rstrip("\r\n"))

def guardincludes(fname):
    """ Adds guards around includes."""
    for l in fileinput.FileInput([fname], inplace = True):
        is_include = include_re.match(l)
        if is_include:
            print mkguard(guard_begin, l),
        print l,
        if is_include:
            print mkguard(guard_end, l),

define_MAX0 = """
/* Header automatically inserted by PYPS for defining MAX, MIN, MOD and others */
#ifndef MAX0
# define MAX0(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MAX
# define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
# define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MOD
# define MOD(a, b) ((a) % (b))
#endif

#ifndef DBLE
# define DBLE(a) ((double)(a))
#endif

#ifndef INT
# define INT(a) ((int)(a))
#endif

#ifdef WITH_TRIGO
#  include <math.h>
#  ifndef COS
#    define COS(a) (cos(a))
#  endif

#  ifndef SIN
#    define SIN(a) (sin(a))
#  endif
#endif
/* End header automatically inserted by PYPS for defining MAX, MIN, MOD and others */
"""

def addMAX0(fname):
    """ Adds #define's for MAX0 and MOD."""
    addBeginnning(fname, define_MAX0)

def addBeginnning(fname, text):
    """Adds a line of text at the beginning of fname"""
    fi = fileinput.FileInput([fname], inplace = True)
    for l in fi:
        if fi.isfirstline():
            print text
        print l,
    
def unincludes(fname):
    """remove the contents of included files"""
    fi = fileinput.FileInput([fname], inplace = True)
    inside_include = False
    included = None
    end_included = None
    for l in fi:
        match = guard_begin_re.match(l)
        if match:
            included = match.group(1)
            inside_include = True
            end_included = mkguard(guard_end, included)
            print l,
            print included
            continue
        if l == end_included:
            inside_include = False
            included = None
            end_included = None
            print l,
            continue
        if inside_include:
            continue
        print l,

def string2file(string, fname):
    f = open(fname, "w")
    f.write(string)
    f.close()
    
def nameToTmpDirName(name): return "." + name + ".tmp"
