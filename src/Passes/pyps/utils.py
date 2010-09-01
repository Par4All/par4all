# A small toolbox for various utility functions
import re
import os

guard_begin = "PIPS include guard begin:"
guard_begin_re = re.compile(r"^/\* %s (.+) \*/$" % guard_begin)
guard_end = "PIPS include guard end:"
include_re = re.compile(r"^\s*#\s*include\s*(\S+)\s*.*$")

def mkguard(guard, line):
    return "/* %s %s */\n" % (guard, line.rstrip("\r\n"))

def guardincludes(fname):
    """ Adds guards around includes."""
    outname = "%s.tmp" % fname
    with open(outname, "w") as outfile:
        with open(fname, "r") as infile:
            for l in infile:
                is_include = include_re.match(l)
                if is_include:
                    outfile.write(mkguard(guard_begin, l))
                outfile.write(l)
                if is_include:
                    outfile.write(mkguard(guard_end, l))
    os.rename(outname, fname)

define_MAX0 = """
/* Header automatically inserted by PYPS */
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
# define MOD(a, b) ((b) % (a))
#endif

#ifndef DBLE
# define DBLE(a) ((double)(a))
#endif

#ifndef INT
# define INT(a) ((int)(a))
#endif
/* End header automatically inserted by PYPS */
"""

def addMAX0(fname):
    """ Adds #define's for MAX0 and MOD."""
    outname = "%s.tmp" % fname
    with open(outname, "w") as outfile:
        outfile.write(define_MAX0)
        with open(fname, "r") as infile:
            for l in infile:
                outfile.write(l)
    os.rename(outname, fname)

# remove the contents of included files
def unincludes(fname):
    with open(fname + ".tmp", "w") as outfile:
        with open(fname, "r") as infile:
            # ``Mixing iteration and read methods would lose data'', python
            # says, so using the while 1: idiom.
            while 1:
                l = infile.readline()
                if not l: break
                match = guard_begin_re.match(l)
                if match:
                    included = match.group(1)
                    outfile.write(l)
                    outfile.write(included + "\n")
                    while 1:
                        l = infile.readline()
                        if l == mkguard(guard_end, included):
                            break
                    outfile.write(mkguard(guard_end, included))
                else:
                    outfile.write(l)
    os.rename(fname + ".tmp", fname)
    
def nameToTmpDirName(name): return "." + name + ".tmp"

