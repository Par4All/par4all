# A workspace that forces all memory allocations to be aligned.
import re
import fileinput
import sys

alignsize = 16

# replace lines containing a = malloc(size) with:
# (posix_memalign(&a, alignsize, size), a)
mallocre = re.compile(r"(?P<id>\w+)\s*=\s*malloc\s*\((?P<size>.*)\)")
# XXX: doesn't handle a, b = malloc(foo)
# XXX: return the pointer for people who do error checking
memalign_tmpl = r"posix_memalign(&(\g<id>), " + str(alignsize) + r", \g<size>)";
def malloc2memalign(line):
    return mallocre.sub(memalign_tmpl, line)

# in lines declarations like "type a[size?], b[size?], add the attribute
# ((aligned (alignsize))) to a and b.

# First match whether we are looking at such a declaration; we assume here that
# we are already looking at the output from pips' prettyprinter, so the
# following is not possible:
# float a[2],
#       b[3];
# Then add the attribute to each item.

# match "a [3 * sizeof(type)]"
onearraydeclre = re.compile(r"(?:(?P<id>\w+)\s*(?P<size>(?:\[[^]]*\]\s*)+))")
# same RE as above, but without the named groups
onearraydecl = re.sub(r"\?P<\w+?>", r"?:", onearraydeclre.pattern)
print onearraydecl

# match "const float a[2], b[5];" (but this is not sufficient)
#arraydeclre = re.compile(r"^(?P<type>[a-z 	]+)\s+%s(?:\s*,\s*%s)*\s*;\s*$" % (onearraydecl, onearraydecl))
# match something like " const float a[2], b, c[] = {1, 2, 3};" (the final
# semicolon is important)
arraydeclre = re.compile(r"^\s*(?P<type>\w[\w\s]*)\s+(?:%s|.*,\s*%s).*;\s*$" % (onearraydecl, onearraydecl))
arrayalign_tmpl = r"\g<id>\g<size> __attribute__ ((aligned ("+str(alignsize)+")))"
funcdeclre = re.compile(r"^(?P<type>[a-z 	]+)\s+(?P<id>\w+)\s*\(.*\)\s*;\s*")
returnstmtre = re.compile(r"^\s*return.*")
def array2arrayalign(line):
    if arraydeclre.match(line) and \
            not funcdeclre.match(line) and \
            not returnstmtre.match(line):
        line = onearraydeclre.sub(arrayalign_tmpl, line)
    return line

class workspace:
    def __init__(self, *args, **kwargs):
        pass

    def post_goingToRunWith(self, files, outdir):
        for l in fileinput.FileInput(files, inplace = True):
            l = malloc2memalign(l)
            l = array2arrayalign(l)
            print l,
