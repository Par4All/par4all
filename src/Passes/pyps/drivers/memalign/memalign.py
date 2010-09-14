# A workspace that forces all memory allocations to be aligned.
import re
import fileinput

alignsize = 16

# replace lines containing a = malloc(size) with posix_memalign(&a,
# alignsize, size)
mallocre = re.compile(r"(?P<id>\w+)\s*=\s*malloc\s*\((?P<size>.*)\)")
# XXX: doesn't handle a, b = malloc(foo)
# XXX: return the pointer for people who do error checking
memalign_tmpl = r"posix_memalign(&(\g<id>), " + str(alignsize) + r", \g<size>)";
def malloc2memalign(line):
    return mallocre.sub(memalign_tmpl, line)

# replace lines containing type a[size?] with a[size?] __attribute__
# ((aligned (alignsize)))
# XXX: does not handle char a[], b[];
arrayre = re.compile(r"^(?P<type>[a-z 	]*)\s+(?P<id>\w+)\s*\[(?P<size>\d*)\]\s*;")
arrayalign_tmpl = r"\g<type> \g<id>[\g<size>] __attribute__ ((aligned ("+str(alignsize)+")));"
def array2arrayalign(line):
    return arrayre.sub(arrayalign_tmpl, line)

class workspace:
    def __init__(self, *args, **kwargs):
        pass

    def post_goingToRunWith(self, files, outdir):
        for l in fileinput.FileInput(files, inplace = True):
            l = malloc2memalign(l)
            # l = array2arrayalign(l)
            print l,
