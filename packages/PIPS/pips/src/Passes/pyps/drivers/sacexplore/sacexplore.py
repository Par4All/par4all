import pyrops
import pyps
import sac
import sys
import Pyro
from copy import deepcopy
from optparse import OptionParser

conditions = [("simdizer_allow_padding", [False, True]),
              ("auto_unroll", [True, False]),
              ("full_unroll", [False, True]),
              ("reduction_detection", [False, True]),
              ("if_conversion", [False, True]),
              ]

def permutations():
    return generate_permutations_aux(deepcopy(conditions), {})

def generate_permutations_aux(conditions, first_perms):
    if conditions == []:
        return [first_perms]
    (cond, values) = conditions.pop()
    permutations = []
    for value in values:
        new_perms = first_perms.copy()
        new_perms[cond] = value
        permutations.extend(generate_permutations_aux(deepcopy(conditions), new_perms))
    return permutations

def main():
    parser = OptionParser(description = "Try several sac optimisations")
    parser.add_option("-f", "--function", dest = "function",
                      help = "function to try and optimize")
    parser.add_option("-v", "--verbose", dest = "verbose", action = "count",
                      default = 0, help = "be verbose")
    parser.add_option("-o", "--outdir", dest = "outdir",
                      help = "directory to store the resulting transformation")
    (args, sources) = parser.parse_args()
    for p in permutations():
        print "Trying permutation", p
        ws = pyrops.pworkspace(sources, verbose = (args.verbose >= 2),
                               parents = [sac.workspace], driver = "sse")
        module = ws[args.function]
        try:
            module.sac(verbose = (args.verbose >= 3), **p)
            if args.verbose >= 1:
                module.display()
            print "OK:", p
            if args.outdir:
                ws.save(rep = args.outdir)
            ws.close()
            exit(0)
        except RuntimeError, e:
            print "NOK", e.args
            if args.verbose: print "".join(Pyro.util.getPyroTraceback(e))
        ws.close()
    print "Found no suitable permutation"
    exit(1)

if __name__ == "__main__":
    main()
