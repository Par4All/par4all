import pyrops
import pyps
import sac
import sys
from optparse import OptionParser

conditions = "reduction_detection auto_unroll simdizer_allow_padding".split(" ")

def permutations():
    conditions = conditions[:]
    return generate_permutations_aux(conditions)

def generate_permutations_aux(conditions, **first_perms):
    if conditions == []:
        return [first_perms]
    cond = conditions[0]
    new_perms_false = first_perms.copy()
    new_perms_false[cond] = False
    new_perms_true = first_perms.copy()
    new_perms_true[cond] = True
    return generate_permutations_aux(conditions[1:], **new_perms_true) + generate_permutations_aux(conditions[1:], **new_perms_false)

def main():
    parser = OptionParser(description = "Try several sac optimisations")
    parser.add_option("-f", "--function", dest = "function",
                      help = "function to try and optimize")
    parser.add_option("-v", "--verbose", dest = "verbose", action = "count",
                      default = 0, help = "be verbose")
    (args, sources) = parser.parse_args()
    good_perms = []
    print permutations()
    for p in permutations():
        ws = pyrops.pworkspace(sources, verbose = (args.verbose >= 2),
                               parents = [sac.workspace], driver = "sse")
        module = ws[args.function]
        try:
            module.sac(verbose = (args.verbose >= 1), **p)
            if args.verbose >= 1:
                module.display()
            good_perms.append(p)
        except:
            pass
        ws.close()
    print "valid paths are:"
    for p in good_perms: print p

if __name__ == "__main__":
    main()
