/* HPFC module by Fabien COELHO
 *
 * $Id$
 * $Log: lattice_extraction.c,v $
 * Revision 1.3  1997/03/20 10:27:04  coelho
 * RCS headers.
 *
 */

#include "defines-local.h"

/* void extract_lattice
 *
 * what: extracts a lattice from a set of equalities
 * how: Hermite form computation on equalities implying scanners
 *  - equalities translated in F and M;
 *  - variables = scanners + others + cst;
 *  - F.scanners = M.others + V;
 *  - H = PFQ; // note that P^{-1} = P^t because P is a permutation.
 *  - (1) scanners = Q.newscs; 
 *  - (2) F.scanners = P^{-1} H.newscs = M.others + V;
 *   then
 *  - (2) => new equalities
 *  - (1) => ddc in some order..., plus replacement in inequalities
 *  - (1) => newscs, but what about the order?
 *
 * input: Psysteme and scanners
 * output: modified system, new scanners and deducables
 * side effects:
 *  - may create some new variables
 *  - 
 * bugs or features:
 */
void 
extract_lattice(
    Psysteme s,                      /* the system is modified */
    list /* of entity */ scanners,   /* variables to be scanned */
    list /* of entity */ *newscs,    /* returned new scanners */
    list /* of expression */ *ddc)   /* old deduction */
{
    /* void implementation: nothing done!
     */
    *newscs = gen_copy_seq(scanners);
    *ddc = NIL;

    pips_user_warning("not implemented yet\n");

    /* - should try to remove deducables before hand?
     */

    return;
}

/* that is all
 */
