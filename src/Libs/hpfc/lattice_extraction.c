/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* HPFC module by Fabien COELHO
 */

#include "defines-local.h"

#include "matrix.h"
#include "matrice.h"
#include "sparse_sc.h"

/*
void normalize_system(Psysteme * ps)
{
    Ptsg sg;
    Psysteme s;

    DEBUG_SYST(1, "entry system", *ps);

    sg = sc_to_sg_chernikova(*ps);
    sc_rm(*ps);

    s = sg_to_sc_chernikova(sg);
    sg_rm(sg);

    DEBUG_SYST(1, "exit system", s);

    *ps = s;
}    
*/

/* blindly appends b2 after b1
 */
static Pbase append_to(Pbase b1, Pbase b2)
{
    Pbase b;

    if (BASE_NULLE_P(b1)) return b2;

    for (b=b1; !BASE_NULLE_P(b->succ); b=b->succ);
    b->succ = b2;
    return b1;
}

/* returns a newly allocated base with the scanners ahead
 */
static Pbase scanners_then_others(
    Pbase initial,           /* full base */
    list /* of entity */ ls) /* scanners */
{
    Pbase sb = BASE_NULLE, ob = BASE_NULLE;
    Pbase b;
    
    for (b=initial; !BASE_NULLE_P(b); b=b->succ)
    {
	Variable v = var_of(b);
	if (gen_in_list_p((entity) v, ls))
	    base_add_dimension(&sb, v);
	else
	    base_add_dimension(&ob, v);
    }

    return append_to(sb, ob);
}

/* void extract_lattice
 *
 * what: extracts a lattice from a set of equalities
 * how: Hermite form computation on equalities implying scanners
 *  - equalities translated in F and M;
 *  - variables = scanners (s) + others (o) + cst (1)
 *  - F.s + M.o + V == 0
 *  - H = PFQ; // I assert P==I, maybe optimistic for this implementation...
 *  - (1) s = Q.y
 *  - (2) H.y + M.o + V == 0 
 *   then
 *  - (2) => new equalities that define some y(r)
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
    /* - should try to remove deducables before hand?
     */
    int nscanners, nothers, ntotal, neq;
    Pbase b, bsorted, byr;
    Pmatrix FM, F, M, V, P, H, Q, Hl, Hli, Ql, Qr, QlHli,
	QlHliM, QlHliV, mQr, I, Fnew;
    Value det_P, det_Q;
    int i;
    list /* of entity */ lns = NIL, ltmp = NIL;
    Pcontrainte eq;

    neq = sc_nbre_egalites(s);

    if (neq==0 || !get_bool_property("HPFC_EXTRACT_LATTICE")) 
    {
	/* void implementation: nothing done!
	 */
	*newscs = gen_copy_seq(scanners);
	*ddc = NIL;
	return;
    }
    /* else do the job */

    DEBUG_SYST(3, "initial system", s);
    DEBUG_ELST(3, "scanners", scanners);

    b = sc_base(s);
    nscanners = gen_length(scanners);
    ntotal = base_dimension(b);
    nothers = ntotal - nscanners;

    message_assert("more scanners than equalities", nscanners>=neq);

    bsorted = scanners_then_others(b, scanners);

    DEBUG_BASE(3, "sorted base", bsorted);
    pips_debug(3, "%d scanners, %d others, %d eqs\n", nscanners, nothers, neq);

    /* FM (so) + V == 0
     */
    FM = matrix_new(neq, ntotal);
    V  = matrix_new(neq, 1);
    
    constraints_to_matrices(sc_egalites(s), bsorted, FM, V);

    DEBUG_MTRX(4, "FM", FM);
    DEBUG_MTRX(4, "V", V);

    /* Fs + Mo + V == 0
     */
    F = matrix_new(neq, nscanners);
    M = matrix_new(neq, nothers);

    ordinary_sub_matrix(FM, F, 1, neq, 1, nscanners);
    ordinary_sub_matrix(FM, M, 1, neq, nscanners+1, ntotal);

    matrix_free(FM);

    DEBUG_MTRX(4, "F", F);
    DEBUG_MTRX(4, "M", M);

    /* H = P * F * Q
     */
    H = matrix_new(neq, nscanners);
    P = matrix_new(neq, neq);
    Q = matrix_new(nscanners, nscanners);

    matrix_hermite(F, P, H, Q, &det_P, &det_Q);

    DEBUG_MTRX(4, "H", H);
    DEBUG_MTRX(4, "P", P);
    DEBUG_MTRX(4, "Q", Q);

    message_assert("P == I", matrix_diagonal_p(P) && det_P==1);

    /* H = (Hl 0)
     */
    Hl = matrix_new(neq, neq);
    ordinary_sub_matrix(H, Hl, 1, neq, 1, neq);
    matrix_free(H);

    DEBUG_MTRX(4, "Hl", Hl);

    if (!matrix_triangular_unimodular_p(Hl, true)) {
	pips_user_warning("fast exit, some yes/no lattice skipped\n");
	/* and memory leak, by the way 
	 */
	*newscs = gen_copy_seq(scanners);
	*ddc = NIL;
	return;
    }

    message_assert("Hl is lower triangular unimodular", 
		   matrix_triangular_unimodular_p(Hl, true));

    /* Hli = Hl^-1
     */
    Hli = matrix_new(neq, neq);
    matrix_unimodular_triangular_inversion(Hl, Hli, true);
    matrix_free(Hl);

    DEBUG_MTRX(4, "Hli", Hli);

    /* Q = (Ql Qr) 
     */
    Ql = matrix_new(nscanners, neq);
    Qr = matrix_new(nscanners, nscanners-neq);

    ordinary_sub_matrix(Q, Ql, 1, nscanners, 1, neq);
    ordinary_sub_matrix(Q, Qr, 1, nscanners, neq+1, nscanners);

    DEBUG_MTRX(4, "Ql", Ql);
    DEBUG_MTRX(4, "Qr", Qr);

    matrix_free(Q);

    /* QlHli = Ql * Hl^-1 
     */
    QlHli = matrix_new(nscanners, neq);
    matrix_multiply(Ql, Hli, QlHli);

    matrix_free(Ql);
    matrix_free(Hli);

    /* QlHliM = QlHli * M
     */
    QlHliM = matrix_new(nscanners, nothers);
    matrix_multiply(QlHli, M, QlHliM);

    matrix_free(M);

    /* QlHliV = QlHli * V
     */
    QlHliV = matrix_new(nscanners, 1);
    matrix_multiply(QlHli, V, QlHliV);

    matrix_free(V);
    matrix_free(QlHli);

    /* I
     */
    I = matrix_new(nscanners, nscanners);
    matrix_identity(I, 0);

    /* mQr = - Qr
     */
    mQr = matrix_new(nscanners, nscanners-neq);
    matrix_uminus(Qr, mQr);
    matrix_free(Qr);

    /* create nscanners-neq new scanning variables... they are the yr's.
     */
    for (i=0; i<nscanners-neq; i++)
	lns = CONS(ENTITY, hpfc_new_variable(node_module, 
					     MakeBasic(is_basic_int)), lns);

    byr = list_to_base(lns);
    bsorted = append_to(byr, bsorted); byr = BASE_NULLE;

    /* We have: mQr yr + I s + QlHliM o + QlHliV == 0
     * yr are the new scanners, s the old ones, deducable from the new ones.
     * the equation must also be used to remove s from the inequalities.
     *
     * Fnew = ( mQr I QlHliM )
     */
    Fnew = matrix_new(nscanners, 2*nscanners-neq+nothers);

    insert_sub_matrix(Fnew, mQr, 1, nscanners, 1, nscanners-neq);
    insert_sub_matrix(Fnew, I, 1, nscanners, nscanners-neq+1, 2*nscanners-neq);
    insert_sub_matrix(Fnew, QlHliM, 1, nscanners, 
		      2*nscanners-neq+1, 2*nscanners-neq+nothers);

    matrix_free(I);
    matrix_free(mQr);
    matrix_free(QlHliM);

    /* Now we have: 
     *   (a) Fnew (yr s o)^t + QlHliV == 0
     *   (b) lns -- the new scanners
     *
     * we must 
     *  (1) generate deducables from (a), 
     *  (2) regenerate inequalities on yr's.
     */

    matrices_to_constraints(&eq, bsorted, Fnew, QlHliV);
    matrix_free(Fnew);
    matrix_free(QlHliV);

    /* clean the new system
     */
    contraintes_free(sc_egalites(s));
    sc_egalites(s) = eq;
    base_rm(sc_base(s));
    sc_creer_base(s);

    /* old scanners are deduced now:
     */
    *ddc = gen_append(*ddc, simplify_deducable_variables(s, scanners, &ltmp));
    pips_assert("no vars left", ENDP(ltmp));

    *newscs = lns;

    base_rm(bsorted);

    DEBUG_SYST(3, "resulting system", s);
    DEBUG_ELST(3, "new scanners", lns);
}

/* that is all
 */
