 /* package sur les systemes de contraintes sc
  *
  * Yi-Qing YANG, 20/05/92
  *
  */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_elim_var(Psysteme sc, Variable v)`
 * This function eliminate all the contraints containing the variable v
 */
Psysteme sc_elim_var(sc, v)
Psysteme sc;
Variable v;

{
    Pcontrainte pr = NULL;
    Pcontrainte eq;
    
    if (sc == NULL) return(sc);
    else {
	eq = sc->egalites;
	while (eq != NULL) {
	    if (vect_coeff(v,eq->vecteur) != 0) {
		if (pr == NULL) {
		    sc->egalites = eq = eq->succ;
		}
		else {
		    pr->succ = eq = eq->succ;
		}
	    }
	    else {
		pr = eq;
		eq = eq->succ;
	    }
	}

	eq = sc->inegalites;
	pr = NULL;
	while (eq != NULL) {
	    if (vect_coeff(v,eq->vecteur) != 0) {
		if (pr == NULL) {
		    sc->inegalites = eq = eq->succ;
		}
		else {
		    pr->succ = eq = eq->succ;
		}
	    }
	    else {
		pr = eq;
		eq = eq->succ;
	    }
	}

	sc->nb_eq = nb_elems_list(sc->egalites);
	sc->nb_ineq = nb_elems_list(sc->inegalites);
	return(sc);
    }
} 

/* void sc_chg_var(Psysteme s, Variable v_old, Variable v_new)
 * this function replace the variable v_old in the system s by the
 * variable v_new.
 */
void sc_chg_var(s, v_old, v_new)
Psysteme s;
Variable v_old, v_new;
{
    Pcontrainte pc;
    if (s != NULL){
	for (pc = s->egalites; pc != NULL; pc = pc->succ) {
	    vect_chg_var(&(pc->vecteur), v_old, v_new);
	}

	for (pc = s->inegalites; pc != NULL; pc = pc->succ) {
	    vect_chg_var(&(pc->vecteur), v_old, v_new);
	}
    }
}

/* the name is self explanatory, I guess. FC 24/11/94
 * the vectors of the system are sorted.
 * see man qsort about the compare functions.
 */
void sc_vect_sort(s, compare)
Psysteme s;
int (*compare)();
{
    if (s==NULL || sc_empty_p(s) || sc_rn_p(s)) 
	return;

    contrainte_vect_sort(sc_egalites(s), compare);
    contrainte_vect_sort(sc_inegalites(s), compare);
}

/* SORT a Psysteme according to sort_base and compare (given to qsort).
 * Each constraint is first sorted according to the compare function.
 * Then list of constraints are sorted. 
 *
 * The only expected property
 * is that two calls to this function with the same system (whatever its
 * order) and same sort_base that covers all variables and same compare
 * function should give the same result. 
 *
 * The function is quite peculiar and the order is relevant for some
 * code generation issues... 
 *
 * Fabien Coelho
 */
void sc_sort(
    Psysteme sc,
    Pbase sort_base,
    int (*compare)(Pvecteur*, Pvecteur*))
{
    sc_vect_sort(sc, compare);
    sc->inegalites = 
	contrainte_sort(sc->inegalites, sc->base, sort_base, TRUE, TRUE);
    sc->egalites = 
	contrainte_sort(sc->egalites, sc->base, sort_base, TRUE, TRUE);
}

/* Minimize first the lexico-graphic weight of each constraint according to
 * the comparison function "compare", and then sort the list of equalities
 * and inequalities by increasing lexico-graphic weight.
 *
 * Francois Irigoin
 */
void sc_lexicographic_sort(
    Psysteme sc,
    int (*compare)(Pvecteur*, Pvecteur*))
{
    if (sc==NULL || sc_empty_p(sc) || sc_rn_p(sc)) 
	return;

    /* sort the system basis and each constraint */
    vect_sort_in_place(&(sc->base), compare);
    contrainte_vect_sort(sc_egalites(sc), compare);
    contrainte_vect_sort(sc_inegalites(sc), compare);

    /* sort equalities and inequalities */
    sc->egalites = constraints_lexicographic_sort(sc->egalites, compare);
    sc->inegalites = constraints_lexicographic_sort(sc->inegalites, compare);
}

/*   That is all
 */
