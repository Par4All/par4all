 /* package sur les systemes de contraintes sc
  *
  * Yi-Qing YANG, 20/05/92
  *
  */

#include <stdio.h>
#include <sys/stdtypes.h>  /* for debug with dbmalloc */
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

/*   That is all
 */
