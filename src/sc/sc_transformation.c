/* Package sc */

#include <string.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "malloc.h"


/* Transform the two constraints A.x <= b and -A.x <= -b of system sc into 
 * an equality  A.x == b.
*/

void sc_transform_ineg_in_eg(sc)
Psysteme sc;
{

    Pcontrainte pc1,pc2,pc1_succ;
    Pcontrainte pc3=CONTRAINTE_UNDEFINED;
    Pcontrainte pc4=CONTRAINTE_UNDEFINED;
    boolean trouve = FALSE;

    for (pc1 = sc->inegalites; 
	 !CONTRAINTE_UNDEFINED_P(pc1);pc1 = pc1_succ) {
	trouve = FALSE;
	for (pc2 = pc1->succ; 
	     !CONTRAINTE_UNDEFINED_P(pc2) && !trouve; 
	     pc2 = pc2->succ) {
	    Pvecteur pv = vect_add(pc1->vecteur,pc2->vecteur); 
	    if (VECTEUR_NUL_P(pv)) {
		/* This condition is true if the constraints are opposed. */
		pc3 = contrainte_dup(pc1);
		pc4=pc2;
		trouve = TRUE;
	    }
	}
	pc1_succ = pc1->succ;
	if (trouve) {
	    sc_add_eg(sc,pc3);	   
	    eq_set_vect_nul (pc4);
	    eq_set_vect_nul (pc1);
	}
    } 
    sc = sc_kill_db_eg(sc);
}


/* Transform each equality in two inequalities */

void sc_transform_eg_in_ineg(sc)
Psysteme sc;
{

    Pcontrainte eg,pc1,pc2;

    for (eg = sc->egalites; 
	 !CONTRAINTE_UNDEFINED_P(eg); 
	 eg=eg->succ) {
	pc1 = contrainte_dup(eg);
	pc2 = contrainte_dup(eg);
	vect_chg_sgn(pc2->vecteur);
	sc_add_ineg(sc,pc1);
	sc_add_ineg(sc,pc2);
    }
    free(sc->egalites);
    sc->nb_eq = 0;
    sc->egalites = CONTRAINTE_UNDEFINED;
}
