#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"



/* This function inserts the constraint ineq at the beginning of the 
 * system of inequalities of sc
 */
void insert_ineq_begin_sc(sc,ineq)
Psysteme sc;
Pcontrainte ineq;
{
    Pcontrainte ineg;
    ineg = contrainte_dup(ineq);
    ineg->succ = sc->inegalites;
    sc->inegalites = ineg;		    
    sc->nb_ineq ++;
}

/* This function inserts two constraints ineq and ineq->succ at the 
 * end of the system of inequalities of sc
 */
void insert_2ineq_end_sc(sc,ineq)
Psysteme sc;
Pcontrainte ineq;
{
    Pcontrainte pc;

    if (CONTRAINTE_UNDEFINED_P(sc->inegalites)){
	sc->inegalites=contrainte_dup(ineq);
	(sc->inegalites)->succ = contrainte_dup(ineq->succ);
    }
    else {
	for (pc = sc->inegalites; 
	     !CONTRAINTE_UNDEFINED_P(pc->succ); 
	     pc=pc->succ);
	pc->succ = contrainte_dup(ineq);
	pc = pc->succ;
	pc->succ = contrainte_dup(ineq->succ);
    }
    sc->nb_ineq +=2;
}



/* This function inserts one constraint ineq  at the 
 * end of the system of inequalities of sc
*/

void insert_ineq_end_sc(sc,ineq)
Psysteme sc;
Pcontrainte ineq;
{
    Pcontrainte pc;

    if (CONTRAINTE_UNDEFINED_P(sc->inegalites)){
	sc->inegalites=contrainte_dup(ineq);
    }
    else {
	for (pc = sc->inegalites; 
	     !CONTRAINTE_UNDEFINED_P(pc->succ); 
	     pc=pc->succ);
	pc->succ = contrainte_dup(ineq);	
    }
    sc->nb_ineq ++;
}

