/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

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

/* The basis of the constraint system is updated. If not, see sc_add_egalite() */
Psysteme
sc_equation_add(Psysteme sc, Pcontrainte c)
{
    sc = sc_constraint_add(sc, c, true);
    return sc;
}

Psysteme
sc_inequality_add(Psysteme sc, Pcontrainte c)
{
    sc = sc_constraint_add(sc, c, false);
    return sc;
}

Psysteme
sc_constraint_add(Psysteme sc, Pcontrainte c, bool equality)
{
    Pbase old_basis;
    Pbase new_basis;

    if(equality)
      sc_add_egalite(sc,c);
    else
      sc_add_inegalite(sc,c);

    /* maintain consistency, although it's expensive; how about a
       sc_update_base function? Or a proper sc_add_inegalite function? */
    old_basis = sc->base;
    sc->base = (Pbase) VECTEUR_NUL;
    sc_creer_base(sc);
    new_basis = sc->base;
    sc->base = base_union(old_basis, new_basis);
    sc->dimension = base_dimension(sc->base);
    base_rm(new_basis);
    base_rm(old_basis);

    return sc;
}
