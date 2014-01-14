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
 /* PACKAGE MOVEMENTS
  *
  * Corinne Ancourt  - septembre 1991
  */

#include <stdio.h>
#include <limits.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "misc.h"
#include "tiling.h"


Pvecteur 
make_loop_indice_equation(
    Pbase loop_indices,
    tiling tile, 
    Pvecteur tile_delay, 
    Pvecteur tile_indices,
    Pvecteur tile_local_indices,
    int rank)
{ 
    matrice P= (matrice) tiling_tile(tile);
     Pvecteur to = (Pvecteur)tiling_origin(tile);
    Pvecteur pv; 
    Pvecteur lti,li,ti= tile_indices;
    int m = vect_size(loop_indices); 
    int n= m;
    int i,j;
    Value val;
    for (i=1,lti =tile_local_indices, li = loop_indices; i<rank;
	 lti =lti->succ,li= li->succ, i++);
    pv = vect_make(NULL,vecteur_var(lti),VALUE_ONE,TCST,VALUE_ZERO);

    for (j=1;j<=m; j++,ti = ti->succ) {
	if (value_ne(ACCESS(P,n,i,j),VALUE_CONST(999))) { 
	    vect_add_elem(&pv,vecteur_var(ti), ACCESS(P,n,i,j));
	    if (value_notzero_p(val = vect_coeff(li->var, tile_delay)))
		vect_add_elem(&pv,TCST,
			      value_uminus(value_mult(val,ACCESS(P,n,i,j))));
	} 
    }
    if (value_notzero_p(val = vect_coeff(li->var,to)))
	vect_add_elem(&pv,TCST,val);
    ifdebug(8) {
	fprintf(stderr, "Equation defining the tiling\n");
	vect_dump(pv);
    }
    return(pv);
}

/* this function returns the system of constraints
*
*   ---------->      ----------->   --------->    ------>  ----------------->
*  loop_indices = P (tile_indices - tile_delay) + origin + tile_local_indices
*
 * P is the matrice describing the tiling. Its determinant is 1.
*
*/

Psysteme loop_bounds_to_tile_bounds(loop_bounds, loop_indices, t, tile_delay, tile_indices, tile_local_indices)
Psysteme loop_bounds;
Pbase loop_indices;
tiling t;
Pvecteur tile_delay;
Pvecteur tile_indices, tile_local_indices;
{
    Psysteme sc = sc_dup(loop_bounds);
    matrice P = (matrice) tiling_tile(t);
    Pcontrainte pc;
    Pvecteur li,to,lti,ti,td,pv;
    int n,m,i,j;
    Value min,max,val;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"loop_bounds_to_tile_bounds","begin\n");

    /* mise a jour de la base du systeme de contraintes */
    for (li=loop_indices,n=0;!VECTEUR_NUL_P(li);n++,li = li->succ);
    for (ti=tile_indices,m=0;
	 !VECTEUR_NUL_P(ti);
	 sc->base = vect_add_variable(sc->base,(char *) ti->var),
	 sc->dimension++,m++,ti = ti->succ);
    to = (Pvecteur)tiling_origin(t);
    td = tile_delay;

    for (i=1,lti =tile_local_indices, li = loop_indices;
	 i<=n; 
	 i++,
	 sc->base = vect_add_variable(sc->base,(Variable) lti->var),
	 sc->dimension++,lti = lti->succ,li=li->succ) {
	j=1;
	ti = tile_indices;
	pv = vect_new(vecteur_var(lti),VALUE_ONE);
	vect_add_elem(&pv,vecteur_var(li),VALUE_MONE);
	for (j=1;j<=m; j++,ti = ti->succ) {
	    if (value_ne(ACCESS(P,n,i,j),VALUE_CONST(999))) { 
		vect_add_elem(&pv,vecteur_var(ti), ACCESS(P,n,i,j));
		if (value_notzero_p(val = vect_coeff(li->var,td)))
		    vect_add_elem(&pv,TCST,
			value_uminus(value_mult(val,ACCESS(P,n,i,j))));
	    }   else {
		sc_force_variable_to_zero(sc,vecteur_var(ti));
		sc_add_egalite(sc,
			       contrainte_make(vect_new(vecteur_var(ti),
							VALUE_ONE)));
	    }
	}
	if (value_notzero_p(val = vect_coeff(li->var,to))) 
	    vect_add_elem(&pv,TCST,val);
	pc= contrainte_make(pv);
	sc_add_egalite(sc,pc);
    }

    /* build the constraints min (=0) <= lti <= max (=ls-1) */
    for ( j=1,lti= tile_local_indices;
	 !VECTEUR_NUL_P(lti) ;
	 lti = lti->succ,j++) {
	min = ACCESS(P,n,1,j);
	max = ACCESS(P,n,1,j);
	for (i=1;i<=n;i++) {
	    Value a = ACCESS(P,n,i,j);
	    value_minimum(min, a);
	    value_maximum(max, a);
	}
	pv = vect_new(vecteur_var(lti), VALUE_MONE);
	vect_add_elem(&pv,TCST,VALUE_ZERO);
	pc = contrainte_make(pv);
	sc_add_ineg(sc,pc);
	pv = vect_new(vecteur_var(lti), VALUE_ONE);
	vect_add_elem(&pv,TCST,value_minus(VALUE_ONE, max));
	pc = contrainte_make(pv);
	sc_add_ineg(sc,pc);
    }

    /* build the constraints    0 <= ti  */
    for (ti =tile_indices; !VECTEUR_NUL_P(ti); ti = ti->succ) {
	pv = vect_new(vecteur_var(ti), VALUE_MONE);
	pc = contrainte_make(pv);
	sc_add_ineg(sc,pc);
    }
    ifdebug(8) { fprintf(stderr,"TILE SYSTEM:\n");
		 sc_fprint(stderr, sc, (get_variable_name_t) entity_local_name);
    }
    debug(8,"loop_bounds_to_tile_bounds","end\n");
    debug_off();
    return(sc);
}
