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
/* Code Generation for Distributed Memory Machines
  *
  * Functions to decide a specific tiling for a loop nest
  *
  * File: tiling.c
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin, Corinne Ancourt, Lei Zhou
  * 1991
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "ri-util.h"
#include "effects-util.h"

#include "matrice.h"
#include "tiling.h"

/* loop_nest_to_tile():
 *
 */
tiling loop_nest_to_tile(sc, ls, base_index,first_parallel_level,last_parallel_level,perfect_nested_level)
Psysteme sc;
int ls;
Pbase base_index;
int first_parallel_level;
int last_parallel_level;
int perfect_nested_level;
{

    tiling tile;
    matrice M;
    Pvecteur porigin = VECTEUR_NUL;
    Pvecteur pv;
    int vs = vect_size((Pvecteur)(base_index));
    Psysteme sc_proj;
    Value min,max;
    bool faisable;
    int i;

    /* Because the number of elements per bank ligne is usefull to
       optimize the tiling,  parameter ls is used temporarily to build it.
       Because ls is the number of bytes of each bank ligne, it is 
       divided by the assumed number of bytes needed for the  
       location of one element (4 for int and real) */

    int lsd = ls/4;

    /* build the diagonal matrix: ls x I */

    M = matrice_new(vs,vs);
    matrice_nulle(M,vs,vs); 
    for (i =1; i<= vs && i < first_parallel_level; i++)
	ACCESS(M,vs,i,i) = VALUE_ONE;
    for (; i<= vs && i <= last_parallel_level; i++)
	ACCESS(M,vs,i,i) = int_to_value(lsd);
    for (i = MIN(perfect_nested_level+1,last_parallel_level+1); i<= vs; i++)
	ACCESS(M,vs,i,i) = VALUE_CONST(999);

  
    /* build origin tile vector as the first iteration to minimize
       the number of partial tiles (heuristics) */

    for (pv = base_index; !VECTEUR_NUL_P(pv); pv = pv->succ) {
	Variable var = vecteur_var(pv);
	sc_proj = sc_dup(sc);
	faisable =  sc_minmax_of_variable(sc_proj,var, &min, &max);
	if (faisable) 
	    /*	    vect_chg_coeff(&porigin,var,min);*/
	    vect_chg_coeff(&porigin,var,VALUE_ONE);
	else
	    pips_internal_error("illegal empty iteration domain");

    }
  
    tile = make_tiling(M, porigin);
    return tile;
}
