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

#include <stdio.h>
#include <values.h>
#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "tiling.h"

#include "ri-util.h"

#include "matrice.h"

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
    boolean faisable;
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
	    pips_error("loop_nest_to_tile","illegal empty iteration domain\n");

    }
  
    tile = make_tiling(M, porigin);
    return tile;
}
