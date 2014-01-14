/*

  $Id$

  Copyright 1989-2014 MINES ParisTech
  Copyright 2010 HPC Project

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

/* functions specific to cell_relations */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "misc.h"

/**
   beware : modifies l1, l2 and their effects

  @param l1 and l2 are two lists of cell_relations.
  @param  cr1_cr2_combinable_p is a bool function that takes two
          individual cell_relations as arguments and renders true when they are
          considered as combinable ;
  @param  cr1_cr2_binary_op is a binary operator that combines two
          individual cell_relations;
  @param  cr1_unary_op is a unary operators that deal with the remnants of l1,
          that is those cell_relations that are not combinable with any effect of l2;
  @param  cr2_unary_op is a unary operators that deal with the remnants of l2,
          that is those cell_relations that are not combinable with any effect of l1;

  @return a list of cell_relations, combination of l1 and l2.

*/
list cell_relations_generic_binary_op(
    list l1,
    list l2,
    bool (*cr1_cr2_combinable_p)(cell_relation,cell_relation),
    list (*cr1_cr2_binary_op)(cell_relation,cell_relation),
    list (*cr1_unary_op)(cell_relation),
    list (*cr2_unary_op)(cell_relation),
    list (*union_op)(list, list))
{
  list l_res = NIL;
  list l_cr1 = list_undefined;
  list l_cr2 = list_undefined;

  debug_on("CELL_RELATIONS_OPERATORS_DEBUG_LEVEL");

  pips_debug_pvs(1, "l1:\n", l1);
  pips_debug_pvs(1, "l2:\n", l2);

  /* we first deal with the elements of l1 : those that are combinable with
   * the elements of l2, and the others, which we call the remnants of l1 */
  for(l_cr1 = l1; !ENDP(l_cr1); POP(l_cr1))
    {
      cell_relation cr1 = CELL_RELATION(CAR(l_cr1));
      list prec_l_cr2 = NIL;
      bool combinable = false;

      pips_debug_pv(2, "dealing with cr1:\n", cr1);

      l_cr2 = l2;
      while(!ENDP(l_cr2))
	{
	  cell_relation cr2 = CELL_RELATION(CAR(l_cr2));

	  pips_debug_pv(2, "considering cr2:\n", cr2);

	  if ( (*cr1_cr2_combinable_p)(cr1,cr2) )
	    {
	      pips_debug(2, "combinable\n");
	      combinable = true;
	      list l_res_tmp = (*cr1_cr2_binary_op)(cr1,cr2);
	      l_res = (*union_op)(l_res, l_res_tmp);

	      /* gen_remove(&l2, EFFECT(CAR(l_cr2))); */
	      if (prec_l_cr2 != NIL)
		CDR(prec_l_cr2) = CDR(l_cr2);
	      else
		l2 = CDR(l_cr2);

	      free(l_cr2); l_cr2 = NIL;
	      /* */
	      //free_cell_relation(cr1); cr1=cell_relation_undefined;
	      free_cell_relation(cr2); cr2=cell_relation_undefined;
	    }
	  else
	    {
	      pips_debug(2, "not combinable\n");
	      prec_l_cr2 = l_cr2;
	      l_cr2 = CDR(l_cr2);
	    }
	}

      pips_debug_pvs(2, "intermediate l_res 1:\n", l_res);

      if(!combinable)
	{
	  /* cr1 belongs to the remnants of l1 : it is combinable
	   * with no effects of l2 */
	  if ( (*cr1_cr2_combinable_p)(cr1,cell_relation_undefined) )
	    l_res = gen_nconc(l_res, (*cr1_unary_op)(cr1));
	}
      else
	{
	  free_cell_relation(cr1); cr1=cell_relation_undefined;
	}
    }

  pips_debug_pvs(2, "intermediate l_res 2:\n", l_res);

  /* we must then deal with the remnants of l2 */
  for(l_cr2 = l2; !ENDP(l_cr2); POP(l_cr2))
    {
      cell_relation cr2 = CELL_RELATION(CAR(l_cr2));

      if ( (*cr1_cr2_combinable_p)(cell_relation_undefined,cr2) )
	l_res = gen_nconc(l_res, (*cr2_unary_op)(cr2));
    }

  pips_debug_pvs(1, "final pvs:\n", l_res);

  /* no memory leaks: l1 and l2 won't be used anymore */
  gen_free_list(l1);
  gen_free_list(l2);

  debug_off();

  return l_res;
}
