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
/* package convex effects :  Be'atrice Creusillet 6/97
 *
 * File: compose.c
 * ~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the composition of convex effects with transformers or preconditions
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"

#include "transformer.h"
#include "effects-generic.h"
#include "effects-convex.h"

list
convex_regions_transformer_compose(list l_reg, transformer trans)
{
    project_regions_with_transformer_inverse(l_reg, trans, NIL);
    return l_reg;
}

list
convex_regions_inverse_transformer_compose(list l_reg, transformer trans)
{
    project_regions_with_transformer(l_reg, trans, NIL);
    return l_reg;
}

list convex_regions_precondition_compose(list l_reg, transformer context)
{
  list l_res = NIL;
  Psysteme sc_context = predicate_system(transformer_relation(context));

  pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc_context));
  ifdebug(8) {
    pips_debug(8, "context: \n");
    sc_syst_debug(sc_context);
  }

  FOREACH(EFFECT, reg, l_reg) {
    /* Only the store effects require preconditions */
    if(store_effect_p(reg)) {
      /* FI: this leads to problems when the context might become
	 later empty: there won't be any way to find out; in one
	 case I do not remember, the IN effects end up wrong;
	 however, adding descriptors for all scalar references may
	 slow down the region computation a lot. */
      if (!effect_scalar_p(reg) ) {
	descriptor reg_d = effect_descriptor(reg);
	Psysteme reg_sc = descriptor_convex_p(reg_d)? descriptor_convex(reg_d) : NULL;

	pips_assert("sc_context is weakly consistent", sc_weak_consistent_p(sc_context));
	pips_assert("reg_sc is weakly consistent (1)", sc_weak_consistent_p(reg_sc));
	region_sc_append(reg, sc_context, false);

	/* remove potential old values that may be found in precondition */
	list l_old_values = NIL;
	reg_sc = region_system(reg);
	for(Pbase b = sc_base(reg_sc); !BASE_NULLE_P(b); b = vecteur_succ(b)) {
	  entity e = (entity) vecteur_var(b);
	  if(global_old_value_p(e)) {
	    l_old_values = CONS(ENTITY, e, l_old_values);
	  }
	}
	region_exact_projection_along_parameters(reg, l_old_values);
	reg_sc = region_system(reg);
	gen_free_list(l_old_values);
	debug_region_consistency(reg);
	pips_debug_effect(8, "region after transformation: \n", reg );

	pips_assert("sc_context is weakly consistent", sc_weak_consistent_p(sc_context));
	pips_assert("reg_sc is weakly consistent (2)", sc_weak_consistent_p(reg_sc));
      }
      if (!region_empty_p(reg))
	l_res = CONS(EFFECT, reg, l_res);
    }
  }

  return l_res;
}
