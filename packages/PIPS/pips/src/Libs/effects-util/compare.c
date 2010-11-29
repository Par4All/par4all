/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/*
  effects-util/compare.c: sorting comparison functions for cells, cell references,
  effects and pointer values
 */

/************* CELLS */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "top-level.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"


int cell_reference_compare(reference *pr1, reference *pr2)
{
  int c1_pos = 0; /* result */
  reference r1 = *pr1;
  reference r2 = *pr2;
  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);

  if(same_entity_p(e1, e2))
    {
      /* same entity, sort on indices values */
      list dims1 = reference_indices(r1);
      list dims2 = reference_indices(r2);

      size_t nb_dims1 = gen_length(dims1);
      size_t nb_dims2 = gen_length(dims2);

      for(;!ENDP(dims1) && !ENDP(dims2) && c1_pos == 0; POP(dims1), POP(dims2))
	{
	  expression e1 = EXPRESSION(CAR(dims1));
	  expression e2 = EXPRESSION(CAR(dims2));

	  if(unbounded_expression_p(e1))
	    if(unbounded_expression_p(e2))
	      c1_pos = 0;
	    else
	      c1_pos = 1;
	  else
	    if(unbounded_expression_p(e2))
	      c1_pos = -1;
	    else
	      {
		syntax s1 = expression_syntax(e1);
		syntax s2 = expression_syntax(e2);
		if (syntax_reference_p(s1)
		    && entity_field_p(reference_variable(syntax_reference(s1)))
		    && syntax_reference_p(s2)
		    && entity_field_p(reference_variable(syntax_reference(s2))))
		  {
		    entity fe1 = reference_variable(syntax_reference(s1));
		    entity fe2 = reference_variable(syntax_reference(s2));
		    if (!same_entity_p(fe1, fe2))
		      c1_pos = strcmp(entity_name(fe1),entity_name(fe2));
		  }
		else
		  {
		    intptr_t i1 = 0;
		    intptr_t i2 = 0;
		    intptr_t diff = 0;

		    int r1 = expression_integer_value(e1, &i1);
		    int r2 = expression_integer_value(e2, &i2);

		    if (r1 && r2)
		      {
			diff = i1 - i2;
			c1_pos = diff==0? 0 : (diff>0?1:-1);
		      }
		  }
	      }
	}

      if (c1_pos == 0)
	c1_pos = (nb_dims1 < nb_dims2) ? -1 : ( (nb_dims1 > nb_dims2) ? 1 : 0);
    }
  else
    {
      /* not same entity, sort on entity name */
      /* sort on module name */
      char * mod1 = strdup(entity_module_name(e1));
      string mod2 = strdup(entity_module_name(e2));

      c1_pos = strcmp(mod1, mod2);
      /* if same module name: sort on entity local name */
      if (c1_pos == 0)
	{
	  c1_pos = strcmp(entity_user_name(e1), entity_user_name(e2));
	}
      /* else: current module and top_level come first, then others in lexicographic order */
      else
	{
	  entity module = get_current_module_entity();
	  string current_mod = module_local_name(module);
	  if (strcmp(current_mod, mod1) == 0)
	    {
	      if (top_level_entity_p(e2))
		c1_pos = strcmp(entity_user_name(e1), entity_user_name(e2));
	      else
		c1_pos = -1;
	    }
	  else if (strcmp(current_mod, mod2) == 0)
	    {
	      if (top_level_entity_p(e1))
		c1_pos = strcmp(entity_user_name(e1), entity_user_name(e2));
	      else
		c1_pos = 1;
	    }
	  else if (top_level_entity_p(e1))
	    c1_pos = -1;
	  else if (top_level_entity_p(e2))
	    c1_pos = 1;
	}
      free(mod1); free(mod2);
    }

  return c1_pos;
}

int cell_compare(cell *c1, cell *c2)
{
  pips_assert("gaps not handled yet (ppv1 first)", !cell_gap_p(*c1));
  pips_assert("gaps not handled yet (ppv2 first)", !cell_gap_p(*c2));

  return cell_reference_compare(&cell_reference(*c1), &cell_reference(*c2));

}

/************* EFFECTS */

/* Compares two effects for sorting. The first criterion is based on names.
 * Local entities come first; then they are sorted according to the
 * lexicographic order of the module name, and inside each module name class,
 * according to the local name lexicographic order. Then for a given
 * entity name, a read effect comes before a write effect. It is assumed
 * that there is only one effect of each type per entity. bc.
 */
int
effect_compare(effect *peff1, effect *peff2)
{
    int eff1_pos = 0;

    eff1_pos = cell_compare(&effect_cell(*peff1), &effect_cell(*peff2));
    if (eff1_pos == 0)
      {
	/* same paths, sort on action, reads first */
	if (effect_read_p(*peff1))
	  eff1_pos = -1;
	else if (effect_read_p(*peff2))
	  eff1_pos = 1;
	else eff1_pos = 0;
      }
    return(eff1_pos);
}

/* int compare_effect_reference(e1, e2):
 *
 * returns -1 if "e1" is before "e2" in the alphabetic order, else
 * +1. "e1" and "e2" are pointers to effect, we compare the names of their
 * reference's entity. */
int
compare_effect_reference(effect * e1, effect * e2)
{
  reference r1 = effect_any_reference(*e1);
  reference r2 = effect_any_reference(*e2);
  return cell_reference_compare(&r1, &r2);
}

/* int compare_effect_reference_in_common(e1, e2):
 *
 * returns -1 if "e1" is before "e2" in the alphabetic order, else
 * +1. "e1" and "e2" are pointers to effect, we compare the names of their
 * reference's entity with the common name in first if the entity belongs
 * to a common */
int
compare_effect_reference_in_common(effect * e1, effect * e2)
{
  entity v1, v2;
  int n1, n2 ,result;
  string name1, name2;
  v1 = reference_variable(effect_any_reference(*e1));
  v2 = reference_variable(effect_any_reference(*e2));
  n1 = (v1==(entity)NULL),
  n2 = (v2==(entity)NULL);
  name1= strdup((entity_in_common_p(v1)) ?
      (string) entity_and_common_name(v1):
      entity_name(v1));
  name2=  strdup((entity_in_common_p(v2)) ?
      (string) entity_and_common_name(v2):
      entity_name(v2));

  result =  (n1 || n2)?  (n2-n1): strcmp(name1,name2);
  free(name1);free(name2);
  return result;
}

/************* POINTER VALUES */

/* Compares two pointer values for sorting. The first criterion is based on names.
 * Local entities come first; then they are sorted according to the
 * lexicographic order of the module name, and inside each module name class,
 * according to the local name lexicographic order. Then for a given
 * entity name, a read effect comes before a write effect. It is assumed
 * that there is only one effect of each type per entity. bc.
 */
int
pointer_value_compare(cell_relation *ppv1, cell_relation *ppv2)
{
  int ppv1_pos = 0; /* result */
  /* compare first references of *ppv1 and *ppv2 */

  cell ppv1_first_c = cell_relation_first_cell(*ppv1);
  cell ppv2_first_c = cell_relation_first_cell(*ppv2);

  pips_assert("there should not be preference cells in pointer values (ppv1 first) \n",
	      !cell_preference_p(ppv1_first_c));
  pips_assert("there should not be preference cells in pointer values (ppv2 first) \n",
	      !cell_preference_p(ppv2_first_c));

  pips_assert("the first cell must have value_of interpretation (ppv1)\n",
	      cell_relation_first_value_of_p(*ppv1));
  pips_assert("the first cell must have value_of interpretation (ppv2)\n",
	      cell_relation_first_value_of_p(*ppv2));

  ppv1_pos = cell_compare(&ppv1_first_c, &ppv2_first_c);

  if (ppv1_pos == 0)       /* same first cells */
    {
      /* put second cells value_of before address_of */
      bool ppv1_second_value_of_p = cell_relation_second_value_of_p(*ppv1);
      bool ppv2_second_value_of_p = cell_relation_second_value_of_p(*ppv2);

      ppv1_pos = (ppv1_second_value_of_p ==  ppv2_second_value_of_p) ? 0 :
	(ppv1_second_value_of_p ? -1 : 1);

      if (ppv1_pos == 0) /* both are value_of or address_of*/
	{
	  /* compare second cells */
	  cell ppv1_second_c = cell_relation_second_cell(*ppv1);
	  cell ppv2_second_c = cell_relation_second_cell(*ppv2);
	  ppv1_pos = cell_compare(&ppv1_second_c, &ppv2_second_c);

	}
    }
  return(ppv1_pos);
}

/************ PVECTEUR */

/** @brief weight function for Pvecteur passed as argument to
 *         sc_lexicographic_sort in prettyprint functions involving cell descriptors.
 *
 * The strange argument type is required by qsort(), deep down in the calls.
 * This function is an adaptation of is_inferior_pvarval in semantics
 */
int
is_inferior_cell_descriptor_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    /* The constant term is given the highest weight to push constant
       terms at the end of the constraints and to make those easy
       to compare. If not, constant 0 will be handled differently from
       other constants. However, it would be nice to give constant terms
       the lowest weight to print simple constraints first...

       Either I define two comparison functions, or I cheat somewhere else.
       Let's cheat? */
    int is_equal = 0;

    if (term_cst(*pvarval1) && !term_cst(*pvarval2))
      is_equal = 1;
    else if (term_cst(*pvarval1) && term_cst(*pvarval2))
      is_equal = 0;
    else if(term_cst(*pvarval2))
      is_equal = -1;
    else if(variable_phi_p((entity) vecteur_var(*pvarval1))
	    && !variable_phi_p((entity) vecteur_var(*pvarval2)))
      is_equal = -1;
    else  if(variable_phi_p((entity) vecteur_var(*pvarval2))
	    && !variable_phi_p((entity) vecteur_var(*pvarval1)))
      is_equal = 1;
    else
	is_equal =
	    strcmp(entity_name((entity) vecteur_var(*pvarval1)),
		   entity_name((entity) vecteur_var(*pvarval2)));

    return is_equal;
}
