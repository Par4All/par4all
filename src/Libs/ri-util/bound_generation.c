/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* Generation of bound expressions from constraint systems of the
   Linear library */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

/* Used for sorting each constraint and in between constraints, hopefully */
int bound_generation_compare_vector_component(Pvecteur *pv1, Pvecteur *pv2)
{
  entity e1 = vecteur_var(*pv1);
  entity e2 = vecteur_var(*pv2);
  // FI: is "less" a well-chosen name?
  int less = strcmp(entity_local_name(e1), entity_local_name(e2));

  if(less==0) {
    Value v1 = vecteur_val(*pv1);
    Value v2 = vecteur_val(*pv2);
    if(value_gt(v1,v2))
      less = 1;
    else if(value_lt(v1,v2))
      less = -1;
  }

  return less;
}

/* void make_bound_expression(variable index, Pbase base, Psysteme sc,
 * expression *lower, expression *upper)
 *
 * build the expressions "lower" and "upper" for the lower and upper
 * bounds of variable "index".  Variable "index" must appear in "base"
 * and have lower and upper bounds in "sc"
 *
 * Beware of degenerated cases where constraints are reduced to
 * equations because the upper and lower bounds are identical.
 */
void make_bound_expression(Variable index,
			   Pbase base,
			   Psysteme sc,
			   expression *lower,
			   expression *upper)
{
    Pcontrainte pc;
    cons *ll = NIL;
    cons *lu = NIL;

    expression ex;
    entity min, max;

    int i;
    int rank_index ;

    /* compute the rank d of the index in the basis */
    rank_index =
      base_find_variable_rank(base, index,
			      (get_variable_name_t) entity_name_or_TCST);

    pips_debug(7, "index :%s\n", entity_name_or_TCST(index));
    pips_debug(8, "rank_index = %d\n", rank_index);

    /* The constraints should be lexicographically sorted to avoid
       secondary variations in linear */
    // contrainte_vect_sort(ll, bound_generation_compare_vector_component);
    ifdebug(7) {
      fprintf(stderr, "Constraints before sorting:\n");
      sc_dump(sc);
    }
    sc_lexicographic_sort(sc, bound_generation_compare_vector_component);
    ifdebug(7) {
      fprintf(stderr, "Constraints after sorting:\n");
      sc_dump(sc);
    }

    /* search constraints referencing "index" among inequalities and
      create the list of expressions for lower and upper bounds */
    for (pc=sc->inegalites; pc!=NULL; pc=pc->succ) {
	i = level_contrainte(pc, base);
	pips_debug(8,"level: %d\n",i);
	if (ABS(i)==rank_index){	/* found */
	    ifdebug(7) {
		(void) fprintf(stderr, "\n constraint before :");
		contrainte_fprint(stderr, pc, true,
				  (get_variable_name_t) entity_name_or_TCST);
	    }
	    ex = make_constraint_expression(pc->vecteur, (Variable) index);
	    ifdebug(7) {
		fprintf(stderr, "\n expression after :");
		print_expression(ex);
	    }
	    /* add the expression to the list of  lower bounds
	       or to the list of upper bounds*/
	    if (i>0)
		lu = CONS(EXPRESSION, ex, lu);
	    else
		ll = CONS(EXPRESSION, ex, ll);
	}
    }

    /* search equation constraints referencing "index" and create the
      list of expressions for lower and upper bounds. We may have to
      generate useless loops with only one iteration. */
    for (pc=sc->egalites; pc!=NULL; pc=pc->succ) {
	i = level_contrainte(pc, base);
	pips_debug(8,"level: %d\n",i);
	if (ABS(i)==rank_index){	/* found */
	    ifdebug(7) {
		(void) fprintf(stderr, "\n constraint before :");
		contrainte_fprint(stderr, pc, true,
				  (get_variable_name_t) entity_name_or_TCST);
	    }
	    if(i>0) {
	      Pvecteur mv = vect_copy(pc->vecteur);
	      mv = vect_multiply(mv, VALUE_MONE);
	      ex = make_constraint_expression(mv, (Variable) index);
	      vect_rm(mv);
	    }
	    else
	      ex = make_constraint_expression(pc->vecteur, (Variable) index);
	    ifdebug(7) {
		fprintf(stderr, "\n expression after :");
		print_expression(ex);
	    }

	    /* add the expression to the list of  lower bounds
	       and to the list of upper bounds*/
	    lu = CONS(EXPRESSION, ex, lu);
	    ll = CONS(EXPRESSION, ex, ll);
	}
    }

    /* Reverse the expression order */
    ll = gen_nreverse(ll);
    lu = gen_nreverse(lu);

    /* build expressions for the lower and upper bounds */
    if(c_language_module_p(get_current_module_entity())) {
      /* To avoid clash with Fortran intrinsics */
      /* pips_min and pips_max are supposed to be part of PIPS
	 run-time. They are varargs and their first argument is the
	 count of arguments */
      min = entity_intrinsic(PIPS_C_MIN_OPERATOR_NAME);
      max = entity_intrinsic(PIPS_C_MAX_OPERATOR_NAME);
    }
    else { // Fortran case
      min = entity_intrinsic(MIN_OPERATOR_NAME);
      max = entity_intrinsic(MAX_OPERATOR_NAME);
    }

    pips_assert("entities for min and max are found",
		min != entity_undefined && max != entity_undefined);

    if (gen_length(ll) > 1) {
      if(c_language_module_p(get_current_module_entity())) {
	int c = gen_length(ll);
	expression ce = int_to_expression(c);
	ll = CONS(EXPRESSION, ce, ll);
      }
      *lower = make_expression(make_syntax(is_syntax_call,
					   make_call(max,ll)),
			       normalized_undefined);
    }
    else {
	*lower = EXPRESSION(CAR(ll)); /* and memory leak... (cons lost) */
	gen_free_list(ll);
    }

    if (gen_length(lu) > 1 ) {
      if(c_language_module_p(get_current_module_entity())) {
	int c = gen_length(lu);
	expression ce = int_to_expression(c);
	lu = CONS(EXPRESSION, ce, lu);
      }
	*upper = make_expression(make_syntax(is_syntax_call,
					     make_call(min,lu)),
				 normalized_undefined );
    }
    else {
	*upper = EXPRESSION(CAR(lu)); /* idem... */
	gen_free_list(lu);
    }

    ifdebug(7) {
	pips_debug(9, "returning: \n");
	print_expression(*lower);
	print_expression(*upper);
    }
}
