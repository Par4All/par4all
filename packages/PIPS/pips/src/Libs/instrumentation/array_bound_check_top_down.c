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
/* -----------------------------------------------------------------
 *
 *                 TOP-DOWN ARRAY BOUND CHECK VERSION
 *
 * -----------------------------------------------------------------
 *
 * This phase is based on the array region analyses
 *  Different strategies :
 * 1. When a bounds violation is detected, we generate a STOP
 * statement  and we don't go down anymore
 * => difficult for debugging all the program's error
 * => finish as soon as possible
 * 2. Generate comments for errors and warnings
 * => convenient for debugging
 * => can not detect bounds violation for dynamic tests
 *    with generated code
 * 3. Test and STOP statement every where
 * => the {0==1} every where , not very intelligent ?
 *
 * The first strategy is implemented for this moment
 *
 *
 * Hypotheses : there is no write effect on the array bound expression.
 *
 * There was a test for write effect on bound here but I put it away (in
 * effect_on_array_bound.c) because it takes time to calculate the effect
 * but in fact this case is rare */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "alias_private.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "transformer.h"
#include "properties.h"
#include "pipsmake.h"
#include "instrumentation.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"

/* The following data structure is the context of top_down_abc:
   The read_marked_list marks if one bound of one array's dimension
   is tested or not, for the READ regions.
   The write_marked_list marks if one bound of one array's dimension
   is tested or not, for the WRITE regions.
   The saved list keeps the status before go down in the tree.
   The hash table statement_check_list associates each statement
   with its list of checks for each bound of array's dimension  */

typedef struct
{
  abc_checked read_marked_list;
  abc_checked write_marked_list;
  hash_table read_saved_list;
  hash_table write_saved_list;
  hash_table statement_check_list;
  persistant_statement_to_control map; // to treat unstructured case
  stack uns;
} top_down_abc_context_t,
* top_down_abc_context_p;

typedef struct Bound_test
{
  bool bound;
  expression test;
} Bound_test;

static string read_or_write(bool a)
{
  if (a)
    return ", READING, ";
  return ", WRITING, ";
}

string bool_to_bound(bool b)
{
  if (b)
    return ", lower bound, ";
  return ", upper bound, ";
}

/* Statistic variables: */

static int number_of_added_tests;
static int number_of_bound_violations;

static void initialize_top_down_abc_statistics()
{
    number_of_added_tests = 0;
    number_of_bound_violations = 0;
}

static void display_top_down_abc_statistics()
{
  if (number_of_added_tests > 0)
    user_log("* There %s %d array bound check%s added *\n",
	     number_of_added_tests > 1 ? "are" : "is",
	     number_of_added_tests,
	     number_of_added_tests > 1 ? "s" : "");

  if (number_of_bound_violations > 0)
    user_log("* There %s %d bound violation%s *\n",
	     number_of_bound_violations > 1 ? "are" : "is",
	     number_of_bound_violations,
	     number_of_bound_violations > 1 ? "s" : "");
}

static abc_checked initiliaze_marked_list()
{
  list retour = NIL;
  // get list of entities in the declaration part
  list ld =
    code_declarations(value_code(entity_initial(get_current_module_entity())));
  MAP(ENTITY,e,
  {
    type t = entity_type(e);
    if (type_variable_p(t))
      {
	list ldim = variable_dimensions(type_variable(t));
	int length = gen_length(ldim);
	if (length > 0)
	  {
	    dimension_checked dc = dimension_checked_undefined ;
	    list dc_list = NIL;
	    int i;
	    array_dimension_checked adc;
	    for(i=1; i <= length; i++ )
	      {
		dc = make_dimension_checked(i,false,false);
		dc_list = gen_nconc(dc_list,
				    CONS(DIMENSION_CHECKED,dc,NIL));
	      }
	    adc = make_array_dimension_checked(e, dc_list);
	    retour = gen_nconc(retour,CONS(ARRAY_DIMENSION_CHECKED,adc,NIL));
	  }
      }
  },
      ld);
  return make_abc_checked(retour);
}

static void set_array_dimension_checked(top_down_abc_context_p context,
	bool action, entity array, int dim, bool bound)
{
  if (action)
    {
      // read region
      MAP(ARRAY_DIMENSION_CHECKED, adc,
      {
	if (same_entity_p(array,array_dimension_checked_array(adc)))
	  MAP(DIMENSION_CHECKED, dc,
	  {
	    if (dimension_checked_dim(dc) == dim)
	      {
		if (bound)
		  dimension_checked_lower(dc) = true;
		else
		  dimension_checked_upper(dc) = true;
	      }
	  },array_dimension_checked_dims(adc));
      }, abc_checked_list(context->read_marked_list));
    }
  else
    {
      // write region
      MAP(ARRAY_DIMENSION_CHECKED, adc,
      {
	if (same_entity_p(array,array_dimension_checked_array(adc)))
	  MAP(DIMENSION_CHECKED, dc,
	  {
	    if (dimension_checked_dim(dc) == dim)
	      {
		if (bound)
		  dimension_checked_lower(dc) = true;
		else
		  dimension_checked_upper(dc) = true;
	      }
	  },array_dimension_checked_dims(adc));
      }, abc_checked_list(context->write_marked_list));
    }
}

Psysteme my_system_projection_along_variables(Psysteme ps, Pvecteur pv)
{
  // handle overflow by the calling procedure (FWD_OFL_CTRL)
  CATCH(overflow_error)
    return SC_UNDEFINED;
  TRY
    {
      bool is_proj_exact = true;
      sc_projection_along_variables_with_test_ofl_ctrl(&ps, pv,
				  &is_proj_exact, FWD_OFL_CTRL);
      UNCATCH(overflow_error);
      if (is_proj_exact)
	return ps;
      return SC_UNDEFINED;
    }
}

static Psysteme my_system_remove_variables(Psysteme ps)
{
  /* Project all PHI and #init variables, ....in the system ps,
     there are 2 cases :
     1. The result is not sure , there are over flow
        Return the SC_UNDEFINED
     2. The result is sure, three small cases:
        2.1 The system is always false sc_empty => no bounds violation
        2.2 The system is always true sc_rn => bounds violation
        2.3 The system is parametric => test to put*/

  if (!sc_empty_p(ps)&& !sc_rn_p(ps))
    {
      list l_phi = phi_entities_list(1,7);
      list l_var = l_phi;
      Pvecteur pv_var = NULL;
      Pbase b = ps->base;
      /* converts the phi list into a Pvecteur */
      MAP(ENTITY, e,
      {
	if (base_contains_variable_p(ps->base, (Variable) e) )
	  vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
      },l_var);

      for(; !VECTEUR_NUL_P(b);b = b->succ)
	{
	  entity e = (entity) vecteur_var(b);
	  if (old_value_entity_p(e))
	    vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
	}
      ps = my_system_projection_along_variables(ps, pv_var);
      vect_rm(pv_var);
      gen_free_list(l_phi);
    }
  return ps;
}

static void top_down_abc_insert_before_statement(statement s,
		statement s1,top_down_abc_context_p context)
{
  /* If s is in an unstructured instruction, we must pay attetion
     when inserting s1 before s.  */
  if (bound_persistant_statement_to_control_p(context->map, s))
    {
      /* take the control qui has s as statement  */
      control c = apply_persistant_statement_to_control(context->map, s);
      if (stack_size(context->uns)>0)
	{
	  /* take the unstructured correspond to the control c */
	  unstructured u = (unstructured) stack_head(context->uns);
	  control newc;
	  /* for a consistent unstructured, a test must have 2 successors,
	     so if s1 is a test, we transform it into sequence in order
	     to avoid this constraint.
	     Then we create a new control for it, with the predecessors
	     are those of c and the only one successor is c.
	     The new predecessors of c are only the new control*/
	  if (statement_test_p(s1))
	    {
	      list seq = CONS(STATEMENT,s1,NIL);
	      statement s2=
		instruction_to_statement(make_instruction(is_instruction_sequence,
							  make_sequence(seq)));
	      newc = make_control(s2,control_predecessors(c),CONS(CONTROL,c,NIL));
	    }
	  else
	    newc = make_control(s1,control_predecessors(c),CONS(CONTROL,c,NIL));
	  // replace c by  newc as successor of each predecessor of c
	  MAP(CONTROL, co,
	  {
	    MAPL(lc,
	    {
	      if (CONTROL(CAR(lc))==c) CONTROL_(CAR(lc)) = newc;
	    }, control_successors(co));
	  },control_predecessors(c));
	  control_predecessors(c) = CONS(CONTROL,newc,NIL);
	  /* if c is the entry node of the correspond unstructured u,
	     the newc will become the new entry node of u */
	  if (unstructured_control(u)==c)
	    unstructured_control(u) = newc;
	  gen_recurse_stop(newc); // ????????????
	}
      else
	// there is no unstructured (?)
	insert_statement(s,s1,true);
    }
  else
    // structured case
    insert_statement(s,s1,true);
}


static list top_down_abc_call(call c, entity array,
		    dimension dim_i, int i, bool bound)
{
  list retour = NIL;
  list args = call_arguments(c);
  MAP(EXPRESSION,e,
  {
    syntax s = expression_syntax(e);
    tag t = syntax_tag(s);
    switch (t){
    case is_syntax_call:
      {
	list tmp = top_down_abc_call(syntax_call(s),array,dim_i,i,bound);
	if (tmp != NIL)
	  // add tmp to retour
	  MAP(EXPRESSION, exp,
	  {
	    if (!same_expression_in_list_p(exp,retour))
	      retour = gen_nconc(retour,CONS(EXPRESSION,exp,NIL));
	  },
	      tmp);
	break;
      }
    case is_syntax_range:
      /* There is nothing to check here*/
      break;
    case is_syntax_reference:
      {
	reference ref = syntax_reference(s);
	entity arr = reference_variable(ref);
	if (same_entity_p(arr,array))
	  {
	    list arrayinds = reference_indices(ref);
	    expression ith = find_ith_argument(arrayinds,i);
	    expression exp = expression_undefined;
	    if (!expression_undefined_p(ith))
	      {
		ifdebug(2)
		  {
		    fprintf(stderr, "\n The ith expression");
		    print_expression(ith);
		    fprintf(stderr, " \n array: %s ",entity_name(array));
		  }
		if (bound)
		  {
		    /* Make expression : ith < lower_bound */
		    exp = lt_expression(copy_expression(ith),
					dimension_lower(copy_dimension(dim_i)));
		    ifdebug(2)
		      {
			fprintf(stderr, "\n The lower bound test");
			print_expression(exp);
		      }
		  }
		else
		  // test if the upper bound is unbounded or not
		  if (!unbounded_dimension_p(dim_i))
		    {
		      /* Make expression  : upper_bound < ith */
		      exp = lt_expression(dimension_upper(copy_dimension(dim_i)),
					  copy_expression(ith));
		      ifdebug(2)
			{
			  fprintf(stderr, "\n The upper bound test");
			  print_expression(exp);
			}
		      }
	      }

	    /* Remark : Doesn't like the 1st version of abc,
	       we don't have to put a test for ith in the indirect case.
	       F.e : for A(B(i)) = 0.0, we have 2 regions :
	       < B(PHI1)-R-EXACT-{PHI1==I} >
	       < A(PHI1)-W-MAY-{} >
	       when we check for the inexact case of A, we don't have
	       to check for B.

	       In case if there is another access of array B in the same
	       statement, the region of B may be MAY => we check it for
	       the read region of B.

	       In case of A(A(i)) , we have the different regions for read
	       and write effect ??????? => ?????
	       ATT : example Indirection.f */

	    /* Test if exp is trivial or not
	       + If exp is always TRUE: there is certainly bound violation,
	       return  make_true_expression
	       + If exp is always false, we don't have to add it to retour
	       + Otherwise, we add it to retour.*/
	    if (!expression_undefined_p(exp))
	      {
		int tr = trivial_expression_p(exp);
		switch(tr){
		case 1:
		  return CONS(EXPRESSION,make_true_expression(),NIL);
		case 0:
		  {
		    // test if exp is already in retour
		    if (!same_expression_in_list_p(exp,retour))
		      retour = gen_nconc(retour,CONS(EXPRESSION,exp,NIL));
		    break;
		  }
		case -1:
		  break;
		}
	      }
	  }
	break;
      }
    }
  },
      args);
  return retour;
}

/* hack: clean all normalize fields...
 */
static void expr_rwt(expression e)
{
  if (!normalized_undefined_p(expression_normalized(e)))
    {
      free_normalized(expression_normalized(e));
      expression_normalized(e) = normalized_undefined;
    }
}

void clean_all_normalized(expression e)
{
  gen_recurse(e, expression_domain, gen_true, expr_rwt);
}

static Bound_test top_down_abc_not_exact_case( statement s,
	       top_down_abc_context_p context, bool action,
	       entity array, dimension dim_i,int i, bool bound)
{
  Bound_test retour;
  retour.test = expression_undefined;
  retour.bound = true;

  /* Test if s is a call (elementary statement)*/
  if (statement_call_p(s))
    {
      /* generate a lower/upper bound test expression for all
	 array reference of this dimension of array*/
      call c = instruction_call(statement_instruction(s));
      list l = top_down_abc_call(c,array,dim_i,i,bound);
      if (l!= NIL)
	retour.test = expression_list_to_binary_operator_call(l,
				   entity_intrinsic(OR_OPERATOR_NAME));
      set_array_dimension_checked(context,action,array,i,bound);
      gen_free_list(l);
    }
  else
    /* s is not a call, no conclusion for this bound,
       continue to go down */
    retour.bound = false;

  return retour;
}

static Bound_test top_down_abc_dimension(statement s,
	   top_down_abc_context_p context, region re,
	   bool action, entity array, int i, bool bound)
{
  Bound_test retour;
  variable var = type_variable(entity_type(array));
  dimension dim_i = find_ith_dimension(variable_dimensions(var),i);
  retour.bound = true;
  retour.test = expression_undefined;
  if (!bound && unbounded_dimension_p(dim_i))
    /* unbounded dimension, we don't have to check for this bound */
    set_array_dimension_checked(context,action,array,i,bound);
  else
    {
      Psysteme P = sc_dup(region_system(re));
      Pcontrainte con_exp = CONTRAINTE_UNDEFINED;
      normalized nexpr = normalized_undefined;
      expression exp = expression_undefined;
      expression exp_1 = int_to_expression(1);
      if (bound)
	exp = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
					  dimension_lower(copy_dimension(dim_i)),
					  exp_1);
      else
	exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					  dimension_upper(copy_dimension(dim_i)),
					  exp_1);
      clean_all_normalized(exp);
      // fast check: PHIi<=lower-1 (or upper+1<=PHIi) is trivial redundant wrt P or not
      // transform exp to Pcontrainte con_exp
      nexpr = NORMALIZE_EXPRESSION(exp);
      if (normalized_linear_p(nexpr))
	{
	  entity phi = make_phi_entity(i);
	  Pvecteur v1 = vect_new((Variable) phi, VALUE_ONE);
	  Pvecteur v2 = normalized_linear(nexpr);
	  Pvecteur vect;
	  if (bound)
	    vect = vect_substract(v1, v2);
	  else
	    vect = vect_substract(v2, v1);
	  vect_rm(v1);
	  con_exp = contrainte_make(vect);
	  switch (sc_check_inequality_redundancy(con_exp, P)) {/* try fast check */
	  case 1: /* ok, redundant => there is certainly bound violation */
	    {
	      set_array_dimension_checked(context,action,array,i,bound);
	      retour.test = make_true_expression();
	      break;
	    }
	  case 2: /* ok, not feasible => there is no bound violation*/
	    {
	      set_array_dimension_checked(context,action,array,i,bound);
	      break;
	    }
	  case 0: /* no result, try slow version */
	    {
	      /* Add the equation  PHIi <= lower-1 (upper+1 <= PHIi)
		 to the predicate of region re */
	      if (sc_add_phi_equation(&P,exp,i,false,bound))
		{
		  /* Every expression is linear.
		   * Test the feasibility of P by using this function:
		   * sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res) in which
		   *
		   * ofl_ctrl = OFL_CTRL means that the overflows are treated in the
		   * called procedure (sc_rational_feasibility_ofl_ctrl())
		   *
		   * ofl_res = true means that if the overflows occur, function
		   * sc_rational_feasibility_ofl_ctrl will return the value TRUE
		   * we don't know if the system is feasible or not
		   *
		   * The function sc_rational_feasibility_ofl_ctrl() is less
		   * expensive than the function sc_integer_feasibility_ofl_ctrl()*/

		  if (!sc_rational_feasibility_ofl_ctrl(P, OFL_CTRL, true))
		    /* The system is not feasible (certainly) => no violation */
		    set_array_dimension_checked(context,action,array,i,bound);
		  else
		    {
		      /* The system is feasible or we don't know it is feasible or not
		       * Test if the region re is EXACT or MAY */
		      if (region_exact_p(re))
			{
			  /* EXACT region
			     Remove all PHI variables, useless variables such as
			     PIPS generated variables V#init, common variable
			     from another subroutine ......

			     SUBROUTINE X
			     CALL Y(I)
			     <P(I) {I == FOO:J}
			     END

			     SUBROUTINE Y(K)
			     COMMON J
			     K=J
			     END  */
			  Psysteme ps = my_system_remove_variables(P);
			  if (ps == SC_UNDEFINED)
			    // the projection is not exact
			    retour = top_down_abc_not_exact_case (s,context,action,array,dim_i,i,bound);
			  else
			    {
				// the projection is exact
			      set_array_dimension_checked(context,action,array,i,bound);
			      if (!sc_empty_p(ps) && !sc_rn_p(ps))
				// there is a test to put
				// sc_normalized or sc_elim_redon ?????
				retour.test = Psysteme_to_expression(ps);
			      else
				if (sc_rn_p(ps))
				  // the system is trivial true, there are bounds violation
				  retour.test = make_true_expression();
				//else, ps=sc_empty, the system is false, no bounds violation
			    }
			}
		      else
			/* MAY region */
			retour = top_down_abc_not_exact_case(s,context,action,array,dim_i,i,bound);
		    }
		}
	      else
		// the exp is not linear, we can't add to P
		retour.bound = false;
	    }
	  }
	  contrainte_free(con_exp);
	}
      sc_rm(P);
    }
  return retour;
}

static entity current_entity = entity_undefined;
static int current_max;
static int current_min = 0;

static bool max_statement_write_flt(statement s)
{
  list effects_list = load_proper_rw_effects_list(s);
  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	ifdebug(4)
	  {
	    fprintf(stderr,"\n MAX Write on entity %s :\n",entity_name(e));
	    fprintf(stderr,"\n MAX Current entity %s :\n",entity_name(current_entity));
	  }
	if (strcmp(entity_name(e),entity_name(current_entity))==0)
	  {
	    int n = statement_ordering(s);
	    ifdebug(4)
	      {
		fprintf(stderr,"a variable = current entity !!");
		fprintf(stderr,"This statement writes on %s with max ordering %d",
			entity_name(current_entity),n);
	      }
	    if (n>current_max) current_max = n;
	    break;
	  }
      }
  },
      effects_list);
  return true;
}

/* search for the maximum ordering of statement (after s) that writes on a*/
static int maximum_ordering(entity a, statement s)
{
  current_entity = a;
  current_max = 0;
  gen_recurse(s, statement_domain, max_statement_write_flt, gen_null);
  ifdebug(4)
    fprintf(stderr, " return current_max = %d of current entity %s ",
	    current_max, entity_name(current_entity));
  current_entity = entity_undefined;
  return current_max;
}

static bool min_statement_write_flt(statement s)
{
  list effects_list = load_proper_rw_effects_list(s);
  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	ifdebug(4)
	  {
	    fprintf(stderr,"\n MIN Write on entity %s :\n",entity_name(e));
	    fprintf(stderr,"\n MIN Current entity %s :\n",entity_name(current_entity));
	  }
	if (strcmp(entity_name(e),entity_name(current_entity))==0)
	  {
	    current_min = statement_ordering(s);
	    ifdebug(4)
	      {
		fprintf(stderr,"a variable = current entity !!");
		fprintf(stderr, " This statement writes on %s with min ordering %d",
			entity_name(current_entity),current_min);
	      }
	    return false;
	  }
      }
  },
      effects_list);
  return true;
}

/* search for the minimum ordering of statement (after s) that writes on a*/
static int minimum_ordering(entity a, statement s)
{
  current_entity = a;
  current_min = 0;
  gen_recurse(s, statement_domain, min_statement_write_flt, gen_null);
  ifdebug(4)
    fprintf(stderr, " return current_min = %d of current entity %s",
	    current_min, entity_name(current_entity));
  current_entity = entity_undefined;
  return current_min;
}

static bool is_first_written_array_p(entity a, list l, statement s)
{
  int max = maximum_ordering(a,s);
  ifdebug(4)
    fprintf(stderr, " max of %s = %d ", entity_name(a), max);
  MAP(ENTITY,other,
  {
    if (strcmp(entity_name(a),entity_name(other))!=0)
      {
	int min = minimum_ordering(other,s);
	ifdebug(4)
	  fprintf(stderr, " min of other %s  = %d ", entity_name(other), min);
	if (max >= min) return false;
      }
  },l);
  return true;
}

/*
  - For each write region, find list of statements (down from s) that write on the array
  - Find order between these written arrays : A <= B <= (C,D,E,F)
    (A <= B if and only if maximum{statement orderings A} < minimum{statement orderings B})
  - Apply algorithm for read and write regions on A and then B at this level (tests inserted
    before s, tests on A before on B)
  - Go down to substatement of s for the other unordered arrays (C,D,E,F)*/

static entity find_first_written_array(list l,statement s)
{
  list l_tmp = gen_full_copy_list(l);
  MAP(ENTITY,a,
  {
    if (is_first_written_array_p(a,l,s))
      return a;
  },l_tmp);
  return entity_undefined;
}

static statement test_sequence = statement_undefined;;
static bool godown = false;
static list lexp = NIL;

static void top_down_abc_array(entity array, region re,statement s, top_down_abc_context_p context)
{
  list marked_list = NIL;
  list dc_list = NIL;
  bool action = region_read_p(re);
  if (action)
    marked_list = abc_checked_list(context->read_marked_list);
  else
    marked_list = abc_checked_list(context->write_marked_list);
  MAP(ARRAY_DIMENSION_CHECKED,adc,
  {
    if (same_entity_p(array_dimension_checked_array(adc),array))
      {
	dc_list = array_dimension_checked_dims(adc);
	break;
      }
  },
      marked_list);
  // traverse each dimension
  while (!ENDP(dc_list))  {
    dimension_checked dc = DIMENSION_CHECKED(CAR(dc_list));
    int i = dimension_checked_dim(dc);
    Bound_test lower, upper;
    lower.test = expression_undefined;
    upper.test = expression_undefined;
    lower.bound = true;
    upper.bound = true;

    /* if we have a region like: <A(PHI)-EXACT-{}>
     * it means that all *declared* elements are touched, although
     * this is implicit. this occurs with io effects of "PRINT *, A".
     * in such a case, the declaration constraints MUST be appended
     * before the translation, otherwise the result might be false.
     *
     * potential bug : if the declaration system cannot be generated,
     *   the region should be turned to MAY for the translation? */
    append_declaration_sc_if_exact_without_constraints(re);
    if (!dimension_checked_upper(dc))
      {
	/* The upper bound of the dimension i is not marked TRUE*/
	upper = top_down_abc_dimension(s,context,re,action,array,i,false);
	if (!expression_undefined_p(upper.test))
	  {
	    statement sta;
	    test t;
	    string message =
	      strdup(concatenate("\'Bound violation:",
				 read_or_write(action), " array ",
				 entity_name(array),
				 bool_to_bound(false),
				 int_to_dimension(i),"\'",NULL));

	    if (true_expression_p(upper.test))
	      {
		/* There is bounds violation !
		   Insert a STOP before s (bug in Examples/perma.f if replace s by STOP*/
		number_of_bound_violations ++;
		user_log("\n Bound violation !!! \n");
		if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
		  sta  = make_print_statement(message);
		else
		  sta  = make_stop_statement(message);
		// top_down_abc_insert_before_statement(s,sta,context);
		if (statement_undefined_p(test_sequence))
		  test_sequence = copy_statement(sta);
		else
		  insert_statement(test_sequence,copy_statement(sta),false);
		//return false;  // follow the first strategy
	      }
	    // test if expression upper.test exists already in test_sequence
	    else
	      if (!same_expression_in_list_p(upper.test,lexp))
		{
		  ifdebug(2)
		    {
		      fprintf(stderr, "\n The upper test");
		      print_expression(upper.test);
		    }
		  number_of_added_tests++;
		  lexp = gen_nconc(lexp,CONS(EXPRESSION,upper.test,NIL));
		  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
		    t = make_test(upper.test,
				  make_print_statement(message),
				  make_block_statement(NIL));
		  else
		    t = make_test(upper.test,
				  make_stop_statement(message),
				  make_block_statement(NIL));
		  sta  = test_to_statement(t);
		  if (statement_undefined_p(test_sequence))
		    test_sequence = copy_statement(sta);
		  else
		    insert_statement(test_sequence,copy_statement(sta),false);
		}
	  }
      }
    if (!dimension_checked_lower(dc))
      {
	/* The lower bound of the dimension i is not marked TRUE*/
	lower = top_down_abc_dimension(s,context,re,action,array,i,true);
	if (!expression_undefined_p(lower.test))
	  {
	    statement sta;
	    test t;
	    string message =
	      strdup(concatenate("\'Bound violation:",
				 read_or_write(action)," array ",
				 entity_name(array),
				 bool_to_bound(true),
				 int_to_dimension(i),"\'",NULL));

	    if (true_expression_p(lower.test))
	      {
		/* There is bounds violation !
		   Insert a STOP before s (bug in Examples/perma.f if replace s by STOP*/
		number_of_bound_violations ++;
		user_log("\n Bound violation !!! \n");
		if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
		  sta =  make_print_statement(message);
		else
		  sta = make_stop_statement(message);
		// top_down_abc_insert_before_statement(s,sta,context);
		if (statement_undefined_p(test_sequence))
		  test_sequence = copy_statement(sta);
		else
		  /* insert the test after the generated tests, the order of tests
		     is important */
		  insert_statement(test_sequence,copy_statement(sta),false);
		// return false;  // follow the first strategy
	      }
	    else
	      // test if expression lower.test exists already in test_sequence
	      if (!same_expression_in_list_p(lower.test,lexp))
		{
		  ifdebug(2)
		    {
		      fprintf(stderr, "\n The lower test");
		      print_expression(lower.test);
		    }
		  number_of_added_tests ++;
		  lexp = gen_nconc(lexp,CONS(EXPRESSION,lower.test,NIL));
		  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
		    t = make_test(lower.test,
				  make_print_statement(message),
				  make_block_statement(NIL));
		  else
		    t = make_test(lower.test,
				  make_stop_statement(message),
				  make_block_statement(NIL));
		  sta  = test_to_statement(t);
		  if (statement_undefined_p(test_sequence))
		    test_sequence = copy_statement(sta);
		  else
		    insert_statement(test_sequence,copy_statement(sta),false);
		}
	  }
      }
	/* If one bound of the dimension is marked false,
	   we have to go down*/
    if ((!lower.bound) || (!upper.bound)) godown = true;
    dc_list = CDR(dc_list);
  }
}

 /* The old algorithm is false in the case of incorrect code, because regions are
     computed with the assumption that the code is correct. Here is an anti-example:

     COMMON ITAB(10),J
     REAL A(10)
C  <A(PHI1)-W-EXACT-{PHI1==11}>
C  <ITAB(PHI1)-W-MAY-{1<=PHI1}>
      READ *, M
      J = 11
C  <ITAB(PHI1)-W-EXACT-{1<=PHI1, PHI1<=M, J==11}>
      DO I = 1, M
C  <ITAB(PHI1)-W-EXACT-{PHI1==I, J==11, 1<=I, I<=M}>
         ITAB(I) = 1
      ENDDO
C  <A(PHI1)-W-EXACT-{PHI1==J, J==11, 1+M<=I, 1<=I}>
      A(J) = 0

  The region for array A can be false if there is a bound violation in
  ITAB for example with M=12, ITAB(11)=1=J, there will be no violation
  on A but on ITAB.  Based on this false region, the algorithm will
  tell that there is a violation on A.

  To keep the algorithm safe, we must take into account the order in
  which arrays are writen (bound violations on read arrays do not make
  transformers and array regions false).  If bound checks on ITAB are
  inserted before checks on A, tests are checked earlier, so the
  region of A is not false any more. The tests must be:

     COMMON ITAB(10),J
     REAL A(10)
     READ *, M
     IF (M.GT.10) STOP "Bound violation ITAB"
     STOP "Bound violation A"
     J = 11
     DO I = 1, M
       ITAB(I) = 1
     ENDDO
     A(J) = 0

  Modify the algorithm:
  At each statement s, take its list of regions:
  - If no array is written => apply the algorithm normally
  - Else
     - Find the writing order
     - Apply the algorithm for the first written array, and then the second, ...
     - If the order is not found at s, go to the substatements of s */

static bool top_down_abc_flt(statement s,top_down_abc_context_p context)
{
  list l_regions = regions_dup(load_statement_local_regions(s));
  list l_copy = gen_full_copy_list(l_regions);
  list l_written_arrays = NIL;
  lexp = NIL;
  test_sequence = statement_undefined;
  godown = false;
  ifdebug(3)
    {
      fprintf(stderr, "\n list of regions ");
      print_effects(l_regions);
      fprintf(stderr, "\n for the statement");
      print_statement(s);
    }
  hash_put(context->read_saved_list,s,
	   copy_abc_checked(context->read_marked_list));
  hash_put(context->write_saved_list,s,
	   copy_abc_checked(context->write_marked_list));

  /* Compute the list of written arrays */
  /*  while (!ENDP(l_copy))
    {
      region re = REGION(CAR(l_copy));
      reference ref = effect_any_reference(re);
      entity array = reference_variable(ref);
      if (array_reference_p(ref)
      && array_need_bound_check_p(array)
      && region_write_p(re))
	l_written_arrays = CONS(ENTITY,array,l_written_arrays);
      l_copy = CDR(l_copy);
    }
  ifdebug(3)
    {
      fprintf(stderr, "\n List of written arrays : \n ");
      print_list_entities(l_written_arrays);
      }*/

  /* If no array is written */
  if (ENDP(l_written_arrays))
    {
      /* check all arrays in l_regions*/
      l_copy = gen_full_copy_list(l_regions);
      while (!ENDP(l_copy))
	{
	  region re = REGION(CAR(l_copy));
	  reference ref = effect_any_reference(re);
	  entity array = reference_variable(ref);
	  if (array_reference_p(ref) && array_need_bound_check_p(array))
	    top_down_abc_array(array,re,s,context);
	  l_copy = CDR(l_copy);
	}
    }
  else
    {
      /* Choose the first written array and then generate checks for
	 the read and write regions of this array. We can check the
	 second array only when all dimensions of the written regions
	 of the first array are checked */
      while (!ENDP(l_written_arrays) && !godown)
	{
	  entity first_array = find_first_written_array(l_written_arrays,s);
	  /* if there is no array that is always written before the others,
	     we have to go down to substatements of s*/
	  if (entity_undefined_p(first_array))
	    godown = true;
	  else
	    {
	      ifdebug(3)
		{
		  fprintf(stderr, "\n First array: ");
		  fprintf(stderr, "%s ", entity_name(first_array));
		}
	      gen_remove_once(&l_written_arrays,first_array);
	      // check for the first array here
	      l_copy = gen_full_copy_list(l_regions);
	      while (!ENDP(l_copy))
		{
		  region re = REGION(CAR(l_copy));
		  reference ref = effect_any_reference(re);
		  entity array = reference_variable(ref);
		  //  if (same_entity_p(array,first_array))
		  if (strcmp(entity_name(array),entity_name(first_array))==0)
		    top_down_abc_array(array,re,s,context);
		  l_copy = CDR(l_copy);
		}
	    }
	}
    }
  if (!statement_undefined_p(test_sequence))
    {
      ifdebug(3)
	{
	  fprintf(stderr, "\n The sequence of test");
	  print_statement(test_sequence);
	}
      if (!godown)
	// godown = false, insert new tests for the statement s here
	top_down_abc_insert_before_statement(s,test_sequence,context);
      else
	// insert new tests in function rwt
	hash_put(context->statement_check_list,s,test_sequence);
    }
  if (!godown)
    {
      context->read_marked_list =
	(abc_checked) hash_get(context->read_saved_list,s);
      context->write_marked_list =
	(abc_checked) hash_get(context->write_saved_list,s);
    }
  gen_free_list(l_regions);
  gen_free_list(lexp);
  lexp = NIL;
  test_sequence = statement_undefined;
  return godown;
}

static void top_down_abc_rwt(statement s,
				 top_down_abc_context_p context)
{
  statement test_sequence = statement_undefined;
  context->read_marked_list =
    (abc_checked) hash_get(context->read_saved_list,s);
  context->write_marked_list =
    (abc_checked) hash_get(context->write_saved_list,s);
  test_sequence = (statement) hash_get(context->statement_check_list,s);
  if (!statement_undefined_p(test_sequence))
    {
      ifdebug(3)
	{
	  fprintf(stderr, "\n Rewrite : The sequence of test");
	  print_statement(test_sequence);
	  fprintf(stderr, "\n of statement");
	  print_statement(s);
	}
      // insert the new sequence of tests before the current statement
      top_down_abc_insert_before_statement(s,test_sequence,context);
    }
}

static bool store_mapping(control c, top_down_abc_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return true;
}

static bool push_uns(unstructured u, top_down_abc_context_p context)
{
  stack_push((char *) u, context->uns);
  return true;
}

static void pop_uns(unstructured __attribute__ ((unused)) u,
		    top_down_abc_context_p context)
{
  stack_pop(context->uns);
}


static void top_down_abc_statement(statement module_statement)
{
  top_down_abc_context_t context;
  context.read_marked_list = initiliaze_marked_list();
  context.write_marked_list = copy_abc_checked(context.read_marked_list);
  context.read_saved_list =  hash_table_make(hash_pointer, 0);
  context.write_saved_list = hash_table_make(hash_pointer, 0);

  context.statement_check_list = hash_table_make(hash_pointer, 0);
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);

  gen_context_multi_recurse(module_statement, &context,
      unstructured_domain, push_uns, pop_uns,
      control_domain, store_mapping, gen_null,
      statement_domain, top_down_abc_flt, top_down_abc_rwt,
      NULL);

  free_abc_checked(context.read_marked_list);
  free_abc_checked(context.write_marked_list);
  hash_table_free(context.read_saved_list);
  hash_table_free(context.write_saved_list);
  hash_table_free(context.statement_check_list);
  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns);
}

bool array_bound_check_top_down(const char* module_name)
{
  statement module_statement;
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  if (!same_string_p(rule_phase(find_rule_by_resource("REGIONS")),
		     "MUST_REGIONS"))
    pips_user_warning("\n MUST REGIONS not selected - "
		      "\n Do not expect wonderful results\n");
  /* set and get the current properties concerning regions */
  set_bool_property("MUST_REGIONS", true);
  set_bool_property("EXACT_REGIONS", true);
  get_regions_properties();
  /* Get the code of the module. */
  module_statement= (statement) db_get_memory_resource(DBR_CODE,
						       module_name,
						       true);
  set_current_module_statement(module_statement);
  set_ordering_to_statement(module_statement);
  /* Get the READ and WRITE regions of the module */
  set_rw_effects((statement_effects)
		 db_get_memory_resource(DBR_REGIONS, module_name, true));
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS,
					       module_name,true));
  debug_on("ARRAY_BOUND_CHECK_TOP_DOWN_DEBUG_LEVEL");
  pips_debug(1, " Region based ABC, Begin for %s\n", module_name);
  pips_assert("Statement is consistent ...",
	      statement_consistent_p(module_statement));
  initialize_top_down_abc_statistics();
  top_down_abc_statement(module_statement);
  display_top_down_abc_statistics();
  /* Reorder the module, because the bound checks have been added */
  module_reorder(module_statement);
  pips_debug(1, "end\n");
  debug_off();
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name, module_statement);
  reset_ordering_to_statement();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_proper_rw_effects();
  reset_rw_effects();
  return true;
}

/* END OF FILE */













