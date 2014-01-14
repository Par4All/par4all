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
/****************************************************************** *
 *
 *		 INTERPROCEDURAL ARRAY BOUND CHECKING
 *
 *
*******************************************************************/
/* This phase checks for out of bound error when passing arrays or array
   elements as arguments in procedure call. It ensures that there is no bound
   violation in every array access in the callee procedure, with respect to
   the array declarations in the caller procedure

   The association rules for dummy and actual arrays in Fortran standard (ANSI) 
   Section 15.9.3.3 are verified by this checking
 
   * 1. If actual argument is an array name : 
           size(dummy_array) <= size(actual_array) (1)
   * 2. Actual argument is an array element name :
   *       size(dummy_array) <= size(actual_array)+1-subscript_value(array element) (2)

  * Remarks to simplify our checking :
  * 1. If the first k dimensions of the actual array and the dummy array are the same, 
  * we have (1) is equivalent with  
  * size_from_position(dummy_array,k+1) <= size_from_position(actual_array,k+1)
  *
  * 2. If the first k dimensions of the actual array and the dummy array are the same, 
  * and the first k subscripts of the array element are equal with their 
  * correspond lower bounds (column-major order), we have (2) is equivalent with:
  *
  * size_from_position(dummy_array,k+1) = size_from_position(actual_array,k+1) +1 
  *                     - subscript_value_from_position(array_element,k+1).

  ATTENTION : FORTRAN standard (15.9.3) says that an association of dummy and actual 
  arguments is valid only if the type of the actual argument is the same as the type 
  of the corresponding dummy argument. But in practice, not much program respect this 
  rule , so we have to multiply the array size by its element size in order to compare 
  2 arrays*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "conversion.h" /* for Psysteme_to_expression*/
#include "alias_private.h"
#include "instrumentation.h"
#include "transformations.h"

/* As we create checks with stop error message who tell us there are 
 * bound violations for which array in which call, the following 
 * typedef array_test permits us to create a sequence of tests 
 * for each statement more easier. 
 * The functions interprocedural_abc_call,
 * interprocedural_abc_expression return results of type 
 * array_test */

#define PREFIX "$IABC"

typedef struct array_test 
{
  list arr;
  list exp; 
} array_test;

/* context data structure for interprocedural_abc newgen recursion */
typedef struct 
{
  persistant_statement_to_control map;
  stack uns;
} 
  interprocedural_abc_context_t, 
* interprocedural_abc_context_p;

static int number_of_added_tests;
static int number_of_bound_violations;
static int total_number_of_added_tests = 0;
static int total_number_of_bound_violations = 0;

static void initialize_interprocedural_abc_statistics()
{
    number_of_added_tests = 0;
    number_of_bound_violations = 0;
}

static void display_interprocedural_abc_statistics()
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
  total_number_of_added_tests = total_number_of_added_tests 
    + number_of_added_tests;
  total_number_of_bound_violations = total_number_of_bound_violations
    + number_of_bound_violations;
}

#define array_test_undefined ((array_test) {NIL,NIL})

static bool array_test_undefined_p(array_test x)
{
  if ((x.arr == NIL) && (x.exp == NIL))
    return true; 
  return false;
}

static array_test 
make_array_test(entity e,  expression exp)
{
  array_test retour =  array_test_undefined;
  if (!expression_undefined_p(exp))
    {
      retour.arr = gen_nconc( CONS(ENTITY,e,NIL),NIL);
      retour.exp = gen_nconc( CONS(EXPRESSION,exp,NIL),NIL);
    }
  return retour;
}

static array_test 
add_array_test(array_test retour, array_test temp)
{
  if (!array_test_undefined_p(temp))
    {
      if (array_test_undefined_p(retour)) 
	return temp;      
      /* If in temp.exp, there are expressions that exist 
       * in retour.exp, we don't have 
       * to add those expressions to retour.exp */
      while (!ENDP(temp.exp))
	{
	  expression exp = EXPRESSION(CAR(temp.exp));
	  if (!same_expression_in_list_p(exp,retour.exp)) 
	    {
	      retour.arr = gen_nconc(CONS(ENTITY,ENTITY(CAR(temp.arr)),NIL),
				     retour.arr);
	      retour.exp = gen_nconc(CONS(EXPRESSION,exp,NIL),retour.exp);
	    }
	  temp.arr = CDR(temp.arr);
	  temp.exp = CDR(temp.exp);
	}	     
    }  
  return retour;
}


static expression size_of_dummy_array(entity dummy_array,int i)
{
  variable dummy_var = type_variable(entity_type(dummy_array));
  list l_dummy_dims = variable_dimensions(dummy_var);
  int num_dim = gen_length(l_dummy_dims),j;
  expression e = expression_undefined;
  for (j=i+1; j<= num_dim; j++)
    {
      dimension dim_j = find_ith_dimension(l_dummy_dims,j);
      expression lower_j = dimension_lower(dim_j);
      expression upper_j = dimension_upper(dim_j);
      expression size_j;
      if (expression_constant_p(lower_j) && (expression_to_int(lower_j)==1))
	size_j = copy_expression(upper_j);
      else 
	{
	  size_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,upper_j,lower_j);
	  size_j = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					       copy_expression(size_j),int_to_expression(1));
	}
      if (expression_undefined_p(e))
	e = copy_expression(size_j);
      else
	e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,e,size_j);  
    }
  ifdebug(2)
    {
      fprintf(stderr, "\n Size of dummy array: \n");
      print_expression(e);
    }
  return e;
}

static expression expression_less_than_in_context(expression e1, expression e2, 
						  transformer context)
{
  /*This function returns a true expression if (e1 < e2) = TRUE
                                 expression undefined if (e1 < e2) = FALSE
				 a test e1 < e2*/
  normalized n1 = NORMALIZE_EXPRESSION(e1);
  normalized n2 = NORMALIZE_EXPRESSION(e2);
  ifdebug(3) 
    {	  
      fprintf(stderr, "\n First expression e1: ");    
      print_expression(e1);	
      fprintf(stderr, "\n Second expression e2: ");    
      print_expression(e2);
      fprintf(stderr, " \n e1 less e2 wrt to the precondition : ");
      fprint_transformer(stderr,context, (get_variable_name_t)entity_local_name);
    }
  if (normalized_linear_p(n1) && normalized_linear_p(n2))
    {
      /* See if e is true or false if we have already the preconditions ps
       * if ps := ps + e 
       * ps = sc_strong_normalize3(ps)
       * if (ps = sc_empty) => not feasible, no bound violation
       * if (ps = sc_rn) => bound violation
       * else => test to put 
       */
      Pvecteur v1 = normalized_linear(n1);
      Pvecteur v2 = normalized_linear(n2);	      
      Pvecteur v_init = vect_substract(v1,v2);
      /* Trivial test : e1 = 1200, e2 = 1 => e1 < e2 = FALSE*/
      if (vect_constant_p(v_init))
	{
	  /* Tets if v_init < 0 */
	  if (VECTEUR_NUL_P(v_init)) return expression_undefined;; /* False => no bound violation*/
	  if (value_neg_p(val_of(v_init))) return make_true_expression();/* True => bound violation*/
	  if (value_posz_p(val_of(v_init))) return expression_undefined;/* False => no bound violation*/
	}
      else 
	{
	  /* Constraint form:  v +1 <= 0*/
	  Pvecteur v_one = vect_new(TCST,1);
	  Pvecteur v = vect_add(v_init,v_one);
	  Psysteme ps = predicate_system(transformer_relation(context));  
	  switch (sc_check_inequality_redundancy(contrainte_make(v), ps)) /* try fast check */
	    {
	    case 1: /* ok, e1<e2 is redundant wrt ps => bound violation*/
	      return make_true_expression();
	    case 2: /* ok, system {ps + {e1<e2}} is infeasible => no bound violation*/
	      return expression_undefined;
	    case 0: /* no result, try slow version */
	      {
		Psysteme sc = sc_dup(ps);
		Pvecteur pv_var = NULL; 
		Pbase b;
		ifdebug(3) 
		  {	  
		    fprintf(stderr, " \n System before add inequality: ");
		    sc_fprint(stderr,sc,(char * (*)(Variable)) entity_local_name);
		  }
		sc_constraint_add(sc, contrainte_make(v), false);
		ifdebug(3) 
		  {	  
		    fprintf(stderr, " \n System after add inequality: ");
		    sc_fprint(stderr,sc,(char * (*)(Variable)) entity_local_name);
		  }
		sc  = sc_strong_normalize2(sc);
		/* Attention : sc_strong_normalize3 returns SC_UNDEFINED for 103.su2cor*/
		ifdebug(3) 
		  {	  
		    fprintf(stderr, " \n System after strong normalize 2: ");
		    sc_fprint(stderr,sc,(char * (*)(Variable)) entity_local_name);
		  }
		if (SC_UNDEFINED_P(sc) || sc_empty_p(sc))
		  return expression_undefined;	
		if (sc_rn_p(sc))
		  return make_true_expression();
		/* Before using the system, we have to porject variables such as V#init from PIPS*/	
		b = sc->base;
		for(; !VECTEUR_NUL_P(b);b = b->succ)
		  {
		    entity e = (entity) vecteur_var(b);
		    if (old_value_entity_p(e))
		      vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
		  }
		if (pv_var!=NULL)
		  {
		    /* There are V#init variables */
		    ifdebug(3) 
		      {	  
			fprintf(stderr, " \n System before #init variables projection: ");
			sc_fprint(stderr,sc,(char * (*)(Variable)) entity_local_name);
		      }
		    sc = my_system_projection_along_variables(sc, pv_var);  
		    ifdebug(3) 
		      {	  
			fprintf(stderr, " \n System after #init variables projection: ");
			sc_fprint(stderr,sc,(char * (*)(Variable)) entity_local_name);
		      }
		    vect_rm(pv_var);  
		    if (!SC_UNDEFINED_P(sc))		  
		      {
			// the projection is exact		
			if (sc_rn_p(sc))
			  // the system is trivial true, there are bounds violation
			  return make_true_expression();
			if (sc_empty_p(sc))
			  // system is infeasible, no bound violations
			  return expression_undefined;
			return Psysteme_to_expression(sc);
		      }
		    // else : the projection is not exact => return the e1.LT.e2
		  }
		else 
		  return Psysteme_to_expression(sc);
	      }
	    }
	}
    }
  return lt_expression(e1,e2);
}

static expression interprocedural_abc_arrays(call c, entity actual_array, 
					     entity dummy_array, 
					     list l_actual_ref, statement s)
{
  expression retour = expression_undefined;
  expression dummy_array_size;
  int same_dim = 0;  
  transformer prec = load_statement_precondition(s);
  transformer context;
  if (statement_weakly_feasible_p(s))
    context = formal_and_actual_parameters_association(c,transformer_dup(prec));
  else 
    /* If statement is unreachable => do we need check here ?*/
    context = formal_and_actual_parameters_association(c,transformer_identity());  
  /* Compute the number of same dimensions of the actual array, dummy array 
     and actual array element, based on association information, common variables,  
     preconditions for more informations) */
  while (same_dimension_p(actual_array,dummy_array,l_actual_ref,same_dim+1,context))
    same_dim ++;
  ifdebug(2)
    fprintf(stderr, "\n Number of same dimensions : %d \n",same_dim);  
  dummy_array_size = size_of_dummy_array(dummy_array,same_dim); 
  if (!expression_undefined_p(dummy_array_size))
    {
      /* same_dim < number of dimension of the dummy array*/
      entity current_callee = call_function(c);
      ifdebug(2)
	{
	  fprintf(stderr, "\n Dummy array size before translation:\n");
	  print_expression(dummy_array_size);
	}
      /* translate the size of dummy array from the current callee to the 
	 frame of current module */
      dummy_array_size = translate_to_module_frame(current_callee,get_current_module_entity(),
						   dummy_array_size,c);
      ifdebug(2)
	{
	  fprintf(stderr, "\n Dummy array size after translation: \n");
	  print_expression(dummy_array_size);
	}
      if (!expression_undefined_p(dummy_array_size))
	{
	  /* The size of the dummy array is translated to the caller's frame*/
	  expression actual_array_size = size_of_actual_array(actual_array,l_actual_ref,same_dim); 
	  /* As the size of the dummy array is translated, we need only precondition of the call
	     in the current context*/
	  /* Now, compare the element size of actual and dummy arrays*/
	  basic b_dummy = variable_basic(type_variable(entity_type(dummy_array)));
	  basic b_actual = variable_basic(type_variable(entity_type(actual_array)));
	  int i_dummy = SizeOfElements(b_dummy);
	  int i_actual = SizeOfElements(b_actual);
	  if (!expression_undefined_p(actual_array_size))
	    {
	      /* same_dim < number of dimension of the actual array*/
	      if (i_dummy != i_actual)
		{
		  /* Actual and formal arguments have different types*/
		  actual_array_size = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,actual_array_size,
								  int_to_expression(i_actual));
		  dummy_array_size = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,dummy_array_size,
								 int_to_expression(i_dummy));
		}
	      if (!same_expression_p(dummy_array_size,actual_array_size))
		retour = expression_less_than_in_context(actual_array_size,dummy_array_size,prec);
	      /* Add to Logfile some information:*/
	      if (!expression_undefined_p(retour))
		user_log("%s\t%s\t%s\t%s\t%s\t%s\t%s\n",PREFIX,
			 entity_module_name(actual_array),
			 entity_local_name(actual_array),
			 words_to_string(words_expression(actual_array_size,NIL)),
			 entity_module_name(dummy_array),
			 entity_local_name(dummy_array),
			 words_to_string(words_expression(dummy_array_size,NIL)));
	    }
	  else
	    {
	      /* same_dim == number of dimension of the actual array
	       * If the size of the dummy array is greater than 1 => violation
	       * For example : actual array A(M,N), dummy array D(M,N,K) or D(M,N,1,K) */
	      if (i_dummy != i_actual)
		{
		  /* Actual and formal arguments have different types*/
		  dummy_array_size = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,dummy_array_size,
								 int_to_expression(i_dummy));
		  retour = expression_less_than_in_context(int_to_expression(i_actual),dummy_array_size,prec);
		}
	      else
		retour = expression_less_than_in_context(int_to_expression(1),dummy_array_size,prec);
	    }
	}
      else
	pips_user_warning("Cannot translate the size of dummy array %s into module %s's frame\n",
			  entity_local_name(dummy_array),entity_local_name(current_callee));
        /* This case is rare, because size of dummy array = constants or formal parameters or commons*/
    }
  /* else, dummy_array_size == expression_undefined because same_dim == number of 
     dimensions of dummy array, the inequation (1) or (2) is always true (suppose
     that there is no intraprocedural bound violation) */
  return retour;
}

static array_test interprocedural_abc_call(call c, statement s);
static array_test interprocedural_abc_expression(expression e, statement s);

static array_test interprocedural_abc_call(call c, statement s)
{
  /* More efficient : same call, same actual array ? */
  array_test retour = array_test_undefined;
  entity f = call_function(c);
  list l_args = call_arguments(c);
  /* Traverse the argument list => check for argument which is a function call*/
  MAP(EXPRESSION,e,
  {
    array_test temp = interprocedural_abc_expression(e,s);
    retour = add_array_test(retour,temp);	
  },	     
      l_args);  
  /* ATTENTION : we have to compute the callgraph before (make CALLGRAPH_FILE[%ALL])
     in order to have the initial value of f is recognized by PIPS as value code.
     If not => core dumped. I should add a user warning here !!!*/ 
  if (value_code_p(entity_initial(f)))
    {
      /* c is a call to a function or subroutine */
      int i =1;
      ifdebug(2)
	{
	  fprintf(stderr, "\n Call to a function/subroutine:");
	  fprintf(stderr, "%s ", entity_name(f));
	}  
      MAP(EXPRESSION,e,
      {
	if (array_argument_p(e))
	  {
	    reference r = expression_reference(e);
	    entity actual_array = reference_variable(r);
	    if (!assumed_size_array_p(actual_array))
	      {
		/* find corresponding formal argument in f : 
		   f -> value -> code -> declaration -> formal variable 
		   -> offset == i ?*/
		list l_actual_ref = reference_indices(r);
		list l_decls = code_declarations(entity_code(f));
		MAP(ENTITY, dummy_array,
		{
		  if (formal_parameter_p(dummy_array))
		    {
		      formal fo = storage_formal(entity_storage(dummy_array));
		      if (formal_offset(fo) == i)
			{
			  /* We have found the corresponding dummy argument*/
			  if (!assumed_size_array_p(dummy_array))
			    {
			      expression check = interprocedural_abc_arrays(c,actual_array,
								dummy_array,l_actual_ref,s);
			      if (!expression_undefined_p(check))
				{
				  array_test tmp = make_array_test(actual_array,check);
				  retour = add_array_test(retour,tmp);	
				}
			    }
			  else 
			    /* Formal parameter is an assumed-size array => what to do ?*/
			    pips_user_warning("Formal parameter %s is an assumed-size array\n",
					      entity_local_name(dummy_array));
			  break;
			}
		    }
		},
		    l_decls);
	      }
	    else
	      /* Actual argument is an assumed-size array => what to do ?*/
	      pips_user_warning("Actual argument %s is an assumed-size array\n",
				entity_local_name(actual_array));
	  }
	i++;
      },
	  l_args);
    }
  return retour;
}

static array_test interprocedural_abc_expression(expression e, statement s)
{
  array_test retour = array_test_undefined;
  if (expression_call_p(e))
    retour = interprocedural_abc_call(syntax_call(expression_syntax(e)),s);
  return retour;
}

static statement make_interprocedural_abc_tests(array_test at)
{
  list la = at.arr,le = at.exp;
  statement retour = statement_undefined;
  while (!ENDP(la))
    {
      entity a = ENTITY(CAR(la));
      expression e = EXPRESSION(CAR(le));
      string stop_message = strdup(concatenate("\"Bound violation: array ",
					  entity_name(a),"\"", NULL));
      string print_message = strdup(concatenate("\'BV array ",entity_name(a)," with ",
						words_to_string(words_syntax(expression_syntax(e),NIL)),
						"\'",print_variables(e), NULL));
      statement smt = statement_undefined;
      if (true_expression_p(e))
	{
	  /* There is a bound violation, we can return a stop statement immediately,
	     but for debugging purpose, it is better to display all bound violations */
	  number_of_bound_violations++;
	  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
	    smt = make_print_statement(print_message);
	  else
	    smt = make_stop_statement(stop_message);
	}
      else
	{
	  number_of_added_tests++;
	  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
	    smt = test_to_statement(make_test(e, make_print_statement(print_message),
					      make_block_statement(NIL)));
	  else
	    smt = test_to_statement(make_test(e, make_stop_statement(stop_message),
					      make_block_statement(NIL)));
	}
      if (statement_undefined_p(retour))
	retour = copy_statement(smt);
      else
	// always structured case
	insert_statement(retour,copy_statement(smt),false);
      la = CDR(la);
      le = CDR(le);
    }
  ifdebug(3) 
    {
      fprintf(stderr, " Returned statement:");
      print_statement(retour);
    }
  return retour;    
}

static void interprocedural_abc_insert_before_statement(statement s, statement s1,
				 interprocedural_abc_context_p context)
{
  /* If s is in an unstructured instruction, we must pay attention 
     when inserting s1 before s.  */
  if (bound_persistant_statement_to_control_p(context->map, s))
    {
      /* take the control that  has s as its statement  */      
      control c = apply_persistant_statement_to_control(context->map, s);
      if (stack_size(context->uns)>0)
	{	
	  /* take the unstructured correspond to the control c */
	  unstructured u = (unstructured) stack_head(context->uns);
	  control newc;
	  ifdebug(2) 
	    {
	      fprintf(stderr, "Unstructured case: \n");
	      print_statement(s);
	    }    	  
	  /* for a consistent unstructured, a test must have 2 successors, 
	     so if s1 is a test, we transform it into sequence in order 
	     to avoid this constraint. 	  
	     Then we create a new control for it, with the predecessors 
	     are those of c and the only one successor is c. 
	     The new predecessors of c are only the new control*/	  
	  if (statement_test_p(s1))
	    {
	      list seq = CONS(STATEMENT,s1,NIL);
	      statement s2=instruction_to_statement(make_instruction(is_instruction_sequence,
								       make_sequence(seq))); 
	      ifdebug(2) 
		{
		  fprintf(stderr, "Unstructured case, insert a test:\n");
		  print_statement(s1);
		  print_statement(s2);
		}      
	      newc = make_control(s2, control_predecessors(c), CONS(CONTROL, c, NIL));
	    }
	  else 
	    newc = make_control(s1, control_predecessors(c), CONS(CONTROL, c, NIL));
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
	}
      else
	// there is no unstructured (?)
	insert_statement(s,s1,true);
    }
  else
    // structured case 
    insert_statement(s,s1,true);     
}


static void interprocedural_abc_statement_rwt(statement s, interprocedural_abc_context_p context)
{
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);  
  switch(t)
    {
    case is_instruction_call:
      {	
	call c = instruction_call(i);
	array_test retour = interprocedural_abc_call(c,s);
	if (!array_test_undefined_p(retour))
	  {
	    statement seq = make_interprocedural_abc_tests(retour);	    
	    if (stop_statement_p(seq))	
	      user_log("Bound violation !!!!\n");
	    // insert the STOP or the new sequence of tests before the current statement
	    interprocedural_abc_insert_before_statement(s,seq,context);	
	  }
      	break;
      }
    case is_instruction_whileloop:
      {
	whileloop wl = instruction_whileloop(i); 
	// check for function call in the  while loop condition 
	expression e = whileloop_condition(wl);
	array_test retour  = interprocedural_abc_expression(e,s);
	if (!array_test_undefined_p(retour))
	  {
	    statement seq = make_interprocedural_abc_tests(retour);	    
	    if (stop_statement_p(seq))	
	      user_log("Bound violation !!!!\n");
	    // insert the STOP or the new sequence of tests before the current statement
	    interprocedural_abc_insert_before_statement(s,seq,context);	
	  } 
	break;
      }
    case is_instruction_test:
      {
	test it = instruction_test(i);
	// check for function call in the test condition
	expression e = test_condition(it);
	array_test retour  = interprocedural_abc_expression(e,s);
	if (!array_test_undefined_p(retour))
	  {
	    statement seq = make_interprocedural_abc_tests(retour);	
	    if (stop_statement_p(seq))
	      user_log("Bound violation !!!!\n");
	    // insert the STOP statement or the new sequence of tests before the current statement
	    interprocedural_abc_insert_before_statement(s,seq,context);	
	  } 
	break;
      }
    case is_instruction_sequence: 
    case is_instruction_loop:
      // There is not function call in loop's range
    case is_instruction_unstructured:
      /* because we use gen_recurse with statement domain, 
       * we don't have to check unstructured  instruction here*/
      break;
    default:
      pips_internal_error("Unexpected instruction tag %d ", t );
      break; 
    }  
}

static bool store_mapping(control c, interprocedural_abc_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return true;
}

static bool push_uns(unstructured u, interprocedural_abc_context_p context)
{
  stack_push((char *) u, context->uns);
  return true;
}

static void pop_uns(unstructured u, interprocedural_abc_context_p context)
{
  stack_pop(context->uns);
}

static void interprocedural_abc_statement(statement module_statement)
{
  interprocedural_abc_context_t context;
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);

  gen_context_multi_recurse(module_statement, &context,
			    unstructured_domain, push_uns, pop_uns,
			    control_domain, store_mapping, gen_null,
			    statement_domain, gen_true,interprocedural_abc_statement_rwt,
			    NULL);

  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns);
}

bool array_bound_check_interprocedural(const char* module_name)
{ 
  statement module_statement;  
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  module_statement= (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,module_name,true));
  set_ordering_to_statement(module_statement);
  debug_on("ARRAY_BOUND_CHECK_INTERPROCEDURAL_DEBUG_LEVEL");
  ifdebug(1)
    {
      debug(1, "Interprocedural array bound check","Begin for %s\n", module_name);
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
    }      
  initialize_interprocedural_abc_statistics();
  interprocedural_abc_statement(module_statement);
  display_interprocedural_abc_statistics();
  user_log("* The total number of added tests is %d *\n", 
	   total_number_of_added_tests );
  user_log("* The total number of bound violation is %d *\n", 
	   total_number_of_bound_violations );
  module_reorder(module_statement); /*as bound checks are added*/
  ifdebug(1)
    {
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
      debug(1, "Interprocedural array bound check","End for %s\n", module_name);
    }
  debug_off(); 
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,module_statement);
  reset_ordering_to_statement();
  reset_precondition_map();
  reset_current_module_entity();
  return true;
}






