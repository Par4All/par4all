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
 *                 BOTTOM-UP ARRAY BOUND CHECK VERSION
 *
 * -----------------------------------------------------------------
 *
 * This program takes as input the current module, adds array range checks
 * (lower and upper bound checks) to every statement that has one or more
 * array accesses. The output is the module with those tests added.

 * Assumptions : there is no write effect on the array bound expression.
 *
 * There was a test for write effect on bound here but I put it away (in
 * effect_on_array_bound.c) because it takes time to calculate the effect
 * but in fact this case is rare.
 *
 * NN 20/03/2002: Attention: tests generated for array whose bounds are modified are not correct
 * See Validation/ArrayBoundCheck/BU_Effect.f

 * Solution : reput the test effect_on_array_bound, instrument the code with
 * SUBROUTINE EFFECT(A,N)
 * REAL A(N)
 * N_PIPS = N
 * N = 2*N
 * DO I =1,N
 *    IF (I. GT. N_PIPS .OR I. LT> 1) STOP
 *    A(I) = I
 * ENNDO
 *
 * Question : other analyses (regions,.. ) are correct ?
 *
 *
 * Version: change the structure of test:
 * IF (I.LT.lower) THEN STOP "Bound violation ..."
 * IF (I.GT.upper) THEN STOP "Bound violation ..."
 * Add statistics
 * Version : Modify the Implied-DO process
 */

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
#include "text-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "instrumentation.h"

/* As we create checks with stop error message who tell us there are bound
 * violations for which array, on which dimension, which bound (lower or upper),
 * the following typedef array_dimension_bound_test permits us to create a
 * sequence of tests for each statement more easier.
 *
 * The functions bottom_up_abc_call,
 * bottom_up_abc_reference,
 * bottom_up_abc_expression return results of type
 * array_dimension_bound_test */
typedef struct array_dimension_bound_test
{
  list arr;
  list dim;
  list bou;
  list exp;
} array_dimension_bound_test;

/* Data structure to support abc Implied DO*/
typedef struct Index_range
{
  list ind;
  list ran;
} Index_range;

/* context data structure for bottom_up_abc newgen recursion */
typedef struct
{
  persistant_statement_to_control map;
  stack uns;
}
  bottom_up_abc_context_t,
* bottom_up_abc_context_p;

#define array_dimension_bound_test_undefined ((array_dimension_bound_test) {NIL,NIL,NIL,NIL} )
#define Index_range_undefined ((Index_range) {NIL,NIL})

/* Statistic variables: */

static int number_of_bound_checks;
static int number_of_added_checks;
static int number_of_bound_violations;

static void initialize_bottom_up_abc_statistics()
{
    number_of_bound_checks = 0;
    number_of_added_checks = 0;
    number_of_bound_violations = 0;
}

static void display_bottom_up_abc_statistics()
{
  if (number_of_bound_checks > 0)
    {
      user_log("* There are %d array bound checks *\n",
	       number_of_bound_checks*2);
    }
  if (number_of_added_checks > 0)
    user_log("* There %s %d array bound check%s added *\n",
	     number_of_added_checks > 1 ? "are" : "is",
	     number_of_added_checks,
	     number_of_added_checks > 1 ? "s" : "");

  if (number_of_bound_violations > 0)
    user_log("* There %s %d bound violation%s *\n",
	     number_of_bound_violations > 1 ? "are" : "is",
	     number_of_bound_violations,
	     number_of_bound_violations > 1 ? "s" : "");
}

static bool array_dimension_bound_test_undefined_p(array_dimension_bound_test x)
{
  if ((x.arr == NIL) && (x.dim == NIL) && (x.bou == NIL) && (x.exp == NIL))
    return true;
  return false;
}

string int_to_dimension(int i)
{
  switch(i)
    {
    case 0 :
      return "wrt common size";
    case 1 :
      return "1st dimension";
    case 2 :
      return "2nd dimension";
    case 3 :
      return "3rd dimension";
    case 4 :
      return "4th dimension";
    case 5 :
      return "5th dimension";
    case 6 :
      return "6th dimension";
    case 7 :
      return "7th dimension";
    default:
      return "Over 7!!!";
    }
}

/* This function returns true, if the array needs bound checks
 *                       false, otherwise.
 *
 * If the arrays are not created by user, for example the arrays
 * of Logical Units : LUNS, END_LUNS, ERR_LUNS, ... in the
 * IO_EFFECTS_PACKAGE_NAME, we don't have to check array references
 * for those arrays.
 * Maybe we have to add other kinds of arrays, not only those in _IO_EFFECTS_ */

bool array_need_bound_check_p(entity e)
{
  const char* s = entity_module_name(e);
  if (strcmp(s,IO_EFFECTS_PACKAGE_NAME)==0)
    return false;
  return true;
}


static array_dimension_bound_test
make_true_array_dimension_bound_test(entity e, int i, bool low)
{
  array_dimension_bound_test retour;
  expression exp = make_true_expression();
  retour.arr = CONS(ENTITY,e,NIL);
  retour.dim = CONS(INT,i,NIL);
  retour.bou = CONS(BOOL,low,NIL);
  retour.exp = CONS(EXPRESSION,exp,NIL);
  return retour;
}

static array_dimension_bound_test
make_array_dimension_bound_test(entity e, int i, bool low, expression exp)
{
  array_dimension_bound_test retour =  array_dimension_bound_test_undefined;
  if (!expression_undefined_p(exp))
    {
      retour.arr = CONS(ENTITY,e,NIL);
      retour.dim = CONS(INT,i,NIL);
      retour.bou = CONS(BOOL,low,NIL);
      retour.exp = CONS(EXPRESSION,exp,NIL);
    }
  return retour;
}

static array_dimension_bound_test
add_array_dimension_bound_test(array_dimension_bound_test retour,
			       array_dimension_bound_test temp)
{
  if (!array_dimension_bound_test_undefined_p(temp))
    {
      if (array_dimension_bound_test_undefined_p(retour))
	{
	  pips_debug(3, "\n Add bound checks for array: %s",entity_local_name(ENTITY(CAR(temp.arr))));
	  return temp;
	}
      /* If in temp.exp, there are expressions that exist
	 in retour.exp, we don't have to add those expressions to retour.exp
	 If temp.exp is a true expression => add to list to debug several real bound violations*/
      while (!ENDP(temp.exp))
	{
	  expression exp = EXPRESSION(CAR(temp.exp));
	  pips_debug(3, "\n Add bound checks for array: %s",entity_local_name(ENTITY(CAR(temp.arr))));
	  if (true_expression_p(exp) || !same_expression_in_list_p(exp,retour.exp))
	    {
	      retour.arr = gen_nconc(CONS(ENTITY,ENTITY(CAR(temp.arr)),NIL),retour.arr);
	      retour.dim = gen_nconc(CONS(INT,INT(CAR(temp.dim)),NIL),retour.dim);
	      retour.bou = gen_nconc(CONS(BOOL,BOOL(CAR(temp.bou)),NIL),retour.bou);
	      retour.exp = gen_nconc(CONS(EXPRESSION,exp,NIL),retour.exp);
	    }
	  else
	    number_of_added_checks--;
	  temp.arr = CDR(temp.arr);
	  temp.dim = CDR(temp.dim);
	  temp.bou = CDR(temp.bou);
	  temp.exp = CDR(temp.exp);
	}
    }
  return retour;
}

/*****************************************************************************
   This function computes the subscript value of an array element

   DIMENSION A(l1:u1,...,ln:un)
   subscript_value(A(s1,s2,...,sn)) =
   1+(s1-l1)+(s2-l2)*(u1-l1+1)+...+ (sn-ln)*(u1-l1+1)*...*(u(n-1) -l(n-1)+1)

*****************************************************************************/

expression subscript_value(entity arr, list l_inds)
{
  expression retour = int_to_expression(1);
  if (!ENDP(l_inds))
    {
      variable var = type_variable(entity_type(arr));
      expression prod = expression_undefined;
      list l_dims = variable_dimensions(var);
      int num_dim = gen_length(l_inds),i;
      for (i=1; i<= num_dim; i++)
	{
	  dimension dim_i = find_ith_dimension(l_dims,i);
	  expression lower_i = dimension_lower(dim_i);
	  expression sub_i = find_ith_argument(l_inds,i);
	  expression upper_i = dimension_upper(dim_i);
	  expression size_i;
	  if ( expression_constant_p(lower_i) && (expression_to_int(lower_i)==1))
	    size_i = copy_expression(upper_i);
	  else
	    {
	      size_i = binary_intrinsic_expression(MINUS_OPERATOR_NAME,upper_i,lower_i);
	      size_i = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						   copy_expression(size_i),int_to_expression(1));
	    }
	  if (!same_expression_p(sub_i,lower_i))
	    {
	      expression sub_low_i = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
								 sub_i,lower_i);
	      expression elem_i;
	      if (expression_undefined_p(prod))
		elem_i = copy_expression(sub_low_i);
	      else
		elem_i = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
						     sub_low_i,prod);
	      retour = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						   retour, elem_i);
	    }
	  if (expression_undefined_p(prod))
	    prod = copy_expression(size_i);
	  else
	    prod = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
					       prod,size_i);
	}
    }
  ifdebug(4)
    {
      pips_debug(4,"\nSubscript value:");
      print_expression(retour);
    }
  return retour;
}


static array_dimension_bound_test bottom_up_abc_reference(reference r);
static array_dimension_bound_test bottom_up_abc_expression (expression e);
static array_dimension_bound_test bottom_up_abc_call(call cal);

/* The test is: 1 <= offset(e) + subscript_value(r) <= area_size(common)*/
static array_dimension_bound_test abc_with_allocation_size(reference r)
{
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  array_dimension_bound_test temp = array_dimension_bound_test_undefined;
  entity e = reference_variable(r);
  list inds = reference_indices(r);
  ram ra = storage_ram(entity_storage(e));
  entity sec = ram_section(ra);
  basic b = variable_basic(type_variable(entity_type(e)));
  int off = ram_offset(ra)/SizeOfElements(b); /* divided by the element size */
  int size = area_size(type_area(entity_type(sec)))/SizeOfElements(b); /* divided by the element size */
  int i;
  expression subval = subscript_value(e,inds);
  expression check = lt_expression(copy_expression(subval),int_to_expression(1-off));
  ifdebug(3)
    {
      pips_debug(3, "\n Lower bound check expression:");
      print_expression(check);
    }
  clean_all_normalized(check);
  i = trivial_expression_p(check);
  switch(i){
  case 1:
    {
      user_log("\n Bound violation on lower bound of array %s\n",entity_local_name(e));
      return make_true_array_dimension_bound_test(e,0,true);
    }
  case -1:
    break;
  case 0:
    {
      number_of_added_checks++;
      temp = make_array_dimension_bound_test(e,0,true,check);
      retour = add_array_dimension_bound_test(retour,temp);
      break;
    }
  }
  check = gt_expression(copy_expression(subval),int_to_expression(size-off));
  ifdebug(3)
    {
      pips_debug(3, "\n Upper bound check expression:");
      print_expression(check);
    }
  clean_all_normalized(check);
  i = trivial_expression_p(check);
  switch(i){
  case 1:
    {
      user_log("\n Bound violation on upper bound of array %s\n",entity_local_name(e));
      return make_true_array_dimension_bound_test(e,0,false);
    }
  case -1:
    break;
  case 0:
    {
      number_of_added_checks++;
      temp = make_array_dimension_bound_test(e,0,false,check);
      retour = add_array_dimension_bound_test(retour,temp);
      break;
    }
  }
  /*if the ith subscript index is also an array reference, we must check this reference*/
  for (i=1;i <= gen_length(inds);i++)
    {
      expression ith = find_ith_argument(inds,i);
      number_of_bound_checks++;
      temp = bottom_up_abc_expression(ith);
      retour = add_array_dimension_bound_test(retour,temp);
    }
  return retour;
}


static array_dimension_bound_test bottom_up_abc_reference(reference r)
{
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  ifdebug(3)
    {
      pips_debug(3, "\n Array bound check for reference:");
      print_reference(r);
    }
  if (array_reference_p(r))
    {
      entity e = reference_variable(r);
      if (array_need_bound_check_p(e))
      	{
	  /* In practice, bound violations often occur with arrays in a common, with the reason that
	     the allocated size of the common is not violated :-)))
	     So this property helps dealing with this kind of bad programming practice*/
	  if (variable_in_common_p(e) && get_bool_property("ARRAY_BOUND_CHECKING_WITH_ALLOCATION_SIZE"))
	    retour = abc_with_allocation_size(r);
	  else
	    {
	      list arraydims = variable_dimensions(type_variable(entity_type(e)));
	      list arrayinds = reference_indices(r);
	      dimension dimi = dimension_undefined;
	      expression ith, check = expression_undefined;
	      int i,k;
	      array_dimension_bound_test temp = array_dimension_bound_test_undefined;
	      for (i=1;i <= gen_length(arrayinds);i++)
		{
		  number_of_bound_checks++;
		  dimi = find_ith_dimension(arraydims,i);
		  ith = find_ith_argument(arrayinds,i);
		  check = lt_expression(copy_expression(ith),dimension_lower(copy_dimension(dimi)));
		  /* Call the function trivial_expression_p(check)
		   * + If the check expression is always TRUE:
		   *        there is certainly bound violation, we return retour
		   *        immediately: retour = ({e},{i},{bound},{.TRUE.})
		   * + If the check expression is always false, we don't have to add check to retour.
		   * + Otherwise, we have to add check to retour.*/
		  ifdebug(3)
		    {
		      pips_debug(3, "\n Lower bound check expression:");
		      print_expression(check);
		    }
		  clean_all_normalized(check);
		  k = trivial_expression_p(check);
		  switch(k){
		  case 1:
		    {
		      user_log("\n Bound violation on lower bound of array %s\n",entity_local_name(e));
		      return make_true_array_dimension_bound_test(e,i,true);
		    }
		  case -1:
		    break;
		  case 0:
		    {
		      number_of_added_checks++;
		      temp = make_array_dimension_bound_test(e,i,true,check);
		      retour = add_array_dimension_bound_test(retour,temp);
		      break;
		    }
		  }
		  /* If the dimension is unbounded , for example T(*), we cannot check the upper bound*/
		  if (!unbounded_dimension_p(dimi))
		    {
		      check = gt_expression(copy_expression(ith),dimension_upper(copy_dimension(dimi)));
		      ifdebug(3)
			{
			  pips_debug(3, "\n Upper bound check expression:");
			  print_expression(check);
			}
		      clean_all_normalized(check);
		      k = trivial_expression_p(check);
		      switch(k){
		      case 1:
			{
			  user_log("\n Bound violation on upper bound of array %s\n",entity_local_name(e));
			  return make_true_array_dimension_bound_test(e,i,false);
			}
		      case -1:
			break;
		      case 0:
			{
			  number_of_added_checks++;
			  temp = make_array_dimension_bound_test(e,i,false,check);
			  retour = add_array_dimension_bound_test(retour,temp);
			  break;
			}
		      }
		    }
		  /*if the ith subscript index is also an array reference, we must check this reference*/
		  temp = bottom_up_abc_expression(ith);
		  retour = add_array_dimension_bound_test(retour,temp);
		}
	    }
	}
    }
  return retour;
}

static array_dimension_bound_test
bottom_up_abc_base_reference_implied_do(reference re,
					expression ind,
					range ran)
{
  array_dimension_bound_test retour =  array_dimension_bound_test_undefined;
  entity ent = reference_variable(re);
  list arrayinds = reference_indices(re);
  list listdims = variable_dimensions(type_variable(entity_type(ent)));
  expression low = range_lower(ran);
  expression up = range_upper(ran);
  expression ith = expression_undefined;
  int i;
  for (i=1;i <= gen_length(arrayinds);i++)
    {
      number_of_bound_checks++;
      ith = find_ith_argument(arrayinds,i);
      if (expression_equal_p(ith,ind))
	{
	  int k;
	  dimension dimi = find_ith_dimension(listdims,i);
	  array_dimension_bound_test temp = array_dimension_bound_test_undefined;
	  expression check = expression_undefined;
	  // make expression e1.LT.lower
	  check = lt_expression(low,dimension_lower(copy_dimension(dimi)));
	  ifdebug(3)
	    {
	      pips_debug(3,"\n Lower bound check expression:");
	      print_expression(check);
	    }
	  clean_all_normalized(check);
	  k = trivial_expression_p(check);
	  switch(k){
	  case 1:
	    {
	      user_log("\n Bound violation on lower bound of array %s\n",entity_local_name(ent));
	      return make_true_array_dimension_bound_test(ent,i,true);
	    }
	  case -1:
	    break;
	  case 0:
	    {		  
	      number_of_added_checks++;
	      temp = make_array_dimension_bound_test(ent,i,true,check);
	      retour = add_array_dimension_bound_test(retour,temp);
	      break;
	    }
	  }
	  if (!unbounded_dimension_p(dimi))
	    {
	      check = gt_expression(up,dimension_upper(copy_dimension(dimi)));
	      ifdebug(3)
		{
		  pips_debug(3,  "Upper bound check expression:");
		  print_expression(check);
		}
	      clean_all_normalized(check);
	      k = trivial_expression_p(check);
	      switch(k){
	      case 1:
		{
		  user_log("\n Bound violation on upper bound of array %s\n",entity_local_name(ent));
		  return make_true_array_dimension_bound_test(ent,i,false);
		}
	      case -1:
		break;
	      case 0:
		{
		  number_of_added_checks++;
		  temp = make_array_dimension_bound_test(ent,i,false,check);
		  retour = add_array_dimension_bound_test(retour,temp);
		  break;
		}
	      }
	    }
	}
    }
  return retour;
}

static array_dimension_bound_test
bottom_up_abc_reference_implied_do(expression e, Index_range ir)
{
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  syntax s = expression_syntax(e);
  tag t = syntax_tag(s);
  switch (t){
  case is_syntax_range:  
    break;
  case is_syntax_call:
    break;
  case is_syntax_reference:
    {
      /* Treat array reference */
      reference re = syntax_reference(s);
      if (array_reference_p(re))
	{
	  entity ent = reference_variable(re);
	  if (array_need_bound_check_p(ent))
	    {
	      /* For example, we have array reference A(I,J)
		 with two indexes I and J and ranges 1:20 and 1:10, respectively*/
	      list list_index = ir.ind, list_range = ir.ran;
	      while (!ENDP(list_index))
		{
		  expression ind = EXPRESSION(CAR(list_index));
		  range ran = RANGE(CAR(list_range));
		  array_dimension_bound_test temp = array_dimension_bound_test_undefined;
		  temp = bottom_up_abc_base_reference_implied_do(re,ind,ran);
		  retour = add_array_dimension_bound_test(retour,temp);
		  list_index = CDR(list_index);
		  list_range = CDR(list_range);
		}
	    }
	}
      break;
    }
  }
  return retour;
}

static array_dimension_bound_test bottom_up_abc_expression_implied_do(expression e, Index_range ir);

static array_dimension_bound_test
bottom_up_abc_expression_implied_do(expression e, Index_range ir)
{
  /* An implied-DO is a call to an intrinsic function named IMPLIED-DO;
   * the first argument is the implied-DO variable (loop index)
   * the second one is a range
   * the remaining arguments are expressions to be written or references
   * to be read or another implied-DO
   * (dlist, i=e1,e2,e3)
   *
   * We treated only the case where e3=1 (or omitted), the bound tests is
   * IF (e1.LT.lower.OR.e2.GT.upper) STOP Bound violation
   * for statement READ *,(A(I),I=e1,e2)
   *
   * An Implied-DO can be occurred in a DATA statement or an input/output
   * statement. As DATA statement is in the declaration, not executable,
   * we do not need to pay attention to it.*/
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  list args = call_arguments(syntax_call(expression_syntax(e)));
  expression arg2 = EXPRESSION(CAR(CDR(args)));  /* range e1,e2,e3*/
  range r = syntax_range(expression_syntax(arg2));
  expression one_exp = int_to_expression(1);
  if (same_expression_p(range_increment(r),one_exp))
    {
      /* The increment is 1 */
      expression arg1 = EXPRESSION(CAR(args)); /* Implied-DO index*/
      args = CDR(CDR(args));
      MAP(EXPRESSION, expr,
      {
	array_dimension_bound_test temp = array_dimension_bound_test_undefined;
	Index_range new_ir;
	new_ir.ind = gen_nconc(CONS(EXPRESSION,arg1,NIL),ir.ind);
	new_ir.ran = gen_nconc(CONS(RANGE,r,NIL),ir.ran);
	if (expression_implied_do_p(expr))
	  temp = bottom_up_abc_expression_implied_do(expr,new_ir);
	else
	  /* normal expression */
	  temp = bottom_up_abc_reference_implied_do(expr,new_ir);
	retour = add_array_dimension_bound_test(retour,temp);
      },args);
    }
  else
    {
      /* increment <> 1,  we have to put a dynamic test function such as :
       * WRITE *, A(checkA(I), e1,e2,e3)
       * where checkA is a function that does the range checking for array A.
       * because it is not integer, the bound tests :
       *
       * if (e3.GT.0).AND.(e1.LT.lower.OR.e1+e3x[e2-e1/e3].GT.upper)
       *    STOP Bound violation
       * if (e3.LT.0).AND.(e1.GT.upper.OR.e1+e3x[e1-e2/e3].LT.lower)
       *    STOP Bound violation */
    }
  return retour;
}

static array_dimension_bound_test bottom_up_abc_expression (expression e)
{
  /* the syntax of an expression can be a reference, a range or a call*/
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  if (expression_implied_do_p(e))
    retour = bottom_up_abc_expression_implied_do(e,Index_range_undefined);
  else
    {
      syntax s = expression_syntax(e);
      tag t = syntax_tag(s);
      switch (t){
      case is_syntax_call:
	{
	  retour = bottom_up_abc_call(syntax_call(s));
	  break;
	}
      case is_syntax_reference:
	{
	  retour = bottom_up_abc_reference(syntax_reference(s));
	  break;
	}
      case is_syntax_range:
	/* There is nothing to check here*/
	break;
      }
    }
  return retour;
}

static bool expression_in_array_subscript(expression e, list args)
{
  MAP(EXPRESSION, exp,
  {
    syntax s = expression_syntax(exp);
    tag t = syntax_tag(s);
    switch (t){
    case is_syntax_call:
      {
	/* We have to consider only the case of Implied DO
	 * if e is in the lower or upper bound of implied DO's range or not */
	if (expression_implied_do_p(exp))
	  {
	    call c = syntax_call(s);
	    list cargs = call_arguments(c);
	    expression arg2 = EXPRESSION(CAR(CDR(cargs)));  /* range e1,e2,e3*/
	    range r = syntax_range(expression_syntax(arg2));
	    expression e1 = range_lower(r);
	    expression e2 = range_upper(r);
	    expression e3 = range_increment(r);
	    ifdebug(3)
	      {
		fprintf(stderr,"\n Implied DO expression:\n");
		print_expression(exp);
	      }
	    if (same_expression_p(e,e1)||same_expression_p(e,e2)||same_expression_p(e,e3))
	      return true;
	  }
	break;
      }
    case is_syntax_reference:
      {
	reference ref = syntax_reference(s);
	if (array_reference_p(ref))
	  {
	    list arrayinds = reference_indices(ref);
	    if (same_expression_in_list_p(e,arrayinds))
	      return true;
	  }
	break;
      }
    case is_syntax_range:
      break;
    }
  },args);
  return false;
}

static bool read_statement_with_write_effect_on_array_subscript(call c)
{
  entity func = call_function(c);
  if (strcmp(entity_local_name(func),READ_FUNCTION_NAME)==0)
    {
      list args = call_arguments(c);
      while (!ENDP(args))
	{
	  expression exp = EXPRESSION(CAR(args));
	  args = CDR(args);
	  if (expression_in_array_subscript(exp,args))
	    return true;
	}
    }
  return false;
}

static array_dimension_bound_test bottom_up_abc_call(call cal)
{
  list args = call_arguments(cal);
  array_dimension_bound_test retour = array_dimension_bound_test_undefined;
  /* We must check a special case:
   * The call is a READ statement with array reference whose index is
   * read in the same statement.
   *
   * Example : READ *,N, A(N) or READ *,N,(A(I),I=1,N) (implied DO expression)
   *
   * In these case, we have to put a dynamic check function such as :
   * READ *,N,(A(abc_check(I,1,10)),I=1,N),M
   * where abc_check is a function that does the range checking by using its parameters */

  if (read_statement_with_write_effect_on_array_subscript(cal))
    user_log("\n Warning : READ statement with write effect on array subscript. \n This case has not been treated yet :-( \n");
  else
    MAP(EXPRESSION, e,
    {
      array_dimension_bound_test temp = bottom_up_abc_expression(e);
      retour = add_array_dimension_bound_test(retour,temp);
    },args);
  return retour;
}

string print_variables(expression e)
{
  syntax s = expression_syntax(e);
  tag t = syntax_tag(s);
  string retour = "";
  switch (t){
  case is_syntax_range:
    break;
  case is_syntax_call:
    {
      call c = syntax_call(s);
      list args = call_arguments(c);
      MAP(EXPRESSION,exp,
      {
	retour = strdup(concatenate(retour,print_variables(exp),NULL));
      },args);
      break;
    }
  case is_syntax_reference:
    {
      reference ref = syntax_reference(s);
      retour = strdup(concatenate(retour,", \', ",words_to_string(words_reference(ref,NIL))," =\',",
				  words_to_string(words_reference(ref,NIL)),NULL));
      break;
    }
  default:
    {
      pips_internal_error("Unexpected expression tag %d ", t );
      break;
    }
  }
  return retour;
}
statement emit_message_and_stop(string stop_message)
{
  statement smt = statement_undefined;

  if(fortran_module_p(get_current_module_entity())) {
    //return make_stop_statement(message);
    smt  = make_stop_statement(stop_message);
  }
  else {
    /* Must be the C version */
    smt  = make_exit_statement(1, stop_message);
  }
  return smt;
}

static statement make_bottom_up_abc_tests(array_dimension_bound_test adt)
{
  list la = adt.arr,ld = adt.dim,lb = adt.bou,le = adt.exp;
  statement retour = statement_undefined;
  string string_delimiter =
    fortran_module_p(get_current_module_entity())?
    "\'" : "\"";
  while (!ENDP(la))
    {
      entity a = ENTITY(CAR(la));
      int d = INT(CAR(ld));
      bool b = BOOL(CAR(lb));
      expression e = EXPRESSION(CAR(le));
      string stop_message =
	strdup(concatenate(string_delimiter,
			   "Bound violation: array \"",
			   entity_user_name(a),"\", ",
			   int_to_dimension(d),bool_to_bound(b),
			   string_delimiter, NULL));
      string print_message =
	strdup(concatenate(string_delimiter,
			   "BV array \"",entity_user_name(a),"\", ",
			   int_to_dimension(d),bool_to_bound(b),"with ",
			   words_to_string(words_syntax(expression_syntax(e),NIL)),
			   string_delimiter,
			   print_variables(e), NULL));
      statement smt = statement_undefined;
      if (true_expression_p(e))
	{
	  number_of_bound_violations++;
	  /* There is a bound violation, we can return a stop statement immediately,
	     but for debugging purpose, it is better to display all bound violations */
	  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
	    //return make_print_statement(message);
	    smt = make_any_print_statement(print_message);
	  else {
	    smt = emit_message_and_stop(stop_message);
	  }
	}
      else
	{
	  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
	    smt = test_to_statement(make_test(e, make_print_statement(print_message),
					      make_block_statement(NIL)));
	  else
	    smt = test_to_statement(make_test(e, emit_message_and_stop(stop_message),
					      make_block_statement(NIL)));
	}
      if (statement_undefined_p(retour))
	retour = copy_statement(smt);
      else
	// always structured case
	insert_statement(retour,copy_statement(smt),false);
      la = CDR(la);
      ld = CDR(ld);
      lb = CDR(lb);
      le = CDR(le);
    }
  ifdebug(2)
    {
      pips_debug(2,"\n With array bound checks:");
      print_statement(retour);
    }
  return retour;
}

static void bottom_up_abc_insert_before_statement(statement s, statement s1,
				 bottom_up_abc_context_p context)
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
	  ifdebug(5)
	    {
	      pips_debug(5,"Unstructured case: \n");
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
	      ifdebug(5)
		{
		  pips_debug(5, "Unstructured case, insert a test:\n");
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
	  if (unstructured_entry(u)==c)
	    unstructured_entry(u) = newc;
	}
      else
	// there is no unstructured (?)
	insert_statement(s,s1,true);
    }
  else
    // structured case
    insert_statement(s,s1,true);
}

/* forward declaration */
static statement cstr_args_check(statement);

static void bottom_up_abc_statement_rwt(
   statement s,
   bottom_up_abc_context_p context)
{
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);
  statement stmt2;

  ifdebug(2)
    {
      pips_debug(2, "\n Current statement");
      print_statement(s);
    }
  switch(t)
    {
    case is_instruction_call:
      {
	call cal = instruction_call(i);
	array_dimension_bound_test adt;

        if ((stmt2 = cstr_args_check(s)) != statement_undefined) {
          printf("Insertion d'un statement ---->\n");
          bottom_up_abc_insert_before_statement(s, stmt2, context);
        }

	adt = bottom_up_abc_call(cal);
	if (!array_dimension_bound_test_undefined_p(adt))
	  { 
	    statement seq = make_bottom_up_abc_tests(adt);
	    bottom_up_abc_insert_before_statement(s,seq,context);
	  }
      	break;
      }
    case is_instruction_whileloop:
      {
	whileloop wl = instruction_whileloop(i); 
	// array bound check of while loop condition 
	expression e1 = whileloop_condition(wl);
	array_dimension_bound_test adt = bottom_up_abc_expression(e1);
	if (!array_dimension_bound_test_undefined_p(adt))
	  {
	    statement seq = make_bottom_up_abc_tests(adt);
	    bottom_up_abc_insert_before_statement(s,seq,context);
	  }
	break;
      }
    case is_instruction_test:
      {
	test it = instruction_test(i);
	// array bound check of the test condition
	expression e1 = test_condition(it);
	array_dimension_bound_test adt = bottom_up_abc_expression(e1);
	if (!array_dimension_bound_test_undefined_p(adt))
	  {
	    statement seq = make_bottom_up_abc_tests(adt);
	    bottom_up_abc_insert_before_statement(s,seq,context);
	  }
	break;
      }
    case is_instruction_sequence:
    case is_instruction_loop:
      // suppose that there are not array references in loop's range in norm
    case is_instruction_forloop:
      // suppose that there are not array references in loop's range in norm
    case is_instruction_unstructured:
      /* because we use gen_recurse with statement domain,
       * we don't have to check unstructured  instruction here*/
      break;
    default:
      pips_internal_error("Unexpected instruction tag %d ", t );
      break;
    }
}

static bool store_mapping(control c, bottom_up_abc_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return true;
}

static bool push_uns(unstructured u, bottom_up_abc_context_p context)
{
  stack_push((char *) u, context->uns);
  return true;
}

static void pop_uns(unstructured u, bottom_up_abc_context_p context)
{
  stack_pop(context->uns);
}

static void bottom_up_abc_statement(statement module_statement)
{
  bottom_up_abc_context_t context;
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);

  gen_context_multi_recurse(module_statement, &context,
      unstructured_domain, push_uns, pop_uns,
      control_domain, store_mapping, gen_null,
      statement_domain, gen_true,bottom_up_abc_statement_rwt,
		    NULL);

  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns);
}

bool array_bound_check_bottom_up(const char* module_name)
{
  statement module_statement;
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* Begin the dynamic array bound checking phase.
   * Get the code from dbm (true resource) */
  module_statement= (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_current_module_statement(module_statement);
  set_ordering_to_statement(module_statement);
  debug_on("ARRAY_BOUND_CHECK_BOTTOM_UP_DEBUG_LEVEL");
  ifdebug(1)
    {
      debug(1, "Array bound check","Begin for %s\n", module_name);
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
    }
  initialize_bottom_up_abc_statistics();
  bottom_up_abc_statement(module_statement);
  display_bottom_up_abc_statistics();
  /* Reorder the module, because the bound checks have been added */
  module_reorder(module_statement);
  ifdebug(1)
    {
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
      debug(1, "array bound check","End for %s\n", module_name);
    }
  debug_off();
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,module_statement);
  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_current_module_entity();
  return true;
}

/*
 * Insert checks for C string manipulation functions (str*).
 */

#define cf_ge_expression(e1, e2) \
    ( 1 ? binary_intrinsic_expression(">=", e1, e2) : ge_expression(e1, e2) )

#define cf_gt_expression(e1, e2) \
    ( 1 ? binary_intrinsic_expression(">", e1, e2) : gt_expression(e1, e2) )

#define cf_lt_expression(e1, e2) \
    ( 1 ? binary_intrinsic_expression("<", e1, e2) : lt_expression(e1, e2) )

#define cf_and_expression(e1, e2) \
    ( 1 ? binary_intrinsic_expression("&&", e1, e2) : and_expression(e1, e2) )

#define cf_or_expression(e1, e2) \
    ( 1 ? binary_intrinsic_expression("||", e1, e2) : or_expression(e1, e2) )

static expression
make_bin_expression(string name, expression e1, expression e2)
{
    entity add_ent = gen_find_entity(name);

    return make_call_expression(add_ent,
            CONS(EXPRESSION, e1, CONS(EXPRESSION, e2, NIL)));
}

#define make_add_expression(e1, e2) make_bin_expression("TOP-LEVEL:+", e1, e2)
#define make_sub_expression(e1, e2) make_bin_expression("TOP-LEVEL:-", e1, e2)
#define make_mul_expression(e1, e2) make_bin_expression("TOP-LEVEL:*", e1, e2)

static expression
make_strlen_expression(expression arg)
{
    entity strlen_ent;

    strlen_ent = FindOrCreateTopLevelEntity("strlen");
    return make_call_expression(strlen_ent, CONS(EXPRESSION, arg, NIL));
}

static statement
make_c_stop_statement(int abort_program)
{
    /*
     * XXX 'entity_intrinsic' not verified ...
     * Pourtant exit() et abort() sont d�clar�s ds bootstrap.c
     */
#if 0
    list l = NIL;
    string stop_func;

    if (abort_program)
        stop_func = "TOP-LEVEL:abort";
    else
        stop_func = "TOP-LEVEL:exit";

    l = CONS(STATEMENT, make_call_statement(stop_func,
                        CONS(EXPRESSION, int_to_expression(1), NIL),
                        entity_undefined, ""),
        l);

    /*
    if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE")) {
        l = CONS(STATEMENT, make_call_statement("TOP-LEVEL:printf",
                            CONS(EXPRESSION, string_to_expression(), NIL)
                            entity_undefined, ""),
            l);
    }
    */

    return make_block_statement(l);
#endif
    return emit_message_and_stop("Buffer overflow!");
}


/*
 * entity_constant_string_size:
 *
 *  If the given entity is representing a constant string, this
 *  function will return it's length as a positive integer, else
 *  -1 will be returned.
 */
static int
entity_constant_string_size(entity e)
{
     basic b;
     type t, result;
     value v;
     constant c;

     t = entity_type(e);
     if (type_functional_p(t)
        && type_variable_p((result = functional_result(type_functional(t))))
        && basic_string_p((b = variable_basic(type_variable(result))))
        && value_constant_p((v = basic_string(b)))
        && constant_int_p((c = value_constant(v))))
         return constant_int(c);
     return -1;
}

/*
 * entity_size_uname:
 *
 *  Return a string usable as a unique global variable name for
 *  the given entity.
 */
static string
entity_size_uname(entity e)
{
    static const char  prefix []= "PIPS_size_";
    string name = entity_name(e);
    string uname;
    int i, len = strlen(name);

    if ((uname = malloc(len + 1 + sizeof(prefix))) == NULL) {
        perror("malloc");
        exit(1);
    }
    sprintf(uname, "%s%s",prefix, name);
    len = strlen(uname);
    for (i = 0; i < len; i++) {
        if (uname[i] == ':' || uname[i] == '-') {
            uname[i] = '_';
        }
    }
    return uname;
}

/*
 * expression_try_find_size:
 *
 * Try to find the storage size available for the given expression.
 */
static expression
expression_try_find_size(expression expr)
{
    syntax syn = expression_syntax(expr);

    /*
     * Case of a reference.
     *
     * Try to find the size of the corresponding buffer.
     */
    if (expression_reference_p(expr)) {
        reference ref = syntax_reference(syn);
        entity e = reference_variable(ref);
        int s;

        if (type_variable_p(entity_type(e))) {
            variable var = type_variable(entity_type(e));
            entity size_holder;
            string e_uname;
            list dims = variable_dimensions(var);
            list acc = reference_indices(ref);

            while (1) {
                if (ENDP(acc) && !ENDP(dims)) {
                    /*
                     * We found the upper bound of the given
                     * expression, we need to add one to get the
                     * size of the buffer
                     */
                    expression upper_bound =
                            dimension_upper(DIMENSION(CAR(dims)));

                    if (expression_constant_p(upper_bound)) {
                        return int_to_expression(1
                                    + expression_to_int(upper_bound));
                    } else
                        return make_add_expression(upper_bound,
                                                int_to_expression(1));
                } else if (!ENDP(acc) && !ENDP(dims)) {
                    acc = CDR(acc);
                    dims = CDR(dims);
                }
                else
                    break;
            }
            /*
             * The memory pointed by the variable was probably
             * allocated by malloc() (or a similar function).
             */
            e_uname = entity_size_uname(e);
            size_holder = FindOrCreateEntity("main",e_uname);

            free(e_uname);
            if (size_holder == entity_undefined)
                return expression_undefined;
            else
                return make_entity_expression(size_holder, NIL);
        }
        /* The expression is a string constant */
        else if ((s = entity_constant_string_size(e)) >= 0)
            return int_to_expression(s);
    }

    /*
     * The expression is a function call.
     *
     * Only two forms are acceptable here:  base + x, or base - y.
     */
    else if (expression_call_p(expr)) {
        call cal = syntax_call(syn);
        entity fun = call_function(cal);
        string fun_name = entity_name(fun);
        list args = call_arguments(cal);
        expression base, base_size, x;

        if (strcmp(fun_name, "TOP-LEVEL:+C") == 0) {
            base = EXPRESSION(CAR(args));
            x = EXPRESSION(CAR(CDR(args)));
            base_size = expression_try_find_size(base);
            if (base_size == expression_undefined)
                return expression_undefined;
            else
                return make_sub_expression(base_size, x);
        } else // XXX traiter le cas de fonctions imbriqu�es !
            return expression_undefined;
    }

    return expression_undefined;
}

/*
 * expression_try_find_string_size:
 *
 *  Try to find the size of the string pointed by the given
 *  expression.  The function will return an expression holding
 *  the length of the string in the case of a constant string or
 *  a call to the strlen() function in the other cases.
 *
 *  The size returned does not include the trailing NUL
 *  character.
 */
static expression
expression_try_find_string_size(expression expr)
{
    entity e;
    int s;

    e = reference_variable(expression_reference(expr));
    if (e != entity_undefined && (s = entity_constant_string_size(e)) >= 0)
        return int_to_expression(s);
    else
        return make_strlen_expression(expr);
}

static expression
array_indices_check(expression expr)
{
    reference ref;
    entity e;
    variable var;
    list dims, acc;
    expression check = expression_undefined, checktmp, indice;

    if (!expression_reference_p(expr))
        return expression_undefined;
    else {
        ref = syntax_reference(expression_syntax(expr));
        e = reference_variable(ref);
        if (!type_variable_p(entity_type(e)))
            return expression_undefined;
    }
    var = type_variable(entity_type(e));
    dims = variable_dimensions(var);
    acc = reference_indices(ref);

    while (1) {
        if (!ENDP(acc) && !ENDP(dims)) {
            /*
             * We found the upper bound of the given
             * expression, we need to add one to get the
             * size of the buffer
             */
            expression upper_bound =
                    dimension_upper(DIMENSION(CAR(dims)));

            if (expression_constant_p(upper_bound)) {
                upper_bound = int_to_expression(1
                              + expression_to_int(upper_bound));
            } else
                upper_bound = make_add_expression(upper_bound,
                                    int_to_expression(1));

            /*
             * Create the check for the current dimension (checktmp):
             *      indice < upper_bound
             * Add it to the global check:
             *      check := check && checktmp
             */
            indice = EXPRESSION(CAR(acc));
            checktmp = cf_lt_expression(indice, upper_bound);
            if (check == expression_undefined)
                check = checktmp;
            else
                check = cf_and_expression(check, checktmp);

            acc = CDR(acc);
            dims = CDR(dims);
        } else
            break;
    }
    return check;
}

static expression
array_check_add(expression array_expr, expression expr)
{
    expression check;

    if ((check = array_indices_check(array_expr)) == expression_undefined)
        return expr;
    else
        return cf_and_expression(check, expr);
}

#define CHECK_NARGS(n)  if (nargs != (n)) return expression_undefined

/*
 * Check for memcpy(dst, src, n):
 *
 *  n > size(src) || n > size(dst)
 */
static expression
memcpy_check_expression(expression args[], int nargs)
{
    expression arg0_size_expr, arg1_size_expr;
    CHECK_NARGS(3);

    arg0_size_expr = expression_try_find_size(args[0]);
    arg1_size_expr = expression_try_find_size(args[1]);
    if (arg0_size_expr == expression_undefined
        || arg1_size_expr == expression_undefined)
        return expression_undefined;

    return cf_or_expression(
                cf_gt_expression(args[2], arg1_size_expr),
                cf_gt_expression(args[2], arg0_size_expr));
}

/*
 * Check for bcopy(src, dst, n):
 *
 *  n > size(src) || n > size(dst)  => same as memcpy()
 */
static expression
bcopy_check_expression(expression args[], int nargs)
{
    CHECK_NARGS(3);
    // The arguments order (for the two first) doesn't matter
    return memcpy_check_expression(args, nargs);
}

/*
 * Check for bcmp(b1, b2, n):
 *
 *  n > size(b1) || n > size(b2)  => same as memcpy()
 */
static expression
bcmp_check_expression(expression args[], int nargs)
{
    CHECK_NARGS(3);
    return memcpy_check_expression(args, nargs);
}

/*
 * Check for bzero(dst, n):
 *
 *  n > size(dst)
 */
static expression
bzero_check_expression(expression args[], int nargs)
{
    expression arg0_size_expr;

    CHECK_NARGS(2);

    arg0_size_expr = expression_try_find_size(args[0]);
    if (arg0_size_expr == expression_undefined)
        return expression_undefined;

    return cf_gt_expression(args[1], arg0_size_expr);
}

/*
 * Check for memcmp(b1, b2, n):
 *
 *  n > size(b1) || n > size(b2)    => same as memcpy()
 */
static expression
memcmp_check_expression(expression args[], int nargs)
{
    CHECK_NARGS(3);
    return memcpy_check_expression(args, nargs);
}

/*
 * Check for memmove(dst, src, n):
 *
 *  n > size(src) || n > size(dst) || abs(dst - src) < n
 *  => same as bcopy(src, dst, n)
 */
static expression
memmove_check_expression(expression args[], int nargs)
{
    expression tmp;
    CHECK_NARGS(3);

    /* Convert memmove(dst, src, n) -> bcopy(src, dest, n) */
    tmp = args[0];
    args[0] = args[1];
    args[1] = tmp;
    return bcopy_check_expression(args, nargs);
}

/*
 * Check for memset(dst, val, n):
 *
 *  n > size(dst)       => same as bzero().
 */
static expression
memset_check_expression(expression args[], int nargs)
{
    CHECK_NARGS(3);

    /* Convert memset(dst, b, n) -> bzero(dst, n) */
    args[1] = args[2];
    return bzero_check_expression(args, 2);
}

/*
 * Check for strcpy(dst, src):
 *
 *  strlen(src) >= size(dst)
 */
static expression
strcpy_check_expression(expression args[], int nargs)
{
    //statement smt;
    expression arg1_size_expr ;

    CHECK_NARGS(2);

    arg1_size_expr = expression_try_find_size(args[0]);
    if (arg1_size_expr == expression_undefined)
        return expression_undefined;
    /*
    expression arg2_size_expr;
    int arg2size;
    entity arg2ent;
    arg2ent = reference_variable(expression_reference(args[1]));

    if ((arg2size = entity_constant_string_size(arg2ent)) >= 0)
        arg2_size_expr = int_to_expression(arg2size);
    else
        arg2_size_expr = make_strlen_expression(args[1]);
        */

    return cf_ge_expression(expression_try_find_string_size(args[1]), arg1_size_expr);
}

/*
 * Check for strncpy(dst, src, n):
 *
 *  n >= size(dst)
 */
static expression
strncpy_check_expression(expression args[], int nargs)
{
    expression arg1_size_expr;

    CHECK_NARGS(3);

    arg1_size_expr = expression_try_find_size(args[0]);
    if (arg1_size_expr == expression_undefined)
        return expression_undefined;

    return cf_ge_expression(args[2], arg1_size_expr);
}

/*
 * Check for strcat(dst, src):
 *
 *  strlen(dst) + strlen(src) >= size(dst)
 */
static expression
strcat_check_expression(expression args[], int nargs)
{
    expression dst_size;

    CHECK_NARGS(2);

    dst_size = expression_try_find_size(args[0]);
    if (dst_size == expression_undefined)
        return expression_undefined;

    return cf_ge_expression(
        make_add_expression(expression_try_find_string_size(args[0]),
                            expression_try_find_string_size(args[1])),
        dst_size);
}

/*
 * Check for strncat(dst, src, n):
 *
 *  n >= size(dst) - strlen(dst)
 */
static expression
strncat_check_expression(expression args[], int nargs)
{
    expression dst_size;

    CHECK_NARGS(3);

    dst_size = expression_try_find_size(args[0]);
    if (dst_size == expression_undefined)
        return expression_undefined;
    return cf_ge_expression(args[2], make_sub_expression(dst_size,
                                    expression_try_find_string_size(args[0])));
}

/*
 * Check for sprintf(dst, fmt, ...):
 *
 *  snprintf(PIPS_tmpbuf, 1, fmt, ...) > size(dst)
 *
 * (snprintf return the size that would have been written in the
 *  case of an infinite buffer).
 */
static expression
sprintf_check_expression(expression args[], int nargs)
{
    int k;
    list snprintf_args = NIL;
    expression dst_size=expression_undefined, tmpbuf_expr;
    entity snprintf_ent, tmpbuf_ent;

    if (nargs < 2)
        return expression_undefined;

    if ((snprintf_ent = FindEntity(TOP_LEVEL_MODULE_NAME,"snprintf"))
                == entity_undefined)
        return expression_undefined;

    if ((tmpbuf_ent = FindEntity("main","PIPS_tmpbuf")) == entity_undefined)
        // XXX module "main" non correct
        tmpbuf_ent = make_scalar_entity("PIPS_tmpbuf", "main",
                                                make_basic_int(1));
    // XXX Il faut &PIPS_tmpbuf
    tmpbuf_expr = make_entity_expression(tmpbuf_ent, NIL);

    for (k = nargs - 1; k >= 2; k--)
        snprintf_args = CONS(EXPRESSION, args[k], snprintf_args);
    snprintf_args = CONS(EXPRESSION, tmpbuf_expr,
                    CONS(EXPRESSION, int_to_expression(1), snprintf_args));

    return cf_gt_expression(make_call_expression(snprintf_ent, snprintf_args),
                            dst_size);
}

/*
 * Check for snprintf(dst, n, fmt, ...):
 *
 *  n >= size(dst)      => same as strncpy(dst, src, n)
 */
static expression
snprintf_check_expression(expression args[], int nargs)
{
    if (nargs < 3)
        return expression_undefined;
    args[2] = args[1];

    return strncpy_check_expression(args, 3);
}

#undef CHECK_NARGS

static struct checkfn {
    string function;
    expression (*check_statement)(expression *, int);
} checkfns[] = {
    { "bcopy", bcopy_check_expression },
    { "bcmp", bcmp_check_expression },
    { "bzero", bzero_check_expression },
    { "memcmp", memcmp_check_expression },
    { "memcpy", memcpy_check_expression },
    { "memmove", memmove_check_expression },
    { "memset", memset_check_expression },
    { "strcpy", strcpy_check_expression },
    { "strncpy", strncpy_check_expression },
    { "strcat", strcat_check_expression },
    { "strncat", strncat_check_expression },
    { "sprintf", sprintf_check_expression },
    { "snprintf", snprintf_check_expression },
    { NULL, NULL }
};

/*
 * Add instrumentation to memory allocation instructions:
 *
 *  m = malloc(n);    =>    m = malloc(n);
 *                          PIPS_size_m = n;
 *
 * The relevant functions are:
 *  malloc(), calloc(), memalign(), realloc()
 */
static statement
alloc_instrumentation(expression args[], int nargs)
{
    expression size_expr, size_holder;
    entity ptr, size_holder_ent;
    string func_name, size_holder_name;
    call func_call;
    list func_args;
    reference ref;
    int func_nargs;

    if (nargs != 2) {
        return statement_undefined;
    }

    if (!expression_reference_p(args[0]) || !expression_call_p(args[1]))
        return statement_undefined;

    ref = syntax_reference(expression_syntax(args[0]));
    ptr = reference_variable(ref);

    /*
     * Don't try to instrument if the pointer is an array access.
     */
    if (reference_indices(ref) != NIL)
        return statement_undefined;

    func_call = syntax_call(expression_syntax(args[1]));
    func_name = entity_name(call_function(func_call));
    func_args = call_arguments(func_call);
    func_nargs = gen_length(func_args);

    if (strcmp(func_name, "TOP-LEVEL:malloc") == 0) {
        if (func_nargs != 1)
            return statement_undefined;
        size_expr = EXPRESSION(CAR(func_args));
    } else if (strcmp(func_name, "TOP-LEVEL:realloc") == 0
     || strcmp(func_name, "TOP-LEVEL:memalign") == 0) {
        if (func_nargs != 2)
            return statement_undefined;
        size_expr = EXPRESSION(CAR(CDR(func_args)));
    } else if (strcmp(func_name, "TOP-LEVEL:calloc") == 0) {
        if (func_nargs != 2)
            return statement_undefined;
        size_expr = make_call_expression(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
					 func_args);
    } else
        return statement_undefined;

    /*
     * Create a new variable to hold the size of the allocated
     * memory.
     */
    size_holder_name = entity_size_uname(ptr);
    size_holder_ent = FindEntity("main",
							   size_holder_name);
    if (size_holder_ent == entity_undefined) {
        size_holder_ent =
	  make_scalar_entity(size_holder_name,
			     // XXX ?? marche pas ...
			     //STATIC_AREA_LOCAL_NAME, make_basic_int(4));
			     "main", make_basic_int(4));
    }
    size_holder = make_entity_expression(size_holder_ent, NIL);

    return make_assign_statement(size_holder, size_expr);
}

static statement
cstr_args_check(statement s)
{
    instruction i = statement_instruction(s);
    statement astmt;
    expression expr, *argstab;
    call cal;
    list args;
    string func_name;
    int nargs = 0, k;
    struct checkfn *p;

    pips_assert("Statement is a call statmt.",
		instruction_tag(i) == is_instruction_call);

    cal = instruction_call(i);
    func_name = entity_name(call_function(cal));
    pips_debug(1, "FONCTION: %s\n", func_name);

    /* Convert the arguments list into an array */
    args = call_arguments(cal);
    nargs = gen_length(args);
    argstab = malloc(sizeof(expression) * nargs);
    for (k = 0; k < nargs; k++) {
        argstab[k] = EXPRESSION(CAR(args));
        args = CDR(args);
    }

    /* Add malloc(), free(), etc. instrumentation */
    if (strcmp(func_name, "TOP-LEVEL:=") == 0
       && (astmt = alloc_instrumentation(argstab, nargs)) != statement_undefined) {
        free(argstab);
        return astmt;
    /*
     * free(m);     =>      PIPS_size_m = 0;
     *                      free(m);
     */
    } else if (strcmp(func_name, "TOP-LEVEL:free") == 0) {
        entity arg_ent, size_holder;
        string e_uname;

        free(argstab);

        if (!expression_reference_p(argstab[0]))
            return statement_undefined;
        else {
            arg_ent = reference_variable(syntax_reference(
                                    expression_syntax(argstab[0])));
            if (!type_variable_p(entity_type(arg_ent)))
                return statement_undefined;
        }
        e_uname = entity_size_uname(arg_ent);
        size_holder = FindEntity("main", e_uname);
        free(e_uname);
        if (size_holder == entity_undefined)
            return statement_undefined;
        return make_assign_statement(make_entity_expression(size_holder, NIL),
                                        int_to_expression(0));
    }

    if (strncmp(func_name, "TOP-LEVEL:", 10) != 0) {
        free(argstab);
        return statement_undefined;
    }

    expr = expression_undefined;
    for (p = checkfns; p->function != NULL; p++) {
        if (strcmp(p->function, func_name + 10) == 0) {
            expr = (p->check_statement)(argstab, nargs);
            break;
        }
    }

    free(argstab);

    if (expr == expression_undefined) {
        /* The target languague is unknown: C or Fortran, NL at the
	   end or not? It will be up to the prettyprinter (FI).
	   FI+LD 21-Oct-2009 : added NL, apparently needed in Fortran too.
	*/
        put_a_comment_on_a_statement(s, strdup("CHECK WAS NOT CREATED\n"));
        return statement_undefined;
    }

    /*
     * Add checks for array indices when necessary.
     */
    for (k = 0; k < nargs; k++)
        expr = array_check_add(argstab[k], expr);
    return test_to_statement(make_test(expr,
                                       make_c_stop_statement(0),
                                       make_block_statement(NIL)));
}

