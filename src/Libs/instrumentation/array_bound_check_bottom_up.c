/* -----------------------------------------------------------------
 *
 *                 BOTTOM-UP ARRAY BOUND CHECK VERSION
 *
 * -----------------------------------------------------------------
 *
 * This program takes as input the current module, adds array range checks
 * (lower and upper bound checks) to every statement that has one or more 
 * array accesses. The output is the module with those tests added.
 *
 *
 * Hypotheses : there is no write effect on the array bound expression.
 *
 * There was a test for write effect on bound here but I put it away (in 
 * effect_on_array_bound.c) because it takes time to calculate the effect
 * but in fact this case is rare.
 *
 *
 * $Id$
 *

 * Version: change the structure of test: 
 * IF (I.LT.lower).OR.(I.GT.upper) THEN STOP "Bound violation ..."

 * Add statistics 
 
 * Version : Modify the Implied-DO process*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "linear.h"

#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"

#include "instrumentation.h"

/* As we create checks with stop error message who tell us there are 
 * bound violations for which array, on which dimension, the following 
 * typedef array_dimension_test permits us to create a sequence of tests 
 * for each statement more easier. 
 *
 * The functions bottom_up_abc_call,
 * bottom_up_abc_reference, 
 * bottom_up_abc_expression return results of type 
 * array_dimension_test */

typedef struct array_dimension_test 
{
  list arr;
  list dim;
  list exp; 
} array_dimension_test;

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


#define array_dimension_test_undefined ((array_dimension_test) {NIL,NIL,NIL} )
#define Index_range_undefined ((Index_range) {NIL,NIL})

/* Statistic variables: */

static int number_of_array_references;
static int number_of_useful_added_tests;
static int number_of_useless_tests_not_added;
static int number_of_statements_having_bound_violation;
static int total_number_of_array_references = 0;

static void initialize_bottom_up_abc_statistics()
{
    number_of_array_references = 0;
    number_of_useful_added_tests = 0;
    number_of_useless_tests_not_added = 0;
    number_of_statements_having_bound_violation = 0;
}

static void display_bottom_up_abc_statistics()
{
  if (number_of_array_references > 0) 
    {
      user_log("* There %s %d array reference%s (per dimension) *\n",
	       number_of_array_references > 1 ? "are" : "is",
	       number_of_array_references,
	       number_of_array_references > 1 ? "s" : "");	
      total_number_of_array_references = total_number_of_array_references + 
	                                 2*number_of_array_references;
    }
  
  if (number_of_useful_added_tests > 0) 
    user_log("* There %s %d array bound check%s added *\n",
	     number_of_useful_added_tests > 1 ? "are" : "is",
	     number_of_useful_added_tests,
	     number_of_useful_added_tests > 1 ? "s" : "");		

  if (number_of_useless_tests_not_added > 0) 
    user_log("* There %s %d useless array bound check%s not added *\n",
	     number_of_useless_tests_not_added > 1 ? "are" : "is",
	     number_of_useless_tests_not_added,
	     number_of_useless_tests_not_added > 1 ? "s" : "");		

  if (number_of_statements_having_bound_violation > 0) 
    user_log("* There %s %d statement%s having bounds violation *\n",
	     number_of_statements_having_bound_violation > 1 ? "are" : "is",
	     number_of_statements_having_bound_violation,
	     number_of_statements_having_bound_violation > 1 ? "s" : "");		
}

static bool array_dimension_test_undefined_p(array_dimension_test x)
{
  if ((x.arr == NIL) && (x.dim == NIL) && (x.exp == NIL))
    return TRUE; 
  return FALSE;
}

string int_to_dimension(int i)
{
  switch(i) 
    {
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

/* This function returns TRUE, if the array needs bound checks 
 *                       FALSE, otherwise.
 *
 * If the arrays are not created by user, for example the arrays
 * of Logical Units : LUNS, END_LUNS, ERR_LUNS, ... in the 
 * IO_EFFECTS_PACKAGE_NAME, we don't have to check array references 
 * for those arrays. 
 * Maybe we have to add other kinds of arrays, not only those in _IO_EFFECTS_ */

bool array_need_bound_check_p(entity e)
{ 
  string s = entity_module_name(e);
  if (strcmp(s,IO_EFFECTS_PACKAGE_NAME)==0)
    return FALSE; 
  return TRUE; 
}

/* This function returns the ith dimension of a list of dimensions */

dimension find_ith_dimension(list dims, int n)
{
  int i;
  pips_assert("find_ith_dimension", n > 0);
  for(i=1; i<n && !ENDP(dims); i++, POP(dims))
    ;
  if(i==n && !ENDP(dims))
    return DIMENSION(CAR(dims));
  
  return dimension_undefined;
}

static array_dimension_test 
make_true_array_dimension_test(entity e, int i)
{
  array_dimension_test retour;
  expression exp = make_true_expression();
  retour.arr = gen_nconc( CONS(ENTITY,e,NIL),NIL);
  retour.dim = gen_nconc(CONS(INT,i,NIL),NIL);
  retour.exp = gen_nconc( CONS(EXPRESSION,exp,NIL),NIL);
  return retour;
}

static array_dimension_test 
make_array_dimension_test(entity e, int i, expression exp)
{
  array_dimension_test retour =  array_dimension_test_undefined;
  if (!expression_undefined_p(exp))
    {
      retour.arr = gen_nconc( CONS(ENTITY,e,NIL),NIL);
      retour.dim = gen_nconc(CONS(INT,i,NIL),NIL);
      retour.exp = gen_nconc( CONS(EXPRESSION,exp,NIL),NIL);
    }
  return retour;
}

static array_dimension_test 
add_array_dimension_test(array_dimension_test retour, 
			 array_dimension_test temp)
{
  if (!array_dimension_test_undefined_p(temp))
    {
      if (array_dimension_test_undefined_p(retour)) 
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
	      retour.dim = gen_nconc(CONS(INT,INT(CAR(temp.dim)),NIL),retour.dim);
	      retour.exp = gen_nconc(CONS(EXPRESSION,exp,NIL),retour.exp);
	    }
	  else 
	    {
	      if (logical_operator_expression_p(exp))
		{
		  number_of_useful_added_tests = number_of_useful_added_tests - 2;
		  number_of_useless_tests_not_added = 
		    number_of_useless_tests_not_added + 2;
		}
	      else
		{
		  number_of_useful_added_tests--;
		  number_of_useless_tests_not_added++; 			
		}		      
	    }
	  
	  temp.arr = CDR(temp.arr);
	  temp.dim = CDR(temp.dim);
	  temp.exp = CDR(temp.exp);
	}	     
    }  
  return retour;
}

static array_dimension_test bottom_up_abc_reference(reference r);
static array_dimension_test bottom_up_abc_expression (expression e);
static array_dimension_test bottom_up_abc_call(call cal);

static array_dimension_test bottom_up_abc_reference(reference r)
{  
  array_dimension_test retour = array_dimension_test_undefined; 
  
  ifdebug(3) 
    {
      fprintf(stderr, "\n Array bound check for reference:");
      print_reference(r);      
    }
 
  if (array_reference_p(r))
    { 
      entity e = reference_variable(r);
      if (array_need_bound_check_p(e)) 
      	{	 
	  list arraydims = variable_dimensions(type_variable(entity_type(e)));
	  list arrayinds = reference_indices(r);
	  dimension dimi = dimension_undefined;
	  expression ith, checklow,checkup,exp = expression_undefined;
	  int i,t,k;
	  array_dimension_test temp = array_dimension_test_undefined;
	  for (i=1;i <= gen_length(arrayinds);i++)
	    { 
	      number_of_array_references++;

	      dimi = find_ith_dimension(arraydims,i);
	 
	      ith = find_ith_argument(arrayinds,i);
 	  
	      checklow = lt_expression( copy_expression(ith),
					dimension_lower(copy_dimension(dimi)));

	      /* Call the function trivial_expression_p(checklow)
	       * + If the checklow expression is always TRUE: 
	       *        there is certainly bound violation, we return retour 
	       *        immediately: retour = ({e},{i},{.TRUE.})
	       * + If the checklow expression is always FALSE, 
	       *        we don't have to add it to exp. 
	       * + Otherwise, we have to add it to exp.*/
	
      	      ifdebug(3) {
		fprintf(stderr, "Checklow expression:");
		print_expression(checklow);
	      }

	      t = trivial_expression_p(checklow);
	      switch(t){
	      case 1:
		{
		  retour = make_true_array_dimension_test(e,i);
		  return retour;
		}
	      case -1: // do not add checklow to exp
		{
		  number_of_useless_tests_not_added++;
		  /* If the dimension is unbounded , for example T(*), we don't
		   * have to check the upper bound*/
		  if (unbounded_dimension_p(dimi))
		    {
		      number_of_useless_tests_not_added++;
		      exp = expression_undefined;
		    }
		  else 
		    {			
		      checkup = gt_expression( copy_expression(ith),
					       dimension_upper(copy_dimension(dimi)));
		      ifdebug(3) {
			fprintf(stderr, "Checkup expression:");
			print_expression(checkup);
		      }
		      k = trivial_expression_p(checkup);
		      switch(k){
		      case 1:
			{
			  retour = make_true_array_dimension_test(e,i);
			  return retour;
			}
		      case -1: // do not add checkup to exp
			{
			  number_of_useless_tests_not_added++;
			  exp = expression_undefined;
			  break;
			}
		      case 0: // add checkup to exp
			{
			  number_of_useful_added_tests++;
			  exp = copy_expression(checkup);
			  break;
			}
		      }
		    }
		  break;
		}
	      case 0: // add checklow to exp
		{		  		
		  number_of_useful_added_tests++;
		  /* If the dimension is unbounded , for example T(*), we don't
		   * have to check the upper bound*/
		  if (unbounded_dimension_p(dimi)) 
		    {
		      number_of_useless_tests_not_added++;
		      exp = copy_expression(checklow);	
		    }
		  else 
		    {			
		      checkup = gt_expression( copy_expression(ith),
					       dimension_upper(copy_dimension(dimi)));
		      ifdebug(3) {
			fprintf(stderr, "Checkup expression:");
			print_expression(checkup);
		      }
		      k = trivial_expression_p(checkup);
		      switch(k){
		      case 1:
			{
			  retour = make_true_array_dimension_test(e,i);
			  return retour;
			}
		      case -1: // do not add checkup to exp
			{
			  number_of_useless_tests_not_added++;
			  exp = copy_expression(checklow);		
			  break;
			}
		      case 0: // add checkup to exp
			{
			  number_of_useful_added_tests++;
			  exp = or_expression( checklow,checkup);
			  break;
			}
		      }			
		    }
		  break;
		}
	      }	
	      ifdebug(3) {
		fprintf(stderr, "Test expression:");
		print_expression(exp);
	      }
	      
	      temp = make_array_dimension_test(e,i,exp);
	      retour = add_array_dimension_test(retour,temp);
	      
	      /* if the ith subscript index is also an array reference, 
	       * we must check this reference*/	          
	      temp = bottom_up_abc_expression(ith);
	      retour = add_array_dimension_test(retour,temp);
	    }			    
	}	
    } 
  return retour;
}

static array_dimension_test 
bottom_up_abc_base_reference_implied_do(reference re, 
					expression ind, 
					range ran)
{
  array_dimension_test retour =  array_dimension_test_undefined;
  entity ent = reference_variable(re);
  list arrayinds = reference_indices(re);
  list listdims = variable_dimensions(type_variable(entity_type(ent)));  
  expression low = range_lower(ran);
  expression up = range_upper(ran);
  expression ith = expression_undefined;
  int i;
  for (i=1;i <= gen_length(arrayinds);i++)
    {    
      number_of_array_references++;
      ith = find_ith_argument(arrayinds,i);
      if (expression_equal_p(ith,ind))
	{
	  int t,k;
	  dimension dimi = find_ith_dimension(listdims,i);
	  array_dimension_test temp = array_dimension_test_undefined;
	  expression exp = expression_undefined, checklow, checkup;
	  // make expression e1.LT.lower
	  checklow = lt_expression(low,dimension_lower(copy_dimension(dimi)));
	  ifdebug(3) 
	    {
	      fprintf(stderr, "Checklow expression:");
	      print_expression(checklow);
	    }		  
	  t = trivial_expression_p(checklow);
	  switch(t){
	  case 1:
	    {
	      retour = make_true_array_dimension_test(ent,i);
	      return retour;
	    }
	  case -1: // do not add checklow to exp
	    {
	      number_of_useless_tests_not_added++;
	      if (unbounded_dimension_p(dimi))
		{
		  number_of_useless_tests_not_added++;
		  exp = expression_undefined;
		}
	      else 
		{			
		  // make expression e2.GT.upper
		  expression checkup = gt_expression(up,dimension_upper(copy_dimension(dimi)));
		  ifdebug(3) 
		    {
		      fprintf(stderr, "Checkup expression:");
		      print_expression(checkup);
		    }
		  k = trivial_expression_p(checkup);
		  switch(k){
		  case 1:
		    {
		      retour = make_true_array_dimension_test(ent,i);
		      return retour;
		    }
		  case -1: // do not add checkup to exp
		    {
		      number_of_useless_tests_not_added++;
		      exp = expression_undefined;
		      break;
		    }
		  case 0: // add checkup to exp
		    {
		      number_of_useful_added_tests++;
		      exp = copy_expression(checkup);
		      break;
		    }
		  }
		}
	      break;
	    }
	  case 0: // add checklow to exp
	    {		  		
	      number_of_useful_added_tests++;
	      if (unbounded_dimension_p(dimi)) 
		{
		  number_of_useless_tests_not_added++;
		  exp = copy_expression(checklow);	
		}
	      else 
		{			
		  checkup = gt_expression(up,dimension_upper(copy_dimension(dimi)));
		  ifdebug(3) 
		    {
		      fprintf(stderr, "Checkup expression:");
		      print_expression(checkup);
		    }
		  k = trivial_expression_p(checkup);
		  switch(k){
		  case 1:
		    {
		      retour = make_true_array_dimension_test(ent,i);
		      return retour;
		    }
		  case -1: // do not add checkup to exp
		    {
		      number_of_useless_tests_not_added++;
		      exp = copy_expression(checklow);		
		      break;
		    }
		  case 0: // add checkup to exp
		    {
		      number_of_useful_added_tests++;
		      exp = or_expression( checklow,checkup);
		      break;
		    }
		  }			
		}
	      break;
	    }
	  }	
	  ifdebug(3) 
	    {
	      fprintf(stderr, "Test expression:");
	      print_expression(exp);
	    }
	  temp = make_array_dimension_test(ent,i,exp);
	  retour = add_array_dimension_test(retour,temp);
	}
    } 
  return retour;
}

static array_dimension_test 
bottom_up_abc_reference_implied_do(expression e, Index_range ir)
{
  array_dimension_test retour =  array_dimension_test_undefined;
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
		  array_dimension_test temp = array_dimension_test_undefined;
		  temp = bottom_up_abc_base_reference_implied_do(re,ind,ran);
		  retour = add_array_dimension_test(retour,temp);
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

static array_dimension_test bottom_up_abc_expression_implied_do(expression e, Index_range ir);

static array_dimension_test 
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
   *
   * An Implied-DO can be occurred in a DATA statement or an input/output 
   * statement. As DATA statement is in the declaration, not executable,
   * we do not need to pay attention to it.*/
  
  array_dimension_test retour = array_dimension_test_undefined;

  list args = call_arguments(syntax_call(expression_syntax(e)));
 
  expression arg2 = EXPRESSION(CAR(CDR(args)));  /* range e1,e2,e3*/ 
  range r = syntax_range(expression_syntax(arg2));
  expression one_exp = int_to_expression(1);

  if (same_expression_p(range_increment(r),one_exp))
    { 
      /* The increment is 1 */
      expression arg1 = EXPRESSION(CAR(args)); /* Implied-DO index*/
      args = CDR(CDR(args));
      while (!ENDP(args))
	{
	  expression expr  = EXPRESSION(CAR(args));
	  array_dimension_test temp = array_dimension_test_undefined;
	  Index_range new_ir;
	  new_ir.ind = gen_nconc(CONS(EXPRESSION,arg1,NIL),ir.ind);
	  new_ir.ran = gen_nconc(CONS(RANGE,r,NIL),ir.ran);
	  if (expression_implied_do_p(expr))
	    temp = bottom_up_abc_expression_implied_do(expr,new_ir);
	  else
	    /* normal expression */
	    temp = bottom_up_abc_reference_implied_do(expr,new_ir);
	  retour = add_array_dimension_test(retour,temp);
	  args = CDR(args);
	}
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

static array_dimension_test bottom_up_abc_expression (expression e)
{
  /* the syntax of an expression can be a reference, a range or a call*/
 
  array_dimension_test retour = array_dimension_test_undefined;
  
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
  list l = args; 
  while (!ENDP(l))
    {
      expression exp = EXPRESSION(CAR(l));
      syntax s = expression_syntax(exp);
      tag t = syntax_tag(s);
      switch (t){ 
      case is_syntax_call:  
	{
	  /* We have to consider only the case of Implied DO 
	  * if e is in the lower or upper bound of implied DO's range or not */
	
	  break;
	}
      case is_syntax_reference:
	{ 
	  reference ref = syntax_reference(s);
	  if (array_reference_p(ref))
	    { 
	      list arrayinds = reference_indices(ref);
	      if (same_expression_in_list_p(e,arrayinds))
		return TRUE;
	    }
	  break;
	}
      case is_syntax_range:
	break;     
      }
      l = CDR(l);
    }
  return FALSE;
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
	    return TRUE;
	}
    }
  return FALSE;
}

static array_dimension_test bottom_up_abc_call(call cal)
{
  list args = call_arguments(cal);
  array_dimension_test retour = array_dimension_test_undefined;
  
  /* We must check two special cases:
   * 
   * 1. The call is a READ statement with array reference whose index is 
   * read in the same statement. 
   * 
   * Example : READ *,N, A(N)
   *
   * 2. The call is an Implied DO statement 
   *
   * Example : WRITE *,((A(I,J),J=1,M),I=1,N)
   *
   * or more complicated: READ *,N,(A(I),I=1,N)
   *
   * In the complicated cases, we can put a dynamic check function such as :
   *
   * READ *,N,(A(checkA(I)),I=1,N),M
     
   * where checkA is a function that does the range checking for array A */

  if (read_statement_with_write_effect_on_array_subscript(cal))
    user_log("\n READ statement with write effect on array subscript.\n This case has not been treated yet :-( \n");
  else
    while (!ENDP(args))
      {
	expression e = EXPRESSION(CAR(args));
	array_dimension_test temp = bottom_up_abc_expression(e);
	retour = add_array_dimension_test(retour,temp);		     
	args = CDR(args);
      }  
  return retour;
}

static statement make_bottom_up_abc_tests(array_dimension_test adt)
{  
  list la = adt.arr,ld = adt.dim,le = adt.exp; 
  statement retour = statement_undefined;

  while (!ENDP(la))
    { 
      entity a = ENTITY(CAR(la));
      int d = INT(CAR(ld));
      expression e = EXPRESSION(CAR(le));     
      string message = strdup(concatenate("\"Bound violation:array ", 
					  entity_name(a),", ", 
					  int_to_dimension(d),"\"", NULL));
      test tes = test_undefined;
      statement temp = statement_undefined;

      if (true_expression_p(e))
	{
	  number_of_statements_having_bound_violation++;
	  // There exists bound violation, we put a stop statement 
	  retour = make_stop_statement(message);
	  return retour;	  
	}

      tes =  make_test(e, make_stop_statement(message),
		       make_block_statement(NIL));
      
      temp = test_to_statement(tes);

      if (statement_undefined_p(retour))
	retour = copy_statement(temp);
      else 
	// always structured case
	insert_statement(retour,copy_statement(temp),FALSE);   
      
      la = CDR(la);
      ld = CDR(ld);
      le = CDR(le);
    }
  ifdebug(3) 
    {
      fprintf(stderr, " Statement returns:");
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
	      if (CONTROL(CAR(lc))==c) CONTROL(CAR(lc)) = newc;
	    }, control_successors(co));
	  },control_predecessors(c));
 
	  control_predecessors(c) = CONS(CONTROL,newc,NIL);

	  /* if c is the entry node of the correspond unstructured u, 
	     the newc will become the new entry node of u */
	      
	  if (unstructured_control(u)==c) 
	    unstructured_control(u) = newc;	 
	}
      else
	{
	  // there is no unstructured (?)
	  insert_statement(s,s1,TRUE);
	}
    }
  else
    // structured case 
    insert_statement(s,s1,TRUE);     
}

static void bottom_up_abc_statement_rwt(
   statement s,
   bottom_up_abc_context_p context)
{ 
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);  

  switch(t)
    {
    case is_instruction_call:
      {	
	call cal = instruction_call(i);

	array_dimension_test adt = bottom_up_abc_call(cal);

	if (!array_dimension_test_undefined_p(adt))
	  { 	   
	    statement seq = make_bottom_up_abc_tests(adt);	    
	    if (stop_statement_p(seq))	
	      user_log("\n Bound violation !!!! \n");
	    // insert the STOP or the new sequence of tests before the current statement
	    bottom_up_abc_insert_before_statement(s,seq,context);	    
	  }
      	break;
      }
 
    case is_instruction_whileloop:
      {
	whileloop wl = instruction_whileloop(i); 
	
	// array bound check of while loop condition 
	
	expression e1 = whileloop_condition(wl);
	
	array_dimension_test adt = bottom_up_abc_expression(e1);
		
	if (!array_dimension_test_undefined_p(adt))
	  { 
	    statement seq = make_bottom_up_abc_tests(adt);
	    if (stop_statement_p(seq))
	      user_log("Bound violation !!! \n");
	    // insert the STOP or the new sequence of tests before the current statement
	    bottom_up_abc_insert_before_statement(s,seq,context);		     
	  }	 
	break;
      }
    case is_instruction_test:
      {
	test it = instruction_test(i);

	// array bound check of the test condition

	expression e1 = test_condition(it);

	array_dimension_test adt = bottom_up_abc_expression(e1);

	if (!array_dimension_test_undefined_p(adt))
	  { 
	    statement seq = make_bottom_up_abc_tests(adt);	

	    /* I put this fraction of code in comments because of bug in PERMA
	     * I replace the current statement which is a test with same true and false
	     * branches by a STOP => assertion failed of unstructured */

	    //    if (stop_statement_p(seq))
	    // {
		/* There are bounds violations, we replace the current 
		 * statement by a 
		 * STOP "Bound violation:array .. dimension .." */
	    //	fprintf(stderr, "Bound violation !!!!\n");
	    //free_instruction(statement_instruction(s));
	    //statement_instruction(s) = statement_instruction(seq);
	    // }
	    // else	
	    // insert the new sequence of tests before the current statement
	    
	    if (stop_statement_p(seq))
	      user_log("Bound violation !!! \n");
	    bottom_up_abc_insert_before_statement(s,seq,context);	           
	  }	
	break;
      }
    case is_instruction_sequence: 
    case is_instruction_loop:
      // suppose that there are not array references in loop's range in norm	
    case is_instruction_unstructured:
      /* because we use gen_recurse with statement domain, 
       * we don't have to check unstructured  instruction here*/
      break;
    default:
      pips_error("", "Unexpected instruction tag %d \n", t );
      break; 
    }  
}

static bool store_mapping(control c, bottom_up_abc_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return TRUE;
}

static bool push_uns(unstructured u, bottom_up_abc_context_p context)
{
  stack_push((char *) u, context->uns);
  return TRUE;
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



bool bottom_up_array_bound_check(char *module_name)
{ 
  statement module_statement;  
  set_current_module_entity(local_name_to_top_level_entity(module_name));

  /* Begin the dynamic array bound checking phase. 
   * Get the code from dbm (true resource) */
  
  module_statement= (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  set_current_module_statement(module_statement);
 
  initialize_ordering_to_statement(module_statement);
      
  debug_on("BOTTOM_UP_ARRAY_BOUND_CHECK_DEBUG_LEVEL");
 
  ifdebug(1)
    {
      debug(1, "Array bound check","Begin for %s\n", module_name);
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
    }      
  initialize_bottom_up_abc_statistics();
  bottom_up_abc_statement(module_statement);
  display_bottom_up_abc_statistics();
  user_log("* The total number of array references in this file is %d *\n", 
	   total_number_of_array_references );
  
  /* Reorder the module, because the bound checks have been added */
  module_reorder(module_statement);
      
  ifdebug(1)
    {
      pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
      debug(1, "array bound check","End for %s\n", module_name);
    }
  debug_off(); 
  
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name),module_statement);
  
  reset_current_module_statement();
  
  reset_current_module_entity();
  
  return TRUE;
 
}


/* END OF FILE */







