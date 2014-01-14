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
/*
 * ------------------------------------------------
 *
 *           ARRAY BOUND CHECK INSTRUMENTATION
 *
 * ------------------------------------------------
 * This phase instruments the code in order to calcul 
 * the number of dynamic bound checks. 
 *
 * New common variable ARRAY_BOUND_CHECK_COUNT is added.
 * For each bound check, we increase ARRAY_BOUND_CHECK_COUNT by one
 * If the module is the main program, we initialize 
 * ARRAY_BOUND_CHECK_COUNT equal to 0 and before the termination 
 * of program, we display the value of ARRAY_BOUND_CHECK_COUNT. 
 *
 *
 * Hypotheses : there is no write effect on the array bound expression.
 *
 * There was a test for write effect on bound here but I put it away (in 
 * effect_on_array_bound.c) because it takes time to calculate the effect
 * but in fact this case is rare. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "alias_private.h"
#include "instrumentation.h"

#define ABC_COUNT "ARRAY_BOUND_CHECK_COUNT"

static int initial_code_abc_reference(reference r);
static int initial_code_abc_expression (expression e);
static int initial_code_abc_call(call cal);
static entity abccount;
static entity mod_ent;

/* context data structure for array_bound_check_instrumentation newgen recursion */
typedef struct 
{
  persistant_statement_to_control map;
  stack uns;
} 
  abc_instrumentation_context_t, 
* abc_instrumentation_context_p;

typedef struct 
{
  int number;
} 
  abc_number_of_operators_context_t, 
* abc_number_of_operators_context_p;

static int initial_code_abc_reference(reference r)
{  
  int retour = 0;  
  if (array_reference_p(r))
    { 
      entity e = reference_variable(r);
      if (array_need_bound_check_p(e)) 
      	{	 
	  list arrayinds = reference_indices(r);
	  int i;
	  for (i=1;i <= gen_length(arrayinds);i++)
	    {  
	      expression ith = find_ith_argument(arrayinds,i);
	      int temp = initial_code_abc_expression(ith);
	      // two bound checks : lower and upper
	      retour = retour + 2;
	      retour = retour + temp;
	    }	
	}	
    } 
  return retour;
}

static int initial_code_abc_expression (expression e)
{
  /* the syntax of an expression can be a reference, a range or a call*/ 
  if (!expression_implied_do_p(e))
    {
      syntax s = expression_syntax(e);
      tag t = syntax_tag(s);
      switch (t){ 
      case is_syntax_call:  
	return initial_code_abc_call(syntax_call(s));
      case is_syntax_reference:
	return initial_code_abc_reference(syntax_reference(s));
      case is_syntax_range:
	/* There is nothing to check here*/
	return 0;
      }
    }    
  return 0;
}

static int initial_code_abc_call(call cal)
{
  list args = call_arguments(cal);
  int retour = 0;  
  while (!ENDP(args))
    {
      expression e = EXPRESSION(CAR(args));
      int temp = initial_code_abc_expression(e);
      retour = retour + temp;     		     
      args = CDR(args);
    }  
  return retour;
}

static statement make_abc_count_statement(int n)
{  
  expression left = reference_to_expression(make_reference(abccount, NIL));
  expression right = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						 left,
						 int_to_expression(n));
  return make_assign_statement(left,right);  
}
 
statement array_bound_check_display()
{ 
  string message = "      PRINT *,\'Number of bound checks:\', ARRAY_BOUND_CHECK_COUNT\n" ;
  /* Attention : no strdup => newgen error */
  //  statement retour = make_call_statement(CONTINUE_FUNCTION_NAME,
  //				 NIL,entity_undefined,message);
  return  make_call_statement(CONTINUE_FUNCTION_NAME,
			      NIL,entity_undefined,strdup(message));
}

static void abc_instrumentation_insert_before_statement(statement s, statement s1,
				 abc_instrumentation_context_p context)
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

static void initial_code_abc_statement_rwt(statement s,abc_instrumentation_context_p context )
{ 
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);  
  switch(t){
  case is_instruction_call:
    {	
      call cal = instruction_call(i);
      int n = initial_code_abc_call(cal);
      if (n > 0)
	{       
	  statement sta = make_abc_count_statement(n);
	  abc_instrumentation_insert_before_statement(s,sta,context);
	  message_assert("statement is consistent",
			 statement_consistent_p(s));
	}
      if (stop_statement_p(s) || (entity_main_module_p(mod_ent) && return_statement_p(s)) )
	{
	  /* There are 2 kinds of statement which cause the execution of the program to terminate
	   1. STOP statement in every module
	   2. END statement in main program
	   ( PIPS considers the END statement of MAIN PROGRAM as a RETURN statement)
	   we display the counter of bound checks before these kinds of statement */
	  statement tmp = array_bound_check_display();
	  abc_instrumentation_insert_before_statement(s,tmp,context);
	  message_assert("statement is consistent",
			 statement_consistent_p(s));
	}
      break;
    }
  case is_instruction_whileloop:
    {
      whileloop wl = instruction_whileloop(i);    
      expression e1 = whileloop_condition(wl);
      int n = initial_code_abc_expression(e1);
      /* This code is not correct !
	 The counting statement must be inserted in the body of the loop !*/
      if (n>0)
	{ 
	  statement sta = make_abc_count_statement(n);
	  abc_instrumentation_insert_before_statement(s,sta,context);	
	  message_assert("statement is consistent",
			 statement_consistent_p(s));
	}	 
      break;
    }
  case is_instruction_test:
    {
      test it = instruction_test(i);
      expression e1 = test_condition(it);
      int n = initial_code_abc_expression(e1);
      if (n>0)
	{ 
	  statement sta = make_abc_count_statement(n);    
	  ifdebug(3) 
	    {	  
	      fprintf(stderr, "\n Statement to be inserted before a test: ");    
	      print_statement(sta);
	      print_statement(s);
	    }
	  /* bug Example abc_ins.f : there is STOP statement in the branch of if statement	     
	     IF (A(1).GE.L)
	       ...
	       STOP
	     ENDIF
	     free_instruction(s) in insert_statement will be not consistent => core dumped 
	     Solution : like pips_code_abc_statement_rwt */
	  abc_instrumentation_insert_before_statement(s,sta,context);	  
	  message_assert("statement is consistent",
			 statement_consistent_p(s));
	}	
      break;
    }
  case is_instruction_sequence:
  case is_instruction_loop:
  case is_instruction_unstructured:
    break;
  default:
    pips_internal_error("Unexpected instruction tag %d ", t );
    break; 
  }
}

static bool abc_bound_violation_stop_statement_p(statement s)
{ 
  if (stop_statement_p(s))
    {
      list l = call_arguments(statement_call(s));
      if (l!= NIL)
	{
	  expression e = EXPRESSION(CAR(l));
	  string name = entity_name(call_function(syntax_call(expression_syntax(e))));	  
	  //  fprintf(stderr, "name = %s\n",name);
	  if (strstr(name,"Bound violation") != NULL) return true;
	  return false;
	}
      return false;
    }
  return false; 
}

static bool number_of_operators_flt(expression e, abc_number_of_operators_context_p context)
{
  if (expression_call_p(e))
    {
      if (logical_operator_expression_p(e))
	{
	  context->number ++;;
	  return true;
	}
      return false;
    }
  return false;
}

static int  number_of_logical_operators(expression e)
{
  abc_number_of_operators_context_t context;
  context.number =1;
  gen_context_recurse(e,&context,
		      expression_domain,
		      number_of_operators_flt,
		      gen_null);
  return context.number;
}

static void pips_code_abc_statement_rwt(statement s, abc_instrumentation_context_p context)
{ 
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);  
  switch(t){
  case is_instruction_test:
    {
      test it = instruction_test(i);      
      statement true_branch = test_true(it);	
      if (abc_bound_violation_stop_statement_p(true_branch))
	{
	  /* s is a bound check generated by PIPS which has this form:
	     IF (e) THEN
	        STOP "Bound violation ...."
	     ENDIF
	     we replace it by:
	     ARRAY_BOUND_CHECK_COUNT = ARRAY_BOUND_CHECK_COUNT + n
	     IF (e) THEN
	        PRINT *,'Number of bound checks : ', ARRAY_BOUND_CHECK_COUNT
	        STOP "Bound violation ...."
	     ENDIF */	  
	  expression e = test_condition(it);
	  int n = number_of_logical_operators(e);
	  statement sta = make_abc_count_statement(n);  	  
	  statement s1 = array_bound_check_display();
	  list ls1 = CONS(STATEMENT,s1,CONS(STATEMENT,true_branch,NIL));	
	  list ls2;
	  test_true(it) = instruction_to_statement(make_instruction(is_instruction_sequence,
								    make_sequence(ls1)));
	  ls2 = CONS(STATEMENT,sta,CONS(STATEMENT,copy_statement(s),NIL));		  
	  statement_instruction(s) = make_instruction(is_instruction_sequence,
						      make_sequence(ls2));	       
	}	
      break;
    }
  case is_instruction_call: 
    {
      if ((stop_statement_p(s) && !abc_bound_violation_stop_statement_p(s)) 
	  || (return_statement_p(s) && entity_main_module_p(mod_ent)))
	{
	  /* There are 2 kinds of statement which cause the execution of the program to terminate
	   1. STOP statement in every module
	   2. END statement in main program
	   ( PIPS considers the END statement of MAIN PROGRAM as a RETURN statement)
	   we display the counter of bound checks before these kinds of statement 
	   The case of STOP "Bound violation" is done separately */
	  statement tmp = array_bound_check_display();
	  abc_instrumentation_insert_before_statement(s,tmp,context);
	}
      break;
    }
  case is_instruction_whileloop:    
  case is_instruction_sequence:
  case is_instruction_loop:
  case is_instruction_unstructured:
    break;
  default:
    pips_internal_error("Unexpected instruction tag %d ", t );
    break;    
  }
}

static bool store_mapping(control c, abc_instrumentation_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return true;
}

static bool push_uns(unstructured u, abc_instrumentation_context_p context)
{
  stack_push((char *) u, context->uns);
  return true;
}

static void pop_uns(unstructured u, abc_instrumentation_context_p context)
{
  stack_pop(context->uns);
}

static void initial_code_abc_statement(statement module_statement)
{
  abc_instrumentation_context_t context;
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);
  
  gen_context_multi_recurse(module_statement,&context,
    unstructured_domain, push_uns, pop_uns,
    control_domain, store_mapping, gen_null,
    statement_domain, gen_true, initial_code_abc_statement_rwt,
    NULL);

  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns);

}

static void  pips_code_abc_statement(statement module_statement)
{
  abc_instrumentation_context_t context;
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);
  
  gen_context_multi_recurse(module_statement,&context,
    unstructured_domain, push_uns, pop_uns,
    control_domain, store_mapping, gen_null,
    statement_domain, gen_true, pips_code_abc_statement_rwt,
    NULL);

  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns); 
}


bool old_array_bound_check_instrumentation(const char* module_name)
{ 
  statement module_statement;  
  /* add COMMON ARRAY_BOUND_CHECK_COUNT to the declaration
     if main program : DATA ARRAY_BOUND_CHECK_COUNT 0*/
  string new_decl = 
    "      INTEGER*8 ARRAY_BOUND_CHECK_COUNT\n"
    "      COMMON /ARRAY_BOUND_CHECK/ ARRAY_BOUND_CHECK_COUNT\n";
  string new_decl_init = 
    "      DATA ARRAY_BOUND_CHECK_COUNT /0/\n";
  string old_decl;
  basic b = make_basic_int(8);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  mod_ent =  get_current_module_entity();
  abccount = make_scalar_entity(ABC_COUNT,module_name,b);
  old_decl = code_decls_text(entity_code(mod_ent));
  // user_log("Old declaration = %s\n", code_decls_text(entity_code(mod_ent)));  
  if (entity_main_module_p(mod_ent))
    // MAIN PROGRAM 
    code_decls_text(entity_code(mod_ent))
      = strdup(concatenate(old_decl, new_decl, new_decl_init,NULL));
  else 
    code_decls_text(entity_code(mod_ent)) 
      = strdup(concatenate(old_decl, new_decl, NULL));  
  free(old_decl), old_decl = NULL; 
  // user_log("New declaration = %s\n", code_decls_text(entity_code(mod_ent)));  
  //   fprintf(stderr, "NEW = %s\n", code_decls_text(entity_code(mod_ent)));  
  /* Begin the array bound check instrumentation phase. 
   * Get the code from dbm (true resource) */  
  module_statement= (statement) 
    db_get_memory_resource(DBR_CODE, module_name, true);  
  set_current_module_statement(module_statement); 
  set_ordering_to_statement(module_statement);      
  debug_on("ARRAY_BOUND_CHECK_INSTRUMENTATION_DEBUG_LEVEL");  
  if (get_bool_property("INITIAL_CODE_ARRAY_BOUND_CHECK_INSTRUMENTATION"))   
    {
      // instrument the initial code 
      // Rewrite Implied_DO code 

      /* Before running the array_bound_check phase, 
       * for the implied-DO expression (in statement 
       * READ, WRITE of Fortran), we will create new Pips' loops 
       * before the READ/WRITE statement, 
       * it means that instead of checking array references 
       * for implied-DO statement (which is not 
       * true if we do it like other statements), we will check 
       * array references in new loops added*/
      
      // rewrite_implied_do(module_statement);     
      /* Reorder the module, because new loops have been added */      
      // module_reorder(module_statement);
      ifdebug(1)
	{
	  debug(1, " Initial code array bound check instrumentation",
		"Begin for %s\n", module_name);
	  pips_assert("Statement is consistent ...", 
		      statement_consistent_p(module_statement));
	}  	
      initial_code_abc_statement(module_statement);
    }
  if (get_bool_property("PIPS_CODE_ARRAY_BOUND_CHECK_INSTRUMENTATION"))
    {
      ifdebug(1)
	{
	  debug(1, "PIPS code array bound check instrumentation ",
		"Begin for %s\n", module_name);
	  pips_assert("Statement is consistent ...", 
		      statement_consistent_p(module_statement));
	}      
      
      pips_code_abc_statement(module_statement);
    }  
  /* Reorder the module, because new statements have been added */  
  module_reorder(module_statement);  
  ifdebug(1)
    {
      pips_assert("Statement is consistent ...", 
		  statement_consistent_p(module_statement));
      debug(1, "Array bound check instrumentation","End for %s\n", module_name);
    }
  debug_off(); 
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,module_statement);
  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_current_module_entity();
  return true;
}

static list l_commons = NIL;
static int number_of_scalar_variables = 0; 
static int number_of_array_variables = 0; 
bool array_bound_check_instrumentation(const char* module_name)
{
  entity mod = local_name_to_top_level_entity(module_name);
  list d = code_declarations(value_code(entity_initial(mod))); 
  MAP(ENTITY,ent,
  {
    if (!formal_parameter_p(ent))
      {
	if (variable_in_common_p(ent))
	  {
	    entity sec = ram_section(storage_ram(entity_storage(ent)));	
	    if (!entity_in_list_p(sec,l_commons)) 
	      {
		area a = type_area(entity_type(sec));
		list l = area_layout(a);
		/*user_log("*\n%d variable in %s *\n",gen_length(l),entity_name(sec));
		  number_of_variables = number_of_variables + gen_length(l);*/
		MAP(ENTITY, e, 
		{
		  type t = entity_type(e);
		  if (type_variable_p(t))
		    {
		      if (entity_scalar_p(e))
			{
			  user_log("*\nCommon and scalar variable %s *\n",entity_name(e));
			  number_of_scalar_variables++;
			}
		      else
			{
			  user_log("*\nCommon and array variable %s *\n",entity_name(e));
			  number_of_array_variables++;
			}
		    }
		},l);
		l_commons = gen_nconc(l_commons,CONS(ENTITY,sec,NIL));
	      }
	  }
	else 
	  {
	    if (local_entity_of_module_p(ent,mod))
	      {		
		type t = entity_type(ent);
		user_log("*\nLocal variable %s of type %d *\n",entity_name(ent),type_tag(t));
		if (type_variable_p(t))
		  {
		    if (entity_scalar_p(ent))
		      {
			user_log("*\nLocal and scalar variable %s *\n",entity_name(ent));
			number_of_scalar_variables++;
		      }
		    else
		      {
			user_log("*\nLocal and array variable %s *\n",entity_name(ent));
			number_of_array_variables++;
		      }
		  }
	      }
	  }
      }
  },d);
  /* Eliminate 4 special variables : MAIN000:*DYNAMIC*, MAIN000:*STATIC*,MAIN000:*HEAP*, MAIN000:*STACK* **/
  user_log("*\nNumber of scalar variables :%d *\n", number_of_scalar_variables);
  user_log("*\nNumber of array variables :%d *\n", number_of_array_variables);
  return true;
} 







