/*
 * $Id$
 *
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
 * ARRAY_BOUND_CHECK_COUNT equal to 0 and before the terminasion 
 * of program, we display the value of ARRAY_BOUND_CHECK_COUNT.  */


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
#include "effects-generic.h"
#include "effects-simple.h"
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
  int retour = 0;
  if (!expression_implied_do_p(e))
    {
      syntax s = expression_syntax(e);
      tag t = syntax_tag(s);
      switch (t){ 
      case is_syntax_call:  
	{
	  retour = initial_code_abc_call(syntax_call(s));
	  break;
	}
      case is_syntax_reference:
	{ 
	  retour = initial_code_abc_reference(syntax_reference(s));
	  break;
	}
      case is_syntax_range:
	/* There is nothing to check here*/
	break;     
      }
    }    
  return retour;
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
						 left,int_to_expression(n));

  statement retour = make_assign_statement(left,right);  
  
  ifdebug(3) 
    {
      fprintf(stderr, " Statement returns:");
      print_statement(retour);
    }
  return retour;    
}
 
statement array_bound_check_display()
{ 
  string message = "      PRINT *,\'Number of bound checks:\', ARRAY_BOUND_CHECK_COUNT \n";
  statement retour = make_call_statement(CONTINUE_FUNCTION_NAME,
					 NIL,entity_undefined,message);
  return retour;
}

static void abc_instrumentation_insert_before_statement(statement s, statement s1,
				 abc_instrumentation_context_p context)
{
  /* If s is in an unstructured instruction, we must pay attetion 
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
	  // there is no untructured (?)
	  insert_statement(s,s1,TRUE);
	}
    }
  else
    // structured case 
    insert_statement(s,s1,TRUE);     
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
	}
      break;
    }
  case is_instruction_whileloop:
    {
      whileloop wl = instruction_whileloop(i);    
      expression e1 = whileloop_condition(wl);
      int n = initial_code_abc_expression(e1);
      if (n>0)
	{ 
	  statement sta = make_abc_count_statement(n);
	  abc_instrumentation_insert_before_statement(s,sta,context);		     
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
	  abc_instrumentation_insert_before_statement(s,sta,context);	           
	}	
      break;
    }
  case is_instruction_sequence:
  case is_instruction_loop:
  case is_instruction_unstructured:
    break;
  default:
    pips_error("", "Unexpected instruction tag %d \n", t );
    break; 
  }
  
  if (entity_main_module_p(mod_ent))
    {
      // MAIN PROGRAM 
      if (stop_statement_p(s) || return_statement_p(s))
	{
	  // s is the STOP or RETURN statement of main program, it causes 
	  // the execution of the program to terminate  
	  // so we display the counter of bound checks 
 
	  statement tmp = array_bound_check_display();
	  abc_instrumentation_insert_before_statement(s,tmp,context);
	}
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
	  if (strstr(name,"Bound violation") != NULL) return TRUE;
	  return FALSE;
	}
      return FALSE;
    }
  return FALSE; 
}

static int  number_of_logical_operators(expression e)
{
  int retour = 1;
  return retour;
}

static void pips_code_abc_statement_rwt(statement s,abc_instrumentation_context_p context)
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
	  expression e = test_condition(it);
	  int n = number_of_logical_operators(e);
	  if (n>0)
	    { 
	      statement sta = make_abc_count_statement(n);     
	      abc_instrumentation_insert_before_statement(s,sta,context);           
	    }	
	}
      break;
    }
  case is_instruction_call:
  case is_instruction_whileloop:    
  case is_instruction_sequence:
  case is_instruction_loop:
  case is_instruction_unstructured:
    break;
  default:
    pips_error("", "Unexpected instruction tag %d \n", t );
    break; 
  }  
  if (entity_main_module_p(mod_ent))
    {
      // MAIN PROGRAM 
      if (stop_statement_p(s) || return_statement_p(s))
	{
	  // s is the STOP or RETURN statement of main program, it causes 
	  // the execution of the program to terminate  
	  // so we display the counter of bound checks 
 
	  statement tmp = array_bound_check_display();
	  abc_instrumentation_insert_before_statement(s,tmp,context);
	}
    }
}

static bool store_mapping(control c, abc_instrumentation_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return TRUE;
}

static bool push_uns(unstructured u, abc_instrumentation_context_p context)
{
  stack_push((char *) u, context->uns);
  return TRUE;
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

bool array_bound_check_instrumentation(char *module_name)
{ 
  statement module_statement;
  list leffects = NIL,ldeclas = NIL;

  /* In case of adjustable array, the value of variable that appears 
   * in a dimension bound expression 
   * can be changed by one or more write effects in the module
   * (see ANSI X3.9-1978 FORTRAN 77 for more details).
   * Here, we use the summary effects to calculate if a dimension 
   * bound expression is modified 
   * by effect(s) and if it is the case,  an user error message 
   * will be displayed. 

   * If we don't make this test, the dynamic array bound checks 
   * added here are not correct any more. We 
   * will change the approach if one day, we find it neccessary */
  
  set_current_module_entity(local_name_to_top_level_entity(module_name));
 
  leffects = effects_to_list((effects) db_get_memory_resource(DBR_SUMMARY_EFFECTS, 
							      module_name, TRUE));

  ldeclas =  code_declarations(entity_code(get_current_module_entity()));

  if (array_dimension_variable_changed_p(ldeclas,leffects)) 
    {
      user_error ("\t ARRAY BOUND CHECK", 
		  "There are write effects on bounds of array!\n ");      
      reset_current_module_entity();      
      return TRUE; 
    }
  else
    {
      // add COMMON /ARRAY_BOUND_CHECK/ ARRAY_BOUND_CHECK_COUNT to the declaration
      // if main program : DATA ARRAY_BOUND_CHECK_COUNT /0/

      string new_decl = 
	"      INTEGER ARRAY_BOUND_CHECK_COUNT\n"
	"      COMMON /ARRAY_BOUND_CHECK/ ARRAY_BOUND_CHECK_COUNT\n";
      string new_decl_init = 
	"      DATA ARRAY_BOUND_CHECK_COUNT /0/\n";

      basic b = make_basic_int(4);

      mod_ent =  get_current_module_entity();

      abccount = make_scalar_entity(ABC_COUNT,module_name,copy_basic(b));
      
      if (entity_main_module_p(mod_ent))
	// MAIN PROGRAM 
	code_decls_text(entity_code(mod_ent))
          = strdup(concatenate(code_decls_text(entity_code(mod_ent)),
			       new_decl,new_decl_init,NULL));
      else 
	code_decls_text(entity_code(mod_ent)) 
	  = strdup(concatenate(code_decls_text(entity_code(mod_ent)),
			       new_decl, NULL));
      
      /* Begin the array bound check instrumentation phase. 
       * Get the code from dbm (true resource) */
  
      module_statement= (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

      set_current_module_statement(module_statement);
 
      initialize_ordering_to_statement(module_statement);
      
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

	  rewrite_implied_do(module_statement);     

	  /* Reorder the module, because new loops have been added */
	  
	  module_reorder(module_statement);
	  ifdebug(1)
	    {
	      debug(1, " Initial code array bound check instrumention",
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
  
      DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name),module_statement);

      reset_current_module_statement();
 
      reset_current_module_entity();
      
      return TRUE;
    } 
}






