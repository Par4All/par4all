/******************************************************************
 *
 * $Id$
 *
 *		     ALIAS PROPAGATION
 *
 *
*******************************************************************/

/* Aliasing occurs when two or more variables refer to the same
storage location at the same program point.

This phase tries to compute as precise as possible the
interprocedural alias information in a whole program.

The basic idea for computing interprocedural aliases is to follow all the
possible chains of argument-parameters and nonlocal variable-parameter
bindings at all call sites. We introduce a naming memory locations technique 
which guarantees the correctness and enhances the
precision of data-flow analysis. */

/* SOME INTELLIGENT POINTS :-)

If a call is unreachable (never be executed), there is no alias caused by 
this call for the callee => if the precondition = {0==-1} => no more check */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "alias_private.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "makefile.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "pipsmake.h"
#include "instrumentation.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"

#include "transformations.h"

/* Define a static stack and related functions to remember the current
   statement and then get the current precondition for
   alias_propagation_caller(): */
DEFINE_LOCAL_STACK(current_statement, statement)

static list l_current_aliases = NIL;
entity current_callee = entity_undefined;
entity current_caller = entity_undefined; 

/* This function computes the subscript value of an array element 
   
   DIMENSION A(l1:u1,...,ln:un)

   subscript_value(A(s1,s2,...,sn)) 
   = 1+(s1-l1)+(s2-l2)*(u1-l1+1)+...+ (sn-ln)*(u1-l1+1)*...*(u(n-1) -l(n-1)+1)

???????????????   Attention : this function compute (1 +) ??? 

   Input : the entity A 
           the indice list (s1,s2,..sn)
*/

static expression subscript_value(entity arr, list l_inds)
{
  expression retour = expression_undefined;
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
	      size_i =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
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
	      if (expression_undefined_p(retour))
		retour = copy_expression(elem_i);
	      else
		retour = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						  retour, elem_i);
	    }
	  if (expression_undefined_p(prod))
	    prod = copy_expression(size_i);
	  else
	    prod = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
					       prod,size_i);
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n i = %d \n",i);
	      fprintf(stderr, "\n prod : \n");
	      print_expression(prod);
	      fprintf(stderr, "\n retour =: \n");
	      print_expression(retour); 
	    }
	}
    }
  return retour;
}
static bool alias_propagation_call_flt(call c)
{
  if(call_function(c) == current_callee)
    {  
      statement stmt = current_statement_head();
      /* if the call is unreachable => do nothing*/
      if (! statement_weakly_feasible_p(stmt))
	fprintf(stderr, " \n The call is unreachable, unfeasible precondition");
      else 
	{
	  list l_actuals = call_arguments(c);
	  int n = gen_length(l_actuals),i;
	  int order = statement_ordering(stmt); 
	  call_site cs = make_call_site(current_caller,order);
	  list c_site = CONS(CALL_SITE,cs,NIL);
	  for (i=1; i<=n; i++)
	    {
	      expression actual_arg = find_ith_argument(l_actuals,i);
	      if (expression_reference_p(actual_arg))
		{
		  reference actual_ref = expression_reference(actual_arg);
		  entity actual_var = reference_variable(actual_ref);
		  list l_actual_inds = reference_indices(actual_ref);
		  /* compute the subscript value, return expression_undefined if
		     if the actual argument is a scalar variable or array name*/
		  expression subval = subscript_value(actual_var,l_actual_inds);
		  /* search for corresponding formal parameter */
		  entity formal_var = find_ith_formal_parameter(current_callee,i);
		  entity sec = entity_undefined;
		  expression lo = expression_undefined;
		  expression uo = expression_undefined;
		  alias_association one_alias;
		  storage s = entity_storage(actual_var);
		  if (storage_ram_p(s))
		    {
		      /* The actual argument has a ram storage */
		      ram r = storage_ram(s);
		      int off = ram_offset(r);
		      sec = ram_section(r);
		      if (expression_undefined_p(subval))
			{
			  /* The offset of actual variable is an integer 
			     that can always be translated into the callee's frame*/
			  lo = int_to_expression(off);	
			  uo = int_to_expression(off);
			}
		      else 
			{
			  /* We must translate the subscript value
			     into the callee's frame by using precondition +
			     binding information. 
			     If possible , lo = uo = off + translated(subval)
			     else , lo = off + lower_approximation(subval)
			            uo = off + upper_approximation(subval)

			  ......implement here */
			  lo =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
							    int_to_expression(off),
							    subval);
			  uo = copy_expression(lo);
			}
		      one_alias = make_alias_association(formal_var,sec,lo,uo,c_site);
		      l_current_aliases = gen_nconc(l_current_aliases,
						    CONS(ALIAS_ASSOCIATION,one_alias,NIL));
		    }
		  if (storage_formal_p(s))
		  {
		    /* The actual argument is a formal parameter of the current caller, 
		       we must take the alias_associations of the caller */
		    string caller_name = module_local_name(current_caller);
		    alias_associations caller_aliases = (alias_associations)
		      db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,
					     caller_name, TRUE);
		    list l_caller_aliases = 
		      alias_associations_list(caller_aliases); 
		    while (!ENDP(l_caller_aliases))
		    {
		      alias_association aa = ALIAS_ASSOCIATION(CAR(l_caller_aliases));
		      entity caller_var = alias_association_variable(aa);
		      if (same_entity_p(caller_var,actual_var))
		      {
			list path = gen_nconc(alias_association_call_chain(aa),
					      c_site);
			sec = alias_association_section(aa);
			
			/* We must translate the lower and upper offsets 
			   of aa to the callee's frame by using precondition +
			   binding information. 
			   
			   ......implement here */

			if (expression_undefined_p(subval))
			{
			  lo = alias_association_lower_offset(aa);
			  uo = alias_association_upper_offset(aa);
			}
			else 
			{
			  /* We must translate the subscript value
			     into the callee's frame 
			     lo = lower_offset + lower_approximation(subval)
			     uo = upper_offset + upper_approximation(subval)
			     
			     ......implement here */
			  lo =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
							    alias_association_lower_offset(aa),
							    subval);
			  uo =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
							    alias_association_upper_offset(aa),
							    subval);
			}
			one_alias = make_alias_association(formal_var,sec,lo,uo,path);
			l_current_aliases = gen_nconc(l_current_aliases,
						      CONS(ALIAS_ASSOCIATION,one_alias,NIL));
		      }
		      l_caller_aliases = CDR(l_caller_aliases);
		    }
		  }
		} 
	    }
	}
    }
  return TRUE;
}

static list alias_propagation_caller()
{
  string caller_name = module_local_name(current_caller);
  statement caller_statement = (statement) db_get_memory_resource
    (DBR_CODE,caller_name, TRUE);
  l_current_aliases = NIL;
  make_current_statement_stack();
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,caller_name,TRUE));
  
  gen_multi_recurse(caller_statement,
		    statement_domain, current_statement_filter,current_statement_rewrite,
		    call_domain, alias_propagation_call_flt, gen_null,
		    NULL);
  
  reset_precondition_map();
  free_current_statement_stack();
  return l_current_aliases;
}

static list alias_propagation_callers(list l_callers)
{
  list retour = NIL,l_tmp=NIL; 
  while (!ENDP(l_callers))
    {
      string caller_name = STRING(CAR(l_callers));
      current_caller = local_name_to_top_level_entity(caller_name);
      l_tmp = alias_propagation_caller();
      if (l_tmp != NIL)
	retour = gen_nconc(retour,l_tmp);
      current_caller = entity_undefined;
      l_callers = CDR(l_callers);
    }
  return retour;
}

bool alias_propagation(char * module_name)
{
  alias_associations aliases; 
  list l_aliases = NIL;
  current_callee = local_name_to_top_level_entity(module_name);
  debug_on("ALIAS_PROPAGATION_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin alias propagation for %s \n", module_name); 
  /* if the current procedure is the main program, do nothing*/
  if (entity_main_module_p(current_callee))
    fprintf(stderr," \n main program, do nothing");
  else
    {
      list l_decls = NIL, l_formals = NIL;
      set_current_module_entity(current_callee);		   
      l_decls = code_declarations(entity_code(current_callee));   
      /* search for formal parameters in the declaration list */      
      while(!ENDP(l_decls))
	{
	  entity e = ENTITY(CAR(l_decls));
	  storage s = entity_storage(e);
	  if (storage_formal_p(s))
	    l_formals = gen_nconc(l_formals,CONS(ENTITY,e,NIL));
	  l_decls = CDR(l_decls);
	}      
      /* if there is no formal parameter, do nothing */
      if (l_formals == NIL)
	fprintf(stderr," \n  no formal parameter, do nothing");
      else
	{
	  /* Take the list of callers */
	  callees callers = (callees) db_get_memory_resource(DBR_CALLERS,
							     module_name,
							     TRUE);
	  list l_callers = callees_callees(callers); 
	  
	  ifdebug(2)
	    {
	      fprintf(stderr," \n The formal parameters list :");
	      my_print_list_entities(l_formals);
	    }

	  /* if there is no caller, do nothing */
	  if (l_callers == NIL)
	    fprintf(stderr," \n  no caller, do nothing");
	  else
	    {
	      ifdebug(2)
		{
		  fprintf(stderr," \n There is/are %d callers : ",
			  gen_length(l_callers));
		  MAP(STRING, caller_name, {
		    (void) fprintf(stderr, "%s, ", caller_name);
		  }, l_callers);
		  (void) fprintf(stderr, "\n");	
		}
	      l_aliases = alias_propagation_callers(l_callers); 
	    }
	}
      reset_current_module_entity();
    }
  current_callee = entity_undefined;
  aliases = make_alias_associations(l_aliases);
  /* save to resource */
  DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_ASSOCIATIONS, module_name, aliases);  
  ifdebug(1)
    fprintf(stderr, " \n End alias propagation for %s \n", module_name);
  debug_off();
  
  return TRUE;
}
