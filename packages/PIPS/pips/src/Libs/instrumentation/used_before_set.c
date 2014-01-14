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
#include "transformer.h"
#include "misc.h"
#include "properties.h"
#include "pipsmake.h"
#include "alias_private.h"
#include "instrumentation.h"
#include "ubs_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"
#include "text-util.h"
#include "alias-classes.h"

/* This analysis checks if the program uses a variable or an array element
   which has not been assigned a value. In this case, anything may happen:
   the program may appear to run normally, or may crash, or may behave unpredictably. 
   We use IN regions that give a set of read variables not previously written. 
   Depending on the nature of the variable: local, formal or global, we have 
   different cases.    
   This is a top-down analysis that process a procedure before all its
   callees. Information given by callers is used to verify if we have to
   check for the formal parameters in the current module or not. In addition,
   we produce information in the resource MODULE.ubs to tell if the formal
   parameters of the called procedures have to be checked or not.  
   used_before_set   > MODULE.ubs
       < PROGRAM.entities 
       < MODULE.code
       < MODULE.in_regions
       < CALLERS.ubs
   Algorithm: take the list of IN regions of the module statement (attention, the IN region
   list contains the IN effect list and more exact, so always take the IN region list, 
   although it is heavier) and check for each variable in this list. 
   Case 1. Local variable
      a. if MUST IN at module statement, insert STOP 'Variable is used before set'
      b. else MAY IN at module statement,
          - insert an INITIALIZE function  
          - insert a VERIFY function before each sub-statement with MUST IN region.        
	    If the sub-statement is a call site and we only have MAY IN, then update UBS
	    resource of the current procedure with the name of the called procedure and 
	    the offset of corresponding formal parameters (the actual variable may occur in
	    several places in the argument list) in order to check these formal variables
	    in the callee's frame.
   Case 2. Formal variable 
         If no UBS resource for this variable is stored in the callers => no need to VERIFY 
	 this variable. Otherwise, which means that there exists at least 1 actual argument 
	 that may not defined (but already INITIALIZED), then :
	  - insert a VERIFY function before each sub-statement with MUST IN region.
	    If the sub-statement is a call site and we only have MAY IN, then update UBS
	    resource of the current procedure with the name of the called procedure and 
	    the offset of corresponding formal parameters (the actual variable may occur in
	    several places in the argument list) in order to check these formal variables
	    in the callee's frame.
   Case 3. Common variables, depend on if we have the main program or not. 
      a. if the current module is the main program, or in general, has no callers (maybe unreachable, or a 
         library case): for all kind of common variables (visible in the module or not), do like case 1.
	 Attention, for the invisible variable, we have to declare the common in its INITIALIZE function.
	 When updating the resource ubs, there are two possibilities:
	  - (called procedure, common variable)
	  - (called procedure, formal parameter) in case if the common variable is passed as actual argument.
      b. if not, the current module has several callers, do like case 2. 
         If the sub-statement is a call site and we only have MAY IN => also update ubs resource 
	 for the current procedure with two possibilities
	 - (called procedure, common variable)
	 - (called procedure, formal parameter) in case if the common variable is passed as actual argument.
   ATTENTION : we have to treat variables initialized by a DATA statement and take care of variables in 
   an EQUIVALENCE statement. Current IN effects/regions do not take into account if an equivalenced variable 
   is defined, there no more IN effects/regions on the other :-) This must be its task in order to keep the
   analysis conservative !!!

   Some possible improvements:
   a.If V is a local variable/or common variable in module with no caller and 
   1.There is no WRITE on V in the current module => no need INITIALIZE + VERIFY, insert STOP before 
     each MUST IN  
   2.If before a MUST IN on V, there is no WRITE on V => no need  VERIFY, insert STOP before this MUST IN
     No no no, because we always VERIFY the formal variable => the actual variable may not be INITIALIZED
     => false 
     We must have a mechanism to tell the called procedure if it have to insert a VERIFY or a STOP. 
     But with some call paths : VERIFY, the others : STOP => what to do ? 
     Safe way => INITIALIZE and VERIFY  all. 
  
   By using IN regions, we can limit the number of variables to be checked, 
   and code instrumentation is only used when we do not have enough information. Only array 
   elements in the MAY IN regions are initialized,  and array elements in the EXACT IN 
   regions are verified, not the whole array.

   do i_pips = k,l
      a(i_pips) = undef
   enddo
   
   <A(PHI)-IN-MAY-{K<=PHI<=L}>
   DO I = M,N
      A(I) =I
   ENDDO
   
   do i_pips = k,l
      if (a(i_pips) == undef) stop "used before set"
   enddo

   <A(PHI)-IN-EXACT-{K<=PHI<=L}>
   DO I = K,L
      X(I) = A(I)
   ENDDO 

   There are two possible implementations :
   1. Put checks in an instrumented file and then use a script to insert them into the initial code 
   2. Insert checks directly in the code, but more complicated with gen_recurse, unstructured, and 
      for different variables */

#define PREFIX1 "$UBS_CHECK"
#define PREFIX2 "$UBS_CHECK_END"
static FILE * out;
static FILE * out2;
static entity current_entity = entity_undefined;
static entity current_mod = entity_undefined;
static statement module_statement = statement_undefined;
static int number_of_may_uninitialized_scalar_variables = 0;
static int number_of_may_uninitialized_array_variables = 0;
//static int number_of_may_uninitialized_array_elements = 0;
static int number_of_uninitialized_scalar_variables = 0;
static int number_of_uninitialized_array_variables = 0;
static int number_of_added_verifications = 0;
static int number_of_processed_modules = 0;
static string file_name;
static string initialization_file;
static list l_ubs_checks = NIL;
static list l_initialized_commons = NIL;

static void display_used_before_set_statistics()
{
  user_log("\n Number of uninitialized scalar variables : %d", 
	   number_of_uninitialized_scalar_variables);
  user_log("\n Number of uninitialized array variables : %d", 
	   number_of_uninitialized_array_variables);
  user_log("\n Number of may uninitialized scalar variables : %d", 
	   number_of_may_uninitialized_scalar_variables);
  user_log("\n Number of may uninitialized array variables : %d",
	   number_of_may_uninitialized_array_variables);
  user_log("\n Number of added verifications : %d",
	   number_of_added_verifications);
  user_log("\n Number of processed modules : %d\n",
	   number_of_processed_modules);
}

static list entities_to_expressions2(list l_ent)
{
  list l_exp = NIL;
  MAP(ENTITY,ent,
  {
    l_exp = gen_nconc(l_exp, CONS(EXPRESSION,entity_to_expression(ent),NIL));
  },l_ent);
  return(l_exp);
}

static string print_list_of_entities(list l)
{
  string retour = "";
  MAP(ENTITY, e,
  {
    retour = strdup(concatenate(retour,",",entity_local_name(e),NULL));
  }, l);
  return retour;
}

static string print_list_of_expressions(list l)
{
  string retour = "";
  MAP(EXPRESSION, exp,
  {
    retour = strdup(concatenate(retour,",",words_to_string(words_expression(exp,NIL)),NULL));
  }, l);
  return retour;
}

static bool common_variable_in_module_scope_p(entity ent,entity mod)
{
  /* Check if a common variable, i.e SUB1:COM1, declared in a common block BLOCK1
     is visible in a module SUB2 or not */
  list d = code_declarations(value_code(entity_initial(mod)));
  MAP(ENTITY,e,
  {
    if (strcmp(entity_local_name(e),entity_local_name(ent)) == 0)
      {
	ram r1 = storage_ram(entity_storage(e));
	ram r2 = storage_ram(entity_storage(ent));
	entity a1 = ram_section(r1);
	entity a2 = ram_section(r2);
	if (a1 == a2) return true;
      }
  },d);
  return false;
}

static bool initialized_by_equivalent_variable_p(entity ent)
{
  /* This is not true !!! We must follow the IN regions of the equivalent variables*/
  storage s = entity_storage(ent);
  list local_shared = ram_shared(storage_ram(s));
  if (gen_length(local_shared)>0) 
    {
      ifdebug(3)
	{
	  pips_debug(3,"List of shared variables of %s \n",entity_name(ent));
	  print_entities(local_shared);
	}
      return true;
    }
  return false;
}

static Psysteme remove_temporal_variables_from_system(Psysteme ps)
{
  /* Project all #init variables from the system ps,
     there are 2 cases :
     1. The result is not sure , there are over flow
        Return the SC_UNDEFINED 
     2. The projection is exact */
  if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
      Pvecteur pv_var = NULL; 
      Pbase b = ps->base; 
      for(; !VECTEUR_NUL_P(b);b = b->succ)
	{
	  entity e = (entity) vecteur_var(b);
	  if (old_value_entity_p(e))
	    vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
	}
      return my_system_projection_along_variables(ps, pv_var); 
    }
  return ps;     
}

static expression make_special_value(entity ent)
{
  basic b = entity_basic(ent);
  switch (basic_tag(b)) {
  case 0: /*integer*/
    return int_to_expression(1000000000);
  case 1: /*float*/
    switch (basic_float(b)) {
    case 4:
      return make_call_expression(make_constant_entity("r_signaling_nan()",is_basic_float,4),NIL);
    case 8:
      return make_call_expression(make_constant_entity("d_signaling_nan()",is_basic_float,8),NIL);
    default:
      user_log("\nInitialize floating number with more than 8 bytes ?");
      return make_call_expression(MakeConstant("which_value",is_basic_float),NIL);
    }
  case 2: /*logical : false = 0, true = 1*/
    return int_to_expression(2);
  case 3: /*overloaded*/
    user_log("\nInitialize overloaded ?");
    return make_call_expression(MakeConstant("which_value",is_basic_overloaded),NIL);
  case 4: /*complex*/
    switch (basic_complex(b)) {
    case 8:
      return make_call_expression(make_constant_entity("CMPLX(r_signaling_nan(),r_signaling_nan())",
						       is_basic_complex,8),NIL);
    case 16:
      return make_call_expression(make_constant_entity("CMPLX(d_signaling_nan(),d_signaling_nan())",
						       is_basic_complex,16),NIL);
    default:
      user_log("\nInitialize complex number with more than 16 bytes ?");
      return make_call_expression(MakeConstant("which_value",is_basic_complex),NIL);
    }
  case 5: /*string*/
    return make_call_expression(MakeConstant("\'Nga Nguyen\'",is_basic_string),NIL);
  default:
    pips_internal_error("Unexpected basic tag");
    return expression_undefined; // just to avoid gcc warning
  }
}

/* This function makes a test like X.EQ.100000 or id_nan(A(PHI1)) or ir_nan(A(I,B(J)) */
static expression make_test_condition(expression exp, entity ent)
{
  basic b = entity_basic(ent);
  expression cond = expression_undefined;
  switch (basic_tag(b)) {
  case 1: /*float*/
    switch (basic_float(b)) {
    case 4:
      cond = make_call_expression(make_constant_entity("ir_isnan",is_basic_float,4),
				  CONS(EXPRESSION,exp,NIL));
      break;
    case 8:
      cond = make_call_expression(make_constant_entity("id_isnan",is_basic_float,8),
				  CONS(EXPRESSION,exp,NIL));
      break;
    default:
      cond = eq_expression(exp,make_special_value(ent));
      break;
    }
    break;
  case 4:/*complex*/
    switch (basic_complex(b)) {
    case 8:
      cond = make_call_expression(make_constant_entity("ir_isnan",is_basic_complex,8),
				  CONS(EXPRESSION,exp,NIL));
      break;
    case 16:
      cond = make_call_expression(make_constant_entity("id_isnan",is_basic_complex,16),
				  CONS(EXPRESSION,exp,NIL));
      break;
    default:
      cond = eq_expression(exp,make_special_value(ent));
      break;
    }
    break;
  default:
    cond = eq_expression(exp,make_special_value(ent));
    break;
  }
  return cond;
}

/* This function generates an assignment that initializes the scalar variable */
static void initialize_scalar_variable(entity ent)
{
  statement s = make_assign_statement(entity_to_expression(ent),make_special_value(ent));  
  print_text(out, text_statement(entity_undefined,0,s,NIL));
}

/* This function generates code that initializes every array element in the array region
   to a special value. 
   We use the algorithm_row_echelon(initial system, list of variables to scan, 
   return condition,  returned enumeration system) to generate the nested loop.  
   The generated code will be like this:
   IF (condition) THEN
      DO PHI1 = low1,up1
	 DO PHI2 = low2,up2
	    A(PHI1,PHI2) = SNan
	 ENDDO
      ENDDO
   ENDIF 
   If bounds can not be computed from the region, we use bounds from the array declaration 
   DO PHI1 = dec_low1,dec_up1
       DO PHI2 = dec_low2,dec_up2
          A(PHI1,PHI2) = SNan
       ENDDO
   ENDDO
   CURRENTLY: we cannot check if all lower/upper bounds can be generated for variables Phii
   from region => use bounds from array declarations 
   ATTENTION: the initialization is expensive, it increases much execution time 
   (number of array variables * size of arrays :-)) */

static void initialize_array_variable(entity ent)
{
  list dims = variable_dimensions(type_variable(entity_type(ent)));
  int n = gen_length(dims);
  list l_phi = phi_entities_list(1,n);
  /* Attention, this analysis uses PHI entities, static variables of region => init_regions*/
  reference ref = make_reference(ent,entities_to_expressions2(l_phi));
  expression exp = reference_to_expression(ref);
  statement smt = make_assign_statement(exp,make_special_value(ent));  
  Psysteme row_echelon = entity_declaration_sc(ent);
  /* The assumed-size case cannot happen, because formal variables are not initialized*/
  smt = systeme_to_loop_nest(row_echelon,l_phi,smt,entity_intrinsic(DIVIDE_OPERATOR_NAME));
  print_text(out,text_statement(entity_undefined,0,smt,NIL));
}

static void verify_scalar_variable(entity ent)
{
  string message = strdup(concatenate("\'Scalar variable ",entity_name(ent)," is used before set\'",NULL));
  expression cond = make_test_condition(entity_to_expression(ent),ent);
  test t = test_undefined;
  statement smt = statement_undefined;
  basic b = entity_basic(ent);
  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
    {
      smt = make_print_statement(message); 
      /*To help debugging, reinitialize the integer and real variables by 0 in order to 
	avoid so much propagated NaN or 1000000 values. By doing this, we cannot
	detect all uninitialized variables but it helps Corinne, with thousand lines of code */
      if ((basic_tag(b)==0) || (basic_tag(b)==1))
	{
	  statement s = make_assign_statement(entity_to_expression(ent),int_to_expression(0));  
	  /*insert the assignment after the PRINT statement*/
	  insert_statement(smt,s,false); 
	}
    }
  else
    smt = make_stop_statement(message); 
  t = make_test(cond,smt,make_block_statement(NIL));
  smt = test_to_statement(t);
  print_text(out, text_statement(entity_undefined,0,smt,NIL));
  free(message), message = NULL;
}

static void verify_array_element(entity ent, expression exp)
{
  statement smt = statement_undefined;
  test t = test_undefined;
  list args = reference_indices(expression_reference(exp));
  string stop_message  = strdup(concatenate("\'Array ",entity_name(ent)," is used before set\'",NULL));
  string print_message  = strdup(concatenate("\'Array element ",entity_name(ent),"(\'",
					     print_list_of_expressions(args),",\') is used before set\'",NULL));
  expression cond = make_test_condition(exp,ent);
  basic b = entity_basic(ent);
  ifdebug(3)
    {
      pips_debug(3,"Verify array element:");
      print_expression(exp);
    }
  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
    {   
      smt = make_print_statement(print_message); 
      /*To help debugging, reinitialize the integer and real array elements by 0 in order to 
	avoid so much propagated NaN or 1000000 values. By doing this, we cannot
	detect all uninitialized variables but it helps Corinne, with thousand lines of code */
      if ((basic_tag(b)==0) || (basic_tag(b)==1))
	{
	  statement s = make_assign_statement(exp,int_to_expression(0));  
	  /*insert the assignment after the PRINT statement*/
	  insert_statement(smt,s,false); 
	}
    }
  else
    smt = make_stop_statement(stop_message); 
  t = make_test(cond,smt,make_block_statement(NIL));
  smt = test_to_statement(t);
  ifdebug(3)
    {
      fprintf(stderr,"\nGenerated statement:");
      print_statement(smt);
    }
  print_text(out,text_statement(entity_undefined,0,smt,NIL));
  free(print_message), print_message = NULL;
  free(stop_message), stop_message = NULL;
}

/* This function generates code that verifies if all array elements in the array region
   are initialized or not. 
   We use the algorithm_row_echelon(initial system, list of variables to scan, 
   return condition,  returned enumeration system) to generate the nested loop.  
   The generated code will be like this:
   IF (condition) THEN
      DO PHI1 = low1,up1
	 DO PHI2 = low2,up2
	    IF (A(PHI1,PHI2).EQ.SNan) STOP "A is used before set"
	 ENDDO
      ENDDO
   ENDIF  */

static void verify_array_variable(entity ent, region reg)
{
  statement smt = statement_undefined;
  statement l = statement_undefined;
  test t = test_undefined;
  list dims = variable_dimensions(type_variable(entity_type(ent)));
  int n = gen_length(dims);
  list l_phi = phi_entities_list(1,n);
  reference ref = make_reference(ent,entities_to_expressions2(l_phi));
  expression exp = reference_to_expression(ref);
  Pbase phi_variables = entity_list_to_base(l_phi);
  Psysteme ps = region_system(reg);
  Psysteme row_echelon = SC_UNDEFINED, condition = SC_UNDEFINED;
  string stop_message  = strdup(concatenate("\'Array ",entity_name(ent)," is used before set\'",NULL));
  string print_message  = strdup(concatenate("\'Array element ",entity_name(ent),"(\'",
					     print_list_of_entities(l_phi),",\') is used before set\'",NULL));
  expression cond = make_test_condition(exp,ent);
  basic b = entity_basic(ent);
  ifdebug(3)
    {
      pips_debug(3,"Verify array region:");
      print_region(reg);
    }
  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
    {   
      smt = make_print_statement(print_message); 
      /*To help debugging, reinitialize the integer and real array elements by 0 in order to 
	avoid so much propagated NaN or 1000000 values. By doing this, we cannot
	detect all uninitialized variables but it helps Corinne, with thousand lines of code */
      if ((basic_tag(b)==0) || (basic_tag(b)==1))
	{
	  statement s = make_assign_statement(exp,int_to_expression(0));  
	  /*insert the assignment after the PRINT statement*/
	  insert_statement(smt,s,false); 
	}
    }
  else
    smt = make_stop_statement(stop_message); 
  t = make_test(cond,smt,make_block_statement(NIL));
  smt = test_to_statement(t);
  ps = remove_temporal_variables_from_system(ps); // remove I#init like variables 

  /*to simplify the test condition, we have to remove preconditions from regions*/

  algorithm_row_echelon(ps,phi_variables, &condition, &row_echelon);
  sc_find_equalities(&condition);
  /* If no bound is found for a variable PHIi => it is not the case because the
     region is MUST => TO PROVE*/
  l = systeme_to_loop_nest(row_echelon,l_phi,smt,entity_intrinsic(DIVIDE_OPERATOR_NAME));
  smt = generate_optional_if(condition,l);
  ifdebug(3)
    {
      fprintf(stderr,"\nGenerated statement:");
      print_statement(smt);
    }
  print_text(out,text_statement(entity_undefined,0,smt,NIL));
  free(print_message), print_message = NULL;
  free(stop_message), stop_message = NULL;
}

static void verify_used_before_set_expression(expression exp,statement s);
static void verify_used_before_set_call(call c,statement s);

static void verify_used_before_set_call(call c,statement s)
{
  entity fun = call_function(c);
  list l_args = call_arguments(c);
  if (entity_module_p(fun))
    {
      /* If c is a procedure call, we have to update UBS resource in order to check 
	 for the corresponding variables in the frame of the called procedure. 
	 There are two possibilities: 
	 - if the current entity is passed as actual arguments 
	   insert (called procedure,offset of actual argument)
	 - the current entity is a common variable and it has IN regions in the called 
	   procedure => insert (called procedure,common variable)*/
      int i = 0;
      ifdebug(3)
	fprintf(stderr,"\nCall to other module\n");
      if (variable_in_common_p(current_entity))
	{
	  /* always add (called procedure,common variable) to ubs resource,
	     although current_entity may not in the IN regions of the called procedure */
	  ubs_check fp = make_ubs_check(fun,current_entity);
	  l_ubs_checks = gen_nconc(CONS(UBS_CHECK,fp,NIL),l_ubs_checks);
	}
      ifdebug(3)
	{
	  fprintf(stderr,"\nCall to %s with argument list:",entity_local_name(fun));
	  print_expressions(l_args);
	}
      MAP(EXPRESSION,exp,
      {
	list l_refs = expression_to_reference_list(exp, NIL);
	i++;
	MAP(REFERENCE,ref,
	{
	  entity var = reference_variable(ref);
	  if (same_entity_p(var,current_entity))
	    {
	      ubs_check fp = make_ubs_check(fun,make_integer_constant_entity(i));
	      ifdebug(4)
		{
		  fprintf(stderr,"\nFound at %d-th argument:",i);
		  print_expression(exp);
		  fprintf(stderr,"\nAdd ubs (%s,%d)\n",entity_local_name(fun),i);
		}
	      l_ubs_checks = gen_nconc(CONS(UBS_CHECK,fp,NIL),l_ubs_checks);
	      break;
	    }
	},l_refs);
      },l_args);		
    }
  else
    {
      /* The current statement is not a call to another routine, so an assignment, 
	 or a READ, a WRITE, ...
	 We have go though the argument list and generate a verification for each 
	 read (= IN region) of the current entity. This is really a redundant work
	 with region's computation :-( 
	 So may be it is better to keep the list of regions, not a convex hull. 
	 
	 Examples :  A = C(J) + FOO(C(I)), X(A(I)) = A(J) + A(K)
	      READ *, A(B(I)), A(B(J)) 

	 SO FOR THE MOMENT, I ONLY TREAT SOME FREQUENT CASES: assignment, WRITE*/
      ifdebug(3)
	fprintf(stderr,"\nMAY IN region of elementary statement\n");
      MAP(EXPRESSION,exp,
      {
	verify_used_before_set_expression(exp,s);
      },l_args);
    }
}

static void verify_used_before_set_expression(expression exp,statement s)
{
  syntax syn = expression_syntax(exp);
  tag t = syntax_tag(syn);  
  ifdebug(4)
    {
      fprintf(stderr,"\nVerify expression\n");
      print_expression(exp);
    }
  switch (t){ 
  case is_syntax_range:  
    break;
  case is_syntax_reference:
    { 
      reference ref = syntax_reference(syn);
      entity var = reference_variable(ref);
      if (same_entity_p(var,current_entity))
	{
	  int order = statement_ordering(s);
	  ifdebug(3)
	    fprintf(stderr,"\nInsert a VERIFY function before a reference\n");
	  number_of_added_verifications++;
	  fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_mod),
		  ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
	  if (entity_scalar_p(var))  /*exists ? MAY IN for a MUST IN?*/
	    verify_scalar_variable(var);
	  else
	    verify_array_element(var,exp);
	  fprintf(out,"%s\n",PREFIX2);
	}	
      break;
    }
  case is_syntax_call:
    {
      call c = syntax_call(syn);
      verify_used_before_set_call(c,s);
      break;
    }
  default:
    {
      pips_internal_error("Unexpected expression tag %d ", t );
      break;
    } 
  }	
}

static bool verify_used_before_set_statement_flt(statement s)
{
  list l_in_regions = load_statement_in_regions(s);
  ifdebug(3)
    {
      pips_debug(3,"Verify the current statement:");
      print_statement(s);
      fprintf(stderr,"with list of IN regions:");
      print_inout_regions(l_in_regions);
    }  
  MAP(REGION,reg,
  {
    /* there are IN regions like IN-EXACT-{0==-1} !!!*/
    if (!region_empty_p(reg))
      {
	entity ent = region_entity(reg);
	if (same_entity_p(ent,current_entity))
	  {
	    approximation app = region_approximation(reg);
	    ifdebug(3)
	      fprintf(stderr,"\nFound variable %s in the IN region list\n",entity_name(ent));
	    if (approximation_exact_p(app) && 
	       (!variable_in_common_p(current_entity) || 
	       (variable_in_common_p(current_entity) && 
		common_variable_in_module_scope_p(current_entity,current_mod))))
	      {	
		/* MUST IN region and variable belonging to current module scope 
		   (local, formal and visible global variable)*/
		int order = statement_ordering(s);
		ifdebug(3)
		  fprintf(stderr,"\nMUST IN region, variable in module scope\n");
		number_of_added_verifications++;
		fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_mod),
			ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
		if (entity_scalar_p(ent))
		  verify_scalar_variable(ent);
		else
		  verify_array_variable(ent,reg);
		fprintf(out,"%s\n",PREFIX2);
		return false;
	      }
	    else 
	      {
		/* MAY IN region or common variable not in current module scope 
		   If the current statement is elementary, we have to treat it, 
		   else, we continue to go down with gen_recurse */
		instruction i = statement_instruction(s);
		tag t = instruction_tag(i); 
		ifdebug(3)
		  fprintf(stderr,"\nMAY IN region or common variable not in current module scope\n");
		switch(t)
		  {
		  case is_instruction_call:
		    {	
		      call c = statement_call(s);
		      verify_used_before_set_call(c,s);
		      return false;
		    } 
		  case is_instruction_whileloop:
		    {
		      whileloop wl = instruction_whileloop(i); 	
		      expression e = whileloop_condition(wl);	
		      verify_used_before_set_expression(e,s);
		    }
		  case is_instruction_test:
		    {
		      test it = instruction_test(i);
		      expression e = test_condition(it);
		      verify_used_before_set_expression(e,s);
		    }
		  case is_instruction_sequence: 
		  case is_instruction_loop:
		    // suppose that there are only MUST IN region in loop's range
		  case is_instruction_unstructured:
		    return true;
		  }  
	      }
	  }
      }
  },l_in_regions);
  return false;
}

static void verify_used_before_set_statement()
{
  gen_recurse(module_statement, 
	      statement_domain,
	      verify_used_before_set_statement_flt,
	      gen_null);
}

static void verify_formal_and_common_variables(entity ent,list l_callers)
{  
  bool check = false;	    
  MAP(STRING,caller_name,
  { 
    list l_caller_ubs = ubs_list((ubs)db_get_memory_resource(DBR_UBS,caller_name,true));
    MAP(UBS_CHECK,fp,
    {
      entity mod = ubs_check_module(fp);	 
      if (same_entity_p(mod,current_mod))
	{
	  entity e = ubs_check_variable(fp);
	  int off;
	  if (integer_constant_p(e,&off))
	    {
	      if (formal_parameter_p(ent)) 
		{
		  int offset = formal_offset(storage_formal(entity_storage(ent)));
		  if (off == offset)
		    {
		      pips_debug(1,"Formal parameter %s must be verified\n",entity_name(ent));
		      check = true;
		      break;
		    }
		}
	    }
	  else
	    {
	      if (entities_may_conflict_p(e,ent))
		{
		  pips_debug(1,"Common variable %s must be verified\n",entity_name(ent));
		  check = true;
		  break;
		}
	    }
	}
    },l_caller_ubs);
    if (check) break;
  },l_callers);
  if (check)
    {
      current_entity = ent;
      verify_used_before_set_statement();
      current_entity = entity_undefined;
    }
}

/* This function prints a common variable with its numerical size, because
   we do not want to generate the PARAMETER declarations
     PARAMETER (NX=50)
     COMMON W(5*NX)
  => COMMON W(250)*/

static list words_numerical_dimension(dimension obj)
{
  list pc = NIL;
  expression low_exp = dimension_lower(obj);
  expression up_exp = dimension_upper(obj);
  normalized n_low = NORMALIZE_EXPRESSION(low_exp);
  normalized n_up = NORMALIZE_EXPRESSION(up_exp);
  int low,up;
  if (EvalNormalized(n_low,&low))
    {
      if (low!=1)
	{
	  pc = CHAIN_SWORD(pc,i2a(low));
	  pc = CHAIN_SWORD(pc,":");
	}
    }
  else
    {
      pc = words_expression(low_exp,NIL);
      pc = CHAIN_SWORD(pc,":");
    }
  if (EvalNormalized(n_up,&up))
    pc = CHAIN_SWORD(pc,i2a(up));
  else
    pc = gen_nconc(pc, words_expression(up_exp,NIL));
  return(pc);
}

static list words_common_variable(entity e)
{
  list pl = NIL;
  pl = CHAIN_SWORD(pl, entity_local_name(e));
  if (type_variable_p(entity_type(e)))
    {
      if (variable_dimensions(type_variable(entity_type(e))) != NIL)
	{
	  list dims = variable_dimensions(type_variable(entity_type(e)));
	  pl = CHAIN_SWORD(pl, "(");
	  MAPL(pd,
	  {
	    pl = gen_nconc(pl, words_numerical_dimension(DIMENSION(CAR(pd))));
	    if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, ",");
	  }, dims);
	  pl = CHAIN_SWORD(pl, ")");
	}
    }
  return(pl);
}

static void insert_common_declaration(entity ent,entity sec)
{
  const char* mod_name = entity_module_name(ent);
  entity mod = local_name_to_top_level_entity(mod_name);
  list entities = common_members_of_module(sec,mod,true);
  ifdebug(3)
    {
      fprintf(stderr,"\nList of entities in the common declaration");
      print_entities(entities);
    }
  if (!ENDP(entities))
    {
      const char* area_name = module_local_name(sec);
      bool comma = false;
      fprintf(out2,"      COMMON ");
      if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) != 0) 
	fprintf(out2,"/%s/ ", area_name);
      MAP(ENTITY, ee, 
      {
	if (comma) fprintf(out2,",");
	else comma = true;
	fprintf(out2,"%s",words_to_string(words_common_variable(ee)));
      }, entities);
      fprintf(out2,"\n");
    }
  gen_free_list(entities);
}	

static void insert_initialization(entity ent)
{
  if (entity_scalar_p(ent))
    {
      number_of_may_uninitialized_scalar_variables++;
      initialize_scalar_variable(ent);
    }
  else
    {
      number_of_may_uninitialized_array_variables++;
      initialize_array_variable(ent);	
    }
}

static void initialize_and_verify_common_variable(entity ent, region reg)
{
  /* Check if ent has already been initialized by a BLOCK DATA subprogram.
     Check if ent can also be initialized by an EQUIVALENCE variable */
  storage s = entity_storage(ent);
  ram r = storage_ram(s);
  entity sec = ram_section(r);
  list l_layout = area_layout(type_area(entity_type(sec)));
  approximation app = region_approximation(reg);
  /* ram_shared does not work so we use common layout*/
  MAP(ENTITY,other,
  {
    if (entities_may_conflict_p(ent,other))
      {
	const char* mod_name = entity_module_name(other);
	entity mod = local_name_to_top_level_entity(mod_name);
	if (entity_blockdata_p(mod))
	  {
	    if (entity_scalar_p(ent))
	      user_log("\nCommon scalar variable %s is initialized by BLOCKDATA\n",entity_name(ent));
	    else 
	      user_log("\nCommon array variable %s is fully initialized by BLOCKDATA???\n",
		       entity_name(ent));
	    return;
	  }
      }
  },l_layout);
  if (approximation_exact_p(app))
    {	
      pips_debug(2,"MUST IN at module statement\n");
      user_log("\nCommon variable %s is used before set\n",entity_name(ent));
      if (entity_scalar_p(ent))
	number_of_uninitialized_scalar_variables++;
      else
	number_of_uninitialized_array_variables++;
      fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_local_name(current_mod));
      fprintf(out,"      STOP 'Variable %s is used before set'\n",entity_name(ent));
      fprintf(out,"%s\n",PREFIX2);
    }
  else
    {
      pips_debug(2,"MAY IN at module statement\n");
      user_log("\nCommon variable %s maybe used before set\n",entity_name(ent));  
      if (common_variable_in_module_scope_p(ent,current_mod))
	{  
	  fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_local_name(current_mod));   
	  insert_initialization(ent);  
	  fprintf(out,"%s\n",PREFIX2);
	}
      else 
	{
	  /* ent is not in the main module scope.
	     Since there may be variables in different common blocks with the same name => it is safe
	     to add CALL INITIALIZATION_COMMONNAME for each common block, then add in subroutine
	     INITIALIZATION_COMMONNAME the common, type, parameter declaration + common variable 
	     initializations */	
	  entity sec = ram_section(storage_ram(entity_storage(ent)));	
	  const char* area_name = module_local_name(sec);
	  ifdebug(1)
	    fprintf(stderr,"\nCommon variable %s not in main module scope\n",entity_name(ent)); 
	  if (!entity_in_list_p(sec,l_initialized_commons)) 
	    {
	      fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_local_name(current_mod));
	      if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) == 0) 
		{
		  fprintf(out,"      CALL INITIALIZE_BLANK\n");
		  fprintf(out2,"      SUBROUTINE INITIALIZE_BLANK\n");
		}
	      else
		{
		  fprintf(out,"      CALL INITIALIZE_%s\n",area_name);
		  fprintf(out2,"      SUBROUTINE INITIALIZE_%s\n",area_name);
		}
	      fprintf(out,"%s\n",PREFIX2);
	      insert_common_declaration(ent,sec);
	      fprintf(out2,"C (0,1)\n");
	      fprintf(out2,"      END\n");
	      l_initialized_commons = gen_nconc(l_initialized_commons,CONS(ENTITY,sec,NIL));
	    }
	  if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) == 0) 
	    fprintf(out,"%s\t%s\tINITIALIZE_BLANK\t(0,1)\n",PREFIX1,initialization_file);   
	  else
	    fprintf(out,"%s\t%s\tINITIALIZE_%s\t(0,1)\n",PREFIX1,initialization_file,area_name);   
	  insert_initialization(ent);
	  fprintf(out,"%s\n",PREFIX2);
	}
      /* TO BE IMPROVED, do not verify unvisible common variable*/
      current_entity = ent;
      verify_used_before_set_statement();
      current_entity = entity_undefined;	  
    }
}

static void initialize_and_verify_local_variable(entity ent,region reg)
{
  /* Check if ent has already been initialized by a DATA statement. There are 3 cases:
     1. ent is not initialized by any DATA statement
     2. ent is fully initialized by DATA statements
     3. ent is partially initialized, insert a sequence of assignments corresponding to 
        the DATAs 
     Check if ent can also be initialized by an EQUIVALENCE variable */
  if (variable_static_p(ent))
    {
      /* Local variable ent is in a DATA or SAVE statement (distinguish DATA vs SAVE ???) */
      if (entity_scalar_p(ent))
	user_log("\nLocal scalar variable %s is initialized by DATA\n",entity_name(ent));
      else 
	user_log("\nLocal array variable %s is fully initialized by DATA???\n",entity_name(ent));
    }
  else
    {
      if (initialized_by_equivalent_variable_p(ent))
	/* test to rewrite */
	user_log("\nLocal variable %s is initialized through EQUIVALENCE\n",entity_name(ent));
      else
	{
	  approximation app = region_approximation(reg);
	  if (approximation_exact_p(app))
	    {	
	      pips_debug(2,"MUST IN at module statement\n");
	      user_log("\nLocal variable %s is used before set\n",entity_name(ent));
	      if (entity_scalar_p(ent))
		number_of_uninitialized_scalar_variables++;
	      else
		number_of_uninitialized_array_variables++;
	      fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_local_name(current_mod));
	      fprintf(out,"      STOP 'Variable %s is used before set'\n",entity_name(ent));
	      fprintf(out,"%s\n",PREFIX2);
	    }
	  else
	    {
	      pips_debug(2,"MAY IN at module statement\n");
	      user_log("\nLocal variable %s maybe used before set\n",entity_name(ent));
	      fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_local_name(current_mod));
	      insert_initialization(ent);
	      fprintf(out,"%s\n",PREFIX2);
	      current_entity = ent;
	      verify_used_before_set_statement();
	      current_entity = entity_undefined;	  
	    }
	}
    }
}
 
bool used_before_set(const char* module_name)
{ 
  list l_in_regions = NIL;
  ubs module_ubs; 
  string user_file = db_get_memory_resource(DBR_USER_FILE,module_name,true);
  string base_name = pips_basename(user_file, NULL);
  /* File instrument.out is used to store ubs checks*/
  string dir_name = db_get_current_workspace_directory();
  string instrument_file = strdup(concatenate(dir_name, "/instrument.out", NULL));
  out = safe_fopen(instrument_file, "a");  
  file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), 
				 "/",base_name,NULL));
  number_of_processed_modules ++;
  if (!same_string_p(rule_phase(find_rule_by_resource("REGIONS")),"MUST_REGIONS"))
    pips_user_warning("\nMUST REGIONS not selected - " "Do not expect wonderful results\n");
  /* Set and get the current properties concerning regions */
  set_bool_property("MUST_REGIONS", true);
  set_bool_property("EXACT_REGIONS", true);
  get_regions_properties();
  current_mod = local_name_to_top_level_entity(module_name);
  set_current_module_entity(current_mod);
  /* Get the code of the module */
  module_statement = (statement) db_get_memory_resource(DBR_CODE,module_name,true);
  set_current_module_statement(module_statement);
  /* Get IN regions of the module */
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS,module_name,true));
  regions_init(); 
  set_ordering_to_statement(module_statement);  
  debug_on("USED_BEFORE_SET_DEBUG_LEVEL");
  l_in_regions = load_statement_in_regions(module_statement);
  ifdebug(2)
    {
      pips_debug(2,"List of IN regions of module %s:",module_name);
      print_inout_regions(l_in_regions);
    }
  fprintf(out,"%s\t%s\t%s\t(0,1)\n",PREFIX1,file_name,module_name);
  fprintf(out,"      EXTERNAL ir_isnan,id_isnan\n");
  fprintf(out,"      LOGICAL*4 ir_isnan,id_isnan\n");
  fprintf(out,"%s\n",PREFIX2);
  if (entity_main_module_p(current_mod))
    {
      /* File xxx.database/Src/initialization.f contains all the initializations of 
	 global variables */
      initialization_file = strdup(concatenate(dir_name, "/Src/initialization.f", NULL));
      out2 = safe_fopen(initialization_file, "a");  
      MAP(REGION,reg,
      {
	entity ent = region_entity(reg);
	if (strcmp(entity_module_name(ent),IO_EFFECTS_PACKAGE_NAME)!=0)
	  {
	    if (variable_in_common_p(ent))
	      /* Common variable in main program */
	      initialize_and_verify_common_variable(ent,reg);
	    else 
	      {
		/* Local variable in main program, but attention, 
		   IN regions contain also static variables of other modules !!!*/		
		if (local_name_to_top_level_entity(entity_module_name(ent)) == current_mod)
		  initialize_and_verify_local_variable(ent,reg);
	      }
	  }
      },l_in_regions);
      safe_fclose(out2,initialization_file);
      free(initialization_file), initialization_file = NULL;
    }
  else
    {    
      callees callers = (callees) db_get_memory_resource(DBR_CALLERS,module_name,true);
      list l_callers = callees_callees(callers); 
      MAP(REGION,reg,
      {
	entity ent = region_entity(reg);
	if (strcmp(entity_module_name(ent),IO_EFFECTS_PACKAGE_NAME)!=0)
	  {
	    if (formal_parameter_p(ent)||variable_in_common_p(ent))
	      /* Formal or common variable */
	      verify_formal_and_common_variables(ent,l_callers);
	    else 
	      {
		/* Local variable */
		if (local_name_to_top_level_entity(entity_module_name(ent)) == current_mod)
		  initialize_and_verify_local_variable(ent,reg);
	      }
	  }
      },l_in_regions); 
    }
  module_ubs = make_ubs(l_ubs_checks);
  //  message_assert("module ubs is consistent",ubs_consistent_p(module_ubs));
  /* save to resource */
  DB_PUT_MEMORY_RESOURCE(DBR_UBS,module_name,copy_ubs(module_ubs));  
  display_used_before_set_statistics();
  debug_off(); 
  safe_fclose(out,instrument_file);
  free(instrument_file), instrument_file = NULL;
  free(dir_name), dir_name = NULL;
  current_mod = entity_undefined;
  reset_ordering_to_statement();
  regions_end();
  reset_in_effects();
  reset_current_module_statement(); 
  reset_current_module_entity();
  return true;
}






