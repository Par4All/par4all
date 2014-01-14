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
/******************************************************************
 *
 *		     ALIAS VERIFICATION
 *
 *
 In Fortran 77, parameters are passed by address in such a way that, 
 as long as the actual argument is associated with a named storage 
 location, the called subprogram can change the value of the actual
 argument by assigning a value to the corresponding formal parameter. 
 So new aliases can be created between formal parameters if the same 
 actual argument is passed to two or more formal parameters, or 
 between formal parameters and global parameters if an actual
 argument is an object in common storage which is also visible in the
 called subprogram or other subprograms in the call chain below it.

 Restrictions on association of entities in Fortran 77 (Section 15.9.3.6 
 ANSI83) say that neither aliased formal parameters nor the variable 
 in the common block may become defined during execution of the called 
 subprogram or the others subprograms in the call chain.

 This phase verifies statically if the program violates the standard 
 restriction on alias or not by using information from the alias 
 propagation phase. If these informations are not known at compile-time, 
 we instrument the code with tests that check the violation dynamically 
 during execution of program.

 For each alias_association of a formal variable F, check for 2 cases:

 Case 1: there is other formal variable F' that has a same section by an
 included call path => test(F,F')

 Case 2: 
 2.1 there is a common variable W in current module that has same section
 with F => test(F,W)
 2.2 there is a common variable V in some callee (direct and indirect) of the 
 current  module that has same section with F => test(F,V)

 Test(A,B):
 1. No intersection between A and B => OK
 2. Intersection
    2.1 no write on one variable => OK
    2.2 write on one varibale => violation
    2.3 Undecidable
 3. Undecidable

 In case of undecidable, we try to repeat Test(A,B) in one higher call site 
 in the call path. And if we return to undecidable cases, we use dynamic checks
 
*******************************************************************/

/* TO AMELIORATE :
   for the moment, we only use trivial_expression_p to compare offsets + array sizes 
   CAN TAKE MORE INFORMATION from array declarations, A(l:u) => u>l
   not precondtitions (corrupted)*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "alias_private.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "properties.h"
#include "effects-generic.h"
#include "instrumentation.h"
#include "text-util.h" /* for words_to_string*/

#define PREFIX1 "$ALIAS_CHECK"
#define PREFIX2 "$ALIAS_CHECK_END"
#define ALIAS_FLAG "ALIAS_FLAG"
#define ALIAS_FUNCTION "DYNAMIC_ALIAS_CHECK"

typedef struct 
{
  entity first_entity;
  entity second_entity;
  expression offset1;
  expression offset2;
  expression condition;
  expression flags;
  list path;
  bool insert;
} alias_context_t,
* alias_context_p;

static entity current_mod = entity_undefined;
static entity current_caller = entity_undefined; 
static const char* caller_name;
static call current_call = call_undefined; 
static list l_module_aliases = NIL;
static statement module_statement = statement_undefined;
static statement current_statement = statement_undefined;
static int current_ordering = 0;
static FILE * out;
static entity alias_flag = entity_undefined;
static entity alias_function = entity_undefined;
/* This list tells us if two variables have been checked dynamically or not*/
static list l_dynamic_check = NIL; 
static int unique_flag_number = 0; // a counter for alias flags
static int number_of_tests = 0;
static int number_of_calls = 0;
static int number_of_processed_modules = 0;


expression simplify_relational_expression(expression e)
{  
  if (relational_expression_p(e))
    {
      /* If e is a relational expression*/
      list args = call_arguments(syntax_call(expression_syntax(e)));
      expression e1 =  EXPRESSION(CAR(args));
      expression e2 = EXPRESSION(CAR(CDR(args)));    
      if (!expression_undefined_p(e1) && !expression_undefined_p(e2))
	{
	  normalized n1 = NORMALIZE_EXPRESSION(e1);
	  normalized n2 = NORMALIZE_EXPRESSION(e2);
	  if (normalized_linear_p(n1) && normalized_linear_p(n2))
	    {
	      entity op = call_function(syntax_call(expression_syntax(e)));
	      Pvecteur v1 = normalized_linear(n1);
	      Pvecteur v2 = normalized_linear(n2);
	      Pvecteur v = vect_substract(v1,v2);
	      expression new_exp;
	      vect_normalize(v);
	      new_exp = Pvecteur_to_expression(v);
	      return binary_intrinsic_expression(entity_local_name(op),new_exp,int_to_expression(0));
	    }
	}
    }
  return e;
}

static void display_alias_check_statistics()
{
  user_log("\n Number of flags : %d", unique_flag_number); 
  user_log("\n Number of tests : %d", number_of_tests);
  user_log("\n Number of calls : %d", number_of_calls);
  user_log("\n Number of processed modules: %d\n",number_of_processed_modules); 
}

static void initialize_dynamic_check_list()
{
  list l_decls = code_declarations(entity_code(current_mod));
  list l_formals = NIL;
  list l_commons = NIL;
  /* search for formal parameters in the declaration list */   
  MAP(ENTITY, e,
  {
    if (formal_parameter_p(e))
      l_formals = gen_nconc(l_formals,CONS(ENTITY,e,NIL));
  },
      l_decls);
  MAP(ENTITY, e,
  {
    if (variable_in_common_p(e))
      l_commons = gen_nconc(l_commons,CONS(ENTITY,e,NIL));
  },
      l_decls);

  MAP(ENTITY,e1,
  {
    MAP(ENTITY,e2,
    {
      dynamic_check dc = make_dynamic_check(e1,e2,false);
      l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
    },l_formals);
    MAP(ENTITY,e2,
    {
      dynamic_check dc = make_dynamic_check(e1,e2,false);
      l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
    },l_commons);
  },l_formals);
}

static bool dynamic_checked_p(entity e1, entity e2)
{
  MAP(DYNAMIC_CHECK,dc,
  { 
    if ((dynamic_check_first(dc)==e1)&&(dynamic_check_second(dc)==e2))
      return dynamic_check_checked(dc);
  }, l_dynamic_check);
  return false;
}

static void set_dynamic_checked(entity e1, entity e2)
{
  MAP(DYNAMIC_CHECK,dc,
  { 
    if ((dynamic_check_first(dc)==e1)&&(dynamic_check_second(dc)==e2))
      dynamic_check_checked(dc) = true;
  }, l_dynamic_check);
}

static bool same_call_site_p(call_site cs1, call_site cs2)
{
  entity f1 = call_site_function(cs1);
  entity f2 = call_site_function(cs2); 
  int o1 = call_site_ordering(cs1);
  int o2 = call_site_ordering(cs2);
  return (same_entity_p(f1,f2) && (o1==o2));
}

/****************************************************************************

 This function returns true if one list is included in other : 
 l1=conc(l2,l) or l2=conc(l1,l)               

*****************************************************************************/

bool included_call_chain_p(list l1, list l2)
{
  while (!ENDP(l1) && !ENDP(l2))
    {
      call_site cs1 = CALL_SITE(CAR(l1));
      call_site cs2 = CALL_SITE(CAR(l2));
      if (!same_call_site_p(cs1,cs2))  return false;
      l1 = CDR(l1);
      l2 = CDR(l2);
    }
  return true;
}

/****************************************************************************

 This function returns true if l1 = conc(cs,l2)               

*****************************************************************************/

static bool tail_call_path_p(call_site cs, list l1, list l2)
{
  if (gen_length(l1) == gen_length(l2)+1)
    {
      call_site cs1 = CALL_SITE(CAR(l1));
      return (same_call_site_p(cs,cs1) && included_call_chain_p(l2,CDR(l1)));
    }
  return false;
}

/* This function prints the call path , including names of caller functions 
   and orderings of call sites in their corresponding functions */
static string print_call_path(list path)
{
  list pc = NIL;
  MAP(CALL_SITE,casi,
  {
    entity casifunc = call_site_function(casi);
    int casiord = call_site_ordering(casi);
    pc = CHAIN_SWORD(pc,"(");
    pc = CHAIN_SWORD(pc,module_local_name(casifunc));
    pc = CHAIN_SWORD(pc,":(");
    pc = CHAIN_SWORD(pc,i2a(ORDERING_NUMBER(casiord)));
    pc = CHAIN_SWORD(pc,",");
    pc = CHAIN_SWORD(pc,i2a(ORDERING_STATEMENT(casiord)));
    pc = CHAIN_SWORD(pc,")) ");
  },path);
  return words_to_string(pc);
}

/*****************************************************************************

 This function prints an alias_association                 

*****************************************************************************/

void  print_alias_association(alias_association aa)
{
  entity e = alias_association_variable(aa);
  entity sec = alias_association_section(aa);
  expression exp = alias_association_offset(aa);
  list path = alias_association_call_chain(aa);
  int l = alias_association_lower_offset(aa);
  int u = alias_association_upper_offset(aa);
  fprintf(stderr,"\n Alias association :");
  fprintf(stderr,"\n Formal variable %s with",entity_name(e));
  fprintf(stderr, "\n section :%s", entity_name(sec));
  fprintf(stderr, "\n offset :");
  print_expression(exp);
  fprintf(stderr, "lower offset :%d, upper offset: %d \n",l,u);
  fprintf(stderr, "call path :%s \n", print_call_path(path)); 
}

/*****************************************************************************

 This function prints the list of alias_association                 

*****************************************************************************/

void  print_list_of_alias_associations(list l)
{
  MAP(ALIAS_ASSOCIATION, aa,{
    print_alias_association(aa);
  },l);    
}

/*****************************************************************************

 This function returns the size of an array minus 1, multiplied by array element 
 (D1*D2*...*Dn-1)* element_size     
 
*****************************************************************************/

static expression array_size_stride(entity ent)
{
  expression exp = expression_undefined;
  if (!assumed_size_array_p(ent) && !pointer_type_array_p(ent))
    {
      variable var = type_variable(entity_type(ent));   
      list l_dims = variable_dimensions(var);
      int num_dim = gen_length(l_dims),j;
      basic b = variable_basic(type_variable(entity_type(ent)));
      expression e_size = int_to_expression(SizeOfElements(b));
      for (j=1; j<= num_dim; j++)
	{
	  dimension dim_j = find_ith_dimension(l_dims,j);
	  expression lower_j = dimension_lower(dim_j);
	  expression upper_j = dimension_upper(dim_j);
	  expression size_j;
	  if (expression_constant_p(lower_j) && (expression_to_int(lower_j)==1))
	    size_j = copy_expression(upper_j);
	  else 
	    {
	      size_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,upper_j,lower_j);
	      size_j =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						    copy_expression(size_j),int_to_expression(1));
	    }
	  if (expression_undefined_p(exp))
	    exp = copy_expression(size_j);
	  else
	    exp = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
					      copy_expression(exp),size_j);  
	}
      exp = binary_intrinsic_expression(MINUS_OPERATOR_NAME,copy_expression(exp),int_to_expression(1));
      exp = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(exp),e_size);
      ifdebug(2)
	{
	  fprintf(stderr, "\n Stride of array size : \n");
	  print_expression(exp);
	}
    }
  /* else return expression_undefined with assumed-size or pointer-type array*/
  return exp;
}

/*****************************************************************************

   This function inserts flags and condition (if exists) before current 
   statement in module : IF (cond && F1 && F2) STOP "Alias violation"
   
*****************************************************************************/

static void insert_test_before_statement(expression flags, expression condition,
					 statement s, entity e1, entity e2,list path)
{
  expression cond;
  statement smt;
  int order = statement_ordering(s);
  string message = strdup(concatenate("\'Alias violation in module ",
				      module_local_name(current_mod),": write on ",
				      entity_local_name(e1),", aliased with ",
				      entity_local_name(e2)," by call path ",
				      print_call_path(path),"\'", NULL));
  const char* mod_name = module_local_name(current_mod);
  string user_file = db_get_memory_resource(DBR_USER_FILE,mod_name,true);
  string base_name = pips_basename(user_file, NULL);
  string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
  if (true_expression_p(condition))
    cond = copy_expression(flags);
  else
    cond = and_expression(condition,flags);
  if (get_bool_property("PROGRAM_VERIFICATION_WITH_PRINT_MESSAGE"))
    smt = test_to_statement(make_test(cond, make_print_statement(message),
				      make_block_statement(NIL)));
  else
    smt = test_to_statement(make_test(cond, make_stop_statement(message),
				      make_block_statement(NIL)));
  fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_mod),
	  ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
  print_text(out, text_statement(entity_undefined,0,smt,NIL));
  fprintf(out,"%s\n",PREFIX2);
  number_of_tests++;
  free(file_name), file_name = NULL;
  free(message), message = NULL;
}

/*****************************************************************************

   This function inserts a flag before each call site in call path :
   ALIAS_FLAG(i) = .TRUE.

*****************************************************************************/

static void insert_flag_before_call_site(list flags,list path)
{
  while (!ENDP(path))
    {
      call_site cs = CALL_SITE(CAR(path));
      expression e_flag = EXPRESSION(CAR(flags));
      entity caller = call_site_function(cs);
      int order = call_site_ordering(cs);
      statement s_flag = make_assign_statement(e_flag,make_true_expression());
      const char* call_name = module_local_name(caller);
      string user_file = db_get_memory_resource(DBR_USER_FILE,call_name,true);
      string base_name = pips_basename(user_file, NULL);
      string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
      fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(caller),
	      ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
      print_text(out,text_statement(entity_undefined,0,s_flag,NIL));
      fprintf(out, "%s\n",PREFIX2);
      path = CDR(path);
      flags = CDR(flags);
      free(file_name), file_name = NULL;
    }
}

/*****************************************************************************

   This function inserts flags and condition (if exists) before current 
   call site : IF (cond) ALIAS_FLAG(i) =.TRUE.
   
*****************************************************************************/

static void insert_test_before_caller(expression condition, expression e_flag)
{
  statement s_flag = make_assign_statement(e_flag,make_true_expression());
  string user_file = db_get_memory_resource(DBR_USER_FILE,caller_name,true);
  string base_name = pips_basename(user_file, NULL);
  string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
  if (true_expression_p(condition))
    {
      fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_caller),
	      ORDERING_NUMBER(current_ordering),ORDERING_STATEMENT(current_ordering));
      print_text(out,text_statement(entity_undefined,0,s_flag,NIL));
      fprintf(out,"%s\n",PREFIX2);
    }
  else
    {
      statement smt = test_to_statement(make_test(condition, s_flag,
						  make_block_statement(NIL)));
      fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_caller),
	      ORDERING_NUMBER(current_ordering),ORDERING_STATEMENT(current_ordering));
      print_text(out, text_statement(entity_undefined,0,smt,NIL));
      fprintf(out,"%s\n",PREFIX2);
      number_of_tests++;
    }
  free(file_name), file_name = NULL;
}

static list make_list_of_flags(list path)
{
  int i;
  list retour = NIL;
  for (i=1;i<=gen_length(path);i++)
    {
      expression e_flag = reference_to_expression(make_reference(alias_flag,
				   CONS(EXPRESSION,int_to_expression(++unique_flag_number),NIL)));
      ifdebug(2)
	{
	  fprintf(stderr, "\n New flag expression: \n");
	  print_expression(e_flag);
	}
      if (retour==NIL)
	retour = CONS(EXPRESSION,e_flag,NIL);
      else
	retour = gen_nconc(retour, CONS(EXPRESSION,e_flag,NIL));
    }
  return retour;
}

/* This function returns true if c is an user-defined function/subroutine*/
bool functional_call_p(call c)
{
  entity fun = call_function(c);
  return entity_module_p(fun);
}

static bool written = false;
static entity current_entity  = entity_undefined;
/* This function returns true if the variable is written directly in the current module, 
   not by its callees (do not take into account the effects on X,Y of statement like 
   CALL FOO(X,Y))*/

static bool variable_is_written_by_statement_flt(statement s)
{
  if (statement_call_p(s))
    {	
      if (!functional_call_p(statement_call(s)))
	{
	  list l_rw = load_proper_rw_effects_list(s);
	  MAP(EFFECT, eff,
	  {
	    action a = effect_action(eff);
	    if (action_write_p(a))
	      {
		reference r = effect_any_reference(eff);
		entity e = reference_variable(r);
		if (same_entity_p(e,current_entity))
		  {
		    ifdebug(3)
		      {
			fprintf(stderr,"\n Write on entity %s :\n",entity_name(e));
			fprintf(stderr,"\n Current entity %s :\n",entity_name(current_entity));
		      }
		    written = true;
		    //gen_recurse_stop(NULL);
		    return false;
		  }
	      }
	  },
	      l_rw); 
	  return false;
	}  
      return false;
    }
  return true; 
}

static bool variable_is_written_p(entity ent)
{
  written = false;
  current_entity = ent;
  gen_recurse(module_statement,statement_domain,
	      variable_is_written_by_statement_flt,gen_null);
  current_entity = entity_undefined;
  return written;
}

static void insert_check_alias_before_statement(entity e1, expression subval,
						entity e2, expression size,
						statement s)
{
  string message = strdup(concatenate("\'Alias violation in module ",
				      module_local_name(current_mod),": write on ",
				      entity_local_name(e1),", aliased with ",
				      entity_local_name(e2),"\'", NULL));
  list l = CONS(EXPRESSION,size,NIL);
  statement smt;
  int order = statement_ordering(s);
  const char* mod_name = module_local_name(current_mod);
  string user_file = db_get_memory_resource(DBR_USER_FILE,mod_name,true);
  string base_name = pips_basename(user_file, NULL);
  string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
  l = CONS(EXPRESSION,entity_to_expression(e2),l);
  l = CONS(EXPRESSION,subval,l);
  l = CONS(EXPRESSION,entity_to_expression(e1),l);
  l = CONS(EXPRESSION,MakeCharacterConstantExpression(message),l);
  smt = call_to_statement(make_call(alias_function,l));
  fprintf(out,"%s\t%s\t%s\t(%d,%d)\n",PREFIX1,file_name,module_local_name(current_mod),
	  ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
  print_text(out, text_statement(entity_undefined,0,smt,NIL));
  fprintf(out,"%s\n",PREFIX2);
  number_of_calls++;
  free(file_name), file_name = NULL;
}

/*****************************************************************************

 This function generates a CALL ALIAS_CHECK(e1,ref1,e2,size2) before each write on
 a may be aliased variable e1. checkalias() is a C function that takes the addresses of
 e1 and e2, writing reference ref1 and offset size2 as input, returns if there is
 alias violation on this writing or not.

 This type of dynamic checking is expensive, because we do not use information from
 static analysese => use it only when we can do nothing else.
 
*****************************************************************************/

static bool dynamic_alias_check_flt(statement s, alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);

  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		list l_inds = reference_indices(r);
		expression subval = subscript_value_stride(context->first_entity,l_inds);
		expression size;
		if (array_entity_p(context->second_entity))
		  size = array_size_stride(context->second_entity);
		else 
		  size = int_to_expression(0);
		if (!expression_undefined_p(size)) // not assumed-size array
		  insert_check_alias_before_statement(context->first_entity,subval,
						      context->second_entity,size,s);
		else
		  /* Assumed-size or pointer-type array, the size of dummy array can not be 
		     derived from the size of actual argument as we have no corresponding call chain*/
		  pips_user_warning("\nAssumed-size or pointer-type array \n");
		return false;
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if an aliased scalar variable is written or not.
 If yes, insert test before statement

*****************************************************************************/

static bool alias_check_scalar_variable_in_module_flt(statement s,
						      alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);
  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		insert_test_before_statement(context->flags,context->condition,s,
					     context->first_entity,context->second_entity,
					     context->path);
		context->insert = true;
		return false;
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if an array element which is aliased with a scalar variable 
 is written or not. Test : ref2 == off1 ? where ref2 = off2 + subval_stride2
 If yes, insert test before statement

*****************************************************************************/

static bool alias_check_array_and_scalar_variable_in_module_flt(statement s,
								alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);

  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		list l_inds = reference_indices(r);
		/* Attention : <may be written> V(*,*) => what kind of indices ???*/
		expression subval2 = subscript_value_stride(context->first_entity,l_inds);
		expression diff;
		int k;
		if (same_expression_p(context->offset1,context->offset2))
		  diff = eq_expression(int_to_expression(0),subval2);
		else
		  {
		    expression ref2;
		    if (expression_equal_integer_p(context->offset2,0))
		      ref2 = copy_expression(subval2);
		    else
		      ref2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,context->offset2,subval2);
		    diff = eq_expression(context->offset1,ref2);
		  }
		clean_all_normalized(diff);
		k = trivial_expression_p(diff);
		switch(k){
		case -1: 
		  /* off1 == ref2 is false =>  Okay, no write on aliased variable */
		  return false;
		default:
		  {
		    /* k = 0 or 1*/
		    if (k==0)
		      insert_test_before_statement(context->flags,simplify_relational_expression(diff),s,
						   context->first_entity,context->second_entity,
						   context->path);
		    else 
		      insert_test_before_statement(context->flags,make_true_expression(),s,
						   context->first_entity,context->second_entity,
						   context->path);
		    
		    context->insert = true;
		    return false;	       
		  }
		}
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if an array variable has written element which is aliased 
 with other array variable's element or not.
 Test : off2 <= ref1 <= off2+size_stride2 ? where ref1 = off1+subval_stride1
 If yes, insert test before statement

*****************************************************************************/

static bool alias_check_array_variable_in_module_flt(statement s,alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);

  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		list l_inds = reference_indices(r);
		/* Attention : <may be written> V(*,*) => what kind of indices ???*/
		expression subval1 = subscript_value_stride(context->first_entity,l_inds);
		expression diff1;
		int k1;
		expression ref1;
		if (expression_equal_integer_p(context->offset1,0))
		  ref1 = copy_expression(subval1);
		else
		  ref1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						     context->offset1,subval1);
		if (same_expression_p(context->offset1,context->offset2))
		  diff1 = ge_expression(subval1,int_to_expression(0));
		else
		  diff1 = ge_expression(ref1,context->offset2);
		clean_all_normalized(diff1);
		k1 = trivial_expression_p(diff1);
		switch(k1){
		case -1: 
		  /* off2 <= ref1 is false =>  Okay, no write on aliased variable */
		  return false;
		default:
		  {
		    expression size2 = array_size_stride(context->second_entity);
		    // test of assumed-size array is carried out before
		    expression diff2;
		    int k2;
		    if (same_expression_p(context->offset1,context->offset2))
		      diff2 = le_expression(subval1,size2);
		    else
		      {
			expression sum2;
			if (expression_equal_integer_p(context->offset2,0))
			  sum2 = copy_expression(size2);
			else
			  sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
							     context->offset2,size2);
			diff2 = le_expression(ref1,sum2);
		      }
		    clean_all_normalized(diff2);
		    k2 = trivial_expression_p(diff2);
		    switch(k2){
		    case -1: 
		      /* ref1 <= off2+size_stride2 is false =>  Okay, no write on aliased variable */
		      return false;
		    default:
		      {
			/* k1 = 0 or 1, k2 = 0 or 1*/
			if (k1+k2==0) // k1=k2=0
			  insert_test_before_statement(context->flags,
						       and_expression(simplify_relational_expression(diff1),
								      simplify_relational_expression(diff2)),
						       s,context->first_entity,context->second_entity,
						       context->path);
			else 
			  {
			    if (k1+k2==2) // k1=k2=1
			      insert_test_before_statement(context->flags,make_true_expression(),s,
							   context->first_entity,context->second_entity,
							   context->path);
			    else
			      {
				if (k1==0) // k1=0, k2=1
				  insert_test_before_statement(context->flags,
							       simplify_relational_expression(diff1),s,
							       context->first_entity,context->second_entity,
							       context->path);
				else // k2=0, k1=1
				  insert_test_before_statement(context->flags,
							       simplify_relational_expression(diff2),s,
							       context->first_entity,context->second_entity,
							       context->path);
			      }
			  }
			context->insert = true;
			return false;
		      }
		    }
		    return false;
		  }
		}
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if there is alias violation between two scalar variables or not.

 IF off1 != off2 => no alias, OK
 ELSE, off1 == off2 => alias, 
   IF e1 or e2 is written => alias violation
   => insert test before every statement where e1 and e2 must be written 

*****************************************************************************/

static void alias_check_two_scalar_variables_in_module(entity e1,entity e2,expression off1, 
						       expression off2,list path)
{
  expression diff = eq_expression(off1,off2);
  int k;
  clean_all_normalized(diff);
  k = trivial_expression_p(diff);
  /* Should we use also formal and actual binding / precondition ??? */
  ifdebug(2)
    fprintf(stderr,"\n Check in module, 2 scalar variables \n");
  switch(k){
  case -1: 
    /* off1 != off2 => Okay, no alias between these 2 variables */
    return;
  default:
    {
      /* insert flags before each call site in call path*/
      list l_flags = make_list_of_flags(path);
      alias_context_t context;
      context.insert = false;
      context.path = path;
      context.flags = expression_list_to_conjonction(l_flags);	  
      if (k==1)
	context.condition = make_true_expression();
      else   
	context.condition = simplify_relational_expression(diff);
      context.first_entity = e1;
      context.second_entity = e2;
      gen_context_recurse(module_statement,&context, statement_domain, 
			  alias_check_scalar_variable_in_module_flt, gen_null);      
      context.first_entity = e2;
      context.second_entity = e1;
      gen_context_recurse(module_statement,&context, statement_domain, 
			  alias_check_scalar_variable_in_module_flt, gen_null);
      if (context.insert) 
	insert_flag_before_call_site(l_flags,path);
      context.first_entity = entity_undefined;
      context.second_entity = entity_undefined;
      context.offset1 = expression_undefined;
      context.offset2 = expression_undefined;
      context.condition = expression_undefined;
      context.flags = expression_undefined;
      context.path = NIL;
    }
  }
}

/*****************************************************************************

 This function checks if there is alias violation between a scalar variable e1 
 and an array variable e2 or not. 

 IF off1 < off2 OR off1 > off2+size_stride2 => no alias, OK
 ELSE : off2 <= off1 <= off2+size_stride2 
       - IF e1 is written => alias violation 
       - IF e2 is written (ref2 == subval_stride2+off2) : 
           - IF ref2 != off1 => no alias violation 
	   - IF ref2 == off1 => alias violation 
     
*****************************************************************************/

static void alias_check_scalar_and_array_variables_in_module(entity e1,entity e2,
				       expression off1,expression off2,list path)
{
  expression diff1 = le_expression(off2,off1);
  int k1;
  clean_all_normalized(diff1);
  k1 = trivial_expression_p(diff1);
  ifdebug(2)
    fprintf(stderr,"\n Check in module, scalar and array variables \n");
  switch(k1){
  case -1: 
    /* off2 <= off1 is false => Okay, no alias between these 2 variables */
    return;
  default:
    {
      expression size2 = array_size_stride(e2);
      if (!expression_undefined_p(size2)) // not assumed-size array
	{
	  expression diff2;
	  int k2;
	  if (same_expression_p(off1,off2))
	    diff2 = le_expression(int_to_expression(0),size2);
	  else
	    {
	      expression sum2;
	      if (expression_equal_integer_p(off2,0))
		sum2 = copy_expression(size2);
	      else
		sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off2,size2);
	      diff2 = le_expression(off1,sum2);
	    }
	  clean_all_normalized(diff2);
	  k2 = trivial_expression_p(diff2);
	  switch(k2){
	  case -1: 
	    /* off1 <= off2+size_stride2 is false => Okay, no alias between these 2 variables */
	    return;
	  default:
	    {
	      /* k1 = 0 or 1, k2 = 0 or 1
		 insert flags before each call site in call path*/
	      list l_flags = make_list_of_flags(path);
	      alias_context_t context;
	      context.insert = false;
	      context.path = path;
	      context.flags = expression_list_to_conjonction(l_flags);
	      if (k1+k2==0) // k1=k2=0
		context.condition = and_expression(simplify_relational_expression(diff1),
						   simplify_relational_expression(diff2));
	      else 
		{
		  if (k1+k2==2) // k1=k2=1
		    context.condition = make_true_expression();
		  else
		    {
		      if (k1==0) // k1=0, k2=1
			context.condition = simplify_relational_expression(diff1); 
		      else // k2=0, k1=1
			context.condition = simplify_relational_expression(diff2);
		    }
		}
	      context.first_entity = e1;
	      context.second_entity = e2;
	      gen_context_recurse(module_statement,&context, statement_domain, 
				  alias_check_scalar_variable_in_module_flt,gen_null);
	      context.offset1 = off1;
	      context.offset2 = off2;
	      context.first_entity = e2;
	      context.second_entity = e1;
	      gen_context_recurse(module_statement,&context, statement_domain, 
				  alias_check_array_and_scalar_variable_in_module_flt,gen_null);	
	      if (context.insert)
		insert_flag_before_call_site(l_flags,path);
	      context.first_entity = entity_undefined;
	      context.second_entity = entity_undefined;
	      context.offset1 = expression_undefined;
	      context.offset2 = expression_undefined;
	      context.condition = expression_undefined;
	      context.flags = expression_undefined;
	      context.path = NIL;
	    }
	  }
	}
      else
	/* Assumed-size or pointer-type array, the size of dummy array can be 
	 derived from the size of actual argument, as we have the corresponding call chain*/
	pips_user_warning("\nAssumed-size or pointer-type array \n");
    }
  }
}

/*****************************************************************************

 These functions check if there is alias violation between two array variables or not.
 
 IF (off2+size_stride2<off1) OR (off2>off1+size_stride1)  => OK, no alias 

 ELSE 
   IF e1 is written (ref1) 
     IF (ref1 < off2) OR (ref1 > off2+size_stride2) => no alias violation
     ELSE (off2 <= ref1 <= off2+size_stride2) => alias violation 
   IF e2 is written (ref2) : symetrical with e1 
     IF (ref2 < off1) OR (ref2 > off1+size_stride1) => no alias violation
     ELSE (off1 <= ref2 <= off1+size_stride1) => alias violation  

*****************************************************************************/

static void alias_check_two_array_variables_in_module(entity e1,entity e2,expression off1, 
						      expression off2,list path)
{
  expression size2 = array_size_stride(e2);
  if (!expression_undefined_p(size2)) // not assumed-size array
    {
      expression sum2,diff1;
      int k1;
      if (expression_equal_integer_p(off2,0))
	sum2 = copy_expression(size2);
      else
	sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off2,size2);
      diff1 = gt_expression(off1,sum2);
      clean_all_normalized(diff1);
      k1 = trivial_expression_p(diff1);
      ifdebug(2)
	fprintf(stderr,"\n Check in module, 2 array variables \n");
      switch(k1){
      case 1: 
	{
	  ifdebug(3) 
	    fprintf(stderr,"\n off1> off2+size_stride2 is true => Okay, no alias between these 2 variables \n");
	  /* off1> off2+size_stride2 is true => Okay, no alias between these 2 variables */
	  return;
	}
      default:
	{ 
	  expression size1 = array_size_stride(e1);
	  if (!expression_undefined_p(size1)) // not assumed-size array
	    {
	      expression sum1,diff2;
	      int k2;
	      if (expression_equal_integer_p(off1,0))
		sum1 = copy_expression(size1);
	      else
		sum1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off1,size1);
	      diff2 = gt_expression(off2,sum1);
	      clean_all_normalized(diff2);
	      k2 = trivial_expression_p(diff2);
	      switch(k2){
	      case 1: 
		{
		  ifdebug(3) 
		    fprintf(stderr,"\n  off2 > off1+size_stride1 is true => Okay, no alias between these 2 variables \n");
		  /* off2 > off1+size_stride1 is true => Okay, no alias between these 2 variables */
		  return;
		}
	      default:
		{
		  /* insert flags before each call site in call path*/
		  list l_flags = make_list_of_flags(path);
		  alias_context_t context;
		  context.insert = false;
		  context.path = path;
		  context.flags = expression_list_to_conjonction(l_flags);
		  context.first_entity = e1;
		  context.second_entity = e2;
		  context.offset1 = off1;
		  context.offset2 = off2;
		  gen_context_recurse(module_statement,&context, statement_domain, 
				      alias_check_array_variable_in_module_flt, gen_null);
		  context.first_entity = e2;
		  context.second_entity = e1;
		  context.offset1 = off2;
		  context.offset2 = off1;
		  gen_context_recurse(module_statement,&context, statement_domain, 
				      alias_check_array_variable_in_module_flt, gen_null);
		  if (context.insert)
		    insert_flag_before_call_site(l_flags,path);
		  context.first_entity = entity_undefined;
		  context.second_entity = entity_undefined;
		  context.offset1 = expression_undefined;
		  context.offset2 = expression_undefined;
		  context.condition = expression_undefined;
		  context.flags = expression_undefined;
		  context.path = NIL;
		}
	      }
	    }
	  else 
	    /* Assumed-size or pointer-type array, the size of dummy array can be 
	       derived from the size of actual argument, as we have the corresponding call chain*/
	    pips_user_warning("\nAssumed-size or pointer-type array \n");
	}
      }
    }
  else 
    /* Assumed-size or pointer-type array, the size of dummy array can be 
       derived from the size of actual argument, as we have the corresponding call chain*/
    pips_user_warning("\nAssumed-size or pointer-type array \n");
}

/*****************************************************************************

   This function checks if there is an alias violation in written reference r 
   of entity 1, which may be aliased with entity 2

*****************************************************************************/

static void alias_check_in_module(entity e1,entity e2,
				  expression off1,expression off2,list path)
{
  /* There are 3 cases:
     - Case 1: e1 and e2 are scalar variables
     - Case 2: e1 is a scalar variable, e2 is an array variable or vice-versa
     - Case 3: e1 and e2 are array variables */

  if (entity_scalar_p(e1) && entity_scalar_p(e2))
    alias_check_two_scalar_variables_in_module(e1,e2,off1,off2,path);
  
  if (entity_scalar_p(e1) && !entity_scalar_p(e2))
    alias_check_scalar_and_array_variables_in_module(e1,e2,off1,off2,path);
  
  if (!entity_scalar_p(e1) && entity_scalar_p(e2))
    alias_check_scalar_and_array_variables_in_module(e2,e1,off2,off1,path);

  if (!entity_scalar_p(e1) && !entity_scalar_p(e2))
    alias_check_two_array_variables_in_module(e1,e2,off1,off2,path);
}


/*****************************************************************************

 This function checks if an aliased scalar variable is written or not.
 If yes, insert test before direct call

*****************************************************************************/

static bool alias_check_scalar_variable_in_caller_flt(statement s,alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);
  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		insert_test_before_statement(context->flags,make_true_expression(),s,
					     context->first_entity,context->second_entity,
					     context->path);
		context->insert = true;
		return false;
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if an array element which is aliased with a scalar variable 
 is written or not. Test : ref2 == off1 ? where ref2 = off2+subval_stride2
 If yes, insert test before direct call

*****************************************************************************/

static bool alias_check_array_and_scalar_variable_in_caller_flt(statement s,alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);

  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {
		list l_inds = reference_indices(r);
		/* Attention : <may be written> V(*,*) => what kind of indices ???*/
		expression subval2 = subscript_value_stride(context->first_entity,l_inds);
		/* Translate subval2 to the frame of caller, in order to compare with offsets*/
		expression new_subval2 = translate_to_module_frame(current_mod,current_caller,subval2,current_call);
		if (!expression_undefined_p(new_subval2))
		  { 
		    expression diff;
		    int k;
		    if (same_expression_p(context->offset1,context->offset2))
		      diff = eq_expression(int_to_expression(0),new_subval2);
		    else
		      {
			expression ref2;
			if (expression_equal_integer_p(context->offset2,0))
			  ref2 = copy_expression(new_subval2);
			else
			  ref2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,context->offset2,new_subval2);
			diff = eq_expression(context->offset1,ref2);
		      }
		    clean_all_normalized(diff);
		    k = trivial_expression_p(diff);
		    switch(k){
		    case -1: 
		      /* off1 == ref2 is false =>  Okay, no write on aliased variable */
		      return false;
		    default:
		      {
			/* k = 0 or 1*/
			expression e_flag = reference_to_expression(make_reference(alias_flag,
				     CONS(EXPRESSION,int_to_expression(++unique_flag_number),NIL)));
			ifdebug(2)
			  {
			    fprintf(stderr, "\n New flag expression: \n");
			    print_expression(e_flag);
			  }
			if (true_expression_p(context->flags))
			  insert_test_before_statement(e_flag,make_true_expression(),s,
						       context->first_entity,context->second_entity,
						       context->path);
			else
			  insert_test_before_statement(and_expression(e_flag,context->flags),
						       make_true_expression(),s,
						       context->first_entity,context->second_entity,
						       context->path);
			if (k==0)
			  insert_test_before_caller(simplify_relational_expression(diff),e_flag);
			else 
			  insert_test_before_caller(make_true_expression(),e_flag);
			context->insert = true;
			return false;
		      }
		    }
		  }
		else
		  {
		    /* We can not translate subval to the frame of caller => use dynamic check*/
		    insert_check_alias_before_statement(context->first_entity,subval2,
							context->second_entity,int_to_expression(0),s);
		    return false;
		    //user_log("\n Warning : Can not translate writing reference to frame of caller \n");
		  }
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if an array variable has written element which is aliased 
 with other array variable's element or not.
 Test : off2 <= ref1 <= off2+size_stride2 ? where ref1 = off1+subval_stride1
 If yes, insert test before statement

*****************************************************************************/

static bool alias_check_array_variable_in_caller_flt(statement s,alias_context_p context)
{
  list l_rw = NIL;
  if (statement_call_p(s))
    l_rw = load_proper_rw_effects_list(s);
  else
    l_rw = load_cumulated_rw_effects_list(s);
  MAP(EFFECT, eff,
  {
    action a = effect_action(eff);
    if (action_write_p(a))
      {
	reference r = effect_any_reference(eff);
	entity e = reference_variable(r);
	if (same_entity_p(e,context->first_entity))
	  {
	    approximation rw = effect_approximation(eff);
	    if (approximation_exact_p(rw))
	      {

		/*TREATED ? CASE READ *,ARRAY */

		list l_inds = reference_indices(r);
		/* Attention : <may be written> V(*,*) => what kind of indices ???*/
		expression subval1 = subscript_value_stride(context->first_entity,l_inds);
		/* Translate subval1 to the frame of caller, in order to compare with offsets*/
		expression new_subval1 = translate_to_module_frame(current_mod,current_caller,
								   subval1,current_call);
		if (!expression_undefined_p(new_subval1))
		  {
		    expression diff1 = expression_undefined;
		    int k1;
		    expression ref1 = expression_undefined;
		    if (same_expression_p(context->offset1,context->offset2))
		      diff1 = ge_expression(new_subval1,int_to_expression(0));
		    else
		      {
			if (expression_equal_integer_p(context->offset1,0))
			  ref1 = copy_expression(new_subval1);
			else
			  ref1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,context->offset1,new_subval1);
			diff1 = ge_expression(ref1,context->offset2);
		      }
		    clean_all_normalized(diff1);
		    k1 = trivial_expression_p(diff1);
		    switch(k1){
		    case -1: 
		      /* off2 <= ref1 is false =>  Okay, no write on aliased variable */
		      break;
		    default:
		      {
			expression size2 = array_size_stride(context->second_entity);
			/* Test of assumed-size array has been carried out before 
			   Translate size2 to the frame of caller, in order to compare with offsets*/
			size2 =  translate_to_module_frame(current_mod,current_caller,size2,current_call);
			if (!expression_undefined_p(size2))
			  {
			    expression diff2;
			    int k2;
			    if (same_expression_p(context->offset1,context->offset2))
			      diff2 = le_expression(new_subval1,size2);
			    else
			      {
				expression sum2;
				if (expression_equal_integer_p(context->offset2,0))
				  sum2 = copy_expression(size2);
				else
				  sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,context->offset2,size2);
				diff2 = le_expression(ref1,sum2);
			      }
			    clean_all_normalized(diff2);
			    k2 = trivial_expression_p(diff2);
			    switch(k2){
			    case -1: 
			      /* ref1 <= off2+size_stride2 is false =>  Okay, no write on aliased variable */
			      break;
			    default:
			      {
				/* k1 = 0 or 1, k2 = 0 or 1*/
				expression e_flag = reference_to_expression(make_reference(alias_flag,
				            CONS(EXPRESSION,int_to_expression(++unique_flag_number),NIL)));
				ifdebug(2)
				  {
				    fprintf(stderr, "\n New flag expression: \n");
				    print_expression(e_flag);
				  }
				if (true_expression_p(context->flags))
				  insert_test_before_statement(e_flag,make_true_expression(),s,
							       context->first_entity,context->second_entity,
							       context->path);
				else
				  insert_test_before_statement(and_expression(e_flag,context->flags),
							       make_true_expression(),s,
							       context->first_entity,context->second_entity,
							       context->path);
				if (k1+k2==0) // k1=k2=0
				  insert_test_before_caller(and_expression(simplify_relational_expression(diff1),
									   simplify_relational_expression(diff2)),e_flag);
				else 
				  {
				    if (k1+k2==2) // k1=k2=1
				      insert_test_before_caller(make_true_expression(),e_flag);
				    else
				      {
					if (k1==0) // k1=0, k2=1
					  insert_test_before_caller(simplify_relational_expression(diff1),e_flag);
					else // k2=0, k1=1
					  insert_test_before_caller(simplify_relational_expression(diff2),e_flag);
				      }
				  }
				context->insert = true;
				break;
			      }
			    }
			  }
			else
			  /* We can not translate size2 to the frame of caller => create new common variable*/
			  pips_user_warning("\nCan not translate size of array to frame of caller \n");
			break;
		      }
		    }
		  }
		else
		  {
		    /* We can not translate subval1 to the frame of caller => use dynamic check*/
		    expression size2 = array_size_stride(context->second_entity);
		    insert_check_alias_before_statement(context->first_entity,subval1,
							context->second_entity,size2,s);
		  }
		return false;
	      }
	    return true;
	  }
      }
  },
      l_rw); 
  return false; /* The variable is not in the list of write effects => OK*/
}

/*****************************************************************************

 This function checks if there is alias violation between two scalar variables or not.

 IF off1 != off2 => no alias, OK
 ELSE, off1 == off2 => alias, 
   IF e1 or e2 is written => alias violation
   => insert test before every statement where e1 and e2 must be written 

*****************************************************************************/

static void alias_check_two_scalar_variables_in_caller(entity e1,entity e2,expression off1, 
						       expression off2,list path)
{
  expression diff = eq_expression(off1,off2);
  int k;
  clean_all_normalized(diff);
  k = trivial_expression_p(diff);
  /* Should we use also formal and actual binding / precondition ??? */
  ifdebug(2)
    fprintf(stderr,"\n Check in caller, 2 scalar variables \n");
  switch(k){
  case -1: 
    /* off1 != off2 => Okay, no alias between these 2 variables */
    return;
  default:
    {
      /* insert flags before each call site in call path and for the
	 current caller, we add the condition */
      list l_flags = make_list_of_flags(path);
      alias_context_t context;
      context.insert = false;
      context.path = path;
      context.flags = expression_list_to_conjonction(l_flags);
      context.first_entity = e1;
      context.second_entity = e2;
      gen_context_recurse(module_statement,&context, statement_domain, 
			  alias_check_scalar_variable_in_caller_flt, gen_null);
      context.first_entity = e2;
      context.second_entity = e1;
      gen_context_recurse(module_statement,&context, statement_domain, 
			  alias_check_scalar_variable_in_caller_flt, gen_null);
      if (context.insert)
	{
	  expression e_flag = EXPRESSION(CAR(l_flags));
	  insert_flag_before_call_site(CDR(l_flags),CDR(path));
	  if (k==1)
	    insert_test_before_caller(make_true_expression(),e_flag);
	  else
	    insert_test_before_caller(simplify_relational_expression(diff),e_flag);
	}
      context.first_entity = entity_undefined;
      context.second_entity = entity_undefined;
      context.offset1 = expression_undefined;
      context.offset2 = expression_undefined;
      context.condition = expression_undefined;
      context.flags = expression_undefined;
      context.path = NIL;
    }
  }
}

/*****************************************************************************

 This function checks if there is alias violation between a scalar variable e1 
 and an array variable e2 or not. 

 IF off1 < off2 OR off1 > off2+size_stride2 => no alias, OK
 ELSE : off2 <= off1 <= off2+size_stride2 
       - IF e1 is written => alias violation 
       - IF e2 is written (ref2 == subval_stride2+off2) : 
           - IF ref2 != off1 => no alias violation 
	   - IF ref2 == off1 => alias violation 
     
*****************************************************************************/

static void alias_check_scalar_and_array_variables_in_caller(entity e1, entity e2,  
				       expression off1, expression off2,list path)
{
  expression diff1 = le_expression(off2,off1);
  int k1;
  clean_all_normalized(diff1);
  k1 = trivial_expression_p(diff1);
  ifdebug(2)
    fprintf(stderr,"\n Check in caller, scalar and array variables \n");
  switch(k1){
  case -1: 
    /* off2 <= off1 is false => Okay, no alias between these 2 variables */
    return;
  default:
    {
      expression size2 = array_size_stride(e2);
      if (!expression_undefined_p(size2))
	{
	  /* Translate size2 to the frame of caller, in order to compare with offsets*/
	  size2 = translate_to_module_frame(current_mod,current_caller,size2,current_call);
	  if (!expression_undefined_p(size2))
	    {
	      expression diff2;
	      int k2;
	      if (same_expression_p(off1,off2))
		diff2 = le_expression(int_to_expression(0),size2);
	      else
		{
		  expression sum2;
		  if (expression_equal_integer_p(off2,0))
		    sum2 = copy_expression(size2);
		  else
		    sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off2,size2);
		  diff2 = le_expression(off1,sum2);
		}
	      clean_all_normalized(diff2);
	      k2 = trivial_expression_p(diff2);
	      switch(k2){
	      case -1: 
		/* off1 <= off2+size_stride2 is false => Okay, no alias between these 2 variables */
		return;
	      default:
		{
		  /* k1 = 0 or 1, k2 = 0 or 1
		     insert flag before each call site in call path and for the current caller, 
		     we add the condition */
		  list l_flags = make_list_of_flags(path);
		  alias_context_t context;
		  context.insert = false;
		  context.path = path;
		  context.flags = expression_list_to_conjonction(l_flags);
		  context.first_entity = e1;
		  context.second_entity = e2;
		  gen_context_recurse(module_statement,&context, statement_domain, 
				      alias_check_scalar_variable_in_caller_flt, gen_null);

		  /* Attention: there are differences between scalar and array variables 
		     when adding test/flag before current statement and current caller. 
		     For array variable, we must take into account the ref (reference),
		     it must be sure if this reference is written or not. So new flag
		     is inserted to guarantee if the reference is written. */

		  if (ENDP(CDR(l_flags)))
		    context.flags = make_true_expression();
		  else
		    context.flags = expression_list_to_conjonction(CDR(l_flags));
		  context.offset1 = off1;
		  context.offset2 = off2;
		  context.first_entity = e2;
		  context.second_entity = e1;
		  gen_context_recurse(module_statement,&context, statement_domain, 
				      alias_check_array_and_scalar_variable_in_caller_flt, gen_null);
		  
		  if (context.insert)
		    {
		      insert_flag_before_call_site(CDR(l_flags),CDR(path));
		      if (variable_is_written_p(e1))
			{
			  expression e_flag = EXPRESSION(CAR(l_flags));
			  if (k1+k2==0) // k1=k2=0
			    insert_test_before_caller(and_expression(simplify_relational_expression(diff1),
								     simplify_relational_expression(diff2)),e_flag);
			  else 
			    {
			      if (k1+k2==2) // k1=k2=1
				insert_test_before_caller(make_true_expression(),e_flag);
			      else
				{
				  if (k1==0) // k1=0, k2=1
				    insert_test_before_caller(simplify_relational_expression(diff1),e_flag);
				  else // k2=0, k1=1
				    insert_test_before_caller(simplify_relational_expression(diff2),e_flag);
				}
			    }
			}
		    }
		  context.first_entity = entity_undefined;
		  context.second_entity = entity_undefined;
		  context.offset1 = expression_undefined;
		  context.offset2 = expression_undefined;
		  context.condition = expression_undefined;
		  context.flags = expression_undefined;
		  context.path = NIL;
		}
	      }
	    }
	  else
	    pips_user_warning("\nCan not translate size of array to frame of caller");
	}
      else
	/* Assumed-size or pointer-type array, the size of dummy array can be 
	   derived from the size of actual argument, as we have the corresponding call chain*/
	pips_user_warning("\nAssumed-size or pointer-type array \n");  
    }
  }
}

/*****************************************************************************

 These functions check if there is alias violation between two array variables or not.
 
 IF (off2+size_stride2<off1) OR (off2>off1+size_stride1)  => OK, no alias 
 ELSE 
   IF e1 is written (ref1) 
     IF (ref1 < off2) OR (ref1 > off2+size_stride2) => no alias violation
     ELSE (off2 <= ref1 <= off2+size_stride2) => alias violation 
   IF e2 is written (ref2) : symetrical with e1 
     IF (ref2 < off1) OR (ref2 > off1+size_stride1) => no alias violation
     ELSE (off1 <= ref2 <= off1+size_stride1) => alias violation  

*****************************************************************************/

static void alias_check_two_array_variables_in_caller(entity e1, entity e2,  
			        expression off1, expression off2,list path)
{
  expression size2 = array_size_stride(e2);
  if (!expression_undefined_p(size2))
    {
      /* Translate size2 to the frame of caller, in order to compare with offsets*/
      size2 = translate_to_module_frame(current_mod,current_caller,size2,current_call);
      if (!expression_undefined_p(size2))
	{
	  expression sum2,diff1;
	  int k1;
	  if (expression_equal_integer_p(off2,0))
	    sum2 = copy_expression(size2);
	  else
	    sum2 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off2,size2);
	  diff1 = gt_expression(off1,sum2);
	  clean_all_normalized(diff1);
	  k1 = trivial_expression_p(diff1);
	  ifdebug(2) 
	    fprintf(stderr,"\n Check in caller, 2 array variables \n");
	  switch(k1){
	  case 1: 
	    /* off1>off2+size_stride2 is true => Okay, no alias between these 2 variables */
	    return;
	  default:
	    { 
	      expression size1 = array_size_stride(e1);
	      if (!expression_undefined_p(size1))
		{
		  /* Translate size1 to the frame of caller, in order to compare with offsets*/
		  size1 = translate_to_module_frame(current_mod,current_caller,size1,current_call);
		  if (!expression_undefined_p(size1))
		    {
		      expression sum1,diff2;
		      int k2;
		      if (expression_equal_integer_p(off1,0))
			sum1 = copy_expression(size1);
		      else
			sum1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,off1,size1);
		      diff2 = gt_expression(off2,sum1);
		      clean_all_normalized(diff2);
		      k2 = trivial_expression_p(diff2);
		      switch(k2){
		      case 1: 
			/* off2 > off1+size_stride1 is true => Okay, no alias between these 2 variables */
			return;
		      default:
			{
			  /* insert flag before each call site in call path */
			  list l_flags = NIL;
			  alias_context_t context;
			  context.insert = false;
			  context.path = path;
			  if (ENDP(CDR(path)))
			    context.flags = make_true_expression();
			  else
			    {
			      l_flags = make_list_of_flags(CDR(path));
			      context.flags = expression_list_to_conjonction(l_flags);
			    }
			  context.first_entity = e1;
			  context.second_entity = e2;
			  context.offset1 = off1;
			  context.offset2 = off2;
			  gen_context_recurse(module_statement,&context, statement_domain, 
					      alias_check_array_variable_in_caller_flt, gen_null);
			  context.first_entity = e2;
			  context.second_entity = e1;
			  context.offset1 = off2;
			  context.offset2 = off1;
			  gen_context_recurse(module_statement,&context, statement_domain, 
					      alias_check_array_variable_in_caller_flt, gen_null);

			  if (context.insert)
			    insert_flag_before_call_site(l_flags,CDR(path));
			  context.first_entity = entity_undefined;
			  context.second_entity = entity_undefined;
			  context.offset1 = expression_undefined;
			  context.offset2 = expression_undefined;
			  context.condition = expression_undefined;
			  context.flags = expression_undefined;
			  context.path = NIL;
			}
		      }
		    }
		  else
		    pips_user_warning("\nCan not translate size of array to frame of caller");
		}
	      else 
		/* Assumed-size or pointer-type array, the size of dummy array can be 
		   derived from the size of actual argument, as we have the corresponding call chain*/
		pips_user_warning("\nAssumed-size or pointer-type array \n");  
	    }
	  }
	}
      else
	pips_user_warning("\nCan not translate size of array to frame of caller");
    }
  else 
    /* Assumed-size or pointer-type array, the size of dummy array can be 
       derived from the size of actual argument, as we have the corresponding call chain*/
    pips_user_warning("\nAssumed-size or pointer-type array \n");      
}


/*****************************************************************************

  
*****************************************************************************/
static void alias_check_in_caller(entity e1,entity e2,expression off1,
				  expression off2,list path)
{
 if (entity_scalar_p(e1) && entity_scalar_p(e2))
    alias_check_two_scalar_variables_in_caller(e1,e2,off1,off2,path);
  
  if (entity_scalar_p(e1) && !entity_scalar_p(e2))
    alias_check_scalar_and_array_variables_in_caller(e1,e2,off1,off2,path);
  
  if (!entity_scalar_p(e1) && entity_scalar_p(e2))
    alias_check_scalar_and_array_variables_in_caller(e2,e1,off2,off1,path);

  if (!entity_scalar_p(e1) && !entity_scalar_p(e2))
    alias_check_two_array_variables_in_caller(e1,e2,off1,off2,path);
}


/*****************************************************************************

   This function computes the offset of a storage ram variable : 
   offset = initial_offset + subscript_value_stride

*****************************************************************************/

static expression storage_ram_offset(storage s,expression subval)
{
  ram r = storage_ram(s);
  int initial_off = ram_offset(r);
  expression exp = int_to_expression(initial_off);
  if (!expression_equal_integer_p(subval,0))
    {
      if (initial_off == 0)
	exp = copy_expression(subval);
      else
	exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					  int_to_expression(initial_off),
					  copy_expression(subval));
    }
  return exp;
}

/****************************************************************************

   This function computes the offset of a formal storage variable : 
   offset = initial_offset + subscript_value_stride

   initial_offset is from alias_association with path' = path - {cs} 

*****************************************************************************/

static expression storage_formal_offset(call_site cs,entity actual_var,
					expression subval,list path)
{
  list l_caller_aliases = alias_associations_list((alias_associations)
        db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,caller_name,true)); 
  expression exp = expression_undefined;
  MAP(ALIAS_ASSOCIATION, aa,
  {
    entity caller_var = alias_association_variable(aa);
    list caller_path = alias_association_call_chain(aa);
    if (same_entity_p(caller_var,actual_var) && tail_call_path_p(cs,caller_path,path))
      {
	expression initial_off = alias_association_offset(aa);
	if (!expression_undefined_p(initial_off)) 
	  {
	    if (expression_equal_integer_p(subval,0))
	      exp = copy_expression(initial_off);
	    else 
	      exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME, 
						copy_expression(initial_off),
						copy_expression(subval));
	  }
	return exp;
      }
  },
      l_caller_aliases);
  return exp;
}


/*****************************************************************************

 If e is a formal parameter, find its rank in the formal parameter list of 
 current module in order to find out the corresponding actual argument and 
 then its offset								      
 If e is a common variable in the current module, offset of e is constant

*****************************************************************************/

static expression offset_in_caller(entity e, call_site cs, list path)
{
  ram ra ;
  if (formal_parameter_p(e))
    {
      formal f = storage_formal(entity_storage(e));
      int rank = formal_offset(f);
      list l_args = call_arguments(current_call);
      expression actual_arg = find_ith_argument(l_args,rank);
      reference actual_ref = expression_reference(actual_arg);
      entity actual_var = reference_variable(actual_ref);
      list l_actual_inds = reference_indices(actual_ref);
      /* compute the subscript value, return expression_undefined if
	 if the actual argument is a scalar variable or array name*/
      expression subval = subscript_value_stride(actual_var,l_actual_inds);
      storage s = entity_storage(actual_var);
      ifdebug(3)
	fprintf(stderr, " \n Current actual argument %s",entity_name(actual_var));	
      if (storage_ram_p(s))
	/* The actual argument has a ram storage */
	return storage_ram_offset(s,subval);
      if (storage_formal_p(s))
	/* The actual argument is a formal parameter of the current caller, 
	   we must take the alias_associations of the caller */
	return storage_formal_offset(cs,actual_var,subval,path);
    }
  // common variable
  ra = storage_ram(entity_storage(e)); 
  return int_to_expression(ram_offset(ra));
}

static bool search_statement_by_ordering_flt(statement s)
{
  if (statement_ordering(s)==current_ordering)
    {
      current_statement = s;
      return false;
    }
  return true;
}

static void alias_check_two_variables(entity e1, entity e2, expression off1, 
				      expression off2, list path)
{
  if (variable_is_written_p(e1) || variable_is_written_p(e2))
    {
      /* e1 or e2 is written => check for alias violation*/
      if (!expression_undefined_p(off1) && !expression_undefined_p(off2))
	alias_check_in_module(e1,e2,off1,off2,path);
      else 
	{
	  /* As we do not have exact offsets of variables, we have to go to the 
	     caller's frame to check for alias violations. The direct caller is
	     CAR(call_path) because of the following concatenation in alias_propagation:
	     path = CONS(CALL_SITE,cs,gen_full_copy_list(alias_association_call_chain(aa)));
	 
	     To find a call site from its ordering, we have to do a gen_recurse 
	     in the caller module. */
	  
	  call_site cs = CALL_SITE(CAR(path));
	  statement caller_statement;
	  current_caller = call_site_function(cs);
	  current_ordering = call_site_ordering(cs);
	  caller_name = module_local_name(current_caller);
	  caller_statement = (statement)db_get_memory_resource(DBR_CODE,caller_name,true);
	  current_statement = statement_undefined;
	  
	  gen_recurse(caller_statement,statement_domain,
		      search_statement_by_ordering_flt,gen_null);
      
	  if (!statement_undefined_p(current_statement) && statement_call_p(current_statement))
	    {
	      expression new_off1, new_off2;
	      current_call = statement_call(current_statement);
	      new_off1 = offset_in_caller(e1,cs,path);
	      new_off2 = offset_in_caller(e2,cs,path);
	      if (!expression_undefined_p(new_off1) && !expression_undefined_p(new_off2))
		alias_check_in_caller(e1,e2,new_off1,new_off2,path);
	      else
		{
		  /* Try with special cases : CALL FOO(R(TR(K)),R(TR(K))) ???????
		     Does this case exist when we create special section + offset 
		     for same actual arguments ??? */
		  /* use dynamic alias check*/
		  alias_context_t context;
		  context.first_entity = e1;
		  context.second_entity = e2;
		  gen_context_recurse(module_statement,&context,statement_domain,
				      dynamic_alias_check_flt,gen_null);
		  context.first_entity = e2;
		  context.second_entity = e1;
		  gen_context_recurse(module_statement,&context,statement_domain,
				      dynamic_alias_check_flt,gen_null);
		  context.first_entity = entity_undefined;
		  context.second_entity = entity_undefined;
		  set_dynamic_checked(e1,e2);
		}
	      current_call = call_undefined;
	    }
	  else 
	    pips_user_warning("Problem with statement ordering *\n"); 
	  current_statement = statement_undefined;
	  current_ordering = 0;
	  current_caller = entity_undefined;
	}
    }
}

/*****************************************************************************
   Take pair of variables (formal and formal or formal and common) that
   may be aliased (in same section, included call path), then check if they 
   are really aliased or not. 
   If no , OK. 
   If yes, gen_recurse on statements that write these variables => check for 
   alias violations 
*****************************************************************************/

bool alias_check(char * module_name)
{
  /* File instrument.out is used to store alias checks and flags*/
  string dir_name = db_get_current_workspace_directory();
  string instrument_file = strdup(concatenate(dir_name, "/instrument.out", NULL));
  free(dir_name), dir_name = NULL;
  out = safe_fopen(instrument_file, "a");  
  number_of_processed_modules++;
  current_mod = local_name_to_top_level_entity(module_name);
  /* We do not add the line "INCLUDE alias_flags.h" into code_decls_text because of
     repeated bug for module with ENTRY.
     09/11/2001 : do not add INCLUDE any more, because there are modules that do not 
     need this INCLUDE => use script: if a module is modified => add INCLUDE line
     fprintf(out, "AC: %s (%d,%d)\n",module_local_name(current_mod),0,1);
     fprintf(out,"      INCLUDE 'alias_flags.h'\n");
     fprintf(out, "ACEND \n");*/
  debug_on("ALIAS_CHECK_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin alias_check for %s \n", module_name); 
  l_module_aliases = alias_associations_list((alias_associations)
		 db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,module_name,true)); 
  /* if the list of alias associations of module is NIL, do nothing*/
  if (l_module_aliases != NIL)
    {
      /* Compute the list of direct and indirect callees of current module */
      // list l_callees = compute_all_callees(current_mod);             
      string alias_flag_name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
						  MODULE_SEP_STRING,ALIAS_FLAG,NULL));
      string alias_function_name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
						      MODULE_SEP_STRING,ALIAS_FUNCTION,NULL));
      alias_flag = gen_find_tabulated(alias_flag_name,entity_domain);
      if (entity_undefined_p(alias_flag))
	alias_flag = make_entity(alias_flag_name, 
				 make_type(is_type_variable,
					   make_variable(make_basic_logical(4),NIL,NIL)),
				 storage_undefined, value_undefined);
      alias_function = gen_find_tabulated(alias_function_name,entity_domain);
      if (entity_undefined_p(alias_function))
	alias_function = make_empty_subroutine(ALIAS_FUNCTION,copy_language(module_language(current_mod)));
      module_statement = (statement) db_get_memory_resource(DBR_CODE,module_name,true);
      ifdebug(2)
	{
	  fprintf(stderr, " \n The list of alias associations for module %s is:\n", module_name); 
	  print_list_of_alias_associations(l_module_aliases);
	}
      set_current_module_entity(current_mod);
      set_ordering_to_statement(module_statement);  
      /* Get the proper and cumulated effects of the module, we have to take both kinds of 
	 effects because of their difference for an elementary statement: 
	 V(I) = I  => cumulated effect : <may be written> V(*)
	           => proper effect : <must be written> V(I) 
         If a cumulated effect for elementary statements = proper effect => we need only 
         cumulated effect */
      set_cumulated_rw_effects((statement_effects) 
			       db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,true)); 
      set_proper_rw_effects((statement_effects) 
			    db_get_memory_resource(DBR_PROPER_EFFECTS,module_name,true));
      initialize_dynamic_check_list();
      while (!ENDP(l_module_aliases))
	{
	  alias_association aa1 = ALIAS_ASSOCIATION(CAR(l_module_aliases));
	  entity e1 = alias_association_variable(aa1);
	  entity sec1 = alias_association_section(aa1);
	  list path1 = alias_association_call_chain(aa1);
	  expression off1 = alias_association_offset(aa1);
	  int l1 = alias_association_lower_offset(aa1);
	  int u1 = alias_association_upper_offset(aa1);
	  l_module_aliases = CDR(l_module_aliases);

	  /* Looking for another formal variable in the list of alias
	     associations that has same section and included call path. 
	     If this variable is checked dynamically with e1 => no need 
	     to continue */
	
	  MAP(ALIAS_ASSOCIATION, aa2,
	  {
	    entity e2 = alias_association_variable(aa2);
	    entity sec2 = alias_association_section(aa2);
	    list path2 = alias_association_call_chain(aa2);
	    if (!same_entity_p(e1,e2) && same_entity_p(sec1,sec2) && 
		!dynamic_checked_p(e1,e2)&& included_call_chain_p(path1,path2))
	      {  
		int l2 = alias_association_lower_offset(aa2);
		int u2 = alias_association_upper_offset(aa2);
		/*
		  Special cases : no alias u1 < l2, u2 <l1, u1< o1, u2 < o1, 
		  o2 + s2 < l1, o1 + s1 < l2
		  So easiest case: If u1,l2 are defined (different to -1) and u1<l2, 
		  there is no alias violation
		  The same for: u2,l1 are defined (different to -1) and u2<l1*/
		if (((u1==-1)||(u1>=l2))&&((u2==-1)||(u2>=l1)))
		  {
		    expression off2 = alias_association_offset(aa2);
		    ifdebug(2) 
		      fprintf(stderr, "\nFound two may be aliased formal parameters: %s, %s. Let's check !\n",
			      entity_name(e1),entity_name(e2));
		    if (gen_length(path1) < gen_length(path2))
		      alias_check_two_variables(e1,e2,off1,off2,path2);
		    else 
		      alias_check_two_variables(e1,e2,off1,off2,path1);
		  }
	      }
	  },
	      l_module_aliases); 
	  
	  /* Looking for common variables in module or callee of modules 
	     to check for alias violation ...*/
	  
	  /* For this moment, only for common variable of module 
	     ========> add for chain of callees

	     Check for write on common variables in callee, keep call path also
	     => to insert test*/
	  
	  MAP(ENTITY, e2,
	  {
	    if (variable_in_common_p(e2))
	      {
		ram ra = storage_ram(entity_storage(e2));
		entity sec2 = ram_section(ra);
		if (!dynamic_checked_p(e1,e2) && same_entity_p(sec1,sec2))
		  {  
		    /* formal parameter has a same section with other common variable*/
		    int l2 = ram_offset(ra);
		    int u2 = l2;
		    if (array_entity_p(e2))
		      {
			int tmp;
			if (SizeOfArray(e2,&tmp))
			  u2 = tmp-SizeOfElements(variable_basic(type_variable(entity_type(e2))))+l2;
			else
			  user_log("Varying size of common variable");
		      }
		    /* If u1 is defined (different to -1) and u1<l2, there is no alias violation
		       The same for: l1 is defined (different to -1) and u2<l1*/
		    if (((u1==-1)||(u1>=l2))&&(u2>=l1))
		      {
			expression off2 = int_to_expression(l2);
			/* The common variable always have a good offset off2 */
			ifdebug(2) 
			  fprintf(stderr,"\n Found may be aliased formal and common variable :%s, %s. Let's check ! \n",
				  entity_name(e1), entity_name(e2));
			alias_check_two_variables(e1,e2,off1,off2,path1);
		      }
		  }
	      }
	  },
	      code_declarations(entity_code(current_mod))); 
	}
      l_dynamic_check = NIL;
      reset_proper_rw_effects();
      reset_cumulated_rw_effects();
      DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,module_statement);
      reset_ordering_to_statement();
      reset_current_module_entity();
      module_statement = statement_undefined;
      // free(alias_flag_name), alias_flag_name = NULL;
      // free(alias_function_name), alias_function_name = NULL;
    }
  else
    user_log("\n No alias for this module \n"); 

  safe_fclose(out,instrument_file);
  free(instrument_file), instrument_file= NULL;
  display_alias_check_statistics();
  pips_debug(1, "end\n");
  debug_off();
  l_module_aliases = NIL;
  current_mod = entity_undefined;
  return true;
}









