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
 *		     ALIAS PROPAGATION
 *
 *
*******************************************************************/

/* Aliasing occurs when two or more variables refer to the same storage 
   location at the same program point. This phase tries to compute as 
   precise as possible the interprocedural alias information in a whole 
   program.
   
   In Fortran 77, there are several ways to create aliases: 

   1. EQUIVALENCE: two or more entities in the same program unit share 
   storages units
   2. COMMON: associates different variables in different subprograms 
   with the same storage
   3. An actual argument is passed to different formal parameters
   4. A global variable is passed as an actual argument and a variable 
   aliased to the global variable is passed as another actual argument 
   => alias between the two corresponding formal parameters.
   5. Formal parameters aliases can be passed through chains of calls.

   The basic idea for computing interprocedural aliases is to follow all 
   possible chains of argument-parameters and nonlocal variable-parameter
   bindings at all call sites. The call graph of program is traversed in
   invocation order, and alias information is accumulated incrementally.

   We use the newgen structure alias_association = (formal_parameter,section,
   offset, call_path) to store alias information for each formal parameter
   of each module. Call_path = list of call_sites, call_site = (caller, 
   ordering of the call site) (this is the only current way to store the 
   location of a call site).

   Let ai be the considering actual argument in the current call site, by 
   separating the treatment of formal parameters from the treatment of 
   global variables, we only have to treat the following case:

   1. Alias between formal parameter and common variable 
   A global variable can only become aliased to a formal parameter in a
   routine in which it is visible and only by its being passed as an 
   actual argument to that formal parameter.

   1.1 Alias created by only one call:
   
   Case 1. ai is a common variable and is visible in the current module or in 
   at least one callee (direct and indirect) of this module => add alias 
   association for fi with section of the common : TOP-LEVEL:~FOO
 
   1.2 Alias created through chain of calls:

   Case 2. ai is a formal variable with a common section and this common is 
   visible in the current module or in at least one callee (direct and 
   indirect) of this module => add alias association for fi with section 
   of the common : TOP-LEVEL:~FOO and path = path(formal ai) + (C,ordering)

   => useless tests between fi and other variables in the same common block with 
   ai, if not take into account the size of ai (assumption: no [interprocedural]
   array bound violation), because the section is not enough (unique)

   2. Alias between formal parameters
 
   2.1 Alias created by only one call:

   Case 3. An actual argument is bound to different formal parameters or there are  
   different actual arguments but equivalent. So for a call site, we can divide the 
   argument list into groups of same actual or equivalence arguments. For 
   example:
   EQUIVALENCE (V1(1,1),V2(5))
   EQUIVALENCE (U,W)
   CALL FOO(V1,A,B(TR(I)),C,B(TR(K)),B(H),V1(I,J),V2(K),C,A,M,U,W)
   SUBROUTINE FOO(F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13)
   => (F1,F7,F8), (F2,F10), (F3,F5,F6),(F4,F9), (F12,F13) 
   
   We add alias associations for these formal parameters, all parameters in a 
   same group have same and unique section, path = {(C,ordering)}.
   The difference among the group (F1,F7,F8), (F12,F13) and the others is that 
   we need to know the initial offsets of F1, F7, F8 and F12, F13 because they 
   can be different variables, and their sections are ram section. For the other 
   cases, we can use section = ALIAS_SPECIAL_i, initial_off = 0.
  
   => useless tests  ??? Not for same variables because the section is unique but
   useless tests for equivalence variables, as U and V1 have the same section
   => test between F8,F12, ...

   2.2 Alias created through chain of calls
   
   Case 4. Actual arguments are formal variables of the caller and have same 
   section from two included call paths. 
   ai, aj : formal variables, same section, call_path(ai) (is) include(s/d)
   call_path(aj); add alias association for fi with call_path = path_formal(ai)
   + (C,ordering)
   
   Case 5. Actual argument is a formal variable that has same section with other 
   actual argument that is a common variable:

   5.1 If ai is the formal variable => add alias association for fi with
   call_path = path_formal(ai) + (C,ordering)

   5.2 If ai is the common variable => add alias association for fi with
   call_path = (C,ordering)

   To compute the offset, we do not use preconditions that may be corrupted by 
   alias violation */

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
#include "properties.h"
#include "transformer.h"
#include "instrumentation.h"
#include "text-util.h"
#include "transformations.h"
#include "alias-classes.h"

#define ALIAS_SECTION "ALIAS_SECTION"

/* Define a static stack and related functions to remember the current
   statement */

DEFINE_LOCAL_STACK(current_statement, statement)

static entity current_mod = entity_undefined;
static entity current_caller = entity_undefined; 
static const char* caller_name;
static list l_current_aliases = NIL;
static list l_traversed = NIL;
static int number_of_alias_associations = 0;
static int number_of_unknown_offsets = 0;
static int number_of_known_offsets = 0;
static int number_of_processed_modules = 0;
static int unique_section_number = 0;/* Special alias section counter*/

static void display_alias_propagation_statistics()
{
  user_log("\n Number of added alias associations: %d",number_of_alias_associations);
  user_log("\n Number of known offsets: %d",number_of_known_offsets);
  user_log("\n Number of unknown offsets: %d",number_of_unknown_offsets);
  user_log("\n Number of processed modules: %d\n",number_of_processed_modules); 
}

/*****************************************************************************
   This function computes the subscript value of an array element 
   minus 1, multiplied by the size of array element. 
 
   [subscript_value(array_element)-1]*array_element_size
   
   DIMENSION A(l1:u1,...,ln:un)
   subscript_value(A(s1,s2,...,sn)) =
   1+(s1-l1)+(s2-l2)*(u1-l1+1)+...+ (sn-ln)*(u1-l1+1)*...*(u(n-1) -l(n-1)+1)

   Input : the entity A 
           the indice list (s1,s2,..sn)
   Output : [subscript_value(A(s1,..sn))-1]*array_element_size 

   If l_inds = NIL => return int_to_expression(0)
   If si = li for all i => return int_to_expression(0) 

*****************************************************************************/

expression subscript_value_stride(entity arr, list l_inds)
{
  expression retour = int_to_expression(0);
  if (!ENDP(l_inds))
    {
      variable var = type_variable(entity_type(arr));
      expression prod = expression_undefined;
      list l_dims = variable_dimensions(var);
      int num_dim = gen_length(l_inds),i;
      basic b = variable_basic(type_variable(entity_type(arr)));
      expression e_size = int_to_expression(SizeOfElements(b));
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
	      if (expression_equal_integer_p(retour,0))
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
	}
      if (!expression_equal_integer_p(retour,0))
	retour = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(retour),e_size);
    }
  ifdebug(4)
    {
      pips_debug(4,"\nStride of subscript value:");
      print_expression(retour);
    }
  return retour;
}

/* This function translates an expression e1 from the frame of module 1 
   to the frame of module 2
   1.If e1 is a reference:    
   1.1. Common variable in mod1 => return corresponding common variable e2 in mod2
   1.2. Special case : e1 is a formal parameter in mod1, mod2 is caller of mod1
   => take corresponding actual argument e2 in call c
   1.3. From the association information, if we have e1 = e2 where e2 
   is constant or contains variables of the mod2 => return e2
   2.If e1 is a call : 
   2.1. Storage rom : numerical constant or symbolic value (PARAMETER) => return e1
   2.2. Recursive : e1 = ex1 * ey1 
	 ex2 = translate_to_module_frame(mod1,mod2,ex1,c) 
	 ey2 = translate_to_module_frame(mod1,mod2,ey1,c)  
	 => return e2=ex2*ey2
   3.If e1 is a range : error

   Return expression_undefined if we can not translate*/

expression translate_to_module_frame(entity mod1, entity mod2, expression e1, call c)
{ 
  if (!expression_undefined_p(e1)) {
  syntax syn = expression_syntax(e1);
  tag t = syntax_tag(syn);
  switch(t){  
  case is_syntax_reference: 
    {
      reference ref = syntax_reference(syn);
      entity en = reference_variable(ref);
      normalized ne;
      if (variable_in_common_p(en))
	{
	  /* Check if the common variable is also declared in the mod2 or not 
	   * We can use ram_shared which contains a list of aliased variables with en 
	   * but it does not work ????  
	   * Another way : looking for a variable in the declaration of the mod2
	   * that has the same offset in the same common block */
	  list l_decls = code_declarations(entity_code(mod2));
	  MAP(ENTITY, enti,{
	    if (same_scalar_location_p(en,enti))
	      {
		pips_debug(4,"\nThe common variable %s is translated\n",entity_local_name(en));
		if (array_entity_p(enti))
		  {
		    /* ATTENTION : enti may be an array, such as A(2):
		       COMMON C1,C2,C3,C4,C5
		       COMMON C1,A(2,2)
		       we must return A(1,1), not A */
		    variable varenti = type_variable(entity_type(enti));   		      
		    int len =  gen_length(variable_dimensions(varenti));
		    list l_inds = make_list_of_constant(1,len);
		    reference refer = make_reference(enti,l_inds);
		    return reference_to_expression(refer);
		  }
		return entity_to_expression(enti);
	      }
	  },l_decls);
	  // return the common variable although it is not declared in the module !!!!!!
	  //	  return entity_to_expression(en);
	}
      if (variable_is_a_module_formal_parameter_p(en,mod1)) 
	{
	  formal fo = storage_formal(entity_storage(en));
	  entity fun = call_function(c);
	  if (same_entity_p(fun,mod1))
	    {
	      /* Special case : e1 is a formal parameter in mod1, mod2 is caller of mod1
	         miss a check : mod2 = caller of mod1 => can be wrong !!!*/
	      int off = formal_offset(fo);
	      list l_args = call_arguments(c);
	      pips_debug(4,"\nThe formal parameter %s is translated to the caller's frame\n", 
			 entity_local_name(en));
	      return find_ith_argument(l_args,off);
	    }
	}
      /* Use the association of the call site:
	 Take only the equalities.
	 Project all variables belonging to mod1 , except the current variable e1
	 there are 2 cases :
	 1. The projection is not exact , there are over flows
	 Return the SC_UNDEFINED => what to do, like before ? 
	 2. The result is exact, three small cases: 
	 2.1 The system is always false sc_empty => unreachable code ?
	 2.2 The system is always true sc_rn => we have nothing ?
	 2.3 The system is parametric =>
	 
	 Look for equality that contain e1
	 Delete e1 from the vector 
	 Check if the remaining of the vectors contains only constant (TCTS) 
	 or variables of mod2 => return*/
      if (expression_equal_integer_p(e1,0)) return e1;
      clean_all_normalized(e1);
      ne =  NORMALIZE_EXPRESSION(e1);
      if (normalized_linear_p(ne))
	{
	  Pvecteur ve = normalized_linear(ne);
	  Variable vare = var_of(ve); 
	  transformer binding_context = formal_and_actual_parameters_association(c,transformer_identity());
	  Psysteme ps_tmp = predicate_system(transformer_relation(binding_context));
	  Pbase b_tmp = ps_tmp->base;
	  /* Attention :   here the transformer binding_context is consistent 
	     but not the system ps_tmp. I do not understand why ?
	     fprintf(stderr, "consistent psystem ps_tmp before");
	     pips_assert("consistent psystem ps_tmp", sc_consistent_p(ps_tmp));*/
	  if (base_contains_variable_p(b_tmp,vare))
	    {
	      Psysteme ps = sc_dup(ps_tmp);
	      Pbase b = ps->base;
	      Pvecteur pv_var = VECTEUR_NUL; 	  
	      for(; !VECTEUR_NUL_P(b); b = b->succ) 
		{
		  Variable var = vecteur_var(b);
		  if ((strcmp(module_local_name(mod1),entity_module_name((entity)var))==0)
		      && (var!=vare))
		    vect_add_elem(&pv_var, var, VALUE_ONE); 
		}
	      ps->inegalites = contraintes_free(ps->inegalites);
	      ps->nb_ineq = 0;
	      ps = my_system_projection_along_variables(ps, pv_var);       
	      vect_rm(pv_var);  
	      if (ps != SC_UNDEFINED)
		{
		  // the projection is exact		       
		  Pcontrainte egal, egal1;
		  for (egal = ps->egalites; egal != NULL; egal = egal1) 
		    {
		      /* Take only the equations of the system */
		      Pvecteur vec = egal->vecteur;
		      if (vect_contains_variable_p(vec,vare))
			{
			  Value vale = vect_coeff(vare,vec);
			  Pvecteur newv = VECTEUR_UNDEFINED; 
			  if (value_one_p(vale) || value_mone_p(vale))
			    newv = vect_del_var(vec,vare);
			  if  (value_one_p(vale))
			    vect_chg_sgn(newv);
			  if (!VECTEUR_UNDEFINED_P(newv))
			    {
			      /*the coefficient of e is 1 or -1.
				Check if the remaining vector contains only constant 
				or variavbles of mod2*/
			      Pvecteur v;
			      bool check = true;
			      for (v = newv; (v !=NULL) && (check); v = v->succ)
				{ 
				  Variable var = v->var;
				  if ((var != TCST) && 
				      (strcmp(module_local_name(mod2),entity_module_name((entity)var))!=0))
				    check = false;
				}
			      if (check)
				{
				  pips_debug(4,"\nThe variable %s is translated by using binding information\n",
					     entity_local_name(en));
				  return Pvecteur_to_expression(newv);
				}
			      vect_rm(newv);
			    }
			}
		      egal1 = egal->succ;
		    }
		}	 
	      sc_rm(ps);
	    }
	}
      break;
    }
  case is_syntax_call:
    {
      call ca = syntax_call(syn);
      entity fun = call_function(ca);
      list l_args = call_arguments(ca);
      if (l_args==NIL)
	{
	  /* Numerical constant or symbolic value (PARAMETER) */
	  ifdebug(4)
	    {
	      pips_debug(4,"\nNumerical constant or symbolic value is translated\n");
	      print_expression(e1);
	    }
	  return e1;
	}
      /* e1 is a call, not a constant 
	 Recursive : with the arguments of the call
	 As our generated expression e1 is a call with operators : +,-,* only,
	 we treat only these cases */
      if (gen_length(l_args)==1)
	{
	  expression ex1 = EXPRESSION(CAR(l_args));
	  expression ex2 = translate_to_module_frame(mod1, mod2,ex1,c);
	  if (!expression_undefined_p(ex2))
	    return MakeUnaryCall(fun,ex2);
	}      
      if (gen_length(l_args)==2)
	{
	  expression ex1 = EXPRESSION(CAR(l_args));
	  expression ey1 = EXPRESSION(CAR(CDR(l_args)));
	  expression ex2 = translate_to_module_frame(mod1,mod2,ex1,c);
	  expression ey2 = translate_to_module_frame(mod1,mod2,ey1,c);
	  if (!expression_undefined_p(ex2) && !expression_undefined_p(ey2))
	    return MakeBinaryCall(fun,ex2,ey2);
	}        
      break;
    }
  default:
    pips_internal_error("Abnormal cases ");
    break;
  }}
  return expression_undefined;
}

static void ram_variable_add_aliases(call c,call_site cs,entity actual_var,
				     entity formal_var,expression subval)
{
  storage s = entity_storage(actual_var);
  ram r = storage_ram(s); 
  entity sec = ram_section(r);
  int initial_off = ram_offset(r),end_off = -1;
  list path = CONS(CALL_SITE,cs,NIL);
  expression off = expression_undefined;
  alias_association one_alias = alias_association_undefined;
  if (array_entity_p(actual_var))
    {
      int tmp;
      if (SizeOfArray(actual_var, &tmp))
	end_off = tmp - SizeOfElements(variable_basic(type_variable(entity_type(actual_var)))) + initial_off;
    }
  else
    end_off = initial_off;
  ifdebug(4)
    {
      fprintf(stderr, "\nActual argument %s is a ram variable",entity_name(actual_var));
      fprintf(stderr,"\nwith initial ram offset %d and end offset %d",initial_off,end_off);
    }
  if (expression_equal_integer_p(subval,0))
    /* The offset of actual variable is an integer 
       that can always be translated into the module's frame*/
    off = int_to_expression(initial_off);	
  else 
    {
      /* We must translate the subscript value from current caller to the 
	 current module's frame by using binding information. This value must be 
	 multiplied by the size of array element (number of numerical/character
	 storage units, according to Fortran standard, in PIPS 1 storage unit=1 byte)*/
      expression new_subval = translate_to_module_frame(current_caller,current_mod,subval,c);
      ifdebug(4)
	{
	  fprintf(stderr, "\nSubval expression before translation:");
	  print_expression(subval);
	  fprintf(stderr, "\nSubval expression after translation:");
	  print_expression(new_subval);
	}
      if (!expression_undefined_p(new_subval))
	{
	  /* subval is translated to the module's frame */
	  if (initial_off ==0)
	    off = copy_expression(new_subval);
	  else 
	    off =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					       int_to_expression(initial_off),
					       copy_expression(new_subval));
	}
    }
  if (expression_undefined_p(off))
    number_of_unknown_offsets++;
  else 
    number_of_known_offsets++;
  /* Attention: normalization of an expression equal to 0 returns a Pvecteur null*/ 	
  /* initial_off <= off <= initial_off + SizeOfArray - SizeOfElement (no bound violation ;-))*/
  one_alias = make_alias_association(formal_var,sec,off,initial_off,end_off,path);
  number_of_alias_associations++;
  message_assert("alias_association is consistent",
		 alias_association_consistent_p(one_alias));
  ifdebug(2)
    print_alias_association(one_alias);
  l_current_aliases = gen_nconc(l_current_aliases, 
				CONS(ALIAS_ASSOCIATION,one_alias, NIL));
}

/* This function tests if a common com (TOP_LEVEL:~FOO) is visible 
   in the module mod or in at least one callee (direct and indirect) 
   of this module */

static bool common_is_visible_p(entity sec, entity mod)
{
  list l_decl = code_declarations(entity_code(mod));
  /* search for the common declaration in the list */	  
  MAP(ENTITY, ei,
  {
    storage si = entity_storage(ei);
    if (storage_ram_p(si))
      {
	entity seci = ram_section(storage_ram(si));
	if (same_entity_p(sec,seci))
	  return true;
      }
  },  l_decl);
  /* search for the common declaration in the callees */	
  if (!entity_main_module_p(mod))
    {
      const char* mod_name = module_local_name(mod);
      callees all_callees = (callees) db_get_memory_resource(DBR_CALLEES,mod_name,true);
      list l_callees = callees_callees(all_callees); 
      MAP(STRING,callee_name,
      {
	entity current_callee = local_name_to_top_level_entity(callee_name);
	if (common_is_visible_p(sec,current_callee)) return true;
      },l_callees);
    }
  return false;
}

/******************************************************************** 
 The following functions are for the case of there is a formal variable 
 that have the same section sec in the argument list l.

 Look for actual arguments that are formal parameters => take 
 alias_associations of the current caller => compare if they have 
 section = sec or not.
*********************************************************************/

static bool same_section_formal_variable_in_list_p(entity actual_var,entity sec,
						   list actual_path,list l,list l_aliases)
{
  MAP(EXPRESSION,exp,
  {
    if (expression_reference_p(exp))
      {
	reference ref = expression_reference(exp);
	entity var = reference_variable(ref);
	if (!same_entity_p(var,actual_var))
	  {
	    storage si =  entity_storage(var);
	    if (storage_formal_p(si))
	      {
		MAP(ALIAS_ASSOCIATION, aa,
		{
		  entity formal_var = alias_association_variable(aa);
		  if (same_entity_p(formal_var,var))
		    {
		      entity formal_sec = alias_association_section(aa);
		      list formal_path = alias_association_call_chain(aa);
		      if (same_entity_p(formal_sec,sec) && 
			  included_call_chain_p(actual_path,formal_path)) 
			{
			  /* INCLUDED CALL CHAIN ????????*/
			  pips_debug(3,"\nAliases from an actual argument that is a formal parameter and has same section with other actual argument (the last one can be a common variable or another formal variable).\n");
			  return true;
			}
		    } 
		},
		    l_aliases);
	      }
	  }
      }
  },l);
  return false;
}

/******************************************************************** 
 The following functions are for the case of there is a common variable 
 that has the same section sec in the argument list l.
*********************************************************************/

static bool same_section_common_variable_in_list_p(entity sec,list l)
{
  MAP(EXPRESSION,exp,
  {
    if (expression_reference_p(exp))
      {
	reference ref = expression_reference(exp);
	entity var = reference_variable(ref);
	if (variable_in_common_p(var))
	  {
	    storage si =  entity_storage(var);
	    entity seci = ram_section(storage_ram(si));
	    if (same_entity_p(seci,sec)) 
	      {
		pips_debug(3,"\nAliases from an actual argument that is a common variable and has same section with other actual argument that is a formal variable.\n");
		return true;
	      }
	  }
      }
  },l);
 return false;
}

/***************************************************************************
 The following functions are for the cases: in the actual argument list,
 there is a formal variable:

 Case 2. that has a common section and this common is visible in the current
 module or in at least one callee (direct and indirect) of this module

 Case 4. that has same section with other formal variable from two included call 
 paths. 

 Case 5.1 that has same section with other common variable in the argument list
****************************************************************************/

static void formal_variable_add_aliases(call c,call_site cs, entity actual_var,
					entity formal_var,expression subval,list l_actuals)
{
  list l_caller_aliases = alias_associations_list((alias_associations)
       db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,caller_name,true)); 
  pips_debug(2,"\nActual argument %s is a formal parameter", 
	     entity_name(actual_var));	
  MAP(ALIAS_ASSOCIATION, aa,
  {
    entity caller_var = alias_association_variable(aa);
    if (same_entity_p(caller_var,actual_var))
      {
	/* a gen_full_copy_list here is to copy the list and its contain 
	   without this : gen_write => gen_trav ....=> bug unknown type 1 
	   because the CDR of path point to 
	   a newgen data in the caller which is freed before , no more in the memory */
	entity sec = alias_association_section(aa);
	list actual_path = alias_association_call_chain(aa); 
	if ((!entity_special_area_p(sec) && common_is_visible_p(sec,current_mod))
	    || same_section_formal_variable_in_list_p(actual_var,sec,actual_path,l_actuals,l_caller_aliases)
	    || same_section_common_variable_in_list_p(sec,l_actuals))
	  {
	    list path = CONS(CALL_SITE,cs,gen_full_copy_list(alias_association_call_chain(aa)));
	    expression off = expression_undefined;
	    alias_association one_alias = alias_association_undefined;
	    expression initial_off = alias_association_offset(aa);
	    /* To be modified : init_off = lower_offset ...*/

	    int init_off = -1;
	    int end_off = -1;
	    // path = gen_nconc(path,gen_full_copy_list(alias_association_call_chain(aa)));
	    ifdebug(3)
	      fprintf(stderr,"\nEntry for %s found in the alias_association", 
		      entity_name(caller_var));
	    /* If offset of aa is not expression_undefined, we must translate 
	       it to the module's frame by using binding information */
	    if (!expression_undefined_p(initial_off))
	      {
		expression new_initial_off = translate_to_module_frame(current_caller,current_mod,
								       initial_off,c);
		ifdebug(3)
		  {
		    fprintf(stderr, "\nInitial offset expression before translation: \n");
		    print_expression(initial_off);
		    fprintf(stderr, "\nInitial offset expression after translation: \n");
		    print_expression(new_initial_off);
		  }		
		if (!expression_undefined_p(new_initial_off))
		  {
		    if (expression_equal_integer_p(subval,0))
		      off = copy_expression(new_initial_off);
		    else 
		      {
			/* We must translate the subscript value from current caller to the 
			   current module's frame by using binding information. This value must be 
			   multiplied by the size of array element (number of numerical/character
			   storage units, according to Fortran standard, in PIPS 1 storage unit=1 byte)*/
			expression new_subval = translate_to_module_frame(current_caller,current_mod,
									  copy_expression(subval),c);
			ifdebug(3)
			  {
			    fprintf(stderr, "\n Subval expression before translation: \n");
			    print_expression(subval);
			    fprintf(stderr, "\n Subval expression after translation: \n");
			    print_expression(new_subval);
			  }
			if (!expression_undefined_p(new_subval))
			  off =  binary_intrinsic_expression(PLUS_OPERATOR_NAME, 
							     copy_expression(new_initial_off), 
							     copy_expression(new_subval));
		      }
		    if (expression_constant_p(new_initial_off))
		      {
			init_off = expression_to_int(new_initial_off);
			if (array_entity_p(actual_var))
			  {
			    int tmp;
			    if (SizeOfArray(actual_var, &tmp))
			      end_off = tmp - SizeOfElements(variable_basic(type_variable(entity_type(actual_var)))) 
				+ init_off;
			  }
			else
			  end_off = init_off;
		      }
		  }
		
	      }
	    if (expression_undefined_p(off))
	      number_of_unknown_offsets++;
	    else 
	      number_of_known_offsets++;
	    /* init_off <= off <= init_off + SizeOfArray - SizeOfElement (no bound violation ;-))*/
	    one_alias = make_alias_association(formal_var,sec,off,init_off,end_off,path);
	    message_assert("alias_association is consistent", 
			   alias_association_consistent_p(one_alias));	
	    ifdebug(2)
	      print_alias_association(one_alias);
	    number_of_alias_associations++;
	    l_current_aliases = gen_nconc(l_current_aliases, 
					  CONS(ALIAS_ASSOCIATION,one_alias, NIL));
	  }
      }
  },
      l_caller_aliases);
}

/******************************************************************** 
 The following functions are for the case of same or equivalence actual arguments 
*********************************************************************/

static list list_of_same_or_equivalence_arguments(entity e,list l)
{
  list retour = NIL;
  int j;
  for (j=1;j<=gen_length(l);j++)
    {
      expression exp = find_ith_argument(l,j);
      if (expression_reference_p(exp))
	{
	  reference ref = expression_reference(exp);
	  entity var = reference_variable(ref);
	  if (same_entity_p(var,e))
	    retour = CONS(INT,j,retour);
	  else 
	    if (entities_may_conflict_p(var,e) && 
		!(variable_in_common_p(e)&&variable_in_common_p(var)))
	      {
		l_traversed = CONS(ENTITY,var,l_traversed);
		retour = CONS(INT,j,retour);
	      }
	}
    }
  return retour;
}

/* Add alias_association for each formal variable whose offset is in the list l*/
static void same_or_equivalence_argument_add_aliases(list l,call c,call_site cs,
						     list l_actual,bool equiv)
{
  if (equiv)
    {
      MAP(INT,k,
      {
	expression actual_arg = find_ith_argument(l_actual,k);
	reference actual_ref = expression_reference(actual_arg);
	entity actual_var = reference_variable(actual_ref);
	entity formal_var = find_ith_formal_parameter(current_mod,k);	     
	list l_actual_inds = reference_indices(actual_ref);
	expression subval = subscript_value_stride(actual_var,l_actual_inds);
	ram_variable_add_aliases(c,cs,actual_var,formal_var,subval);
      },l);
    }
  else
    {
        string istr = i2a(unique_section_number++);
        string ename = strdup(concatenate(ALIAS_SECTION,istr,NULL));
      entity sec = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, ename);
      free(ename);
      free(istr);
      MAP(INT,k,
      {
	expression actual_arg = find_ith_argument(l_actual,k);
	reference actual_ref = expression_reference(actual_arg);
	entity actual_var = reference_variable(actual_ref);
	entity formal_var = find_ith_formal_parameter(current_mod,k);	     
	list l_actual_inds = reference_indices(actual_ref);
	expression subval = subscript_value_stride(actual_var,l_actual_inds);
	expression off = expression_undefined;
	int end_off = -1;
	alias_association one_alias = alias_association_undefined;
	list path = CONS(CALL_SITE,cs,NIL);
	if (array_entity_p(actual_var))
	  {
	    int tmp;
	    if (SizeOfArray(actual_var, &tmp))
	      end_off = tmp - SizeOfElements(variable_basic(type_variable(entity_type(actual_var))));
	  }
	else
	  end_off = 0;
	if (expression_equal_integer_p(subval,0))
	  /* The offset of the actual variable is an integer 
	     that can always be translated into the module's frame*/
	  off = int_to_expression(0);	
	else 
	  {
	    off  = translate_to_module_frame(current_caller,current_mod,subval,c);
	    ifdebug(4)
	      {
		fprintf(stderr, "\nSubval expression before translation: \n");
		print_expression(subval);
		fprintf(stderr, "\nSubval expression after translation: \n");
		print_expression(off);
	      }
	  }
	if (expression_undefined_p(off))
	  number_of_unknown_offsets++;
	else 
	  number_of_known_offsets++;
	/* 0 <= off <= 0 + array_size_stride (no bound violation ;-))*/
	one_alias = make_alias_association(formal_var,sec,off,0,end_off,path);
	number_of_alias_associations++;
	message_assert("alias_association is consistent",
		       alias_association_consistent_p(one_alias));
	ifdebug(2)
	  print_alias_association(one_alias);
	l_current_aliases = gen_nconc(l_current_aliases, 
				      CONS(ALIAS_ASSOCIATION,one_alias, NIL));
      },l);
    }
}

static bool add_aliases_for_current_call_site(call c)
{
  if(call_function(c) == current_mod)
    {  
      statement stmt = current_statement_head();
      list l_actuals = call_arguments(c);
      list l = gen_full_copy_list(l_actuals);
      int order = statement_ordering(stmt); 
      int i = 0;
      call_site cs = make_call_site(current_caller,order);
      //  list path = gen_full_copy_list(CONS(CALL_SITE,cs,NIL));
      ifdebug(2)
	{
	  pips_debug(2,"\nCurrent caller: %s", caller_name);
	  fprintf(stderr,"\nCurrent call site:");
	  print_statement(stmt);
	}
      message_assert("call_site is consistent", call_site_consistent_p(cs));	
      l_traversed = NIL;
      MAP(EXPRESSION,actual_arg,
      {
	i++;
	if (expression_reference_p(actual_arg))
	  {
	    /* Correspond to different cases of alias, we make the following groups and order :
	       Case 3. list_of_same_or_equivalence_arguments
	       Case 1 + Case 5.2. common variable
	       Case 2 + Case 4 + Case 5.1. formal variable */
	    reference actual_ref = expression_reference(actual_arg);
	    entity actual_var = reference_variable(actual_ref);     
	    list l_actual_inds = reference_indices(actual_ref);
	    expression subval = subscript_value_stride(actual_var,l_actual_inds);
	    entity formal_var = find_ith_formal_parameter(current_mod,i);	
	    list l_same_or_equiv = NIL;
	    /* To distinguish between equivalence or same argument cases*/
	    int j = gen_length(l_traversed);
	    bool equiv = false;
	    if (!variable_in_list_p(actual_var,l_traversed))
	      {
		l_same_or_equiv = list_of_same_or_equivalence_arguments(actual_var,l);
		if (gen_length(l_traversed)>j) equiv = true;
		l_traversed = CONS(ENTITY,actual_var,l_traversed);
	      }
	    ifdebug(3)
	      {
		if (equiv)
		  fprintf(stderr,"\nList of equivalent arguments: ");
		else
		  fprintf(stderr,"\nList of same arguments: ");
		MAP(INT,l,
		{
		  fprintf(stderr,"%d,",l);
		},l_same_or_equiv);
	      }
	    if (gen_length(l_same_or_equiv) > 1)
	      same_or_equivalence_argument_add_aliases(l_same_or_equiv,c,cs,l_actuals,equiv);
	    if (variable_in_common_p(actual_var))
	      {
		storage s = entity_storage(actual_var);
		entity sec = ram_section(storage_ram(s)); 
		list l_caller_aliases = alias_associations_list((alias_associations)
								db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,caller_name, true)); 
		if (common_is_visible_p(sec,current_mod) || 
		    same_section_formal_variable_in_list_p(actual_var,sec,NIL,l_actuals,l_caller_aliases))
		  ram_variable_add_aliases(c,cs,actual_var,formal_var,subval);
	      }
	    if (storage_formal_p(entity_storage(actual_var)))
	      formal_variable_add_aliases(c,cs,actual_var,formal_var,subval,l_actuals); 
	  } 
      },l);
      l_traversed = NIL;
      gen_free_list(l);
    }
  return true;
}

static void add_aliases_for_current_caller()
{  
  statement caller_statement = (statement) db_get_memory_resource
    (DBR_CODE,caller_name, true);
  set_ordering_to_statement(caller_statement); 
  make_current_statement_stack();
  gen_multi_recurse(caller_statement,
		    statement_domain, current_statement_filter,current_statement_rewrite,
		    call_domain, add_aliases_for_current_call_site, gen_null,
		    NULL);
  free_current_statement_stack();
  reset_ordering_to_statement();
}

static list alias_propagation_callers(list l_callers)
{
  /* Traverse each caller and add aliases to the list of aliases (l_current_aliases) */
  l_current_aliases = NIL;
  MAP(STRING, c_name,
  {
    current_caller = local_name_to_top_level_entity(c_name);
    caller_name = module_local_name(current_caller);
    if (get_bool_property("ALIAS_CHECKING_USING_MAIN_PROGRAM") &&
	(! module_is_called_by_main_program_p(current_caller)))
      /* If the current caller is never called by the main program =>
	 no need to follow this caller*/
      pips_user_warning("Module %s is not called by the main program \n",caller_name);
    else
      add_aliases_for_current_caller();
  }, l_callers);
  return l_current_aliases;
}
 
bool alias_propagation(char * module_name)
{
  list l_aliases = NIL;
  current_mod = local_name_to_top_level_entity(module_name);
  set_current_module_entity(current_mod);	
  //  number_of_alias_associations = 0;
  number_of_processed_modules++;
  debug_on("ALIAS_PROPAGATION_DEBUG_LEVEL");
  pips_debug(1,"\nBegin alias propagation for module %s \n", module_name); 
  /* No alias for main program*/
  if (!entity_main_module_p(current_mod))
    {
      if (get_bool_property("ALIAS_CHECKING_USING_MAIN_PROGRAM") &&
	  (!module_is_called_by_main_program_p(current_mod)))
	/* If the current module is never called by the main program => 
	   don't need to compute aliases for this module*/
	pips_user_warning("Module %s is not called by the main program \n",module_name);
      else 
	{
	  list l_decls = code_declarations(entity_code(current_mod)); 
	  list l_formals = NIL; 
	  /* search for formal parameters in the declaration list */   
	  MAP(ENTITY, e,
	  {
	    if (formal_parameter_p(e))
	      l_formals = gen_nconc(l_formals,CONS(ENTITY,e,NIL));
	  },l_decls);
	  /* if there is no formal parameter, do nothing */
	  if (l_formals != NIL)
	    {
	      /* Take the list of callers, if there is no caller, do nothing */
	      callees callers = (callees) db_get_memory_resource(DBR_CALLERS,module_name,true);
	      list l_callers = callees_callees(callers); 	  	      
	      if (l_callers != NIL)
		{
		  ifdebug(2)
		    {
		      fprintf(stderr,"The list of formal parameters:");
		      print_entities(l_formals);
		      fprintf(stderr,"\nThe list of callers: ");
		      MAP(STRING, caller_name, {
			(void) fprintf(stderr, "%s, ", caller_name);
		      }, l_callers);
		      (void) fprintf(stderr, "\n");	
		    }
		  l_aliases = alias_propagation_callers(l_callers); 
		}
	      else 
		/* The module has no caller => don't need to compute aliases for this module*/
		pips_user_warning("\n Module %s has no caller \n",module_name );
	    }
	}
    }
  /* save to resource */
  DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_ASSOCIATIONS,module_name,make_alias_associations(l_aliases));  
  display_alias_propagation_statistics();
  reset_current_module_entity();
  current_mod = entity_undefined;
  debug_off();  
  return true;
}








