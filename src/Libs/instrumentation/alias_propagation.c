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

The basic idea for computing interprocedural aliases is to follow 
all the possible chains of argument-parameters and nonlocal 
variable-parameter bindings at all call sites. We introduce a naming 
memory locations technique which guarantees the correctness and 
enhances the precision of data-flow analysis. */

/* A possibility: if a call is unreachable (never be executed), 
there is no alias caused by this call => if the precondition is 
unfeasible ({0==-1}) => no more check 

This approach is not correct, because we may have a corrupted 
preconditions that is false because of an alias violation as 
the following example
 
     SUBROUTINE Q(K,L)
     K = 5
C    P(K){K==5} ==> may not right because K and L may be aliased
C                  we have no right to write on K
     IF (K.NE.5) THEN
C    P(K) {0==1}
        CALL P(K)
     ENDIF */

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
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"
#include "transformations.h"

#define ALIAS_SECTION "ALIAS_SECTION"

/* Define a static stack and related functions to remember the current
   statement  (and then get the current precondition for
   alias_propagation_caller()  =====> no need any more 12/06/2001 NN) */

DEFINE_LOCAL_STACK(current_statement, statement)

static entity current_mod = entity_undefined;
static entity current_caller = entity_undefined; 
static string caller_name;
static list l_current_aliases = NIL;
static int number_of_alias_associations = 0;
static int number_of_unknown_offsets = 0;
static int number_of_known_offsets = 0;

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
  //  else 
  // user_log("\n Warning : Assumed-size A(N,*) or pointer-type A(N,1) array \n");
  return exp;
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
  ifdebug(2)
    {
      fprintf(stderr, "\n Stride of subscript value : \n");
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
	  MAP(ENTITY, enti,
	  {
	    if (same_scalar_location_p(en,enti))
	      {
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
	  },
	      l_decls);
	  // return the common variable although it is not declared in the module !!!!!!
	  return entity_to_expression(en);
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
			      bool check = TRUE;
			      for (v = newv; (v !=NULL) && (check); v = v->succ)
				{ 
				  Variable var = v->var;
				  if ((var != TCST) && 
				      (strcmp(module_local_name(mod2),entity_module_name((entity)var))!=0))
				    check = FALSE;
				}
			      if (check)
				return Pvecteur_to_expression(newv);
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
	/* Numerical constant or symbolic value (PARAMETER) */
	return e1;
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
    pips_error("", "Abnormal cases \n");
    break;
  }}
  return expression_undefined;
}



/* This function returns TRUE if there exists another actual argument
   in the argument list that is the same variable with e
   CALL P(K,K) or CALL P(X(I),X(J)) */

static bool exist_same_actual_argument_p(int i, list l_actuals, entity actual_var)
{
  int j;
  for (j=1;j<=gen_length(l_actuals);j++)
    {
      if (j!=i)
	{
	  expression exp = find_ith_argument(l_actuals,j);
	  if (expression_reference_p(exp))
	    {
	      reference ref = expression_reference(exp);
	      entity var = reference_variable(ref);
	      if (same_entity_p(var,actual_var)) 
		{
		  ifdebug(3)
		    fprintf(stderr, " \n Same actual argument");
		  return TRUE;
		}
	    }
	}
    }
  return FALSE;
}

/* This function adds the alias association of current
   ram variable to the l_current_aliases */

static void storage_ram_add_aliases(call c, storage s, call_site cs, entity formal_var, 
				    expression subval,int i, list l_actuals,entity actual_var)
{
  /* We add new alias_association only : 
     If ai is a common variable  (!SPECIAL_AREA_P(section))
     If ai is a local variable of current caller :
     - if ai is in a EQUIVALENCE (by using ram_shared) 
     - if ai is not in a EQUIVALENCE but exists aj=ai (j!=i)
     (CALL P(K,K) or CALL P(X(I),X(J)) => add */

  ram r = storage_ram(s); 
  entity sec = ram_section(r);
  list shared = ram_shared(r);
  if (!SPECIAL_AREA_P(sec) || (shared !=NIL) || 
      exist_same_actual_argument_p(i,l_actuals,actual_var))
    {
      int initial_off = ram_offset(r);
      list path = CONS(CALL_SITE,cs,NIL);
      expression off = expression_undefined;
      alias_association one_alias = alias_association_undefined;
      ifdebug(3)
	{
	  fprintf(stderr, " \n Actual argument %s is a ram variable of the current caller", 
		  entity_name(actual_var));
	  fprintf(stderr,"\n Initial ram offset %d",initial_off);
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
	  ifdebug(3)
	    {
	      fprintf(stderr, "\n Subval expression before translation: \n");
	      print_expression(subval);
	      fprintf(stderr, "\n Subval expression after translation: \n");
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
      /* Attention : bug : if I normalize an expression that is equal to 0,
	 I will have a Pvecteur null 
	 clean_all_normalized(off);
	 n = NORMALIZE_EXPRESSION(off);
	 if (normalized_linear_p(n)) 
	 {
	 Pvecteur v = normalized_linear(n);
	 off = Pvecteur_to_expression(v); }*/
      ifdebug(3)
	{
	  fprintf(stderr, "\n Offset :\n");
	  print_expression(off);
	}		
      one_alias = make_alias_association(formal_var,sec,off,path);
      number_of_alias_associations++;
      message_assert("alias_association is consistent",
		     alias_association_consistent_p(one_alias));
      ifdebug(3)
	print_alias_association(one_alias);
      l_current_aliases = gen_nconc(l_current_aliases, 
				    CONS(ALIAS_ASSOCIATION,one_alias, NIL));
    }
}

/* This function returns TRUE if there exists another actual argument
   in the argument list that is a formal parameter of current caller 
   and has an alias_association entry with same section 

   SUB R
   CALL Q(V1,V2)
   ....

   SUB Q(F1,F2)
   CALL P(F1,K,F2)  => F1 and F2 in this case 
   ....

   SUB P(X1,X2,X3) */

static bool exist_actual_argument_formal_parameter_with_same_section_p
                (int i, list l_actuals, entity sec, list l_caller_aliases)
{
  int j;
  for (j=1;j<=gen_length(l_actuals);j++)
    {
      if (j!=i)
	{
	  expression exp = find_ith_argument(l_actuals,j);
	  if (expression_reference_p(exp))
	    {
	      reference ref = expression_reference(exp);
	      entity var = reference_variable(ref);
	      MAP(ALIAS_ASSOCIATION, aa,
	      {
		entity formal_var = alias_association_variable(aa);
		if (same_entity_p(formal_var,var))
		  {
		    entity formal_sec = alias_association_section(aa);
		    if (same_entity_p(formal_sec,sec)) return TRUE;
		  } 
	      },
		  l_caller_aliases);
	    }
	}
    }
  return FALSE;
}

/* This function adds the alias association of current
   formal variable to the l_current_aliases  */

static void storage_formal_add_aliases(call c, call_site cs, entity actual_var,
				       entity formal_var, expression subval, 
				       int i, list l_actuals)
{
  /* We add new alias_association only:
     - if ai has a section of common variable in alias_asociation 
     - if there exists another actual argument aj (j!=i) which is 
       also a formal parameter of current caller that has the same 
       section with ai in some alias_association 
     - if exists aj=ai (j!=i) (CALL P(K,K) or CALL P(X(I),X(J)) => add
       but we may not have section of ai, because it may not in alias_association 
       => section = ALIAS_SPECIAL, initial_off = 0, path = {(C,ordering)}*/

  list l_caller_aliases =  
    alias_associations_list((alias_associations)
			    db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,
						   caller_name, TRUE)); 
  ifdebug(3)
    fprintf(stderr, " \n Actual argument %s is a formal parameter of the current caller", 
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
	if (!SPECIAL_AREA_P(sec) || exist_same_actual_argument_p(i,l_actuals,actual_var)||
	    exist_actual_argument_formal_parameter_with_same_section_p(i,l_actuals,sec,
								       l_caller_aliases))
	  {
	    list path = CONS(CALL_SITE,cs,gen_full_copy_list(alias_association_call_chain(aa)));
	    expression off = expression_undefined;
	    alias_association one_alias = alias_association_undefined;
	    expression initial_off = alias_association_offset(aa);
	    ifdebug(3)
	      fprintf(stderr, " \n Entry for %s found in the alias_association", 
		      entity_name(caller_var));
	    /* If offset of aa is not expression_undefined, we must translate 
	       it to the module's frame by using binding information */
	    if (!expression_undefined_p(initial_off))
	      {
		expression new_initial_off = translate_to_module_frame(current_caller,current_mod,
								       initial_off,c);
		ifdebug(3)
		  {
		    fprintf(stderr, "\n Initial offset expression before translation: \n");
		    print_expression(initial_off);
		    fprintf(stderr, "\n Initial offset expression after translation: \n");
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
		  }
	      }
	    if (expression_undefined_p(off))
	      number_of_unknown_offsets++;
	    else 
	      number_of_known_offsets++;
	    ifdebug(3)
	      {
		fprintf(stderr, "\n Offset :\n");
		print_expression(off);
	      }		
	    one_alias = make_alias_association(formal_var,sec,off,path);
	    message_assert("alias_association is consistent", 
			   alias_association_consistent_p(one_alias));	
	    ifdebug(3)
	      print_alias_association(one_alias);
	    number_of_alias_associations++;
	    l_current_aliases = gen_nconc(l_current_aliases, 
					  CONS(ALIAS_ASSOCIATION,one_alias, NIL));
	  }
      }
  },
      l_caller_aliases);

  if (exist_same_actual_argument_p(i,l_actuals,actual_var))
    {
      /* For 3rd case : 
	 - if exists aj = ai (j!=i) (CALL P(K,K) or CALL P(X(I),X(J)) and 
	 we do not have section of ai, because it is not in alias_association 
	 => create new section = TOP_LEVEL::ALIAS_SPECIAL, 
            initial_off = 0, path = {(C,ordering)}*/
      entity sec = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,ALIAS_SECTION);
      list path = CONS(CALL_SITE,cs,NIL);
      expression off = expression_undefined;
      alias_association one_alias = alias_association_undefined;
      ifdebug(3)
	fprintf(stderr, " \n Same actual arguments but not in alias_association");
      if (expression_equal_integer_p(subval,0))
	/* The offset of actual variable is an integer 
	   that can always be translated into the module's frame*/
	off = int_to_expression(0);	
      else 
	{
	  off  = translate_to_module_frame(current_caller,current_mod,subval,c);
	  ifdebug(3)
	    {
	      fprintf(stderr, "\n Subval expression before translation: \n");
	      print_expression(subval);
	      fprintf(stderr, "\n Subval expression after translation: \n");
	      print_expression(off);
	    }
	}
      if (expression_undefined_p(off))
	number_of_unknown_offsets++;
      else 
	number_of_known_offsets++;
      ifdebug(3)
	{
	  fprintf(stderr, "\n Offset :\n");
	  print_expression(off);
	}		
      one_alias = make_alias_association(formal_var,sec,off,path);
      number_of_alias_associations++;
      message_assert("alias_association is consistent",
		     alias_association_consistent_p(one_alias));
      ifdebug(3)
	print_alias_association(one_alias);
      l_current_aliases = gen_nconc(l_current_aliases, 
				    CONS(ALIAS_ASSOCIATION,one_alias, NIL));
    }
}

static bool add_aliases_for_current_call_site(call c)
{
  if(call_function(c) == current_mod)
    {  
      statement stmt = current_statement_head();
      list l_actuals = call_arguments(c);
      int n = gen_length(l_actuals),i;
      int order = statement_ordering(stmt); 
      call_site cs = make_call_site(current_caller,order);
      ifdebug(3)
	{
	  fprintf(stderr, " \n Current caller: %s ", caller_name);
	  fprintf(stderr, " \n Current call site:\n");
	  print_statement(stmt);
	}
      message_assert("call_site is consistent", call_site_consistent_p(cs));		
      for (i=1; i<=n; i++)
	{
	  expression actual_arg = find_ith_argument(l_actuals,i);
	  if (expression_undefined_p(actual_arg))
	    pips_user_warning(" \n Problem with the argument list\n"); 
	  else if (expression_reference_p(actual_arg))
	    {
	      /* search for corresponding formal parameter */
	      entity formal_var = find_ith_formal_parameter(current_mod,i);
	      if (entity_undefined_p(formal_var))
		pips_user_warning(" \n The actual and formal argument lists do not
                                       have the same number of arguments\n");
	      else 
		{
		  reference actual_ref = expression_reference(actual_arg);
		  entity actual_var = reference_variable(actual_ref);
		  list l_actual_inds = reference_indices(actual_ref);
		  /* compute the subscript value stride, return expression_undefined if
		     if the actual argument is a scalar variable or array name*/
		  expression subval = subscript_value_stride(actual_var,l_actual_inds);
		  storage s = entity_storage(actual_var);
		  ifdebug(3)
		    {
		      fprintf(stderr, " \n Subval expression: \n ");
		      print_expression(subval);
		    }

		  /* To optimize the alias_association list, we need only to treat
		     the following case :
		     If ai is a common variable  => add
		     If ai is a local variable of current caller :
		       - if ai is in a EQUIVALENCE (by using ram_shared) => add
		       - if ai is not in a EQUIVALENCE but exists aj=ai (j!=i)
		         (CALL P(K,K) or CALL P(X(I),X(J)) => add
		     If ai is a formal parameter :
		       - if ai has a common section in alias_asociation => add
		       - if there exists aj (j!=i) which is also a formal parameter 
		         of current caller that has the same section => add
		       - if exists aj=ai (j!=i) (CALL P(K,K) or CALL P(X(I),X(J)) => add 
		         but we may not have section of ai, because it may not in alias_association 
			 => section = ALIAS_SPECIAL, initial_off = 0, path = {(C,ordering)} */

		  if (storage_ram_p(s))
		    /* The actual argument has a ram storage */
		    storage_ram_add_aliases(c,s,cs,formal_var,subval,i,l_actuals,actual_var);
		  else 
		    {
		      if (storage_formal_p(s))
			/* The actual argument is a formal parameter of the current caller, 
			   we must take the alias_associations of the caller */
			storage_formal_add_aliases(c,cs,actual_var,formal_var,subval,i,l_actuals);
		      else
			pips_user_warning(" \n Reference variable does not have formal/ram storage !!!\n"); 
		    }
		}
	    } 
	}
    }
  return TRUE;
}

static void add_aliases_for_current_caller()
{  
  statement caller_statement = (statement) db_get_memory_resource
    (DBR_CODE,caller_name, TRUE);
  initialize_ordering_to_statement(caller_statement); 
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
  /* we traverse all callers to find all call sites,
   * and fill in the list of aliases (l_current_aliases)
   */
  l_current_aliases = NIL;
  MAP(STRING, c_name,
  {
    current_caller = local_name_to_top_level_entity(c_name);
    caller_name = module_local_name(current_caller);
    add_aliases_for_current_caller();
  },
      l_callers);
  return l_current_aliases;
}
 

bool alias_propagation(char * module_name)
{
  list l_aliases = NIL;
  current_mod = local_name_to_top_level_entity(module_name);
  set_current_module_entity(current_mod);		
  number_of_alias_associations = 0;

  debug_on("ALIAS_PROPAGATION_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin alias_propagation for %s \n", module_name); 

  /* if the current procedure is the main program, do nothing*/
  if (! entity_main_module_p(current_mod))
    {
      list l_decls = code_declarations(entity_code(current_mod)); 
      list l_formals = NIL;
      /* search for formal parameters in the declaration list */   
      MAP(ENTITY, e,
      {
	if (formal_parameter_p(e))
	  l_formals = gen_nconc(l_formals,CONS(ENTITY,e,NIL));
      },
	  l_decls);
      /* if there is no formal parameter, do nothing */
      if (l_formals != NIL)
	{
	  /* Take the list of callers */
	  callees callers = (callees) db_get_memory_resource(DBR_CALLERS,
							     module_name,
							     TRUE);
	  list l_callers = callees_callees(callers); 	  
	  ifdebug(2)
	    {
	      fprintf(stderr," \n The formal parameters list :");
	      print_entities(l_formals);
	    }
	  /* if there is no caller, do nothing */
	  if (l_callers != NIL)
	    l_aliases = alias_propagation_callers(l_callers); 
	}
    }
  /* save to resource */
  DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_ASSOCIATIONS, 
			 module_name, 
			 (char*) make_alias_associations(l_aliases));  
  user_log(" \n The number of added alias associations for this module : %d \n",
	   number_of_alias_associations);
  user_log(" \n Total number of known offsets : %d \n",
	   number_of_known_offsets);
  user_log(" \n Total number of unknown offsets : %d \n",
	   number_of_unknown_offsets);
  reset_current_module_entity();
  current_mod = entity_undefined;
  ifdebug(1)
    fprintf(stderr, " \n End \n");
  debug_off();  
  return TRUE;
}








