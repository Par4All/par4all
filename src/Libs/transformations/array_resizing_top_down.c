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
 *		     TOP DOWN ARRAY RESIZING
 *
 *
*******************************************************************/
/* Example : 
      PROGRAM MAIN
      PARAMETER (N=10,M=20)
      REAL A(N,M)
      CALL FOO(A,N,M)
      END
      SUBROUTINE FOO(X,Y,Z)
      REAL X(Y,*) // REAL X(Y,1)
      DO I=1,10
         X(I,I)=1.
      ENDDO
      END
 In fact, the * or 1 could be replaced by its right value Z

 This phase is based on the association rules for dummy and actual arrays, in Fortran standard (ANSI) 
  * Section 15.9.3.3
  * 
  * Association of dummy and actual argument : arrays as dummy argument
  * 1. If actual argument is an array name : size(dummy_array) <= size(actual_array)
  * 2. Actual argument is an array element name :
  *    size(dummy_array) <= size(actual_array)+ 1 - subscript_value(array element)
  *
  * So the new value is derived from the 2 equations : 
  * size(dummy_array) = size(actual_array)   (1) or 
  * size(dummy_array) = size(actual_array)+ 1 - subscript_value(array element) (2)
  *
  * Our remarks to simplify these equations :
  * 1. If the first k dimensions of the actual array and the dummy array are the same, 
  * we have (1) is equivalent with  
  *
  * size_from_position(dummy_array,k+1) = size_from_position(actual_array,k+1)
  *
  * 2. If the first k dimensions of the actual array and the dummy array are the same, 
  * and the first k subscripts of the array element are equal with their 
  * correspond lower bounds (column-major order), we have (2) is equivalent with:
  *
  * size_from_position(dummy_array,k+1) = size_from_position(actual_array,k+1) +1 
  *                     - subscript_value_from_position(array_element,k+1)

  ATTENTION : FORTRAN standard (15.9.3) says that an association of dummy and actual 
  arguments is valid only if the type of the actual argument is the same as the type 
  of the corresponding dummy argument. But in practice, not much program respect this 
  rule , so we have to take into account the element size when computing the new value.
  We have an option when computing the array size : multiply the array size with the 
  element size or not
  
  For example : SPEC95, benchmark 125.turb3d : 
  SUBROUTINE TURB3D
  IMPLICIT REAL*8 (A-H,O-Z)
  COMMMON /ALL/ U(IXPP,IY,IZ)

  CALL TGVEL(U,V,W)

  SUBROUTINE TGVEL(U,V,W)
  COMPLEX*16 U(NXHP,NY,*)
 
  SUBROUTINE FOO(U,V,W)
  REAL*8 U(2,NXHP,NY,*)

  If we take into account the different types, as we have NXHP=IXPP/2, IY=NY, NZ=IZ => *=NZ  */

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
#include "alias-classes.h"
#include "alias_private.h"
#include "instrumentation.h"
#include "resources.h"
#include "misc.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "text-util.h" /* for words_to_string*/
#include "transformations.h"

#define PREFIX1  "$ARRAY_DECLARATION"
#define PREFIX2  "$COMMON_DECLARATION"
#define PREFIX3  "$COMMON_DECLARATION_END"
#define NEW_DECLARATIONS  ".new_declarations"

/* Define a static stack and related functions to remember the current
   statement and then get the current precondition for top_down_adn_flt(): */
DEFINE_LOCAL_STACK(current_statement, statement)

static list l_values_of_current_caller = NIL;
static entity current_callee = entity_undefined;
static entity current_caller = entity_undefined;
static entity current_dummy_array = entity_undefined;
static entity current_variable_caller = entity_undefined;
static FILE * instrument_file; /*To store new array declarations and assignments*/
static int number_of_unnormalized_arrays_without_caller = 0;
static int number_of_replaced_array_declarations = 0;
static int number_of_instrumented_array_declarations = 0;
static int number_of_array_size_assignments = 0;
static int number_of_processed_modules = 0;
static string file_name_caller= NULL;

static int opt = 0; /* 0 <= opt <= 7*/

bool module_is_called_by_main_program_p(entity mod)
{
  if (!entity_main_module_p(mod))
    {
      const char* mod_name = module_local_name(mod);
      callees callers = (callees) db_get_memory_resource(DBR_CALLERS,mod_name,true);
      list l_callers = callees_callees(callers); 
      while (!ENDP(l_callers))
	{
	  string caller_name = STRING(CAR(l_callers));
	  entity current_caller = module_name_to_entity(caller_name);
	  if (module_is_called_by_main_program_p(current_caller)) return true;
	  l_callers = CDR(l_callers);
	}
      return false;
    }
  return true;
}

static void display_array_resizing_top_down_statistics()
{
  user_log("* Number of unnormalized arrays without caller: %d *\n",
	   number_of_unnormalized_arrays_without_caller);	
  user_log("* Number of right array declarations replaced: %d*\n",
	   number_of_replaced_array_declarations);	
  user_log("* Number of array declarations instrumented: %d *\n",
	   number_of_instrumented_array_declarations);
  user_log("* Number of assignments added: %d *\n",
	   number_of_array_size_assignments);
  user_log("\n Number of processed modules: %d \n"
	   ,number_of_processed_modules); 
}


static bool scalar_argument_p(entity e)
{
  type t = entity_type(e);
  return(ENDP(variable_dimensions(type_variable(t))));
}




static list my_list_intersection(list l1, list l2)
{
  /* if l1 = NIL then return l2 
     returns a list of expressions that are in both lists l1 and l2 */
  if (l1 != NIL) 
    {
      list l_tmp = NIL;
      MAP(EXPRESSION,e1,
      {
	if (same_expression_in_list_p(e1,l2))
	  l_tmp = gen_nconc(l_tmp,CONS(EXPRESSION,e1,NIL));
      },
	  l1);
      return l_tmp;
    }
  return l2;
}

/* Multiply each element of list l by e*/
static list my_list_multiplication(list l, expression e)
{
  list l_tmp = NIL;
  while (!ENDP(l))
    {
      expression e1= EXPRESSION(CAR(l));
      e1 = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,e1,e);
      l_tmp = gen_nconc(l_tmp,CONS(EXPRESSION,e1,NIL));
      l = CDR(l);
    }
  return l_tmp;
}

/* Divide each element of list l by e*/
static list my_list_division(list l, expression e)
{
  list l_tmp = NIL;
  while (!ENDP(l))
    {
      expression e1= EXPRESSION(CAR(l));
      e1 = binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,e1,e);
      l_tmp = gen_nconc(l_tmp,CONS(EXPRESSION,e1,NIL));
      l = CDR(l);
    }
  return l_tmp;
}

/* Replace each element e of list l1 by "op e"*/
static list my_list_change(list l1, entity op)
{
  list l = NIL;
  while (!ENDP(l1))
    {
      expression e = MakeUnaryCall(op,EXPRESSION(CAR(l1)));
      l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
      l1 = CDR(l1);
    } 
  return l;
}

/* Create new list of expressions "e1 op e2" where e1 is in l1, e2 is in l2*/
static list my_list_combination(list l1, list l2, entity op)
{
  list l = NIL;
  while (!ENDP(l1))
    {
      expression e1 = EXPRESSION(CAR(l1));
      list l_tmp = gen_copy_seq(l2);
      while (!ENDP(l_tmp))
	{
	  expression e2 = EXPRESSION(CAR(l_tmp));
	  expression e = MakeBinaryCall(op,e1,e2);

	  /* attention : add the expression_in_list_p test ??? */
	  l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
	  l_tmp = CDR(l_tmp);
	}
      l1=CDR(l1);
    } 
  return l;
}

static list entity_to_formal_integer_parameters(entity f )
{
  /* get unsorted list of formal integer parameters of module f by declaration
     filtering; these parameters may not be used by the callee's
     semantics analysis, but we have no way to know it because
     value mappings are not available */
  
  list formals = NIL;
  list decl = list_undefined;
  
  pips_assert("entity_to_formal_parameters",entity_module_p(f));
  
  decl = code_declarations(entity_code(f));
  MAPL(ce, {entity e = ENTITY(CAR(ce));
  if(storage_formal_p(entity_storage(e)) &&
     entity_integer_scalar_p(e))
    formals = CONS(ENTITY, e, formals);},
       decl);
  
  return formals;
}


/* formal_and_actual_parameters_association(call c, transformer pre):
 * Add equalities between actual and formal parameters binding by call c
 * to pre 
 * pre := pre  U  {f  = expr }
 *                  i       i
 * for all i such that formal fi is an integer scalar variable and 
 * expression expr-i is affine
 */
transformer formal_and_actual_parameters_association(call c, transformer pre)
{
  entity f = call_function(c);
  list pc = call_arguments(c);
  list formals = entity_to_formal_integer_parameters(f);
  cons * ce;
  
  ifdebug(6) {
    debug(6,"formal_and_actual_parameters_association",
	  "begin for call to %s pre=%x\n", module_local_name(f), pre);
    dump_transformer(pre);
  }
  
  pips_assert("formal_and_actual_parameters_association", 
	      entity_module_p(f));
  pips_assert("formal_and_actual_parameters_association", 
	      pre != transformer_undefined);
  
  /* let's start a long, long, long MAPL, so long that MAPL is a pain */
  for( ce = formals; !ENDP(ce); POP(ce)) {
    entity e = ENTITY(CAR(ce));
    int r = formal_offset(storage_formal(entity_storage(e)));
    expression expr;
    normalized n;
    
    if((expr = find_ith_argument(pc, r)) == expression_undefined)
      user_error("formal_and_actual_parameters_association",
		 "not enough args for formal parm. %d\n", r);
    
    n = NORMALIZE_EXPRESSION(expr);
    if(normalized_linear_p(n)) {
      Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
      entity e_new = external_entity_to_new_value(e);
      
      vect_add_elem(&v, (Variable) e_new, -1);
      pre = transformer_equality_add(pre, v);
    }
  }
  
  free_arguments(formals);
  
  ifdebug(6) {
    debug(6,"formal_and_actual_parameters_association",
	  "new pre=%x\n", pre);
    dump_transformer(pre);
    debug(6,"formal_and_actual_parameters_association","end for call to %s\n",
	  module_local_name(f));
  }  
  return pre;
}


static bool expression_equal_in_context_p(expression e1, expression e2, transformer context)
{
  /* Fast checks : 
     + e1 and e2 are same expressions 
     + e1 and e2 are equivalent because they are same common variables 
     Slow check:
     If NOT(e1=e2) + prec = infeasible, we have e1=e2 is always true*/
  normalized n1;
  normalized n2;
  if (same_expression_p(e1,e2)) return true;
  if (expression_reference_p(e1) && expression_reference_p(e2))
    {
      reference ref1 = expression_reference(e1);
      reference ref2 = expression_reference(e2);
      entity en1 = reference_variable(ref1);
      entity en2 = reference_variable(ref2);
      if (same_scalar_location_p(en1, en2)) return true;
    }
  clean_all_normalized(e1);
  clean_all_normalized(e2);
  n1 = NORMALIZE_EXPRESSION(e1);
  n2 = NORMALIZE_EXPRESSION(e2);
  ifdebug(4) 
    {	  
      fprintf(stderr, "\n First expression : ");    
      print_expression(e1);	
      fprintf(stderr, "\n Second expression : ");    
      print_expression(e2);
      fprintf(stderr, " \n equal in the context ?");
      fprint_transformer(stderr,context, (get_variable_name_t)entity_local_name);
    }
  if (normalized_linear_p(n1) && normalized_linear_p(n2))
    {
      Pvecteur v1 = normalized_linear(n1);
      Pvecteur v2 = normalized_linear(n2);	      
      Pvecteur v_init = vect_substract(v1,v2);
      /* Trivial test e1=N+M, e2=M+N => e1=e2.
	 This test may be subsumed by the above test same_expression_p()*/
      if (vect_constant_p(v_init))
	{
	  /* Tets if v_init == 0 */
	  if (VECTEUR_NUL_P(v_init)) return true;
	  if (value_zero_p(val_of(v_init))) return true;
	  if (value_notzero_p(val_of(v_init))) return false;
	}
      else 
	{
	  Pvecteur v_one = vect_new(TCST,1);
	  Pvecteur v_not_e = vect_add(v_init,v_one);
	  Pvecteur v_temp = vect_multiply(v_init,-1);
	  Pvecteur v_not_e_2 = vect_add(v_temp,v_one);
	  Psysteme ps = predicate_system(transformer_relation(context));
	  if (!efficient_sc_check_inequality_feasibility(v_not_e,ps) && 
	      !efficient_sc_check_inequality_feasibility(v_not_e_2,ps))
	    {
	      vect_rm(v_one);
	      return true;
	    }
	  vect_rm(v_one);
	}
    }
  return false;
}
  
bool same_dimension_p(entity actual_array, entity dummy_array, 
		      list l_actual_ref, size_t i, transformer context)
{
  /* This function returns true if the actual array and the dummy array
   * have the same dimension number i, with respect to the current context 
   * (precondition + association)
   * In case if the actual argument is an array element, we have to add 
   * the following condition: the i-th subscript of the array element 
   * is equal to its correspond lower bound.*/

  variable actual_var = type_variable(entity_type(actual_array));
  variable dummy_var = type_variable(entity_type(dummy_array));
  list l_actual_dims = variable_dimensions(actual_var);
  list l_dummy_dims = variable_dimensions(dummy_var);
  
  /* The following test is necessary in case of array reshaping */
  if ((i <= gen_length(l_actual_dims)) && (i < gen_length(l_dummy_dims)))
    {
      dimension dummy_dim = find_ith_dimension(l_dummy_dims,i);
      expression dummy_lower = dimension_lower(dummy_dim);
      expression dummy_upper = dimension_upper(dummy_dim);
      dimension actual_dim = find_ith_dimension(l_actual_dims,i);
      expression actual_lower = dimension_lower(actual_dim);
      expression actual_upper = dimension_upper(actual_dim);
      expression dummy_size, actual_size;
      if (expression_equal_integer_p(dummy_lower,1) && 
	  expression_equal_integer_p(actual_lower,1))
	{
	  dummy_size = copy_expression(dummy_upper);
	  actual_size = copy_expression(actual_upper);
	}
      else
	{
	  dummy_size = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
						   dummy_upper,dummy_lower);
	  actual_size = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
						    actual_upper,actual_lower);
	}
      if (expression_equal_in_context_p(actual_size, dummy_size, context))
	{
	  if (l_actual_ref == NIL)
	    /* case : array name */
	    return true;
	  else
	    {
	      /* the actual argument is an array element name, 
	       * we have to calculate the subscript value also*/
	      expression actual_sub = find_ith_argument(l_actual_ref,i);
	      if (same_expression_p(actual_sub,actual_lower))
		return true;
	    }
	}
    }
  return false;
}

static list translate_reference_to_callee_frame(expression e, reference ref, transformer context)
{
  /* There are 2 cases for a reference M
     1. Common variable : CALLER::M = CALLEE::M or M'
     2. Precondition +  Association : M =10 or M = FOO::N -1 */
  list l = NIL;
  entity en = reference_variable(ref);
  normalized ne;
  if (variable_in_common_p(en))
    {
      /* Check if the COOMON/FOO/ N is also declared in the callee or not 
       * We can use ram_shared which contains a list of aliased variables with en 
       * but it does not work ????  
       
       * Another way : looking for a variable in the declaration of the callee
       * that has the same offset in the same common block */
      list l_callee_decl = code_declarations(entity_code(current_callee));
      bool in_callee = false;
      /* search for equivalent variable in the list */	  
      FOREACH(ENTITY, enti,l_callee_decl)
      {
	if (same_scalar_location_p(en, enti))
	  {
	    expression expr;
	    /* ATTENTION : enti may be an array, such as A(2):
	       COMMON C1,C2,C3,C4,C5
	       COMMON C1,A(2,2)
	       we must return A(1,1), not A */
	    if (array_entity_p(enti))
	      {
		variable varenti = type_variable(entity_type(enti));   		      
		int len =  gen_length(variable_dimensions(varenti));
		list l_inds = make_list_of_constant(1,len);
		reference refer = make_reference(enti,l_inds);
		expr = reference_to_expression(refer);
	      }
	    else 
	      expr = entity_to_expression(enti);
	    ifdebug(2)
	      {
		fprintf(stderr, "\n Syntax reference: Common variable, add to list: \n");
		print_expression(expr);
	      } 
	    in_callee = true;
	    l = gen_nconc(l,CONS(EXPRESSION,copy_expression(expr),NIL));
	    break;
	  }
      }
      
      /* If en is a pips created common variable, we can add this common declaration
	 to the callee's declaration list => use this value.
	 If en is an initial program's common variable => do we have right to add ? 
	 confusion between local and global varibales that have same name  ??? */
      
      if (!in_callee && strstr(entity_local_name(en),"I_PIPS_") != NULL)
	{
	  const char* callee_name = module_local_name(current_callee);
	  string user_file = db_get_memory_resource(DBR_USER_FILE,callee_name,true);
	  string base_name = pips_basename(user_file, NULL);
	  string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
	  const char* pips_variable_name = entity_local_name(en);
	  string pips_common_name = strstr(entity_local_name(en),"PIPS_");
	  string new_decl = strdup(concatenate("      INTEGER*8 ",pips_variable_name,"\n",
					       "      COMMON /",pips_common_name,"/ ",pips_variable_name,"\n",NULL));
	  /* Attention, miss a test if this declaration has already been added or not 
	     => Solutions : 
	     1. if add pips variable declaration for current module => add for all callees 
	     where the array is passed ???
	     2. Filter when using script array_resizing_instrumentation => simpler ??? */
	  fprintf(instrument_file,"%s\t%s\t%s\t(%d,%d)\n",PREFIX2,file_name,callee_name,0,1);
	  fprintf(instrument_file,"%s", new_decl);
	  fprintf(instrument_file,"%s\n",PREFIX3);
	  free(file_name), file_name = NULL;
	  free(new_decl), new_decl = NULL;
	  l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
	}
    }      
  /* Use the precondition + association of the call site:
     Take only the equalities.
     Project all variables belonging to the caller, except the current variable (from e)
     there are 2 cases :
     1. The projection is not exact , there are over flows
     Return the SC_UNDEFINED => what to do, like before ? 
     2. The result is exact, three small cases: 
     2.1 The system is always false sc_empty => unreachable code ?
     2.2 The system is always true sc_rn => we have nothing ?
     2.3 The system is parametric =>
     
     Look for equality that contain e
     Delete e from the vector 
     Check if the remaining of the vectors contains only constant (TCTS) 
     or formal variable of the callee,
     Add to the list the expression (= remaining vertor)*/
  clean_all_normalized(e);
  ne =  NORMALIZE_EXPRESSION(e);
  if (normalized_linear_p(ne))
    {
      Pvecteur ve = normalized_linear(ne);
      Variable vare = var_of(ve); 
      Psysteme ps_tmp = predicate_system(transformer_relation(context));
      Pbase b_tmp = ps_tmp->base;
      /* Attention :   here the transformer current_con text is consistent 
	 but not the system ps_tmp. I do not understand why ?
	 fprintf(stderr, "consistent psystem ps_tmp before");
	 pips_assert("consistent psystem ps_tmp", sc_consistent_p(ps_tmp));*/
      if (base_contains_variable_p(b_tmp,vare))
	{
	  Psysteme ps = sc_dup(ps_tmp);
	  Pbase b = ps->base;
	  Pvecteur pv_var = VECTEUR_NUL; 	  
	  ifdebug(4)
	    {
	      fprintf(stderr, "\n Syntax reference : using precondition + association \n");
	      fprint_transformer(stderr,context,(get_variable_name_t)entity_local_name);
	    }
	  for(; !VECTEUR_NUL_P(b); b = b->succ) 
	    {
	      Variable var = vecteur_var(b);
	      if ((strcmp(module_local_name(current_caller),entity_module_name((entity)var))==0)
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
			    Check if the remaining vector contains only constant or formal argument of callee*/
			  Pvecteur v;
			  bool check = true;
			  for (v = newv; (v !=NULL) && (check); v = v->succ)
			    { 
			      Variable var = v->var;
			      if ((var != TCST) && (!variable_is_a_module_formal_parameter_p((entity)var,current_callee)) )
				check = false;
			    }
			  if (check)
			    {
			      expression new_exp = Pvecteur_to_expression(newv);
			      ifdebug(2)
				{
				  fprintf(stderr, "\n Add new expression/reference to list by using prec+ asso : \n");
				  print_expression(new_exp);
				} 
			      l = gen_nconc(l,CONS(EXPRESSION,new_exp,NIL));
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
  return l;
}

static list translate_to_callee_frame(expression e, transformer context);

static list translate_call_to_callee_frame(call ca, transformer context)
{
  list l = NIL;
  entity fun = call_function(ca);
  list l_args = call_arguments(ca);
  if (l_args==NIL)
    {
      /* Numerical constant or symbolic value (PARAMETER (M=2000)) . 
	 There are 2 cases:
	 1. Constant : 2000 => add to list
	 2. Precondition + Association : 2000 = M = FOO:N -1=> add N-1 to the list  
	 => trade-off : we have more chances vs more computations */
      value val = entity_initial(fun);
      constant con = constant_undefined;
      int i;
      /*  ifdebug(2)
	  {
	  fprintf(stderr, "\n Add  symbolic value to list: \n");
	  print_expression(e);
	  } */
      /* There is something changed in PIPS ? 
	 PARAMETER M =10 
	 add M, not 10 as before ??? */
      // l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
      
      if (value_symbolic_p(val))
	{
	  /* Symbolic constant: PARAMETER (Fortran) or CONST (Pascal) */
	  symbolic sym = value_symbolic(val);
	  con = symbolic_constant(sym);
	}
      if (value_constant_p(val))
	con = value_constant(val);
      if (!constant_undefined_p(con))
	{
	  if (constant_int_p(con))
	    {
	      /* Looking for a formal parameter of the callee that equals 
		 to i in the Precondition + Association information 
		 Add this formal parameter to the list
		 We have to project all variables of the caller
		 Attention : bug in PerfectClub/mdg :
		 looking for formal parameter equal to 1 in system: {==-1} */
	      Psysteme ps_tmp = predicate_system(transformer_relation(context));
	      Psysteme ps = sc_dup(ps_tmp);
	      Pbase b = ps->base;
	      Pvecteur pv_var = VECTEUR_NUL; 
	      /* There is something changed in PIPS ? 
		 PARAMETER M =10 
		 add M, not 10 as before ??? */
	      int j = constant_int(con);
	      ifdebug(2)
		{
		  fprintf(stderr, "\n Add numerical constant to list: \n");
		  print_expression(int_to_expression(j));
		} 
	      l = gen_nconc(l,CONS(EXPRESSION,int_to_expression(j),NIL));
	      for(; !VECTEUR_NUL_P(b); b = b->succ) 
		{
		  Variable var = vecteur_var(b);
		  if (!variable_is_a_module_formal_parameter_p((entity)var,current_callee))
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
		  i = constant_int(con);
		  ifdebug(4)
		    {
		      fprintf(stderr, "\n Call : using Precondition + Association to find formal parameter equal to %d \n", i);
		      fprint_transformer(stderr,context, (get_variable_name_t)entity_local_name);
		    }
		  for (egal = ps->egalites; egal != NULL; egal = egal1) 
		    {
		      /* Take the equations of the system */
		      Pvecteur vec = egal->vecteur,v;
		      for (v = vec; v !=NULL; v = v->succ)
			{ 
			  if (term_cst(v))
			    {
			      Value valu = v->val;
			      if (value_eq(value_abs(valu),int_to_value(i)))
				{
				  Pvecteur newv = vect_del_var(vec,TCST);
				  expression new_exp;
				  if (value_pos_p(valu))
				    vect_chg_sgn(newv);
				  new_exp = Pvecteur_to_expression(newv);
				  ifdebug(2)
				    {
				      fprintf(stderr, "\n Add new expression/constant to list by prec+asso: \n");
				      print_expression(new_exp);
				    } 
				  l = gen_nconc(l,CONS(EXPRESSION,new_exp,NIL));
				  vect_rm(newv);
				}
			    }
			}
		      egal1 = egal->succ;
		    } 
		}
	      sc_rm(ps);
	    }
	}
    }
  else 
    {
      /* e is a call, not a constant 
	 Recursive : with the arguments of the call
	 As our generated expression e is a call with operators : +,-,* only,
	 we treat only these cases */
      if (gen_length(l_args)==1)
	{
	  expression e1 = EXPRESSION(CAR(l_args));
	  list l1 = translate_to_callee_frame(e1, context);
	  l1 = my_list_change(l1,fun);
	  l = gen_nconc(l,l1);
	}
      if (gen_length(l_args)==2)
	{
	  expression e1 = EXPRESSION(CAR(l_args));
	  expression e2 = EXPRESSION(CAR(CDR(l_args)));
	  list l1 = translate_to_callee_frame(e1, context);
	  list l2 = translate_to_callee_frame(e2, context);
	  list l3 = my_list_combination(l1,l2,fun);
	  l = gen_nconc(l,l3);
	} 
    }
  return l;
}

static list translate_to_callee_frame(expression e, transformer context)
{
  /* Check if the expression e can be translated to  the frame of the callee or not.      
     Return list of all possible translated expressions.
     Return list NIL if the expression can not be translated.

     BE CAREFUL when adding expressions to list => may have combination exploration 

     If e is a reference:    
        1. Common variable => replace e by the corresponding variable 
	2. From the current context of the call site (precondition + association):
	   e = e1 where e1 is constant or e1 contains formal variables of the callee
	   => l = {e1}

     If e is a call : 
        1. Storage rom : 
	   n1 = numerical constant or  symbolic value (PARAMETER) 
	   n2 = equivalent formal variable (if exist)
	   l = {n1,n2}
	2. Recursive : e = e1 * e2 
	   translate_to_callee_frame(e1) = l1 = (e11,e12,e13)
	   translate_to_callee_frame(e2) = l2 = (e21,e22)
	   translate_to_callee_frame(e) = (e11*e21, e11*e22, e12*e21,e12*....)
	   
	   If l1 or l2 = NIL => l = NIL

     If e is a range : error, size of array can not be a range */

  list l= NIL;
  syntax syn = expression_syntax(e);
  tag t = syntax_tag(syn);
  switch(t){  
  case is_syntax_reference: 
    {
      reference ref = syntax_reference(syn);
      ifdebug(2)
	{
	  fprintf(stderr, "\n Syntax reference \n");
	  print_expression(e);
	} 
      return translate_reference_to_callee_frame(e,ref,context);
    }
  case is_syntax_call:
    {
      call ca = syntax_call(syn);
      ifdebug(2)
	{
	  fprintf(stderr, "\n Syntax call \n");
	  print_expression(e);
	} 
      return translate_call_to_callee_frame(ca,context);
    }
  default:
    pips_internal_error("Abnormal cases ");
    break;
  }
  return l;
}

/*****************************************************************************

 This function returns the size of an unnormalized array, from position i+1:
  (D(i+1)*...*D(n-1))
 
*****************************************************************************/

static expression size_of_unnormalized_dummy_array(entity dummy_array,int i)
{
  variable dummy_var = type_variable(entity_type(dummy_array));
  list l_dummy_dims = variable_dimensions(dummy_var);
  int num_dim = gen_length(l_dummy_dims),j;
  expression e = expression_undefined;
  for (j=i+1; j<= num_dim-1; j++)
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
	  size_j =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						copy_expression(size_j),int_to_expression(1));
	}
      if (expression_undefined_p(e))
	e = copy_expression(size_j);
      else
	e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(e),size_j);  
    }
  ifdebug(2)
    {
      fprintf(stderr, "\n Size of unnormalized dummy array: \n");
      print_expression(e);
    }
  return e;
}

/*****************************************************************************

 This function returns the size of an array, from position i+1, minus the 
 subscript value of array reference from position i+1: 
 (D(i+1)*...*Dn - (1+ s(i+1)-l(i+1) + (s(i+2)-l(i+2))*d(i+1)+...-1))
 
*****************************************************************************/
expression size_of_actual_array(entity actual_array,list l_actual_ref,int i)
{
  expression e = expression_undefined;
  variable actual_var = type_variable(entity_type(actual_array));   
  list l_actual_dims = variable_dimensions(actual_var);
  int num_dim = gen_length(l_actual_dims),j;
  for (j=i+1; j<= num_dim; j++)
    {
      dimension dim_j = find_ith_dimension(l_actual_dims,j);
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
      if (expression_undefined_p(e))
	e = copy_expression(size_j);
      else
	e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(e),size_j);  
    }
  ifdebug(3)
    {
      fprintf(stderr, "\n Size of actual array without subscript value \n");
      print_expression(e);
    }  
  if (l_actual_ref!=NIL)
    {
      /* the actual argument is an array element name, 
       * we have to compute the subscript value also*/
      expression sum = expression_undefined, prod = expression_undefined;
      ifdebug(3)
	fprintf(stderr, "\n actual argument is an array element name:");
      for (j=i+1; j<= num_dim; j++)
	{
	  dimension dim_j = find_ith_dimension(l_actual_dims,j);
	  expression lower_j = dimension_lower(dim_j);
	  expression sub_j = find_ith_argument(l_actual_ref,j);
	  expression upper_j = dimension_upper(dim_j);
	  expression size_j,sub_low_j,elem_j;
	  if ( expression_constant_p(lower_j) && (expression_to_int(lower_j)==1))
	    size_j = copy_expression(upper_j);
	  else 
	    {
	      size_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,upper_j,lower_j);
	      size_j =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						    copy_expression(size_j),int_to_expression(1));
	    }    
	  /* ATTENTION : heuristic 
	     We can distinguish or not the special case: lower_bound = subscript, 
	     1. If not, we do not lose information such as in SPEC95/applu.f :
	          real u(5,33,33,33)
	          call exact(i,j,1,u(1,i,j,1))
		    => size = 5.33.33.33 -((i-1)5 +(j-1)5.33 +(k-1)5.33.33), 
		    not 5.33.33.33 -((i-1)5 +(j-1)5.33) (as k=1)
		  subroutine exact(i,j,k,u000ijk)
		  We will have more combinations, but more chance to translate 
		  the size of actual array to the callee's frame
	     2. If yes, there are so much combinations => it takes long time to compute,
	        such as in SPEC95/turb3d.f 
	     CONCLUSION : for the efficiency of time, we distinguish this case*/

	  if (!same_expression_p(sub_j,lower_j)) 
	    {
	      sub_low_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
						      sub_j,lower_j);
	      if (expression_undefined_p(prod))
		elem_j = copy_expression(sub_low_j);
	      else
		elem_j = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
						     sub_low_j,prod);
	      if (expression_undefined_p(sum))
		sum = copy_expression(elem_j);
	      else
		sum = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						  sum, elem_j);
	    }
	  if (expression_undefined_p(prod))
	    prod = copy_expression(size_j);
	  else
	    prod = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
					       prod,size_j);
	  ifdebug(4)
	    {
	      fprintf(stderr, "\n j = %d \n",j);
	      fprintf(stderr, "\n prod : \n");
	      print_expression(prod);
	      fprintf(stderr, "\n sum =: \n");
	      print_expression(sum); 
	    }
	}
      if (!expression_undefined_p(sum))
	{
	  e =  binary_intrinsic_expression(MINUS_OPERATOR_NAME,
					   copy_expression(e),sum);
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n Size of actual array - subscript value : \n");
	      print_expression(e);
	    }
	}
    }
  ifdebug(2)
    {
      fprintf(stderr, "\n Size of actual array:\n");
      print_expression(e);
    }
  return e;
}

/* This function computes a list of translated values corresponding to the current call site,
   and then modify the list of values of the current caller*/
static bool top_down_adn_call_flt(call c)
{
  if(call_function(c) == current_callee)
    {    
      int off = formal_offset(storage_formal(entity_storage(current_dummy_array)));
      list l_actual_args = call_arguments(c);
      expression actual_arg = find_ith_argument(l_actual_args,off);
      if (! expression_undefined_p(actual_arg))
	{	
	  entity actual_array = expression_to_entity(actual_arg);
	  int same_dim = 0;
	  statement stmt = current_statement_head();
	  expression actual_array_size = expression_undefined;
	  list l_values_of_current_call_site = NIL;
	  transformer context;
	  ifdebug(3) 
	    {	  
	      fprintf(stderr, " \n Current statement : \n");
	      print_statement(stmt);
	    }
	  if (statement_weakly_feasible_p(stmt))
	    {
	      transformer prec = load_statement_precondition(stmt); 
	      ifdebug(4) 
		{	  
		  fprintf(stderr, " \n The precondition before \n");
		  fprint_transformer(stderr,prec, (get_variable_name_t)entity_local_name);
		}
	      // context = formal_and_actual_parameters_association(c,prec);
	      context = formal_and_actual_parameters_association(c,transformer_dup(prec));
	      ifdebug(4) 
		{	  
		  fprintf(stderr, " \n The precondition after \n");
		  fprint_transformer(stderr,prec,(get_variable_name_t)entity_local_name);
		}
	    }
	  else 
	    context = formal_and_actual_parameters_association(c,transformer_identity());
	  if (array_argument_p(actual_arg))
	    {
	      /* Actual argument is an array */     
	      if (!unnormalized_array_p(actual_array))
		{
		  /* The actual array is not an assumed_sized array nor a pointer-type array
		     Attention : there may exist a declaration REAL A(1) which is good ? */	
		  reference actual_ref = expression_reference(actual_arg);
		  list l_actual_ref = reference_indices(actual_ref);
		  while (same_dimension_p(actual_array,current_dummy_array,l_actual_ref,same_dim+1,context))
		    same_dim ++;
		  ifdebug(2)
		    fprintf(stderr, "\n Number of same dimensions : %d \n",same_dim);
		  actual_array_size = size_of_actual_array(actual_array,l_actual_ref,same_dim); 
		}
	      else
		{
		  /* Actual argument is an unnormalized array => 
		     Pointer case => compute actual array size => try to translate 
		     if not ok => code instrumentation*/
		  l_values_of_current_caller = NIL;
		  return false;
		}
	    }
	  else
	    {
	      /* Actual argument is not an array*/
	      if (scalar_argument_p(actual_array))
		{
		  /* Actual argument is a scalar variable like in PerfectClub/spc77
		     SUBROUTINE MSTADB
		     REAL T
		     CALL SATVAP(T,1)
		     SUBROUTINE SATVAP(T,JMAX)
		     REAL T(1)
		     T(1:JMAX)*/
		  ifdebug(2)
		    fprintf(stderr,"Actual argument is a scalar variable");
		  actual_array_size = int_to_expression(1);
		}
	      else
		{
		  if (value_constant_p(entity_initial(actual_array)) &&
		      constant_call_p(value_constant(entity_initial(actual_array))))
		    { 
		      /* Actual argument can be a string as in SPEC/wave5 
			 CALL ABRT("BAD BY",6)
			 SUBROUTINE ABRT(MESS,NC)
			 CHARACTER MESS(1)
			 The argument's name is TOP-LEVEL:'name'*/
		      ifdebug(2)
			fprintf(stderr,"Actual argument is a string");
		      actual_array_size = int_to_expression(strlen(entity_name(actual_array))-12);
		    }
		  else 
		    {
		      /* Actual argument is not an array, not a scalar variable, not a string
			 => code instrumentation*/
		      l_values_of_current_caller = NIL;
		      return false;
		    }
		}
	    }
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n Size of actual array before translation : \n");
	      print_expression(actual_array_size);
	    }  
	  l_values_of_current_call_site = translate_to_callee_frame(actual_array_size,context);
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n Size of actual array after translation (list of possible values) : \n");
	      print_expressions(l_values_of_current_call_site);
	    }  
	  if (l_values_of_current_call_site != NIL)
	    {
	      /* we have a list of translated actual array sizes (in the callee's frame)*/
	      expression dummy_array_size = size_of_unnormalized_dummy_array(current_dummy_array,same_dim);
	      if (value_constant_p(entity_initial(actual_array)))
		{
		  /* String case : do not compare the element size*/
		  if (!expression_undefined_p(dummy_array_size))
		    l_values_of_current_call_site = my_list_division(l_values_of_current_call_site,
								     dummy_array_size);
		}
	      else 
		{
		  /* Now, compare the element size of actual and dummy arrays*/
		  basic b_dummy = variable_basic(type_variable(entity_type(current_dummy_array)));
		  basic b_actual = variable_basic(type_variable(entity_type(actual_array)));
		  int i_dummy = SizeOfElements(b_dummy);
		  int i_actual = SizeOfElements(b_actual);
		  if (i_dummy == i_actual)
		    {
		      if (!expression_undefined_p(dummy_array_size))
			l_values_of_current_call_site = my_list_division(l_values_of_current_call_site,
									 dummy_array_size);
		    }
		  else
		    {
		      l_values_of_current_call_site = my_list_multiplication(l_values_of_current_call_site,
									     int_to_expression(i_actual));
		      if (expression_undefined_p(dummy_array_size))
			l_values_of_current_call_site = my_list_division(l_values_of_current_call_site,
									 int_to_expression(i_dummy));
		      else
			{
			  dummy_array_size = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,dummy_array_size,
									 int_to_expression(i_dummy));
			  l_values_of_current_call_site = my_list_division(l_values_of_current_call_site,
									   dummy_array_size);
			}
		    }
		}
	      l_values_of_current_caller = my_list_intersection(l_values_of_current_caller,
								l_values_of_current_call_site); 
	      ifdebug(2)
		{
		  fprintf(stderr, "\n List of values of the current caller (after intersection): \n");
		  print_expressions(l_values_of_current_caller); 
		}  
	      if (l_values_of_current_caller == NIL)
		/* There is no same value for different call sites 
		   => code instrumentation  */
		return false;
	      /* We have a list of same values for different call sites => continue 
		 to find other calls to the callee*/
	      return true;
	    }	 
	}
      else 
	/* Actual argument is an undefined expression => code instrumentation*/
      l_values_of_current_caller = NIL;
      return false;
    }
  return true;
}

/* Insert "I_PIPS_SUB_ARRAY = actual_array_size" before each call to the current callee*/
static void instrument_call_rwt(call c)
{
  if(call_function(c) == current_callee)
    {    
      int off = formal_offset(storage_formal(entity_storage(current_dummy_array)));
      list l_actual_args = call_arguments(c);
      expression actual_arg = find_ith_argument(l_actual_args,off);
      if (! expression_undefined_p(actual_arg))
	{  
	  entity actual_array = expression_to_entity(actual_arg);
	  expression actual_array_size = expression_undefined;
	  if (array_argument_p(actual_arg))
	    {
	      if (!unnormalized_array_p(actual_array))
		{
		  /* The actual array is not an assumed_sized array nor a pointer-type array
		     Attention : there may exist a declaration REAL A(1) which is good ? */
		  reference actual_ref = expression_reference(actual_arg);
		  list l_actual_ref = reference_indices(actual_ref);
		  actual_array_size = size_of_actual_array(actual_array,l_actual_ref,0);
		}
	      else
		pips_user_warning("Array %s in module %s has unnormalized declaration\n",
				  entity_local_name(actual_array),module_local_name(current_caller)); 
	      /* How to instrument code ??? Pointer cases ??? 
		 Caller is not called by the main program => already excluded*/
	    }
	  else
	    {
	      /* Actual argument is not an array*/
	      if (scalar_argument_p(actual_array))
		{
		  ifdebug(2)
		    fprintf(stderr,"Actual argument is a scalar variable");
		  actual_array_size = int_to_expression(1);
		}
	      else
		{
		  if (value_constant_p(entity_initial(actual_array)) &&
		      constant_call_p(value_constant(entity_initial(actual_array))))
		    { 
		      /* Actual argument can be a string whose name is TOP-LEVEL:'name'*/
		      ifdebug(2)
			fprintf(stderr,"Actual argument is a string");
		      actual_array_size = int_to_expression(strlen(entity_name(actual_array))-12);
		    }
		  else 
		    /* Abnormal case*/
		    pips_user_warning("Actual argument %s is not an array, not a scalar variable, not a string.\n",entity_local_name(actual_array));
		}
	    }
	  if (!expression_undefined_p(actual_array_size))
	    {
	      /* The actual array size is computable */
	      statement stmt = current_statement_head();
	      int order = statement_ordering(stmt);
	      expression left = entity_to_expression(current_variable_caller);
	      statement new_s;
	      if (!value_constant_p(entity_initial(actual_array)))
		{
		  /* Compare the element size*/
		  basic b_dummy = variable_basic(type_variable(entity_type(current_dummy_array)));
		  basic b_actual = variable_basic(type_variable(entity_type(actual_array)));
		  int i_dummy = SizeOfElements(b_dummy);
		  int i_actual = SizeOfElements(b_actual);
		  if (i_dummy != i_actual)
		    {
		      actual_array_size = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,actual_array_size,
								      int_to_expression(i_actual));
		      actual_array_size = binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,actual_array_size,
								      int_to_expression(i_dummy));
		    }
		}
	      new_s = make_assign_statement(left,actual_array_size);
	      ifdebug(2)
		{
		  fprintf(stderr, "\n Size of actual array: \n");
		  print_expression(actual_array_size);
		}
	      /* As we can not modify ALL.code, we cannot use a function like :
		 insert_statement(stmt,new_s,true).
		 Instead, we have to stock the assignment as weel as the ordering
		 of the call site in a special file, named TD_instrument.out, and
		 then use a script to insert the assignment before the call site.*/
	      ifdebug(2)
		{
		  fprintf(stderr, "\n New statements: \n");
		  print_statement(new_s);
		  print_statement(stmt);
		}
	      fprintf(instrument_file, "%s\t%s\t%s\t(%d,%d)\n",PREFIX2,file_name_caller,
		      module_local_name(current_caller),ORDERING_NUMBER(order),ORDERING_STATEMENT(order));
	      print_text(instrument_file, text_statement(entity_undefined,0,new_s,NIL));
	      fprintf(instrument_file,"%s\n",PREFIX3);
	      number_of_array_size_assignments++;
	    }
	}
      else
	/* Abnormal case*/
	pips_user_warning("Actual argument is an undefined expression\n");
    }
}

/* This function computes a list of translated values for the new size of the formal array*/

static list top_down_adn_caller_array()
{
  const char* caller_name = module_local_name(current_caller);
  statement caller_statement = (statement) db_get_memory_resource(DBR_CODE,caller_name,true);  
  l_values_of_current_caller = NIL;
  make_current_statement_stack();
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,caller_name,true));  
  gen_multi_recurse(caller_statement,
		    statement_domain, current_statement_filter,current_statement_rewrite,
		    call_domain, top_down_adn_call_flt, gen_null,
		    NULL);  
  reset_precondition_map();
  free_current_statement_stack(); 
  ifdebug(2)
    {
      fprintf(stderr, "\n List of values of the current caller is :\n ");
      print_expressions(l_values_of_current_caller);
    }
  return l_values_of_current_caller;
}

static void instrument_caller_array()
{
  const char* caller_name = module_local_name(current_caller);
  statement caller_statement = (statement) db_get_memory_resource(DBR_CODE,caller_name,true);  
  make_current_statement_stack();
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,caller_name,true));  
  gen_multi_recurse(caller_statement,
		    statement_domain, current_statement_filter,current_statement_rewrite,
		    call_domain,gen_true,instrument_call_rwt,NULL);  
  reset_precondition_map();
  free_current_statement_stack(); 
  return;
}

static void top_down_adn_callers_arrays(list l_arrays,list l_callers)
{
  /* For each unnormalized array:
     For each call site in each caller, we compute a list of possible values 
     (these values have been translated to the frame of the callee). 
     If this list is NIL => code instrumentation
     Else, from different lists of different call sites in different callers, 
     try to find a same value that becomes the new size of the unnormalized array. 
     If this value does not exist  => code instrumentation */

  /* Find out the name of the printed file in Src directory: database/Src/file.f */
  const char* callee_name = module_local_name(current_callee);
  string user_file = db_get_memory_resource(DBR_USER_FILE,callee_name,true);
  string base_name = pips_basename(user_file, NULL);
  string file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), "/",base_name,NULL));
  
  while (!ENDP(l_arrays))
    {
      list l = gen_copy_seq(l_callers),l_values = NIL,l_dims;
      bool flag = true;
      variable v;
      int length;
      dimension last_dim;
      expression new_value;
      current_dummy_array = ENTITY(CAR(l_arrays));      
      v = type_variable(entity_type(current_dummy_array));   
      l_dims = variable_dimensions(v);
      length = gen_length(l_dims);
      last_dim =  find_ith_dimension(l_dims,length);
      while (flag && !ENDP(l))
	{
	  string caller_name = STRING(CAR(l));
	  current_caller = module_name_to_entity(caller_name);
	  if ( (opt%8 >= 4) && (! module_is_called_by_main_program_p(current_caller)))
	    /* If the current caller is never called by the main program => 
	       no need to follow this caller*/
	    pips_user_warning("Module %s is not called by the main program\n",caller_name);
	  else
	    {
	      list l_values_of_one_caller = top_down_adn_caller_array();
	      ifdebug(2)
		{
		  fprintf(stderr, "\n List of values computed for caller %s is:\n ",caller_name);
		  print_expressions(l_values_of_one_caller);
		}
	      if (l_values_of_one_caller == NIL)
		flag = false;
	      else
		{
		  ifdebug(2)
		    {
		      fprintf(stderr, "\n List of values (before intersection):\n ");
		      print_expressions(l_values);
		    }
		  l_values = my_list_intersection(l_values,l_values_of_one_caller);
		  ifdebug(2)
		    {
		      fprintf(stderr, "\n List of values (after intersection):\n ");
		      print_expressions(l_values);
		    }
		  if (l_values == NIL)
		    flag = false;
		}
	    }
	  current_caller = entity_undefined;
	  l = CDR(l);
	} 
      if (flag && (l_values!=NIL))
	{
	  /* We have l_values is the list of same values for different callers 
	     => replace the unnormalized upper bound by 1 value in this list */
	  normalized n;
	  new_value = EXPRESSION (CAR(l_values));
	  clean_all_normalized(new_value);
	  n = NORMALIZE_EXPRESSION(new_value);
	  if (normalized_linear_p(n))
	    {
	      Pvecteur ve = normalized_linear(n);
	      new_value = Pvecteur_to_expression(ve);
	    }
	  else 
	    {
	      // Try to normalize the divide expression
	      if (operator_expression_p(new_value,DIVIDE_OPERATOR_NAME))
		{
		  call c = syntax_call(expression_syntax(new_value));
		  list l = call_arguments(c);
		  expression e1 = EXPRESSION(CAR(l));
		  expression e2 = EXPRESSION(CAR(CDR(l)));
		  normalized n1;
		  clean_all_normalized(e1);
		  n1 = NORMALIZE_EXPRESSION(e1);
		  if (normalized_linear_p(n1))
		    {
		      Pvecteur v1 = normalized_linear(n1);
		      expression e11 = Pvecteur_to_expression(v1);
		      new_value = binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,e11,e2);
		    }
		}
	    }
	  number_of_replaced_array_declarations++;
	}
      else 
	{
	  /* We have different values for different callers, or there are variables that 
	     can not be translated to the callee's frame => use code instrumentation:
	     ......................
	     Insert "INTERGER I_PIPS_SUB_ARRAY
	     COMMON /PIPS_SUB_ARRAY/ I_PIPS_SUB_ARRAY"
	     in the declaration of current callee and every caller that is called by the main program
	     Modify array declaration ARRAY(,,I_PIPS_SUB_ARRAY/dummy_array_size)
	     Insert "I_PIPS_SUB_ARRAY = actual_array_size" before each call site
	     ......................*/
	  
	  /* Insert new declaration in the current callee*/
	  string pips_common_name = strdup(concatenate("PIPS_",callee_name,"_",
						       entity_local_name(current_dummy_array),NULL));
	  string pips_variable_name = strdup(concatenate("I_",pips_common_name,NULL));
	  entity pips_common = make_new_common(pips_common_name,current_callee);
	  entity pips_variable = make_new_integer_scalar_common_variable(pips_variable_name,
									 current_callee,pips_common);
	  string new_decl = strdup(concatenate("      INTEGER*8 ",pips_variable_name,"\n",
					       "      COMMON /",pips_common_name,"/ ",pips_variable_name,"\n",NULL));
	  //  string old_decl = code_decls_text(entity_code(current_callee));
	  expression dummy_array_size = size_of_unnormalized_dummy_array(current_dummy_array,0);
	  new_value = entity_to_expression(pips_variable);
	  if (!expression_undefined_p(dummy_array_size))
	    new_value = binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,new_value,dummy_array_size);
	  
	  /* We do not modify  code_decls_text(entity_code(module)) because of
	     repeated bugs with ENTRY */
	  
	  /* ifdebug(2)
	     fprintf(stderr,"\n Old declaration of %s is %s\n",callee_name,
	     code_decls_text(entity_code(current_callee)));  
	     code_decls_text(entity_code(current_callee)) = strdup(concatenate(old_decl,new_decl,NULL));  
	     ifdebug(2)
	     fprintf(stderr,"\n New declaration of %s is %s\n",callee_name,
	     code_decls_text(entity_code(current_callee)));  
	     free(old_decl), old_decl = NULL; */
	  
	  fprintf(instrument_file,"%s\t%s\t%s\t(%d,%d)\n",PREFIX2,file_name,callee_name,0,1);
	  fprintf(instrument_file,"%s", new_decl);
	  fprintf(instrument_file,"%s\n",PREFIX3);
	  
	  /* Insert new declaration and assignments in callers that are called by the main program*/
	  l = gen_copy_seq(l_callers);
	  while (!ENDP(l))
	    {
	      string caller_name = STRING(CAR(l));
	      current_caller = module_name_to_entity(caller_name);
	      if ((opt%8 >= 4)&& (! module_is_called_by_main_program_p(current_caller)))
		/* If the current caller is never called by the main program => 
		   no need to follow this caller*/
		pips_user_warning("Module %s is not called by the main program\n",caller_name);
	      else
		{
		  string user_file_caller = db_get_memory_resource(DBR_USER_FILE,caller_name,true);
		  string base_name_caller = pips_basename(user_file_caller, NULL);
		  file_name_caller = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE),
							"/",base_name_caller,NULL));
		  current_variable_caller = make_new_integer_scalar_common_variable(pips_variable_name,current_caller,
										    pips_common);
		  fprintf(instrument_file, "%s\t%s\t%s\t(%d,%d)\n",PREFIX2,file_name_caller,caller_name,0,1);
		  fprintf(instrument_file, "%s", new_decl);
		  fprintf(instrument_file, "%s\n",PREFIX3);

		  /* insert "I_PIPS_SUB_ARRAY = actual_array_size" before each call site*/
		  instrument_caller_array();
		  current_variable_caller = entity_undefined;
		  free(file_name_caller), file_name_caller = NULL;
		}
	      current_caller = entity_undefined;
	      l = CDR(l);
	    }
	  number_of_instrumented_array_declarations++;
	  free(new_decl), new_decl = NULL;
	}
      fprintf(instrument_file,"%s\t%s\t%s\t%s\t%d\t%s\t%s\n",PREFIX1,file_name,
	      callee_name,entity_local_name(current_dummy_array),length,
	      words_to_string(words_expression(dimension_upper(last_dim),NIL)),
	      words_to_string(words_expression(new_value,NIL)));
      dimension_upper(last_dim) = new_value;
      l_arrays = CDR(l_arrays);
    }
  free(file_name), file_name = NULL;
}

/* The rule in pipsmake permits a top-down analysis

array_resizing_top_down         > MODULE.new_declarations
                                > PROGRAM.entities
        < PROGRAM.entities
        < CALLERS.code
	< CALLERS.new_declarations

Algorithm : For each module that is called by the main program
- Take the declaration list. 
- Take list of unnormalized array declarations   
  - For each unnormalized array that is formal variable. 
       - save the offset of this array in the formal arguments list
       - get the list of callers of the module 
       - for each caller, get the list of call sites 
       - for each call site, compute the new size for this array
          - base on the offset 
          - base on the actual array size (if the actual array is assumed-size => return the * value, 
	    this case violates the standard norm but it exists in many case (SPEC95/applu,..)
	    NN 26/10/2001: But now our goal is 100% resized arrays, by using code instrumentation 
	    as a complementary phase, so this case does not exist anymore)
	  - base on the dummy array size
	  - base on the subscript value of the array element
	  - base on the binding information
	  - base on the precondition of the call site
	  - base on the translation from the caller's to the callee's name space
	  - if success (the size can be translated to the callee's frame) => take this new value
	  - if fail => using code instrumentation  
       - For all call sites and all callers, if there exists a same value 
           => take this value as the new size for the unnormalized array
       - Else, using code instrumentation
       - Modify the upper bound of the last dimension of the unnormalized declared array entity 
         by the new value.
       - Put MODULE.new_declarations = "Okay, normalization has been done with right value"
- If the list is nil => put MODULE.new_declarations, "Okay, there is nothing to normalize"*/

bool array_resizing_top_down(const char* module_name)
{ 
  /* instrument_file is used to store new array declarations and assignments which
     will be used by a script to insert these declarations in the source code in:
     xxx.database/Src/file_name.f

     declaration_file is only used to make the top-down mechanics of pipsmake possible*/

  FILE * declaration_file;
  string new_declarations = db_build_file_resource_name(DBR_NEW_DECLARATIONS, 
							module_name,NEW_DECLARATIONS);
  string dir_name = db_get_current_workspace_directory();
  string declaration_file_name = strdup(concatenate(dir_name, "/", new_declarations, NULL));
  string instrument_file_name = strdup(concatenate(dir_name, "/TD_instrument.out", NULL));
  instrument_file = safe_fopen(instrument_file_name, "a");  
  current_callee = module_name_to_entity(module_name);

  number_of_processed_modules++;
  debug_on("ARRAY_RESIZING_TOP_DOWN_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin top down array resizing for %s \n", module_name); 
  if (!entity_main_module_p(current_callee))
    {
      list l_callee_decl = NIL, l_formal_unnorm_arrays = NIL;
      set_current_module_entity(current_callee);		   
      l_callee_decl = code_declarations(entity_code(current_callee));  
 
      opt = get_int_property("ARRAY_RESIZING_TOP_DOWN_OPTION");
      /* opt in {0,1,2,3} => Do not use MAIN program 
	 opt in {4,5,6,7} => Use MAIN program 	 
	 => (opt mod 8) <= 3 or not  

	 opt in {0,1,4,5} => Compute new declarations for assumed-size and one arrays only
	 opt in {2,3,6,7} => Compute new declarations for all formal array arguments
	 => (opt mod 4) <= 1 or not 
	 
	 opt in {0,2,4,6} => Compute new declarations for assumed-size and one arrays 
	 opt in {1,3,5,7} => Compute new declarations for assumed-size arrays only
	 => (opt mod 2) = 0 or not */

      /* Depending on the option, take the list of arrays to treat*/

      MAP(ENTITY, e,{
	if (opt%4 <= 1)
	  {
	    /* Compute new declarations for assumed-size and one arrays only */
	    if (opt%2 == 0)
	      {
		/* Compute new declarations for assumed-size and one arrays */
		if (unnormalized_array_p(e) && formal_parameter_p(e))
		  l_formal_unnorm_arrays = gen_nconc(l_formal_unnorm_arrays,CONS(ENTITY,e,NIL));
	      }
	    else 
	      {
		/* Compute new declarations for assumed-size arrays only*/
		if (assumed_size_array_p(e) && formal_parameter_p(e))
		  l_formal_unnorm_arrays = gen_nconc(l_formal_unnorm_arrays,CONS(ENTITY,e,NIL));
	      }
	  }
	else
	  {
	    /* Compute new declarations for all formal array arguments 
	       To be modified, the whole C code: instrumentation, assumed-size checks,...
	       How about multi-dimensional array ? replace all upper bounds ?
	       => different script, ...*/

	    // if (array_entity_p(e) && formal_parameter_p(e))
	    // l_formal_unnorm_arrays = gen_nconc(l_formal_unnorm_arrays,CONS(ENTITY,e,NIL));
	    user_log("\n This option has not been implemented yet");
	  }
      }, l_callee_decl);     
      
      if (l_formal_unnorm_arrays != NIL)
	{
	  if ((opt%8 >= 4) && (!module_is_called_by_main_program_p(current_callee)))
	    {
	      /* Use MAIN program */
	      pips_user_warning("Module %s is not called by the main program\n",module_name);
	      number_of_unnormalized_arrays_without_caller += 
		gen_length(l_formal_unnorm_arrays);
	    }
	  else
	    {
	      /* Do not use MAIN program or module_is_called_by_main_program.
		 Take all callers of the current callee*/
	      callees callers = (callees) db_get_memory_resource(DBR_CALLERS,module_name,true);
	      list l_callers = callees_callees(callers); 
	      if (l_callers == NIL)
		{
		  pips_user_warning("Module %s has no caller\n",module_name);
		  number_of_unnormalized_arrays_without_caller += 
		    gen_length(l_formal_unnorm_arrays);
		}
	      ifdebug(2)
		{
		  fprintf(stderr," \n The formal unnormalized array list :");
		  print_entities(l_formal_unnorm_arrays);
		  fprintf(stderr," \n The caller list : ");
		  MAP(STRING, caller_name, {
		    (void) fprintf(stderr, "%s, ", caller_name);
		  }, l_callers);
		  (void) fprintf(stderr, "\n");	
		} 
	      top_down_adn_callers_arrays(l_formal_unnorm_arrays,l_callers); 
	    }
	}
      reset_current_module_entity();
    }
  display_array_resizing_top_down_statistics();
  declaration_file = safe_fopen(declaration_file_name, "w");
  fprintf(declaration_file, "/* Top down array resizing for module %s. */\n", module_name);
  safe_fclose(declaration_file, declaration_file_name);
  safe_fclose(instrument_file, instrument_file_name);
  free(dir_name), dir_name = NULL;
  free(declaration_file_name), declaration_file_name = NULL;
  free(instrument_file_name), instrument_file_name = NULL;
  current_callee = entity_undefined;
  DB_PUT_FILE_RESOURCE(DBR_NEW_DECLARATIONS, module_name, new_declarations);
  ifdebug(1)
    fprintf(stderr, " \n End top down array resizing for %s \n", module_name);
  debug_off();
  return true;
}








