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
  rule , so we have to multiply the array size by its element size in order to find out
  new value.
  
  For example : SPEC95, benchmark 125.turb3d : 
  SUBROUTINE TURB3D
  IMPLICIT REAL*8 (A-H,O-Z)
  COMMMON /ALL/ U(IXPP,IY,IZ)

  CALL TGVEL(U,V,W)

  SUBROUTINE TGVEL(U,V,W)
  COMPLEX*16 U(NXHP,NY,*)
 
  SUBROUTINE FOO(U,V,W)
  REAL*8 U(2,NXHP,NY,*)

  If we take into account the different types, as we have NXHP=IXPP/2, IY=NY, NZ=IZ => *=NZ  
*/

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
#include "transformer.h"
#include "text-util.h" /* for words_to_string*/
#include "instrumentation.h"
#include "transformations.h"

#define PREFIX_DEC  "$DEC"
#define NEW_DECLARATIONS  ".new_declarations"

/* Define a static stack and related functions to remember the current
   statement and then get the current precondition for top_down_adn_flt(): */
DEFINE_LOCAL_STACK(current_statement, statement)

static list l_current_caller_values = NIL;
static entity current_callee = entity_undefined;
static entity current_caller = entity_undefined;
static entity current_dummy_array = entity_undefined;
static call current_callsite = call_undefined;
static int number_of_right_array_declarations = 0;
static int number_of_one_and_assumed_array_declarations_but_no_caller = 0;

bool unbounded_expression_p(expression e)
{
  syntax s = expression_syntax(e);
  if (syntax_call_p(s)) 
    {
      string n = entity_local_name(call_function(syntax_call(s)));
      if (same_string_p(n, UNBOUNDED_DIMENSION_NAME))
	return TRUE;
    }
  return FALSE;  
}
 
expression make_unbounded_expression()
{
  return MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
}

bool array_entity_p(entity e)
{
  if (entity_variable_p(e))
    {
      variable var = type_variable(entity_type(e));   
      if (!ENDP(variable_dimensions(var)))  return TRUE;
    }
  return FALSE;
}

bool array_argument_p(expression e)
{
  if (expression_reference_p(e))
    {
      reference ref = expression_reference(e);
      entity ent = reference_variable(ref);
      if (array_entity_p(ent)) return TRUE;
    }
  return FALSE;
}

bool assumed_size_array_p(entity e)
{  
  /* return TRUE if e has an assumed-size array declarator  
     (the upper bound of the last dimension is equal to * : REAL A(*) )*/
  if (entity_variable_p(e))
    {
      variable v = type_variable(entity_type(e));   
      list l_dims = variable_dimensions(v);
      if (l_dims != NIL)
	{
	  int length = gen_length(l_dims);
	  dimension last_dim =  find_ith_dimension(l_dims,length);
	  if (unbounded_dimension_p(last_dim)) 
	    return TRUE;
	}
    }
  return FALSE;
}

bool pointer_type_array_p(entity e)
{  
  /* return TRUE if e has a pointer-type array declarator  
     (the upper bound of the last dimension is  equal to 1: REAL A(1) )*/
  if (entity_variable_p(e))
    {
      variable v = type_variable(entity_type(e));   
      list l_dims = variable_dimensions(v);
      if (l_dims != NIL)
	{
	  int length = gen_length(l_dims);
	  dimension last_dim =  find_ith_dimension(l_dims,length);
	  expression exp = dimension_upper(last_dim);
	  if (expression_equal_integer_p(exp,1)) 
	    return TRUE;
	}
    }
  return FALSE;
}

bool unnormalized_array_p(entity e)
{  
  /* return TRUE if e is an assumed-size array or a pointer-type array*/
  if (assumed_size_array_p(e) || pointer_type_array_p(e))
    return TRUE;
  return FALSE;
}

void print_array_declaration(entity e)
{
  /* This function prints only the dimensions of an array 
   => to use the script $PIPS_ROOT/Src/Script/misc/normalization.pl*/
  variable v = type_variable(entity_type(e));   
  list l_dims = variable_dimensions(v);
  user_log("(");
  while (!ENDP(l_dims)) 
    {
      dimension dim = DIMENSION(CAR(l_dims));
      string s = words_to_string(words_dimension(dim));
      user_log("%s", s);  
      l_dims = CDR(l_dims);
      if (l_dims == NIL)  user_log(")");
      else user_log(",");
      free(s);
    }
}

void print_entities(list l)
{
  MAP(ENTITY, e, { 
    fprintf(stderr, "%s ", entity_name(e));
  }, l);
}

list words_dimension(dimension obj)
{   
  list pc;
  pc = words_expression(dimension_lower(obj));
  pc = CHAIN_SWORD(pc,":");
  pc = gen_nconc(pc, words_expression(dimension_upper(obj)));
  return(pc);
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
	  l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
	  l_tmp = CDR(l_tmp);
	}
      l1=CDR(l1);
    } 
  return l;
}

list entity_to_formal_integer_parameters(entity f )
{
  /* get unsorted list of formal integer parameters for f by declaration
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


bool expression_equal_in_context_p(expression e1, expression e2, transformer context)
{
  /* Fast checks : 
     + e1 and e2 are same expressions 
     + e1 and e2 are equivalent because they are same common variables 
     Slow check:
     If NOT(e1=e2) + prec = infeasible, we have e1=e2 is always true*/
  normalized n1;
  normalized n2;
  if (same_expression_p(e1,e2)) return TRUE;
  if (expression_reference_p(e1) && expression_reference_p(e2))
    {
      reference ref1 = expression_reference(e1);
      reference ref2 = expression_reference(e2);
      entity en1 = reference_variable(ref1);
      entity en2 = reference_variable(ref2);
      if (same_scalar_location_p(en1, en2)) return TRUE;
    }
  clean_all_normalized(e1);
  clean_all_normalized(e2);
  n1 = NORMALIZE_EXPRESSION(e1);
  n2 = NORMALIZE_EXPRESSION(e2);
  ifdebug(3) 
    {	  
      fprintf(stderr, "\n First expression : ");    
      print_expression(e1);	
      fprintf(stderr, "\n Second expression : ");    
      print_expression(e2);
      fprintf(stderr, " \n equal in the context ?");
      fprint_transformer(stderr,context, entity_local_name);
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
	  if (VECTEUR_NUL_P(v_init)) return TRUE;
	  if (value_zero_p(val_of(v_init))) return TRUE;
	  if (value_notzero_p(val_of(v_init))) return FALSE;
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
	      return TRUE;
	    }
	  vect_rm(v_one);
	}
    }
  return FALSE;
}
  
bool same_dimension_p(entity actual_array, entity dummy_array, 
		      list l_actual_ref, int i, transformer context)
{
  /* This function returns TRUE if the actual array and the dummy array
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
	    return TRUE;
	  else
	    {
	      /* the actual argument is an array element name, 
	       * we have to calculate the subscript value also*/
	      expression actual_sub = find_ith_argument(l_actual_ref,i);
	      if (same_expression_p(actual_sub,actual_lower))
		return TRUE;
	    }
	}
    }
  return FALSE;
}


static list translate_to_callee_frame(expression e, transformer context)
{
  /* Check if the expression e can be translated to  the frame of the callee or not.      
     Return list of all possible translated expressions.
     Return list NIL if the expression can not be translated.

     If e is a reference:    
        1. Common variable => replace e by the corresponding variable 
	2. From the current context of the call site (precondition + association):
	   e = e1 where e1 is constant or e1 contains formal variable of the callee
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
      /* There are 3 cases for a reference M
	 Precondition : M =10
	 Association : M = FOO::N -1
	 Common variable : CALLER::M = CALLEE::M or M'*/
      reference ref = syntax_reference(syn);
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
	  /* search for equivalent variable in the list */	  
	  MAP(ENTITY, enti,
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
		    fprintf(stderr, "\n Common variable, add to list: \n");
		    print_expression(expr);
		  } 
		l = gen_nconc(l,CONS(EXPRESSION,copy_expression(expr),NIL));
		break;
	      }
	  },  l_callee_decl);
	}      
      /* Use the precondition + association of the call site:
	 Take only the equalities.
	 Project all variables belonging the the caller, except the current variable (from e)
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
	      ifdebug(3)
		{
		  fprintf(stderr, "\n Reference : using precondition + association \n");
		  fprint_transformer(stderr,context,entity_local_name);
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
		  ifdebug(3)
		    {
		      fprintf(stderr, "\n Check if e can be translated by using precondition + association: \n");
		      print_expression(e);
		      fprint_transformer(stderr,context, entity_local_name);
		    } 
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
			      bool check = TRUE;
			      for (v = newv; (v !=NULL) && (check); v = v->succ)
				{ 
				  Variable var = v->var;
				  if ((var != TCST) && (!variable_is_a_module_formal_parameter_p((entity)var,current_callee)) )
				    check = FALSE;
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
	      ifdebug(3)
		{
		  fprintf(stderr, "\n Reference : using precondition + association \n");
		  fprint_transformer(stderr,context, entity_local_name);
		}
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
	  /* Numerical constant or symbolic value (PARAMETER) . 
	     There are two cases:
	     1. Constant : 2000 => add to list
	     2. Association : 2000 = FOO:N (formal variable) => add N to the list
	     3. Precondition + Association : 2000 = M = FOO:N -1=> add N-1 to the list*/
	  value val = entity_initial(fun);
	  constant con = constant_undefined;
	  int i;
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n Add numerical constant or symbolic value to list: \n");
	      print_expression(e);
	    } 
	  l = gen_nconc(l,CONS(EXPRESSION,e,NIL));
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
		      ifdebug(3)
			{
			  fprintf(stderr, "\n Call : using Precondition + Association to find formal parameter equal to %d \n", i);
			  fprint_transformer(stderr,context, entity_local_name);
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
      break;
    }
  default:
    pips_error("", "Abnormal cases \n");
    break;
  }
  return l;
}

/*****************************************************************************

 This function returns the size of an unnormalized array, from position i+1, 
 multiplied by array element size: (D(i+1)*...*D(n-1))* element_size    
 
*****************************************************************************/

static expression size_of_unnormalized_dummy_array(entity dummy_array,int i)
{
  variable dummy_var = type_variable(entity_type(dummy_array));
  list l_dummy_dims = variable_dimensions(dummy_var);
  int num_dim = gen_length(l_dummy_dims),j;
  expression e = expression_undefined;
  basic b = variable_basic(type_variable(entity_type(dummy_array)));
  expression e_size = int_to_expression(SizeOfElements(b));
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
  if (!expression_undefined_p(e))
    e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(e),e_size);
  else 
    /* A(..,Di,*), as the size of actual array is multiplied by its element size,
       we have to mutilply the dummy array size by its element size also*/
    e = copy_expression(e_size);
  ifdebug(2)
    {
      fprintf(stderr, "\n Size of unnormalized dummy array: \n");
      print_expression(e);
    }
  return e;
}

/*****************************************************************************

 This function returns the size of an array, from position i+1, minus the 
 subscript value of array reference from position i+1, multiplied by array 
 element size : 
 (D(i+1)*...*Dn - (1+ s(i+1)-l(i+1) + (s(i+2)-l(i+2))*d(i+1)+...-1))* element_size   
 
*****************************************************************************/
expression size_of_actual_array(entity actual_array,list l_actual_ref,int i)
{
  expression e = expression_undefined;
  variable actual_var = type_variable(entity_type(actual_array));   
  list l_actual_dims = variable_dimensions(actual_var);
  int num_dim = gen_length(l_actual_dims),j;
  basic b = variable_basic(type_variable(entity_type(actual_array)));
  expression e_size = int_to_expression(SizeOfElements(b));
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
	  expression size_j;
	  if ( expression_constant_p(lower_j) && (expression_to_int(lower_j)==1))
	    size_j = copy_expression(upper_j);
	  else 
	    {
	      size_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,upper_j,lower_j);
	      size_j =  binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						    copy_expression(size_j),int_to_expression(1));
	    }    
	  if (!same_expression_p(sub_j,lower_j))
	    {
	      expression sub_low_j = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
								 sub_j,lower_j);
	      expression elem_j;
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
  if (!expression_undefined_p(e))
    e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,copy_expression(e),e_size);
  else 
    e = copy_expression(e_size);
  ifdebug(2)
    {
      fprintf(stderr, "\n Size of actual array:\n");
      print_expression(e);
    }
  return e;
}

static bool top_down_adn_call_flt(call c)
{
  current_callsite = c;
  if(call_function(current_callsite) == current_callee)
    {    
      int off = formal_offset(storage_formal(entity_storage(current_dummy_array)));
      list l_actual_args = call_arguments(current_callsite);
      expression actual_arg = find_ith_argument(l_actual_args,off);
      if (! expression_undefined_p(actual_arg) && array_argument_p(actual_arg))
	{
	  reference actual_ref = expression_reference(actual_arg);
	  entity actual_array = reference_variable(actual_ref);
	  if (!unnormalized_array_p(actual_array))
	    {
	      /* The actual array is not an assumed_sized array nor a pointer-type array
		 Attention : there may exist a declaration REAL A(1) which is good ? */
	      statement  stmt = current_statement_head();
	      expression actual_array_size = expression_undefined;
	      list l_new_caller_values = NIL;
	      int same_dim = 0;
	      list l_actual_ref = reference_indices(actual_ref);
	      transformer context;
	      if (statement_weakly_feasible_p(stmt))
		{
		  transformer prec = load_statement_precondition(stmt); 
		  ifdebug(3) 
		    {	  
		      fprintf(stderr, " \n Does the precondition is modified ? Before \n");
		      fprint_transformer(stderr,prec, entity_local_name);
		    }
		  // context = formal_and_actual_parameters_association(c,prec);
		  context = formal_and_actual_parameters_association(c,transformer_dup(prec));

		  ifdebug(3) 
		    {	  
		      fprintf(stderr, " \n Does the precondition is modified ? After \n");
		      fprint_transformer(stderr,prec, entity_local_name);
		    }
		}
	      else 
		context = formal_and_actual_parameters_association(c,transformer_identity());	     
	      while (same_dimension_p(actual_array,current_dummy_array,l_actual_ref,same_dim+1,context))
		same_dim ++;
	      ifdebug(2)
		fprintf(stderr, "\n Number of same dimensions : %d \n",same_dim);
	      actual_array_size = size_of_actual_array(actual_array,l_actual_ref,same_dim); 
	      ifdebug(2)
		{
		  fprintf(stderr, "\n Size of actual array before translation : \n");
		  print_expression(actual_array_size);
		}  
	      l_new_caller_values = translate_to_callee_frame(actual_array_size,context);
	      ifdebug(2)
		{
		  fprintf(stderr, "\n List of values after translation : \n");
		  print_expressions(l_new_caller_values);
		}  
	      if (l_new_caller_values != NIL)
		{
		  /* we have a list of translated actual array size expressions 
		     (in the frame of callee)*/
		  expression dummy_array_size = size_of_unnormalized_dummy_array(current_dummy_array,same_dim);
		  l_new_caller_values = my_list_division(l_new_caller_values,dummy_array_size);  
		  l_current_caller_values = my_list_intersection(l_current_caller_values,
								 l_new_caller_values); 
		  ifdebug(2)
		    {
		      fprintf(stderr, "\n List after intersection (new caller + current caller) : \n");
		      print_expressions(l_new_caller_values);
		    }  
		  if (l_current_caller_values == NIL)
		    /* There is no same values for different call sites => STOP  
		       gen_stop ?????*/
		    return FALSE;
		  /* We have a list of same values for different call sites => continue 
		     to find other calls to the callee*/
		  return TRUE;
		}
	    }
	  else
	    pips_user_warning(" The array declaration of the caller is *\n"); 
	}
      else 
	pips_user_warning("The actual and formal argument lists do not have the same number of arguments OR arguments are not corresponding !!! \n"); 
      l_current_caller_values = NIL;
      return FALSE;
    }
  current_callsite = call_undefined;
  return TRUE;
}

static list top_down_adn_caller_array()
{
  string caller_name = module_local_name(current_caller);
  statement caller_statement = (statement) db_get_memory_resource(DBR_CODE,caller_name,TRUE);  
  l_current_caller_values = NIL;
  make_current_statement_stack();
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,caller_name,TRUE));  
  gen_multi_recurse(caller_statement,
		    statement_domain, current_statement_filter,current_statement_rewrite,
		    call_domain, top_down_adn_call_flt, gen_null,
		    NULL);  
  reset_precondition_map();
  free_current_statement_stack(); 
  ifdebug(2)
    {
      fprintf(stderr, "\n Current list of values of the caller is :\n ");
      print_expressions(l_current_caller_values);
    }
  return l_current_caller_values;
}

static void top_down_adn_callers_arrays(list l_arrays,list callers)
{
  /* For one array, we may have different values from different call sites in different callers
     The algorithm tries to translate these values to the callee's frame by using the formal  
     parameters if it is possible. So :

     For all call sites in all callers, we have a list of possible values. If this list is NIL, 
     the value can not be translated into the callee's frame => STOP */

  string callee_name = entity_local_name(current_callee);
  while (!ENDP(l_arrays))
    {
      entity e = ENTITY(CAR(l_arrays));
      list l_callers = gen_copy_seq(callers);
      bool flag = TRUE;
      list l_old_values = NIL;
      variable v = type_variable(entity_type(e));   
      list l_dims = variable_dimensions(v);
      int length = gen_length(l_dims);
      dimension last_dim =  find_ith_dimension(l_dims,length);
      current_dummy_array = e;
      if (callers == NIL) 
	{
	  user_log(" \n Module without caller: %s \n", callee_name);
	  number_of_one_and_assumed_array_declarations_but_no_caller ++;
	}
      while (flag && !ENDP(l_callers))
	{
	  string caller_name = STRING(CAR(l_callers));
	  list l_new_values = NIL;
	  current_caller = local_name_to_top_level_entity(caller_name);
	  l_new_values = top_down_adn_caller_array();
	  ifdebug(2)
	    {
	      fprintf(stderr, "\n List of new values :\n ");
	      print_expressions(l_new_values);
	      fprintf(stderr, "\n List of old values (before intersection):\n ");
	      print_expressions(l_old_values);
	    }
	  if (l_new_values == NIL)
	    flag = FALSE;
	  else
	    {
	      l_old_values = my_list_intersection(l_old_values,l_new_values);
	      ifdebug(2)
		{
		  fprintf(stderr, "\n List of old values (after intersection):\n ");
		  print_expressions(l_old_values);
		}
	      if (l_old_values == NIL)
		flag = FALSE;
	    }
	  current_caller = entity_undefined;
	  l_callers = CDR(l_callers);
	}
      user_log("%s\t%s\t%s\t%s\t%d\t", PREFIX_DEC, 
	       db_get_memory_resource(DBR_USER_FILE,callee_name,TRUE), 
	       callee_name,entity_local_name(e),length);     
      if (flag && (l_old_values!=NIL))
	{
	  /* We have l_old_values is the list of same values for different callers 
	     => replace the unnormalized upper bound by 1 value in the list */
	  expression exp = EXPRESSION (CAR(l_old_values));
	  normalized n;
	  clean_all_normalized(exp);
	  n = NORMALIZE_EXPRESSION(exp);
	  if (normalized_linear_p(n))
	    {
	      Pvecteur v = normalized_linear(n);
	      exp = Pvecteur_to_expression(v);
	    }
	  else 
	    {
	      // Try to normalize the divide expression
	      if (operator_expression_p(exp,DIVIDE_OPERATOR_NAME))
		{
		  call c = syntax_call(expression_syntax(exp));
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
		      exp = binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,e11,e2);
		    }
		}
	    }
	  // print new bound
	  user_log("%s\t",words_to_string(words_expression(exp)));
	  // print old declaration
	  print_array_declaration(e);
	  dimension_upper(last_dim) = exp;
	  if (!unbounded_expression_p(exp))
	    number_of_right_array_declarations++;
	}
      else 
	{
	  /* We have different value for different callers, or there are variables that 
	     can not be translated to the callee's frame => we must leave * as the last
	     bound of this array*/
	  // print new bound
	  user_log("%s\t",words_to_string(words_expression(make_unbounded_expression())));
	  // print old declaration
	  print_array_declaration(e);
	  dimension_upper(last_dim) = make_unbounded_expression(); 
	}      
      user_log("\n");
      user_log("---------------------------------------------------------------------------------------\n");
      current_dummy_array = entity_undefined;
      l_arrays = CDR(l_arrays);
    }
}

/* The rule in pipsmake permits a top-down analyses  
     
top_down_array_declaration_normalization         > MODULE.new_declarations
                                                 > PROGRAM.entities
        < PROGRAM.entities
        < CALLERS.code
	< CALLERS.new_declarations
        < CALLERS.new_declarations

Algorithm : For each module: 
- Take the declaration list. 
- Take list of unnormalized array declarations, if this list is not nil      
  - Take the list of unnormalized array that are formal variable 
       - save the offset of each array in the formal argument list
       - get the list of callers of the module 
       - for each caller, get the list of call sites 
       - for each call site, calculate the normalized bounds 
          - base on offset 
          - base on actual array size (if the actual array is assumed-size  => return the * value, 
	    this case violates the standard norm but it exists in many case (SPEC95/applu,..))
	  - base on dummy array size
	  - base on subscript value of array element
	  - base on binding informations
	  - base on preconditions of the call site
	  - if the normalized bound expression contain only visible variables of the callee, 
	     => take this new value
	  - if not =>  it will become * 
       - if the normalized bounds for one array are different for different call sites and different caller 
         =>  it will become *
       - if they are all the same => take the new value
       - Modify the upper bound of the last dimension of the unnormalized declared array entity 
         by the new value or the *.
       - Put MODULE.new_declarations = "Okay, normalization has been done with right value"
           or "Okay, normalization has been done with * value" (for the case of *)
   - Take other unnormalized array => what to do with ? 
- If the list is nil => put MODULE.new_declarations, "Okay, there is nothing to normalize"*/

bool array_resizing_top_down(char *module_name)
{ 
  FILE * out;
  string new_declarations = db_build_file_resource_name(DBR_NEW_DECLARATIONS, 
							module_name, NEW_DECLARATIONS);
  string dir = db_get_current_workspace_directory();
  string filename = strdup(concatenate(dir, "/", new_declarations, NULL));
  current_callee = local_name_to_top_level_entity(module_name);
  debug_on("ARRAY_RESIZING_TOP_DOWN_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin top down array resizing for %s \n", module_name); 
  if (!entity_main_module_p(current_callee))
    {
      list l_callee_decl = NIL, l_formal_unnorm_arrays = NIL, l_other_unnorm_arrays = NIL;
      set_current_module_entity(current_callee);		   
      l_callee_decl = code_declarations(entity_code(current_callee));
   
      /* search for unnormalized array declarations in the list */
      MAP(ENTITY, e,
      {
	if (unnormalized_array_p(e))
	  {
	    if (formal_parameter_p(e))
	      l_formal_unnorm_arrays = gen_nconc(l_formal_unnorm_arrays,CONS(ENTITY,e,NIL));
	    else
	      /* it can not be return or rom storage 
	       * it can be ram storage ( local variables,  COMMON A(1) => i.e PerfectClub/SPICE)*/
	      l_other_unnorm_arrays = gen_nconc(l_other_unnorm_arrays,CONS(ENTITY,e,NIL));
	  }
      }, l_callee_decl);     
      ifdebug(2)
	{
	  fprintf(stderr," \n The formal unnormalized arrays list :");
	  print_entities(l_formal_unnorm_arrays);
	  fprintf(stderr," \n The other unnormalized arrays list :");
	  print_entities(l_other_unnorm_arrays);
	}
      if (l_formal_unnorm_arrays != NIL)
	{
	  /* Look for all call sites in the callers */
	  callees callers = (callees) db_get_memory_resource(DBR_CALLERS,module_name,TRUE);
	  list l_callers = callees_callees(callers); 
	  user_log("\n-------------------------------------------------------------------------------------\n");
	  user_log("Prefix \tFile \tModule \tArray \tNdim \tNew declaration\tOld declaration\n");
	  user_log("---------------------------------------------------------------------------------------\n");
	  ifdebug(2)
	    {
	      fprintf(stderr," \n There is/are %d callers : ",
		      gen_length(l_callers));
	      MAP(STRING, caller_name, {
		(void) fprintf(stderr, "%s, ", caller_name);
	      }, l_callers);
	      (void) fprintf(stderr, "\n");	
	    }	  
	  top_down_adn_callers_arrays(l_formal_unnorm_arrays,l_callers);  
	}
      reset_current_module_entity();
    }
   /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "/* Top down array resizing for module %s. */\n", module_name);
  safe_fclose(out, filename);
  free(dir);
  free(filename);
  current_callee = entity_undefined;
  user_log("\n The total number of right array declarations replaced: %d \n"
	  ,number_of_right_array_declarations );
  user_log(" \n The total number of unnormalized declarations without calller: %d \n"
	  ,number_of_one_and_assumed_array_declarations_but_no_caller );
  DB_PUT_FILE_RESOURCE(DBR_NEW_DECLARATIONS, module_name, new_declarations);
  ifdebug(1)
    fprintf(stderr, " \n End top down adn for %s \n", module_name);
  debug_off();
  return TRUE;
}








