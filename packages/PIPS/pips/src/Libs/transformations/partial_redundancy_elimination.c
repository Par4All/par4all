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
 *
 *          PARTIAL REDUNDANCY ELIMINATION
 *
 *
 *******************************************************************/

#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "text-util.h"
#include "database.h"
#include "resources.h"
#include "control.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "misc.h"
#include "expressions.h"
#include "alias_private.h"
#include "semantics.h" /* for module_to_value_mappings() */
#include "transformations.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "ricedg.h"

/* Statistic variables: */
static int number_of_simplified_expressions;
static int number_of_simplified_assign_expressions;
static int number_of_simplified_if_conditions;
static int number_of_false_while_conditions;
static int number_of_false_if_conditions;
static int number_of_true_if_conditions;

static void initialize_partial_redundancy_elimination_statistics()
{
    number_of_simplified_expressions = 0;
    number_of_simplified_assign_expressions = 0;
    number_of_simplified_if_conditions = 0;
    number_of_false_while_conditions = 0;
    number_of_false_if_conditions = 0;
    number_of_true_if_conditions = 0;
}

static void display_partial_redundancy_elimination_statistics()
{
  number_of_simplified_expressions = 
    number_of_simplified_assign_expressions +
    number_of_false_while_conditions +
    number_of_simplified_if_conditions;

  if (number_of_simplified_expressions > 0) 
    {
      user_log("* There %s %d simplified expression%s including :*\n",
	       number_of_simplified_expressions  > 1 ? "are" : "is",
	       number_of_simplified_expressions ,
	       number_of_simplified_expressions  > 1 ? "s" : "");		
    }
  if (number_of_simplified_assign_expressions > 0) 
    {
      user_log("*\t %d simplified assign expression%s *\n",	     
	       number_of_simplified_assign_expressions ,
	       number_of_simplified_assign_expressions  > 1 ? "s" : "");		
    }
  if (number_of_simplified_expressions > 0) 
    {
      user_log("*\t %d simplified if condition%s *\n",
	       number_of_simplified_if_conditions ,
	       number_of_simplified_if_conditions  > 1 ? "s" : "");		
    }
  if (number_of_false_while_conditions > 0) 
    {
      user_log("*\t %d false while loop condition%s *\n",
	       number_of_false_while_conditions,
	       number_of_false_while_conditions > 1 ? "s" : "");		
    }
  if (number_of_false_if_conditions > 0) 
    {
      user_log("* There %s %d if statement%s with false condition *\n",
	       number_of_false_if_conditions > 1 ? "are" : "is",
	       number_of_false_if_conditions,
	       number_of_false_if_conditions > 1 ? "s" : "");		
    }
  if (number_of_true_if_conditions > 0) 
    {
      user_log("* There %s %d if statement%s with true condition *\n",
	       number_of_true_if_conditions > 1 ? "are" : "is",
	       number_of_true_if_conditions,
	       number_of_true_if_conditions > 1 ? "s" : "");		
    }
}

/* should be moved to linear. FC.
 * returns whether ineq is redundant or not wrt prec.
 * how : first fast check, then actual feasibility.
 * default to feasible
   This function removes all the inequalities
   with big coefficient (> MAXCOEFFICIENT) in the system to avoid overflow */

static bool all_variables_in_precondition(Pvecteur v, Psysteme p)
{
  Pbase b = p->base;
  Pvecteur vec;
  for (vec = v; vec != NULL; vec = vec->succ)
    { 
      Variable var = vec->var;
      if ((var !=TCST) && (!base_contains_variable_p(b,var)))
	return false;
    }
  return true;
}

static Psysteme simplify_big_coeff(Psysteme sc)
{
  int MAXCOEFFICIENT = 1000;
  Value val_max = int_to_value(MAXCOEFFICIENT);
  Pcontrainte ineg, ineg1; 
  for (ineg = sc->inegalites; ineg != NULL; ineg = ineg1) 
    {
      Pvecteur vec = ineg->vecteur,v;
      for (v = vec; v !=NULL && ineg->vecteur != VECTEUR_NUL; v = v->succ)
	{ 
	  Value val = v->val;
	  Variable var =v->var;
	  if (value_gt(value_abs(val),val_max) && (var!=TCST))
	    eq_set_vect_nul(ineg);
	}
      ineg1 = ineg->succ;
    }
  sc_rm_empty_constraints(sc,0);
  return(sc);
}

bool efficient_sc_check_inequality_feasibility(Pvecteur v, Psysteme prec)
{
  bool retour = true;
  ifdebug(3) 
    {	  
      fprintf(stderr, "\n Efficient check feasibility : ");    
      vect_fprint(stderr,v, (char * (*)(Variable)) entity_local_name);
      fprintf(stderr, " \n with the Precondition : ");
      sc_fprint(stderr,prec, (char * (*)(Variable)) entity_local_name);
    }
  switch (sc_check_inequality_redundancy(contrainte_make(v), prec)) /* try fast check */
    {
    case 1: /* ok, feasible because ineq is redundant wrt prec*/
      return true;
    case 2: /* ok, system {prec + ineq} is infeasible */
      return false;
    case 0: /* no result, try slow version. default is feasible. */
      {

	/* NN: 25/09/2001, try to avoid overflows by using a smaller system: 
	   only constraints transitively connected to a constraint referencing a variable
	   of interest in v are taken from prec.
	   sc_restricted_to_variables_transitive_closure(sc,vars) */
	Psysteme s_dup = sc_dup(prec);
	Pbase vars = make_base_from_vect(v);
	Psysteme s_res = sc_restricted_to_variables_transitive_closure(s_dup,vars);

	/* nofoverflows is used to save the number of overflows before the call
	   to sc_integer_feasibility_ofl_ctrl*/ 

	int nofoverflows = linear_number_of_exception_thrown;
	//	Psysteme s = sc_dup(prec);
	Psysteme s = sc_dup(s_res);
	ifdebug(3) 
	  {	  
	    fprintf(stderr, "\n Slow check of feasibility: ");    
	    fprintf(stderr, " \n with the new Preconditions : ");
	    sc_fprint(stderr,s, (char * (*)(Variable)) entity_local_name);
	  }
	/* add the inegality to the system*/
	sc_constraint_add(s, contrainte_make(v), false);
	ifdebug(3) 
	  {	
	    fprintf(stderr, " \n After add constraint : ");
	    sc_fprint(stderr,s, (char * (*)(Variable)) entity_local_name);
	  }
	retour = sc_integer_feasibility_ofl_ctrl(s, OFL_CTRL,true);
	ifdebug(2) 
	  fprintf(stderr, " Retour: %d", retour); 	
	sc_rm(s);
      
      /* if retour = true : the system is feasible or there are overflows
	 (our goal is to verify if retour is false or not)
	 if there are overflows (the value of nofoverflows will be changed), 
	 we simplify the system and recalcul the feasibility*/
	
      if ((retour == true) && 
	  (nofoverflows < linear_number_of_exception_thrown))
	{
	  /* there has been an overflow, let us try something else... */
	  Psysteme new_s = sc_dup(prec);
	  ifdebug(3) 
	    {	  
	      fprintf(stderr, "\n Slow check of feasibility with overflow: ");    
	      fprintf(stderr, " \n with the new Preconditions : ");
	      sc_fprint(stderr,new_s, (char * (*)(Variable)) entity_local_name);
	    }
	  new_s = simplify_big_coeff(new_s); 
	  sc_constraint_add(new_s, contrainte_make(v), false);
	  ifdebug(3) 
	    {	
	      fprintf(stderr, " \n After add constraint : ");
	      sc_fprint(stderr,new_s, (char * (*)(Variable)) entity_local_name);
	    }
	  retour = sc_integer_feasibility_ofl_ctrl(new_s, OFL_CTRL,true);
	  sc_rm(new_s);
	}
      break;
      }
    default: pips_internal_error("unexpected return...");
    }
  ifdebug(2) 	  
    fprintf(stderr, " Retour: %d", retour); 	
  return retour;
}

static expression 
partial_redundancy_elimination_expression(expression e, Psysteme prec)
{
  if (relational_expression_p(e) || logical_operator_expression_p(e))
    { 
      /* Treat all possible cases :
	 e1 < e2,  e1 <= e2, e1 >= e2, e1 > e2, e1.EQ.e2, e1.NE.e2
	 .NOT.e1, e1.AND.e2, e1.OR.e2, e1.EQV.e2,  e1.NEQV.e2, */
      list args = call_arguments(syntax_call(expression_syntax(e)));
      entity op = call_function(syntax_call(expression_syntax(e)));
      expression e1 = EXPRESSION(CAR(args));
      expression e2 = expression_undefined;
      if (!ENDP(CDR(args)))
	/* if op = .NOT. => there is only one argument e1*/
	e2 = EXPRESSION(CAR(CDR(args)));
      ifdebug(2) 
	{	  
	  fprintf(stderr, " Logical expression to be simplified:"); 
	  print_expression(e);
	  fprintf(stderr, " \n with precondition : ");
	  sc_fprint(stderr,prec, (char * (*)(Variable)) entity_local_name);
	}
      if (relational_expression_p(e))
	{
	  /* Fast check : check if e is trivial true or FALSE
	   * Slow check : see if e is true or false wrt the precondition prec
	   * if (e + prec = infeasible) => e = false 
	   * if (NOT(e) + prec = infeasible) => e = TRUE
	   * if not => no conclusion, return e itself
	   *
	   * ATTENTION : to calcul the feasibility or not of a system of contrainsts
	   * this function uses the function: 
	   *  sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res) in which: 
	   * 
	   * ofl_ctrl = OFL_CTRL means that the overflows are treated in the 
	   * called procedure (sc_rational_feasibility_ofl_ctrl())
	   *
	   * ofl_res = true means that if the overflows occur, function 
	   * sc_rational_feasibility_ofl_ctrl will return the value TRUE
	   * we have no conclusion : retour = copy_expression (e)
	   *
	   * The function sc_rational_feasibility_ofl_ctrl() is less expensive 
	   * than the function sc_integer_feasibility_ofl_ctrl() 
	   *
	   * 4 December 2000 :  I (Nga Nguyen) try to replace sc_rational 
	   * by sc_integer ===> re-measure the speed
	   * But maybe sc_integer is suitable for PRE because our goal is to 
	   * reduce the number of  array bound check as much as possible ?
	   */	 
	  normalized n1 = NORMALIZE_EXPRESSION(e1);
	  normalized n2 = NORMALIZE_EXPRESSION(e2);	 
	  if (normalized_linear_p(n1) && normalized_linear_p(n2))
	    {
	      Pvecteur v1 = normalized_linear(n1);
	      Pvecteur v2 = normalized_linear(n2);	      
	      Pvecteur v = vect_substract(v1,v2);	
	      Pvecteur v_one = vect_new(TCST,1);
	      ifdebug(2) 
		{	  
		  fprintf(stderr, "\n Relational expression and linear normalizations: ");    
		  print_expression(e);	
		}
	      /* If there exists a variable belonging to e but not to Prec 
		 => we can not simplify e => stop*/
	      if (all_variables_in_precondition(v,prec))
		{
		  ifdebug(2) 
		    {	  
		      fprintf(stderr, "\n All variables are in the base of the precondition: ");   
		      vect_fprint(stderr,v, (char * (*)(Variable)) entity_local_name);
		    }
		  /* The normal form of a vecteur is :
		   *	 a x + b <= 0 (inegalite)
		   *     a x + b == 0 (egalite)
		   * So we have to transform an expression to the normal form, 
		   * depending on the operator of the expression (op= {<=,<,>=,>,==,!=})
		   * 
		   * Before checking the feasibility of the 2 following cases, 
		   * we do a fast and trivial check if the vector is constant.
		   * + Normal form of the expression e
		   * + Normal form of the negation expression of the  expression e  */
		  if (ENTITY_NON_EQUAL_P(op))
		    {
		      /* Initial expression :  v != 0 
		       * Form +             :  v + 1 <= 0 || -v + 1 <= 0
		       * Negation           :  v == 0
		       * Form -             :  v == 0 */ 
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_zero_p(val_of(v))) return make_false_expression();
			  if (value_notzero_p(val_of(v))) return make_true_expression();	
			}
		      else 
			{
			  Psysteme prec_dup = sc_dup(prec);
			  sc_constraint_add(prec_dup,contrainte_make(v),true);
			  if (!sc_integer_feasibility_ofl_ctrl(prec_dup, OFL_CTRL,true))
			    {
			      /* Not e + prec = infeasible => e = TRUE*/
			      sc_rm(prec_dup);
			      return make_true_expression();
			    }
			  else 
			    {
			      Pvecteur v_1 =  vect_add(v,v_one);      
			      Pvecteur v_temp = vect_multiply(vect_dup(v),-1);
			      Pvecteur v_2 = vect_add(v_temp,v_one);
			      sc_rm(prec_dup);
			      if (!efficient_sc_check_inequality_feasibility(v_1, prec) &&
				  !efficient_sc_check_inequality_feasibility(v_2, prec))
				/* e + prec = infeasible => e = FALSE*/
				return make_false_expression();
			    } 
			}
		    }
		  if (ENTITY_EQUAL_P(op))
		    {
		      /* Initial expression :  v == 0 
		       * Form +             :  v == 0
		       * Negation           :  v != 0
		       * Form -             :  v + 1 <= 0 || -v + 1 <= 0*/	
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_zero_p(val_of(v))) return make_true_expression();
			  if (value_notzero_p(val_of(v))) return make_false_expression();
			}	
		      else
			{
			  Psysteme prec_dup = sc_dup(prec);
			  sc_constraint_add(prec_dup,contrainte_make(v),true);
			  /* for union5.f, sc_rational_feasibility can not eliminate the second test 
			   * it works with sc_integer_feasibility */
			  if (!sc_integer_feasibility_ofl_ctrl(prec_dup, OFL_CTRL,true))
			    {
			      /* e + prec = infeasible => e = FALSE*/
			      sc_rm(prec_dup);
			      return make_false_expression();
			    }
			  else
			    {
			      Pvecteur v_1 =  vect_add(v,v_one);      
			      Pvecteur v_temp = vect_multiply(vect_dup(v),-1);
			      Pvecteur v_2 = vect_add(v_temp,v_one);
			      sc_rm(prec_dup);
			      if (!efficient_sc_check_inequality_feasibility(v_1,prec) && 
				  !efficient_sc_check_inequality_feasibility(v_2,prec))
				/* Not e + prec = infeasible => e = TRUE*/
				return make_true_expression();
			    }
			}
		    }
		  if (ENTITY_GREATER_OR_EQUAL_P(op))
		    {
		      /* Initial expression :  v >= 0 
		       * Form +             :  -v <= 0
		       * Negation           :  v <= -1
		       * Form -             :  v + 1 <= 0*/
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_posz_p(val_of(v)) ) return make_true_expression();
			  if (value_neg_p(val_of(v))) return make_false_expression();	
			}	
		      else
			{
			  Pvecteur v_1 = vect_multiply(vect_dup(v),-1);
			  Pvecteur v_2 = vect_add(v,v_one);
			  if (!efficient_sc_check_inequality_feasibility(v_1, prec)) 
			    /* e + prec = infeasible => e = FALSE*/
			    return make_false_expression();	
			  if (!efficient_sc_check_inequality_feasibility(v_2, prec))
			    /* Not e + prec = infeasible => e = TRUE*/
			    return make_true_expression();
			}
		    }
		  if (ENTITY_LESS_OR_EQUAL_P(op))
		    {
		      /* Initial expression :  v <= 0
		       * Form +             :  v <= 0
		       * Negation           :  v >= 1
		       * Form -             :  -v + 1 <= 0*/
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_negz_p(val_of(v))) return make_true_expression();
			  if (value_pos_p(val_of(v))) return make_false_expression();	
			}	
		      else
			{
			  Pvecteur v_1 = vect_dup(v);
			  Pvecteur v_temp = vect_multiply(vect_dup(v),-1);
			  Pvecteur v_2 = vect_add(v_temp,v_one);
			  if (!efficient_sc_check_inequality_feasibility(v_1, prec)) 
			    /* e + prec = infeasible => e = FALSE*/
			    return make_false_expression();	
			  if (!efficient_sc_check_inequality_feasibility(v_2, prec))
			    /* Not e + prec = infeasible => e = TRUE*/
			    return make_true_expression();
			}
		    }
		  if (ENTITY_LESS_THAN_P(op))
		    {
		      /* Initial expression :  v < 0 
		       * Form +             :  v +1 <= 0
		       * Negation           :  v >= 0
		       * Form -             :  -v <= 0*/
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_posz_p(val_of(v)) ) return make_false_expression();
			  if (value_neg_p(val_of(v))) return make_true_expression();  
			}	
		      else
			{
			  Pvecteur v_1 = vect_add(v,v_one);
			  Pvecteur v_2 = vect_multiply(vect_dup(v),-1);
			  if (!efficient_sc_check_inequality_feasibility(v_1, prec)) 
			    /* e + prec = infeasible => e = FALSE*/
			    return make_false_expression();	
			  if (!efficient_sc_check_inequality_feasibility(v_2, prec))
			    /* Not e + prec = infeasible => e = TRUE*/
			    return make_true_expression();
			}
		    }
		  if (ENTITY_GREATER_THAN_P(op))
		    {
		      /* Initial expression :  v > 0 
		       * Form +             :  -v + 1 <= 0
		       * Negation           :  v <= 0
		       * Form -             :  v <= 0*/
		      if (vect_constant_p(v))
			{
			  if (VECTEUR_NUL_P(v) || value_negz_p(val_of(v)) ) return make_false_expression();
			  if (value_pos_p(val_of(v))) return make_true_expression();	
			}	
		      else
			{
			  Pvecteur v_temp = vect_multiply(vect_dup(v),-1);
			  Pvecteur v_1 = vect_add(v_temp,v_one);
			  Pvecteur v_2 = vect_dup(v);
			  if (!efficient_sc_check_inequality_feasibility(v_1, prec)) 
			    /* e + prec = infeasible => e = FALSE*/
			    return make_false_expression();	
			  if (!efficient_sc_check_inequality_feasibility(v_2, prec))
			    /* Not e + prec = infeasible => e = TRUE*/
			    return make_true_expression();
			}
		    }
		}
	    }
	}
      else if (logical_operator_expression_p(e))	
	{
	  if (ENTITY_NOT_P(op))
	    {
	      expression retour1 = partial_redundancy_elimination_expression(e1,prec);
	      if (true_expression_p(retour1)) return make_false_expression();
	      if (false_expression_p(retour1)) return make_true_expression();
	      return not_expression(retour1); 
	    }
	  if (ENTITY_AND_P(op))
	    {
	      expression retour1 = partial_redundancy_elimination_expression(e1,prec);
	      expression retour2 = partial_redundancy_elimination_expression(e2,prec);
	      if (false_expression_p(retour1) || false_expression_p(retour2)) 
		return make_false_expression();
	      if (true_expression_p(retour1)) return retour2;
	      if (true_expression_p(retour2)) return retour1;
	      return and_expression(retour1,retour2);
	    }		
	  if (ENTITY_OR_P(op))
	    {
	      expression retour1= partial_redundancy_elimination_expression(e1,prec);
	      expression retour2= partial_redundancy_elimination_expression(e2,prec);
	      ifdebug(2) 
		{	  
		  fprintf(stderr, " Simplified OR expression: retour1 + retour2"); 
		  print_expression(retour1);
		  print_expression(retour2);			
		}
	      if (true_expression_p(retour1) || true_expression_p(retour2)) 
		return make_true_expression();
	      if (false_expression_p(retour1)) return retour2;
	      if (false_expression_p(retour2)) return retour1;
	      return or_expression(retour1,retour2);
	    }
	  if (ENTITY_EQUIV_P(op))
	    {
	      expression retour1 = partial_redundancy_elimination_expression(e1,prec);
	      expression retour2 = partial_redundancy_elimination_expression(e2,prec);
	      if ((true_expression_p(retour1) && true_expression_p(retour2)) ||
		  (false_expression_p(retour1) &&  false_expression_p(retour2)) ) 
		return make_true_expression();
	      if ((true_expression_p(retour1) && false_expression_p(retour2)) ||
		  (false_expression_p(retour1) && true_expression_p(retour2)) ) 
		return make_false_expression();
	      return binary_intrinsic_expression(EQUIV_OPERATOR_NAME, retour1, retour2);
	    }	   
	  if (ENTITY_NON_EQUIV_P(op))
	    {
	      expression retour1 = partial_redundancy_elimination_expression(e1,prec);
	      expression retour2 = partial_redundancy_elimination_expression(e2,prec);
	      if ((true_expression_p(retour1) && true_expression_p(retour2)) ||
		  (false_expression_p(retour1) && false_expression_p(retour2)) ) 
		return make_false_expression();
	      if ((true_expression_p(retour1) && false_expression_p(retour2)) ||
		  (false_expression_p(retour1) && true_expression_p(retour2)) ) 
		return make_true_expression();
	      return binary_intrinsic_expression(NON_EQUIV_OPERATOR_NAME, retour1, retour2); 
	    }
	}
    }
  /* return e itself if the expression e can not be simplified */
  return copy_expression(e);
}

static void 
partial_redundancy_elimination_rwt(statement s, 
				   persistant_statement_to_control map)
{
  Psysteme prec = stmt_prec(s);
  if (!sc_empty_p(prec) && !sc_rn_p(prec))
    {
      /* Else :  P = False (dead code) => all tests in s are redundant
	 or P = True, we can not simplify anything */
      instruction i = statement_instruction(s);
      tag t = instruction_tag(i);
      switch(t)
	{
	case is_instruction_call:
	  {
	    call c = instruction_call(i);
	    entity func = call_function(c);
	    if (strcmp(entity_local_name(func),ASSIGN_OPERATOR_NAME)==0)
	      {  
		list args = call_arguments(c);
		expression e = EXPRESSION(CAR(CDR(args)));
		if (logical_expression_p(e))
		  {
		    expression retour = partial_redundancy_elimination_expression(e,prec);
		    ifdebug(3) 
		      {	  
			fprintf(stderr, " Assign statement with logical expression:"); 
			print_statement(s);
			fprintf(stderr, " \n with non empty / not true precondition : ");
			sc_fprint(stderr,prec, (char * (*)(Variable)) entity_local_name);
		      }
		    if (!expression_equal_p(retour,e))
		      { 
			/* e is simplified, replace e by retour in the call c*/	
			instruction new = make_assign_instruction(copy_expression(EXPRESSION(CAR(args))),
								  copy_expression(retour));
			free_instruction(statement_instruction(s));
			statement_instruction(s) = copy_instruction(new);
			number_of_simplified_assign_expressions++;
		      }
		  }
	      }
	    break;
	  }
	case is_instruction_whileloop:
	  {
	    whileloop wl = instruction_whileloop(i); 
	    expression e = whileloop_condition(wl);
	    expression retour = partial_redundancy_elimination_expression(e,prec);
	    ifdebug(3) 
	      {	  
		fprintf(stderr, " Whileloop statement:"); 
		print_statement(s);
		fprintf(stderr, " \n with non empty / not true precondition : ");
		sc_fprint(stderr,prec, (char * (*)(Variable)) entity_local_name);
	      }
	    /* Try to simplify the whileloop's condition
	       if (retour=.FALSE.) then 
	          eliminate this while loop (a trivial case of dead code elimination)  
	       else keep the loop */
	    if (false_expression_p(retour))
	      {
		number_of_false_while_conditions++;
		//	free_instruction(statement_instruction(s));
		statement_instruction(s) = make_instruction_block(NIL);
		fix_sequence_statement_attributes(s);
	      }
	    break;
	  }
	case is_instruction_test:
	  {
	    test it = instruction_test(i);
	    expression e = test_condition(it);
	    expression retour = partial_redundancy_elimination_expression(e,prec);
	    ifdebug(3) 
	      {	  
		fprintf(stderr, " Test statement:"); 
		print_statement(s);
		fprintf(stderr, " Simplified expression:"); 
		print_expression(retour);
		fprintf(stderr, " \n with non empty / not true precondition : ");
		sc_fprint(stderr,prec, (char * (*)(Variable)) entity_local_name);
	      }
	    if (!expression_equal_p(retour,e))
	      {
		number_of_simplified_if_conditions++;
		/* the test's condition e is simplified,
		   + if (retour=.FALSE.) then delete the test and its true branch 
		   + if (retour=.TRUE.) then delete the test and its false branch  
		   (a trivial case of dead code elimination) 
		   + otherwise, replace e by retour.
		   
		   Attention : in the unstructured case, the test it has 
		   2 successors, we need to remove the dead control link for the 
		   first two cases (true branch or false branch is dead) */
		
		if (false_expression_p(retour))
		  {
		    // the true branch is dead
		    number_of_false_if_conditions++;
		    if (bound_persistant_statement_to_control_p(map, s))
		      { 
			// unstructured case   
			control c = apply_persistant_statement_to_control(map,s);
			if (!ENDP(CDR(control_successors(c))))
			  {
			    control true_control = CONTROL(CAR(control_successors(c)));
			    control false_control = CONTROL(CAR(CDR(control_successors(c))));
				// remove the link to the THEN control node
			    gen_remove(&control_successors(c),true_control);
			    gen_remove(&control_predecessors(true_control),c);
				// replace c by  false_control as successor of each predecessor of c 
			    MAP(CONTROL, co,
			    {
			      MAPL(lc, 
			      {
				if (CONTROL(CAR(lc))==c) 
				  CONTROL_(CAR(lc)) = false_control;
			      }, control_successors(co));
			    },control_predecessors(c));
			  }
		      }
		    else 
		      {
			//structured case
			instruction new = statement_instruction(copy_statement(test_false(it)));
			update_statement_instruction(s,new);	
		      }
		  }
		else 
		  if (true_expression_p(retour))
		    {
		      // the false branch is dead
		      number_of_true_if_conditions++;
		      if (bound_persistant_statement_to_control_p(map, s))
			{ 
			  // unstructured case   
			  control c = apply_persistant_statement_to_control(map,s);
			  if (!ENDP(CDR(control_successors(c))))
			    {
			      control false_control = CONTROL(CAR(CDR(control_successors(c))));
			      control true_control = CONTROL(CAR(control_successors(c)));
			      // remove the link to the ELSE control node
			      gen_remove(&control_successors(c),false_control);
			      gen_remove(&control_predecessors(false_control),c);
			      // replace c by true_control as successor of each predecessor of c 
			      MAP(CONTROL, co,
			      {
				MAPL(lc, 
				{
				  if (CONTROL(CAR(lc))==c) 
				    CONTROL_(CAR(lc)) = true_control;
				}, control_successors(co));
			      },control_predecessors(c));   
			    }
			}
		      else 
			{
			  //structured case 
			  instruction new = statement_instruction(copy_statement(test_true(it)));
			  update_statement_instruction(s,new);
			}	
		    }
		  else 
		    {		                            
		      test  new = make_test(copy_expression(retour),
					    copy_statement(test_true(it)),
					    copy_statement(test_false(it)));
		      free_instruction(statement_instruction(s));
		      statement_instruction(s) = make_instruction(is_instruction_test,new);
		    }
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
}

static bool store_mapping(control c, persistant_statement_to_control map)
{
  extend_persistant_statement_to_control(map,control_statement(c),c);
  return true;
}

static void 
partial_redundancy_elimination_statement(statement module_statement)
{  
  persistant_statement_to_control map;  
  map = make_persistant_statement_to_control();  
  gen_context_multi_recurse(module_statement, map,			   
			    control_domain, store_mapping, gen_null,
			    statement_domain, gen_true, partial_redundancy_elimination_rwt,
			    NULL);  
  free_persistant_statement_to_control(map);
}


/* This transformation may be used to reduce the number of logical
   expressions generated by array_bound_check.

   Logical expressions such as the condition in: IF (1.LE.I .AND.I.LE.N)
   THEN ...  are simplified if {1>I } can be proved false wrt the
   precondition. So the simplified test is: IF (1.LE.N) THEN ....

   Logical assignment statement such as: FLAG =
   (A+B).GT.C.AND.(B+C).GT.A.AND.(C+A).GT.B where we have (C+A).LE.B will
   be FLAG = .FALSE.

   If test conditions are simplified to true or false, the test statement
   is replaced by the true or the false branch right away to avoid a
   re-computation of transformers and preconditions. FORMAT statements in
   the eliminated branch are preserved by moving them in the remaining
   statement. Some FORMAT statements may become useless but this is not
   tested.

   If a WHILE condition is simplified to false, the WHILE is eliminated.

   @param[in] module_name is the name of the module we want to apply
   partial_redundancy_elimination on

   @return true since we are confident that everything goes fine
 */
bool partial_redundancy_elimination(const char* module_name)
{
  statement module_statement;
  set_current_module_entity(module_name_to_entity(module_name));
  module_statement= (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_current_module_statement(module_statement);

  set_ordering_to_statement(module_statement);
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,
					      module_name,
					      true));
  debug_on("PARTIAL_REDUNDANCY_ELIMINATION_DEBUG_LEVEL");
  ifdebug(1){
    debug(1, "Partial redundancy elimination for logical expressions","Begin for %s\n", module_name);
    pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
  }
  initialize_partial_redundancy_elimination_statistics();
  partial_redundancy_elimination_statement(module_statement);
  display_partial_redundancy_elimination_statistics();
  module_reorder(module_statement);
  ifdebug(1){
    pips_assert("Statement is consistent ...", statement_consistent_p(module_statement));
    debug(1, "Partial redundancy elimination","End for %s\n", module_name);
  }
  debug_off();
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,module_statement);
  reset_ordering_to_statement();
  reset_precondition_map();
  reset_current_module_statement();
  reset_current_module_entity();
  return true;
}
