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
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */


#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "misc.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "matrix.h"
#include "sparse_sc.h"
#include "tiling.h"
#include "movements.h"

/* Print the complex expression 
 * [coeff1 * (exp1 / coeff2) + exp2 ] / coeff3
 * 
 * where exp(s) are Pvecteur(s) corresponding to parts of complex bound 
 */

expression
complex_bound_generation(Value coeff1,
			 Value coeff2,
			 Value coeff3,
			 Pvecteur exp1,
			 Variable __attribute__ ((unused)) var1,
			 Pvecteur exp2,
			 Variable __attribute__ ((unused)) var2)
{
    expression ex1,ex2,ex4,ex5,ex6,expr,expr1,expr2;
    expression ex3=expression_undefined;
    entity operateur_multi = gen_find_tabulated("TOP-LEVEL:*",entity_domain);
    entity operateur_add = gen_find_tabulated(
					      make_entity_fullname(TOP_LEVEL_MODULE_NAME,
								   PLUS_OPERATOR_NAME), 
					      entity_domain);
    cons * lex2;
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"complex_bound_generation","begin\n");
    expr1 = make_vecteur_expression(exp1);
    expr2 = make_vecteur_expression(exp2);
    ex1 = int_to_expression(
	VALUE_TO_INT(value_abs(coeff1)));
    ex2 = int_to_expression(
	VALUE_TO_INT(value_abs(coeff2)));
    if (value_notone_p(coeff3))
	ex3 = int_to_expression(
	    VALUE_TO_INT(value_abs(coeff3)));
    lex2 = CONS(EXPRESSION,ex2,NIL);
    ex4 = make_div_expression(expr1,lex2);
    lex2 = CONS(EXPRESSION,ex4,NIL);

    ex5 = make_op_expression(operateur_multi
			     ,CONS(EXPRESSION,ex1,lex2));
    lex2 = CONS(EXPRESSION,expr2,NIL);

    expr = ex6 = make_op_expression(operateur_add,
				    CONS(EXPRESSION,ex5,lex2));
    if (value_notone_p(coeff3)) {
	lex2 = CONS(EXPRESSION,ex3,NIL);
	expr= make_div_expression(ex6,lex2);	
    }

    debug(8,"complex_bound_generation","end\n");
    debug_off();
    return(expr);
}
/* Compute the complex bounds associated to the variable of higher rank, 
 * after the variable "el_var" of rank "rank".
 * 
 * variable "el_var" is eliminated from the two inequations ineq1 and ineq2.
 * Bound expression of the higher variable belonging to ineq1 or ineq2 is 
 * then computed.
*/  



expression
complex_bound_computation(Psysteme __attribute__ ((unused)) sc,
			  Pbase index_base,
			  Pcontrainte ineq1,
			  Pcontrainte ineq2,
			  int rank) {

    Variable right_var,left_var;
    Value right_coeff,left_coeff;
    int right_rank,left_rank;
    Value coeff_l=VALUE_ZERO, coeff_r=VALUE_ZERO;
    Variable el_var= variable_of_rank(index_base,rank);
    Pvecteur right_exp,left_exp;
    expression expr;
    Value coeff=VALUE_ZERO;
    int sign =0;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"complex_bound_computation","begin\n");

    constraint_integer_combination(index_base,ineq1,ineq2,rank,
				   &right_var,&right_rank,&right_coeff,
				   &left_var,&left_rank,&left_coeff);

    /* the right_exp is assigned to the inequation where the 
       coefficient of the variable "el_var" is positive. 
       the left_exp is assigned to the inequation where the 
       coefficient of the variable "el_var" is negative.
       */

    if (value_pos_p(coeff = vect_coeff(el_var,ineq1->vecteur))) {
	right_exp=vect_dup(ineq1->vecteur);
	left_exp=vect_dup(ineq2->vecteur);  }
    else { 
	if (value_pos_p(coeff = vect_coeff(el_var,ineq2->vecteur))) {
	    right_exp=vect_dup(ineq2->vecteur);
	    left_exp=vect_dup(ineq1->vecteur);
	} 
    }
    if (value_pos_p(coeff)) {
	coeff_r = vect_coeff(el_var,right_exp);
	coeff_l= vect_coeff(el_var,left_exp);
    }
    vect_chg_coeff(&right_exp,el_var,VALUE_ZERO);
    vect_chg_coeff(&left_exp,el_var,VALUE_ZERO);

    if (left_rank > right_rank) {
	/* computation of the bound expression of the variable of 
	   higher rank, after el_var, when this variable 
	   belongs to left_exp */
  
	vect_chg_coeff(&left_exp,left_var,VALUE_ZERO);
	sign = value_sign(left_coeff);
	if (sign==1) { /* ??? bug? was sign */
	    vect_chg_sgn(left_exp);
	    vect_chg_sgn(right_exp);
	}
	else
	    vect_add_elem(&left_exp,TCST,
			  value_uminus(value_plus(left_coeff,VALUE_ONE)));
	expr = complex_bound_generation(
	    value_uminus(coeff_l),coeff_r,
	    value_abs(left_coeff),
	    right_exp,el_var,left_exp,el_var);
    }
    else {			
	/* computation of the bound expression of the variable of 
	   higher rank, after el_var, when this variable 
	   belongs to right_exp */
	vect_add_elem(&left_exp,TCST,
		      value_uminus(value_plus(coeff_l,VALUE_ONE)));
	vect_chg_coeff(&right_exp,right_var,VALUE_ZERO);
	sign = value_sign(right_coeff);
	if (sign==1) /* ??? was sign */
	    vect_chg_sgn(right_exp);
	else
	    vect_add_elem(&right_exp,TCST,
			  value_uminus(value_plus(right_coeff,VALUE_ONE)));
	expr = complex_bound_generation(
	    value_mult(int_to_value(-sign), coeff_r),
	    value_uminus(coeff_l), 
	    value_mult(sign,right_coeff),
	    left_exp,el_var,right_exp,el_var);
    }
 
    debug(8,"complex_bound_computation","end\n");
    debug_off();
    return (expr);
}




