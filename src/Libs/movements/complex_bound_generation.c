/*
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */


#include <stdio.h>
#include "genC.h"
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
extern Variable variable_of_rank();


/* Print the complex expression 
 * [coeff1 * (exp1 / coeff2) + exp2 ] / coeff3
 * 
 * where exp(s) are Pvecteur(s) corresponding to parts of complex bound 
 */

expression complex_bound_generation(coeff1,coeff2,coeff3,exp1,var1,exp2,var2)
int coeff1,coeff2,coeff3;
Pvecteur exp1;
Variable var1;
Pvecteur exp2;
Variable var2;
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
    ex1 = make_integer_constant_expression(ABS(coeff1));
    ex2 = make_integer_constant_expression(ABS(coeff2));
    if (coeff3 !=1)
	ex3 = make_integer_constant_expression(ABS(coeff3));
    lex2 = CONS(EXPRESSION,ex2,NIL);
    ex4 = make_div_expression(expr1,lex2);
    lex2 = CONS(EXPRESSION,ex4,NIL);

    ex5 = make_op_expression(operateur_multi
			     ,CONS(EXPRESSION,ex1,lex2));
    lex2 = CONS(EXPRESSION,expr2,NIL);

    expr = ex6 = make_op_expression(operateur_add,
				    CONS(EXPRESSION,ex5,lex2));
    if (coeff3 !=1) {
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



expression complex_bound_computation(sc,index_base,ineq1,ineq2,rank)
Psysteme sc;
Pbase index_base;
Pcontrainte ineq1;
Pcontrainte ineq2;
int rank;
{

    Variable right_var,left_var;
    int right_coeff,right_rank,left_coeff,left_rank;
    int coeff_l=0;
    int coeff_r=0;
    Variable el_var= variable_of_rank(index_base,rank);
    Pvecteur right_exp,left_exp;
    expression expr;
    int coeff=0;
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

    if ((coeff = vect_coeff(el_var,ineq1->vecteur))>0) {
	right_exp=vect_dup(ineq1->vecteur);
	left_exp=vect_dup(ineq2->vecteur);  }
    else { 
	if ((coeff = vect_coeff(el_var,ineq2->vecteur))>0) {
	    right_exp=vect_dup(ineq2->vecteur);
	    left_exp=vect_dup(ineq1->vecteur);
	} 
    }
    if (coeff>0) {
	coeff_r = vect_coeff(el_var,right_exp);
	coeff_l= vect_coeff(el_var,left_exp);
    }
    vect_chg_coeff(&right_exp,el_var,0);
    vect_chg_coeff(&left_exp,el_var,0);

    if (left_rank > right_rank) {
	/* computation of the bound expression of the variable of 
	   higher rank, after el_var, when this variable 
	   belongs to left_exp */
  
	vect_chg_coeff(&left_exp,left_var,0);
	sign = (left_coeff >0) ? 1 :-1;
	if (sign) {
	    vect_chg_sgn(left_exp);
	    vect_chg_sgn(right_exp);
	}
	else
	    vect_add_elem(&left_exp,TCST,-left_coeff -1);
	expr = complex_bound_generation(-coeff_l,coeff_r,sign * left_coeff,
					right_exp,el_var,left_exp,el_var);
    }
    else {			
	/* computation of the bound expression of the variable of 
	   higher rank, after el_var, when this variable 
	   belongs to right_exp */
	vect_add_elem(&left_exp,TCST,-coeff_l -1);
	vect_chg_coeff(&right_exp,right_var,0);
	sign = (right_coeff >0) ? 1 :-1;
	if (sign)
	    vect_chg_sgn(right_exp);
	else
	    vect_add_elem(&right_exp,TCST,-right_coeff -1);
	expr = complex_bound_generation(-1 * sign* coeff_r,-coeff_l, 
					sign *right_coeff,
					left_exp,el_var,right_exp,el_var);
    }
 
    debug(8,"complex_bound_computation","end\n");
    debug_off();
    return (expr);
}




