/*
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */


#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "matrice.h"
#include "sparse_sc.h"
#include "tiling.h"
#include "movements.h"
#include "text-util.h"
#include "polyedre.h"
#include "dg.h"

static char *noms_var(entity i)
{
    return(local_name(entity_name(i)));
}


/* This fonction generates the lower bounds of the "loop_rank"-th loop.
*/

expression 
lower_bound_generation(sc_neg,index_base,number_of_lower_bounds,loop_rank)
Psysteme sc_neg;
Pbase index_base;
int number_of_lower_bounds,loop_rank;
{
    Pcontrainte ineq;
    Variable var = variable_of_rank(index_base,loop_rank); 
    Pvecteur pv2 = VECTEUR_NUL; 
    expression expr,ex1,ex2, lower=expression_undefined ;
    cons * lex2,* lexpr = NIL;
    entity max = local_name_to_top_level_entity("MAX");
    int higher_rank,nlb = 0; 
    Value coeff;
    boolean reductible = FALSE;
 
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"lower_bound_generation","begin\n");

    if (number_of_lower_bounds>=1) {
	for (ineq = sc_neg->inegalites; 
	     !CONTRAINTE_UNDEFINED_P(ineq); 
	     ineq=ineq->succ) {
	    Variable var2;
	    Value coeff2,coeff3;
	    reductible = FALSE;
	    higher_rank=search_higher_rank(ineq->vecteur,index_base);
	    if (higher_rank > loop_rank ) {
		Pcontrainte pc2 = ineq->succ;
		var2 = variable_of_rank(index_base,higher_rank);
		coeff2 = vect_coeff(var2,ineq->vecteur);
		coeff3 = vect_coeff(var2,pc2->vecteur);
		
		if (value_one_p(value_abs(coeff2)) || 
		    value_one_p(value_abs(coeff3))) {
		    reductible = TRUE;
		    pv2 = vect_cl2(value_abs(coeff2),pc2->vecteur,
				   value_abs(coeff3),ineq->vecteur);
		    ineq= ineq->succ;
		}
	    }
	    else pv2 = vect_dup(ineq->vecteur);
	    if (higher_rank <= loop_rank || reductible ) {  
		/* this condition is TRUE if the constraint constrains 
		   directly the variable */
		if (value_notmone_p(coeff= vect_coeff(var,pv2)))
		    vect_add_elem(&pv2, TCST,
				  value_uminus(value_plus(coeff, VALUE_ONE)));
		vect_chg_coeff(&pv2,var,VALUE_ZERO);
		expr = ex1 = make_vecteur_expression(pv2);
    		if (value_notmone_p(coeff)) {
		    ex2 = make_integer_constant_expression(
			VALUE_TO_INT(value_abs(coeff)));
		    lex2 =CONS(EXPRESSION,ex2,NIL);
		    expr=make_div_expression(ex1,lex2);
		    vect_add_elem(&pv2,TCST,value_plus(coeff,VALUE_ONE));
		}
	    }
	    else {				
		/* In that case the bound expression results from 
		   the  combination of two constraints. The variable of 
		   rank "higher_rank" will be eliminated from these two 
		   constraints in order to give only one bound for 
		   the "loop_rank" index. */
		expr = complex_bound_computation(sc_neg,index_base,
						 ineq,ineq->succ,
						 higher_rank);
		ineq = ineq->succ;

	    }
	    lexpr = CONS(EXPRESSION, expr,lexpr);
	    nlb ++;
	}
	if (nlb > 1)
	    lower = make_op_expression(max,lexpr);
	else lower = EXPRESSION(CAR(lexpr));
    }
    debug(8,"lower_bound_generation","end\n");
    debug_off();

    return(lower);
}



/* This fonction generates the upper bounds of the "loop_rank"-th loop.
*/


expression 
upper_bound_generation(sc_pos,index_base,number_of_upper_bounds,loop_rank)
Psysteme sc_pos;
Pbase index_base;
int number_of_upper_bounds, loop_rank;
{
    Pcontrainte ineq; 
    Pvecteur pv2=VECTEUR_NUL;
    int higher_rank, nub = 0;
    Value coeff;
    Variable var = variable_of_rank(index_base,loop_rank);
    expression expr, ex2, ex1, upper = expression_undefined;
    cons * lex2,* lexpr = NIL;
    entity min=local_name_to_top_level_entity("MIN");
    boolean reductible = FALSE;
  
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"upper_bound_generation","begin\n");


    if (number_of_upper_bounds) {  
	for (ineq = sc_pos->inegalites;
	     !CONTRAINTE_UNDEFINED_P(ineq); 
	     ineq=ineq->succ) {
	    Variable var2;
	    Value coeff2,coeff3;
	    reductible = FALSE;
	    higher_rank=search_higher_rank(ineq->vecteur,index_base);
	    if (higher_rank > loop_rank ) {
		var2 = variable_of_rank(index_base,higher_rank);
		coeff2 = vect_coeff(var2,ineq->vecteur);
		coeff3 = vect_coeff(var2,(ineq->succ)->vecteur);
		if (value_one_p(value_abs(coeff2)) || 
		    value_one_p(value_abs(coeff3))) {
		    reductible = TRUE;
		    pv2 = vect_cl2(value_abs(coeff2),(ineq->succ)->vecteur,
				   value_abs(coeff3),ineq->vecteur);

		    ineq= ineq->succ;
		}
	    }
	    else pv2 = vect_dup(ineq->vecteur);

	    if (higher_rank <=loop_rank || reductible) {
		/* this condition is TRUE if the constraint constrains 
		   directly the variable */
		coeff= vect_coeff(var,pv2);
		vect_chg_sgn(pv2);
		vect_chg_coeff(&pv2,var,VALUE_ZERO);
		expr = ex1 =make_vecteur_expression(pv2);
		vect_chg_sgn(pv2);
	      
		if (value_notone_p(coeff)) {
		    ex2 = make_integer_constant_expression(
			VALUE_TO_INT(value_abs(coeff)));
		    lex2 = CONS(EXPRESSION,ex2,NIL);
		    expr= make_div_expression(ex1,lex2);
		}
	    }
	    else {  
		/* In that case the bound expression results from 
		   the  combination of two constraints. The variable of 
		   rank "higher_rank" will be eliminated from these two 
		   constraints in order to give only one bound for 
		   the "loop_rank" index. */
		expr = complex_bound_computation(sc_pos,index_base,
						 ineq,ineq->succ,
						 higher_rank);
		ineq = ineq->succ;
	    }
	    lexpr = CONS(EXPRESSION, expr,lexpr);
	    nub ++;
	}
	if (nub > 1)	
	    upper = make_op_expression(min,lexpr);
	else upper = EXPRESSION(CAR(lexpr));
    }
    debug(8,"upper_bound_generation","end\n");
    debug_off();
    return (upper);

}


/* This function generates the expressions of the guard if it exists.
 * All the expressions of the guards are computed from the combination of 
 * two constraints where the variable of higher_rank is eliminated. 
*/

expression 
test_bound_generation(sc_test,index_base)
Psysteme sc_test;
Pbase index_base;
{

    Pcontrainte ineq,right_ineq,left_ineq;
    int rank;
    Value coeff;
    Variable var;
    boolean debut;
    expression expr,ex1,ex2,ex3,ex4;
    expression exl= expression_undefined;
    expression exr= expression_undefined;
    expression lexpr= expression_undefined;
  
    cons * lex2;
    entity inf =local_name_to_top_level_entity(LESS_OR_EQUAL_OPERATOR_NAME);
    entity an =local_name_to_top_level_entity(AND_OPERATOR_NAME);
   
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"test_bound_generation","begin\n");


    debut = TRUE;
    for (ineq = sc_test->inegalites; 
	 !CONTRAINTE_UNDEFINED_P(ineq); 
	 ineq=ineq->succ) {
	rank = search_higher_rank(ineq->vecteur,index_base);
	var = variable_of_rank(index_base,rank);

	if (value_pos_p(coeff= vect_coeff(var,ineq->vecteur))) {
	    right_ineq = contrainte_dup(ineq);
	    left_ineq = contrainte_dup(ineq->succ);
	} 
	else {
	    right_ineq = contrainte_dup(ineq->succ);
	    left_ineq= contrainte_dup(ineq);
	}

	/* generation of the left hand side of the guard */	    
	if (value_notmone_p(coeff= vect_coeff(var,left_ineq->vecteur)))
	    vect_add_elem(&(left_ineq->vecteur),TCST,
			  value_uminus(value_plus(coeff,VALUE_ONE)));
	
	vect_chg_coeff(&(left_ineq->vecteur),var,VALUE_ZERO);
	ex1 = make_vecteur_expression(left_ineq->vecteur);
	if (value_notmone_p(coeff)) {
	    ex2 = make_integer_constant_expression(
		VALUE_TO_INT(value_abs(coeff)));
	    lex2 = CONS(EXPRESSION,ex2,NIL);
	    exl = make_div_expression(ex1,lex2);	
	    vect_add_elem(&(left_ineq->vecteur),TCST,
			  value_plus(coeff,VALUE_ONE));
	}

	/* generation of the right hand side of the guard */	    

	coeff= vect_coeff(var,right_ineq->vecteur);
	vect_chg_sgn(right_ineq->vecteur);
	vect_chg_coeff(&(right_ineq->vecteur),var,VALUE_ZERO);
	ex3= make_vecteur_expression(right_ineq->vecteur);
	vect_chg_sgn(right_ineq->vecteur);
    
	if (value_notone_p(coeff)) {
	    ex4 = make_integer_constant_expression(
		VALUE_TO_INT(value_abs(coeff)));
	    lex2 =CONS(EXPRESSION,ex4,NIL);
	    exr = make_div_expression(ex3,lex2);	
	}
	lex2 = CONS(EXPRESSION,exr,NIL);

	/* generation of the inequality */
	expr =  make_op_expression(inf,CONS(EXPRESSION,exl,lex2));	

	if (debut) lexpr =expr;
	else {
	    lex2 = CONS(EXPRESSION,expr,NIL);
	    lexpr=make_op_expression(an,CONS(EXPRESSION,lexpr,lex2));
	}
	ineq = ineq->succ;
	debut = FALSE;
    }
    
    debug(8,"test_bound_generation","end\n");
    debug_off();	
    return(lexpr); 
}

/* Generation of the new loop nest characterizing the new domain.
 * The set of systems lsystem describes the set of constraints 
 * of each loop index. New loop bounds are deduced from these sytems.
 * 
 */
statement 
bound_generation(
    entity module,
    boolean bank_code,
    boolean receive_code,
    entity ent,
    Pbase loop_body_indices,
    Pbase var_id,
    Psysteme *lsystem,
    Pbase index_base,
    int n,
    int sc_info[][3])
{
 
    Psysteme ps,*sc_pos,*sc_neg,sc_test;
    Pvecteur pv;
    Variable var;
    int n0_loop,i,first_loop;
    int number_of_lower_bounds = 0;
    int number_of_upper_bounds =0;	
    expression lower_bound;
    expression upper_bound;
    expression test_bound;
    range looprange;
    statement stat, cs;
    statement loopbody =statement_undefined;
    test tst;
    entity looplabel;
    loop newloop;
    int space = (n+1) * sizeof(Ssysteme);
    boolean debut = TRUE;
    entity mod;
    int j;
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"bound_generation","begin\n");

    /* Initialisation des systemes */

    sc_neg = (Psysteme *) malloc((unsigned)(space));
    sc_pos = (Psysteme *) malloc((unsigned)(space));
    ps = lsystem[1];
    sc_test =  sc_init_with_sc(ps);

    bound_distribution(lsystem,index_base,sc_info,n,
		       sc_neg,sc_pos,sc_test);


    if (!CONTRAINTE_UNDEFINED_P(sc_test->inegalites))  {
	test_bound = test_bound_generation(sc_test,index_base);
	loopbody = make_datum_movement(module,receive_code,ent,
				       loop_body_indices,
				       var_id);
	tst =  make_test(test_bound,loopbody,
			 make_continue_statement(entity_empty_label()));
	stat = test_to_statement(tst);
	loopbody=make_block_statement(CONS(STATEMENT,stat,NIL));
	debut=FALSE;
    }
  
    ifdebug(8) {
	(void) fprintf(stderr, "base des indices \n");
	for (pv = index_base; pv!=NULL; pv=pv->succ)
	    fprintf (stderr,"%s,",entity_local_name((entity) pv->var));

	for (i=1;i<=n;i++) {
	    for (j=1; j<=3;j++) {
		printf ("%d,",sc_info[i][j]); }
	    printf ("\n");
	}
    }

    first_loop = (bank_code) ?  2 : 1;

    for (n0_loop = vect_size(index_base); n0_loop >= first_loop; n0_loop --) {
	var = variable_of_rank(index_base,n0_loop);

	if (sc_info[rank_of_variable(index_base,var)][1]) {
	    ps = lsystem[n0_loop];
	    ifdebug(8) {
		(void) fprintf(stderr,"LE SYSTEME  est :\n");
		(void) sc_fprint(stderr,ps,noms_var);
	    }
	    number_of_lower_bounds = 0;
	    number_of_upper_bounds = 0;
	    if (debut) {
		loopbody = make_movements_loop_body_wp65(module,
							 receive_code,ent,
							 loop_body_indices,
							 var_id,
							 sc_neg[n0_loop],
							 sc_pos[n0_loop],
							 index_base,n0_loop,
							 sc_info[n0_loop][3],
							 sc_info[n0_loop][2]);
		debut = FALSE;
	    }
	    else {
		/* make new bound expression and new range loop */

		number_of_upper_bounds=sc_info[n0_loop][2];
		number_of_lower_bounds=sc_info[n0_loop][3];
		lower_bound = lower_bound_generation(sc_neg[n0_loop],
						     index_base,
						     number_of_lower_bounds,
						     n0_loop);

		upper_bound = upper_bound_generation(sc_pos[n0_loop],
						     index_base,
						     number_of_upper_bounds,
						     n0_loop);

		looprange = make_range(lower_bound,upper_bound,
				       make_integer_constant_expression(1));

		/* looplabel = make_new_label(initial_module_name); */
		/* new code by FI to add continue statements starts here */
		looplabel = make_loop_label(9000, entity_local_name(module));
		cs = make_continue_statement(looplabel);
		if(instruction_block_p(statement_instruction(loopbody)))
		    (void) gen_nconc(instruction_block(statement_instruction(loopbody)),
				     CONS(STATEMENT, cs, NIL));
		else {
		    loopbody = make_block_statement(CONS(STATEMENT, loopbody,
							 CONS(STATEMENT, cs, NIL)));
		}
		/* end of new code by FI */

		if (n0_loop ==1)
		    newloop = make_loop((entity) var, 
					looprange,
					loopbody,
					looplabel, 
					make_execution(is_execution_parallel,
						       UU),
					NIL);
		else newloop = make_loop((entity) var, 
					 looprange,
					 loopbody,
					 looplabel, 
				 make_execution(is_execution_sequential,UU),
					 NIL);

		stat = loop_to_statement(newloop);
		mod = local_name_to_top_level_entity(entity_local_name(module));
	
		loopbody = make_block_statement(CONS(STATEMENT,stat,NIL));
		ifdebug(8) {
		    wp65_debug_print_text(entity_undefined, loopbody);
		    pips_debug(8,"systeme pour la %d boucle \n",n0_loop);
		    (void) sc_fprint(stderr,sc_neg[n0_loop],noms_var);
		    (void) sc_fprint(stderr,sc_pos[n0_loop],noms_var);
		    (void) sc_fprint(stderr,sc_test,noms_var);
		}
	    }
	}
    }
   
    free((Psysteme) sc_neg);
    free((Psysteme) sc_pos);

    debug(8,"bound_generation","end\n");
    debug_off();
    return(loopbody);

}








