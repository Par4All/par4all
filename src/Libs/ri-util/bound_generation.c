/*
 * $Id$
 *
 * $Log: bound_generation.c,v $
 * Revision 1.10  1998/10/26 10:35:39  irigoin
 * Function bound_generation() slightly updated. Constraints are not yet
 * sorted. The MIN and MAX expressions are not deterministics.
 *
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "misc.h" 
#include "ri.h"

#include "ri-util.h"

/* hum... */
string 
nom_de_variable(string e)
{
    if (e!=NULL) return entity_name((entity) e);
    else return "TCST";
}

/* expression make_contrainte_expression(Pcontrainte pc, variable index)
 * make an expression for constraint of index I 
 * for example: for a constraint of index I : aI + expr_linear(J,K,TCST) <=0
 *              the new expression for I will be : -expr_linear(J,K,TCST)/a 
 */
expression make_contrainte_expression(pc, index)
Pcontrainte pc;
Variable index;
{    
    Pvecteur pv;
    expression ex1,ex2,ex;
    entity div;
    Value coeff;

    /*search the couple (var,val) where var is equal to index and extract it */
    pv = vect_dup(pc->vecteur);
    coeff = vect_coeff(index,pv);
    vect_erase_var(&pv,index);

    if (value_pos_p(coeff)) 
	vect_chg_sgn(pv);
    else
    {
	value_absolute(coeff);
	vect_add_elem(&pv,TCST,value_minus(coeff,VALUE_ONE));
    }

    if(vect_size(pv)==1 && vecteur_var(pv)==TCST)
    {
	vecteur_val(pv) = value_pdiv(vecteur_val(pv), coeff);
	return make_vecteur_expression(pv);
    }

    if(VECTEUR_NUL_P(pv))
	return make_integer_constant_expression(0);

    ex1 = make_vecteur_expression(pv); 
    
    if (value_gt(coeff,VALUE_ONE))
    {
	/* FI->YY: before generating a division, you should test if it could
	   not be performed statically; you have to check if ex1 is not
	   a constant expression, which is fairly easy since you still
	   have its linear form, pv */
	div = gen_find_tabulated("TOP-LEVEL:/",entity_domain);
	
	pips_assert("make_contraitne_expression",div != entity_undefined);
	ex2 = make_integer_constant_expression(VALUE_TO_INT(coeff));
	
	ex = make_expression(make_syntax(is_syntax_call,
					 make_call(div,
						   CONS(EXPRESSION,ex1,
							CONS(EXPRESSION,
							     ex2,NIL)))
	    ),normalized_undefined);	
	return(ex);
    }
    else 
	return(ex1);

}

typedef char * (*variable_name_type)(Variable);

/* void make_bound_expression(variable index, Pbase base, Psysteme sc,
 * expression *lower, expression *upper)
 * make the  expression of the  lower and  upper bounds of  "index"
 * which is in "base" and referenced in "sc"
 */
void make_bound_expression(index, base, sc, lower, upper)
Variable index;
Pbase base;
Psysteme sc;
expression *lower;
expression *upper;
{
    Pcontrainte pc;
    cons *ll = NIL;
    cons *lu = NIL;

    expression ex;
    entity min, max;

    int i;
    int rank_index ;

    /* compute the rank d of the  index in the basis */  
    rank_index = base_find_variable_rank(base,index,nom_de_variable);
    debug(7, "make_bound_expression", "index :%s\n", nom_de_variable(index));
    debug(8, "make_bound_expression", "rank_index = %d\n", rank_index);

    /*search constraints referencing "index" and create the list of 
      expressions for lower and upper bounds */
    for (pc=sc->inegalites; pc!=NULL; pc=pc->succ) {
	i = level_contrainte(pc, base);
	debug(8,"make_bound_expression","level: %d\n",i);
	if (ABS(i)==rank_index){	/* found */
	    ifdebug(7) {
		(void) fprintf(stderr, "\n constraint before :");
		contrainte_fprint(stderr, pc, TRUE, 
				  (variable_name_type) entity_local_name);
	    }
	    ex = make_contrainte_expression(pc, (Variable) index);
	    ifdebug(7) {
		fprintf(stderr, "\n expression after :");
		print_expression(ex);
	    }
	    /* add the expression to the list of  lower bounds
	       or to the list of upper bounds*/
	    if (i>0)
		lu = CONS(EXPRESSION, ex, lu);
	    else
		ll = CONS(EXPRESSION, ex, ll);
	}
    }

    /* make expressions of  lower and  upper  bounds*/
    min = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						  "MIN"), 
			     entity_domain);
    max = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						  "MAX"), 
			     entity_domain);

    pips_assert("some entities",
		min != entity_undefined && max != entity_undefined);

    if (gen_length(ll) > 1) {
	*lower = make_expression(make_syntax(is_syntax_call,
					     make_call(max,ll)),
				 normalized_undefined);
    }
    else {
	*lower = EXPRESSION(CAR(ll)); /* and memory leak... (cons lost) */
	gen_free_list(ll);
    }

    if (gen_length(lu) > 1 ) {
	*upper = make_expression(make_syntax(is_syntax_call,
					     make_call(min,lu)),
				 normalized_undefined );
    }
    else {
	*upper = EXPRESSION(CAR(lu)); /* idem... */
	gen_free_list(lu);
    }

    ifdebug(9) {
	pips_debug(9, "returning: \n");
	print_expression(*lower);
	print_expression(*upper);
    }
}


 
