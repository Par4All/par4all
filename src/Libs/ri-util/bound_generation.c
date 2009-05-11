/*
 * $Id$
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "transformer.h"
#include "ri-util.h"


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
    rank_index = base_find_variable_rank(base, index, (get_variable_name_t) entity_name_or_TCST);
    debug(7, "make_bound_expression", "index :%s\n", entity_name_or_TCST(index));
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
				  (get_variable_name_t) entity_name_or_TCST);
	    }
	    ex = make_constraint_expression(pc->vecteur, (Variable) index);
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
