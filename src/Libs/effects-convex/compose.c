/* package convex effects :  Be'atrice Creusillet 6/97
 *
 * $Id$
 *
 * File: compose.c
 * ~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the composition of convex effects with transformers or preconditions
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "effects-generic.h"
#include "effects-convex.h"

list
convex_regions_transformer_compose(list l_reg, transformer trans)
{
    project_regions_with_transformer_inverse(l_reg, trans, NIL);
    return l_reg;
}

list
convex_regions_inverse_transformer_compose(list l_reg, transformer trans)
{
    project_regions_with_transformer(l_reg, trans, NIL);
    return l_reg;
}

list 
convex_regions_precondition_compose(list l_reg, transformer context)
{
    list l_res = NIL;
    Psysteme sc_context = predicate_system(transformer_relation(context));
    
    ifdebug(8)
	{
	    pips_debug(8, "context: \n");
	    sc_syst_debug(sc_context);
	}
	
    MAP(EFFECT, reg,
	{
	    if (! effect_scalar_p(reg) )
		region_sc_append(reg, sc_context, FALSE);

	    if (!region_empty_p(reg))
		l_res = CONS(EFFECT, reg, l_res);    
	},
	l_reg);

    return l_res;
}
