/*
 * $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"

#include "effects-generic.h"
#include "effects-convex.h"

bool region_consistent_p(region reg)
{
#define MY_MAX_CHECK VALUE_CONST(100000000)
    bool consistent = TRUE;
    Psysteme sc;
    Pbase b, t;
    Pcontrainte c;

    pips_assert("it is a region",  effect_consistent_p(reg));

    /* the system must be defined */
    consistent = consistent && !SC_UNDEFINED_P(region_system(reg));
    pips_assert("the region system is defined", consistent);

    /* check the Psysteme consistency */
    sc = region_system(reg);
    consistent = consistent && sc_weak_consistent_p(sc);
    pips_assert("the region system is consitent", consistent);

    /* the TCST variable must not belong to the base */
    b = sc_base(sc);
    for( t = b; !BASE_UNDEFINED_P(t) && consistent; t = t->succ)
    {
	consistent = consistent && !term_cst(t);
    }    
    pips_assert("no TCST variable in the base", consistent);


    /* and there must be no high coefficient (or it should have raised an
     * exception before
     */
    c = sc_egalites(sc);
    while (c != (Pcontrainte) NULL)
    {
	Pvecteur v;
	for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) 
	{ 
	    pips_assert("no hich coefficient in region system.\n",
			value_lt(vecteur_val(v),MY_MAX_CHECK));
	}
	c = c->succ;
    } 
    c = sc_inegalites(sc);
    while (c != (Pcontrainte) NULL)
    {
	Pvecteur v;
	for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) 
	{ 
	    pips_assert("no hich coefficient in region system.\n",
			value_lt(vecteur_val(v),MY_MAX_CHECK));
	}
	c = c->succ;
    } 
    return consistent;
}


bool regions_consistent_p(list l_reg)
{
    bool consistent = TRUE;
    MAP(EFFECT, reg, 
	consistent = consistent && region_consistent_p(reg),l_reg);
    return consistent;
}

