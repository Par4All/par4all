/* package convex effects :  Be'atrice Creusillet 6/97
 *
 * File: unary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the computation of all types of simple effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "properties.h"

#include "effects-generic.h"
#include "effects-convex.h"


effect
reference_to_convex_region(reference ref, action ac)
{
    return make_reference_region(ref, action_tag(ac));
}

list
convex_regions_descriptor_variable_rename(list l_reg,
					  entity old_ent, entity new_ent)
{
    all_regions_variable_rename(l_reg, old_ent, new_ent);
    return(l_reg);
}

descriptor
loop_convex_descriptor_make(loop l)
{
    descriptor loop_desc;
    Psysteme sc_loop = sc_loop_proper_precondition(l);

    loop_desc = make_descriptor(is_descriptor_convex, sc_loop);
    return loop_desc;
}

list
convex_regions_loop_normalize(list l_reg, entity index, range r, 
			      entity *new_index, descriptor d, 
			      bool descriptor_normalize_p)
{
    bool normalized_p;
    Psysteme sc_loop = SC_UNDEFINED;

    pips_assert("descriptor must be initialized", !descriptor_undefined_p(d));

    if (descriptor_normalize_p)
	sc_loop = descriptor_convex(d);
    
    *new_index = loop_regions_normalize(l_reg, index, r, &normalized_p,
					descriptor_normalize_p, &sc_loop);
    if (descriptor_normalize_p)
	descriptor_convex_(d) = newgen_Psysteme(sc_loop);
    return l_reg;
}

list 
convex_regions_union_over_range(list l_reg, entity index, range r, 
				descriptor d)
{
    if (range_undefined_p(r) && descriptor_undefined_p(d))
    {
	list l_var = CONS(ENTITY, index, NIL);
	project_regions_along_variables(l_reg, l_var);
	gen_free_list(l_var);
    }
    else if (descriptor_undefined_p(d))
    {
	project_regions_along_loop_index(l_reg, index, r);
    }
    else /* range information is included in descriptor */
    {
	Psysteme sc_range = descriptor_convex(d);
	l_reg = array_regions_sc_append(l_reg, sc_range, FALSE);
	ifdebug(6){
	    pips_debug(6, "regions before elimination of index %s.\n",
		       entity_minimal_name(index));
	    (*effects_prettyprint_func)(l_reg);
	}
	project_regions_along_loop_index(l_reg, index, r);
	ifdebug(6){
	    pips_debug(6, "regions after elimination of index.\n");
	    (*effects_prettyprint_func)(l_reg);
	}
    }

    return l_reg;
}


descriptor
vector_to_convex_descriptor(Pvecteur v)
{
    descriptor d;

    if (VECTEUR_UNDEFINED_P(v))
	d = descriptor_undefined;
    else
    {
	Psysteme sc = sc_new();
	Pcontrainte contrainte = contrainte_make(v);
	sc_add_inegalite(sc, contrainte);
	sc->base = BASE_NULLE;
	sc_creer_base(sc);
	ifdebug(8){
	    pips_debug(8, "resulting system:\n");
	    sc_syst_debug(sc);
	}
	d = make_descriptor(is_descriptor_convex, sc);
    }
    
    return d;

}

void
convex_effects_descriptor_normalize(list l_eff)
{
    MAP(REGION, eff,
	{
	    region_system_(eff) = 
		newgen_Psysteme(region_sc_normalize(region_system(eff),1));
	},
	    l_eff);
}
