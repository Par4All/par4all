/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: mappings.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains the mappings necessary for the computation of effects.
 *
 */
#include "genC.h"
#include "linear.h"
#include "ri.h"

/* REFERENCES */
GENERIC_GLOBAL_FUNCTION(proper_references, statement_effects)
GENERIC_GLOBAL_FUNCTION(cumulated_references, statement_effects)

/* READ AND WRITE EFFECTS */
GENERIC_GLOBAL_FUNCTION(proper_rw_effects, statement_effects)
GENERIC_GLOBAL_FUNCTION(rw_effects, statement_effects)
GENERIC_GLOBAL_FUNCTION(invariant_rw_effects, statement_effects)
/* for external use - identical to rw_effects */
GENERIC_GLOBAL_FUNCTION(cumulated_rw_effects, statement_effects)


list 
load_proper_rw_effects_list(statement s)
{
    return(effects_effects(load_proper_rw_effects(s)));
}

void store_proper_rw_effects_list(statement s, list l_eff)
{
    store_proper_rw_effects(s, make_effects(l_eff));
}

void update_proper_rw_effects_list(statement s, list l_eff)
{
    update_proper_rw_effects(s, make_effects(l_eff));
}


list 
load_rw_effects_list(statement s)
{
    return(effects_effects(load_rw_effects(s)));
}

void
store_rw_effects_list(statement s, list l_eff)
{
    store_rw_effects(s, make_effects(l_eff));
}

void
update_rw_effects_list(statement s, list l_eff)
{
    update_rw_effects(s, make_effects(l_eff));
}

list 
load_invariant_rw_effects_list(statement s)
{
    return(effects_effects(load_invariant_rw_effects(s)));
}

void
store_invariant_rw_effects_list(statement s, list l_eff)
{
    store_invariant_rw_effects(s, make_effects(l_eff));
}

void
update_invariant_rw_effects_list(statement s, list l_eff)
{
    update_invariant_rw_effects(s, make_effects(l_eff));
}

list 
load_cumulated_rw_effects_list(statement s)
{
    return(effects_effects(load_cumulated_rw_effects(s)));
}

void
store_cumulated_rw_effects_list(statement s, list l_eff)
{
    store_cumulated_rw_effects(s, make_effects(l_eff));
}

void
update_cumulated_rw_effects_list(statement s, list l_eff)
{
    update_cumulated_rw_effects(s, make_effects(l_eff));
}


/* IN AND OUT EFFECTS */
GENERIC_GLOBAL_FUNCTION(in_effects, statement_effects)
GENERIC_GLOBAL_FUNCTION(cumulated_in_effects, statement_effects)
GENERIC_GLOBAL_FUNCTION(invariant_in_effects, statement_effects)

GENERIC_GLOBAL_FUNCTION(out_effects, statement_effects)

list 
load_in_effects_list(statement s)
{
    return(effects_effects(load_in_effects(s)));
}

void
store_in_effects_list(statement s, list l_eff)
{
    store_in_effects(s, make_effects(l_eff));
}

void
update_in_effects_list(statement s, list l_eff)
{
    update_in_effects(s, make_effects(l_eff));
}


list 
load_cumulated_in_effects_list(statement s)
{
    return(effects_effects(load_cumulated_in_effects(s)));
}

void
store_cumulated_in_effects_list(statement s, list l_eff)
{
    store_cumulated_in_effects(s, make_effects(l_eff));
}

void
update_cummulated_in_effects_list(statement s, list l_eff)
{
    update_cumulated_in_effects(s, make_effects(l_eff));
}

list 
load_invariant_in_effects_list(statement s)
{
    return(effects_effects(load_invariant_in_effects(s)));
}

void
store_invariant_in_effects_list(statement s, list l_eff)
{
    store_invariant_in_effects(s, make_effects(l_eff));
}

void
update_invariant_in_effects_list(statement s, list l_eff)
{
    update_invariant_in_effects(s, make_effects(l_eff));
}


list 
load_out_effects_list(statement s)
{
    return(effects_effects(load_out_effects(s)));
}

void
store_out_effects_list(statement s, list l_eff)
{
    store_out_effects(s, make_effects(l_eff));
}

void
update_out_effects_list(statement s, list l_eff)
{
    update_out_effects(s, make_effects(l_eff));
}

/* BACKWARD COMPATIBILITY */

list load_statement_local_regions(statement s)
{
    return(load_rw_effects_list(s));
}
void store_statement_local_regions(statement s, list t)
{
    store_rw_effects_list(s, t);
}
void update_statement_local_regions(statement s, list t)
{
    update_rw_effects_list(s, t);
}

list load_statement_inv_regions(statement s)
{
    return(load_invariant_rw_effects_list(s));
}
void store_statement_inv_regions(statement s, list t)
{
    store_invariant_rw_effects_list(s, t);
}
void update_statement_inv_regions(statement s, list t)
{
    update_invariant_rw_effects_list(s, t);
}

list load_statement_proper_regions(statement s)
{
    return(load_proper_rw_effects_list(s));
}
void store_statement_proper_regions(statement s, list t)
{
    store_proper_rw_effects_list(s, t);
}


list load_statement_in_regions(statement s)
{
    return(load_in_effects_list(s));
}
void store_statement_in_regions(statement s, list t)
{
    store_in_effects_list(s, t);
}

list load_statement_inv_in_regions(statement s)
{
    return(load_invariant_in_effects_list(s));
}
void store_statement_inv_in_regions(statement s, list t)
{
    store_invariant_in_effects_list(s, t);
}

void update_statement_inv_in_regions(statement s, list t)
{
    update_invariant_in_effects_list(s, t);
}

list load_statement_cumulated_in_regions(statement s)
{
    return(load_cumulated_in_effects_list(s));
}
void store_statement_cumulated_in_regions(statement s, list t)
{
    store_cumulated_in_effects_list(s, t);
}

list load_statement_out_regions(statement s)
{
    return(load_out_effects_list(s));
}
void store_statement_out_regions(statement s, list t)
{
    store_out_effects_list(s, t);
}

