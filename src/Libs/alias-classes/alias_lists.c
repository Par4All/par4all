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
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"
#include "union.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "semantics.h"
#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

static list l_alias_lists;
static list unmatched_alias_pairs;
static list matched_alias_pairs;


/* tests if reg1 and reg2 are the same,
 * including same action_tags (IN/OUT)
 * and precision (may/exact)
 */
static bool
same_reg(region reg1, region reg2)
    {
    Psysteme reg1_sys, reg2_sys;
    bool result = false;

    pips_debug(4,"begin\n");

    if (effect_undefined_p(reg1) || effect_undefined_p(reg2)) return result;

    if (effect_entity(reg1) == effect_entity(reg2))
    {
/*	pips_debug(1,"same entity\n"); */

	if (effect_approximation_tag(reg1) == 
		effect_approximation_tag(reg2))
	    {
/*		pips_debug(1,"same approx\n"); */

		if (effect_action_tag(reg1) == 
		    effect_action_tag(reg2))
		{
/*		pips_debug(1,"same action\n"); */

		    ifdebug(1)
			{
			    set_action_interpretation(ACTION_IN,ACTION_OUT);
			    pips_debug(1,"compare:\n\t");
			    print_region(reg1);
			    pips_debug(1,"with:\n\t");
			    print_region(reg2);
			    reset_action_interpretation();
			}
		    
		    reg1_sys = region_system(reg1);
		    reg2_sys = region_system(reg2);
		    if ( sc_equal_p_ofl(reg1_sys,reg2_sys) )
		    {
			result = true;
			
			pips_debug(1,"same region\n");
		    }
		    else
			pips_debug(1,"not same region\n");
		}
	    }
	}
    pips_debug(4,"end\n");

    return result;
    }


/* global variables IN: unmatched_alias_pairs, l_alias_lists
 * modifies global variable: l_alias_lists
 */
static void
add_unmatched_alias_pairs()
{
    pips_debug(4,"begin\n");

    MAP(LIST,alias_pair,
	{
	    l_alias_lists =
		CONS(EFFECTS,
		     make_effects(regions_dup(alias_pair)),
		     l_alias_lists);
	},unmatched_alias_pairs);

    pips_debug(4,"end\n");
}


/* global variables IN: matched_alias_pairs, l_alias_lists
 * modifies global variable: l_alias_lists
 */
static bool
compare_matched_alias_pairs(bool result,
			    region alias_list_reg,
			    list callee_alias_list)
{
    region formal_reg;

    pips_debug(4,"begin\n");

    MAP(LIST,alias_pair,
	{
	    formal_reg = EFFECT(CAR(alias_pair));

	    ifdebug(9)
		{
		    set_action_interpretation(ACTION_IN,ACTION_OUT);
		    pips_debug(9,"compare to:\n");
		    print_region(formal_reg);
		    reset_action_interpretation();
		}

/* must take the action of the regions into account here because
 * we are checking whether formal_reg is the result of the
 * propagation of alias_list_reg
 */
	    if ( same_reg(formal_reg,alias_list_reg) )
	    {
		pips_debug(9,"match\n");

		l_alias_lists =
		    CONS(EFFECTS,
			 make_effects(gen_nconc(regions_dup(callee_alias_list),
						regions_dup(CDR(alias_pair)))),
			 l_alias_lists);
		result = true;
	    }
	},matched_alias_pairs);

    pips_debug(4,"end\n");

    return result;
}


/* global variables IN: matched_alias_pairs,
 *                      unmatched_alias_pairs,
 *                      l_alias_lists
 * modifies global variables: matched_alias_pairs,
 *                            unmatched_alias_pairs,
 *                            l_alias_lists
 */
static bool
compare_unmatched_alias_pairs(region alias_list_reg, list callee_alias_list)
{
    region formal_reg;
    list rest_unmatched_alias_pairs = unmatched_alias_pairs;
    bool result = false;

    ifdebug(4)
	{
	    set_action_interpretation(ACTION_IN,ACTION_OUT);
	    pips_debug(4,"begin for alias_list_reg:\n");
	    print_region(alias_list_reg);
	    reset_action_interpretation();
	}

    unmatched_alias_pairs = NIL;

    MAP(LIST,alias_pair,
	{
	    formal_reg = EFFECT(CAR(alias_pair));

	    ifdebug(9)
		{
		    set_action_interpretation(ACTION_IN,ACTION_OUT);
		    pips_debug(9,"compare to:\n");
		    print_region(formal_reg);
		    reset_action_interpretation();
		}

/* must take the action of the regions into account here because
 * we are checking whether formal_reg is the result of the
 * propagation of alias_list_reg
 */
	    if ( same_reg(formal_reg,alias_list_reg) )
	    {
		pips_debug(9,"match\n");

		l_alias_lists =
		    CONS(EFFECTS,
			 make_effects(gen_nconc(regions_dup(callee_alias_list),
						regions_dup(CDR(alias_pair)))),
			 l_alias_lists);
		matched_alias_pairs =
		    CONS(LIST,alias_pair,matched_alias_pairs);
		result = true;
	    }
	    else
		unmatched_alias_pairs =
		    CONS(LIST,alias_pair,unmatched_alias_pairs);
	},rest_unmatched_alias_pairs);

    pips_debug(4,"end\n");

    return result;
}


/* global variables IN: matched_alias_pairs,
 *                      unmatched_alias_pairs,
 *                      l_alias_lists
 * modifies global variables: matched_alias_pairs,
 *                            unmatched_alias_pairs,
 *                            l_alias_lists
 */
static void
add_alias_lists_callee( string callee_name )
{
    list callee_alias_lists;
    bool result;
    region alias_list_reg;

    pips_debug(4,"begin for callee %s\n",callee_name);

    callee_alias_lists =
	effects_classes_classes((effects_classes)
				db_get_memory_resource(DBR_ALIAS_LISTS,
						       callee_name,
						       true));
    MAP(EFFECTS,callee_alias_list_effects,
	    {
		list callee_alias_list =
		    effects_effects(callee_alias_list_effects);

		ifdebug(9)
		    {
			pips_debug(9,"add list:\n");
			print_inout_regions(callee_alias_list);
		    }

		if (callee_alias_list != NIL)
		{
		    alias_list_reg = EFFECT(CAR(gen_last(callee_alias_list)));
		    result =
			compare_unmatched_alias_pairs(alias_list_reg,
						      callee_alias_list);
		    result =
			compare_matched_alias_pairs(result,
						    alias_list_reg,
						    callee_alias_list);
		    if (result == false)
			l_alias_lists =
			    CONS(EFFECTS,
				 make_effects(regions_dup(callee_alias_list)),
				 l_alias_lists);
		}
	    },callee_alias_lists);

    pips_debug(4,"end\n");
}


/* global variables IN: unmatched_alias_pairs,
 *                      l_alias_lists
 * modifies global variables: matched_alias_pairs,
 *                            unmatched_alias_pairs,
 *                            l_alias_lists
 */
static void
add_alias_lists_callees(const char* module_name)
{
    callees l_callees;

    pips_debug(4,"begin\n");

    matched_alias_pairs = NIL;

    l_callees = (callees) db_get_memory_resource(DBR_CALLEES,
						 module_name,
						 true);

    MAP(STRING, callee_name,
	{
	    add_alias_lists_callee(callee_name);
	},callees_callees(l_callees));

    pips_debug(4,"end\n");
}


/* modifies global variables: matched_alias_pairs,
 *                            unmatched_alias_pairs,
 *                            l_alias_lists
 */
bool
alias_lists( const char* module_name )
    {
    list in_alias_pairs, out_alias_pairs;
    entity module;

    l_alias_lists = NIL;
    unmatched_alias_pairs = NIL;

    debug_on("ALIAS_LISTS_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);

    ifdebug(1)
	{
	    /* ATTENTION: we have to do ALL this
	     * just to call print_inout_regions for debug !!
	     */
	    set_current_module_entity(
		local_name_to_top_level_entity(module_name) );
	    module = get_current_module_entity();
	    set_current_module_statement( (statement)
					  db_get_memory_resource(DBR_CODE,
								 module_name,
								 true) );
	    set_cumulated_rw_effects((statement_effects)
				     db_get_memory_resource(
					 DBR_CUMULATED_EFFECTS,
					 module_name,
					 true));
	    module_to_value_mappings(module);
	    /* that's it, but we musn't forget to reset everything below */
	}

    /* make alias lists from the IN_alias_pairs */

    /* DBR_IN_ALIAS_PAIRS is a newgen structure of type effects_classes
     * which has one field called classes
     * which is a list of newgen structures of type effects
     * (and each newgen structure of type effects
     * has one field called effects which is a list of elements
     * of type effect)
     */

    in_alias_pairs =
	effects_classes_classes((effects_classes)
				db_get_memory_resource(DBR_IN_ALIAS_PAIRS,
						       module_name,
						       true));


/* wrong but did work:

    in_alias_pairs =
	effects_to_list((effects)
			db_get_memory_resource(DBR_IN_ALIAS_PAIRS,
					       module_name,
					       true));
*/

/*    MAP(LIST, alias_pair,
	{
	    list in_alias_pair = regions_dup(alias_pair);
	    */

    MAP(EFFECTS, alias_pair_effects,
	{
	    list alias_pair = effects_effects(alias_pair_effects);

	    ifdebug(9)
		{
		    pips_debug(9,"IN alias pair : \n");
		    print_inout_regions(alias_pair);
		}
	    unmatched_alias_pairs =
		CONS(LIST,alias_pair,unmatched_alias_pairs);
	},in_alias_pairs);

    /* make alias lists from the OUT_alias_pairs */
    out_alias_pairs =
	effects_classes_classes((effects_classes)
				db_get_memory_resource(DBR_OUT_ALIAS_PAIRS,
						       module_name,
						       true));

/*    MAP(LIST, alias_pair,
	{
	    list out_alias_pair = regions_dup(alias_pair);*/

    MAP(EFFECTS, alias_pair_effects,
	{
	    list alias_pair = effects_effects(alias_pair_effects);

	    ifdebug(9)
		{
		    pips_debug(9,"OUT alias pair : \n");
		    print_inout_regions(alias_pair);
		}

	    unmatched_alias_pairs =
		CONS(LIST,alias_pair,unmatched_alias_pairs);
	},out_alias_pairs);

    ifdebug(9)
	{
	    pips_debug(9,"unmatched_alias_pairs:\n");
	    MAP(LIST,alias_pair,
		{
		    print_inout_regions(alias_pair);
		    pips_debug(9,"---\n");
		},unmatched_alias_pairs);
	}

    add_alias_lists_callees(module_name);

    ifdebug(9)
	{
	    pips_debug(9,"l_alias_lists:\n");
	    MAP(EFFECTS,alias_list,
		{
		    print_inout_regions(effects_effects(alias_list));
		    pips_debug(9,"---\n");
		},l_alias_lists);
	    pips_debug(9,"matched_alias_pairs:\n");
	    MAP(LIST,alias_pair,
		{
		    print_inout_regions(alias_pair);
		    pips_debug(9,"---\n");
		},matched_alias_pairs);
	    pips_debug(9,"unmatched_alias_pairs:\n");
	    MAP(LIST,alias_pair,
		{
		    print_inout_regions(alias_pair);
		    pips_debug(9,"---\n");
		},unmatched_alias_pairs);
	}

    add_unmatched_alias_pairs();

    ifdebug(9)
	{
	    pips_debug(9,"l_alias_lists:\n");
	    MAP(EFFECTS,alias_list,
		{
		    print_inout_regions(effects_effects(alias_list));
		    pips_debug(9,"---\n");
		},l_alias_lists);
	    }

    DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_LISTS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_alias_lists));    

    ifdebug(1)
	{
	    free_value_mappings();
	    reset_current_module_statement();
	    reset_cumulated_rw_effects();
	    reset_current_module_entity();
	}
    pips_debug(4,"end\n");
    debug_off();

    return(true);
}

