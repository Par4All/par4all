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

#include "alias-classes.h"

static list l_alias_lists, l_alias_classes;
static list other_lists, class, rest_list, rest_other_lists;

/* The algorithms below depend on the following properties
 * of Alias Lists and Alias Classes:
 * - an Alias List contains no repeated elements
 * - two Alias Lists of the same module never have the same
 *   head (distinct heads)
 * - an element from one Alias List of a particular module can also
 *   be present in another Alias List of the same module (not disjoint sets)
 * - no element from one Alias Class of a particular module is also
 *   present in another Alias Class of the same module (disjoint sets)
 */


/* global variables IN: l_alias_lists
 * gloabl variables modified: l_alias_lists
 * add a copy of each element in additional_list
 * not already present in initial_reg_list
 * to the end of initial_reg_list
 */
static list
append_all_not_present(list initial_reg_list, list additional_list)
{
    list new_reg_list;

    pips_debug(4,"begin\n");

    MAP(EFFECT,additional_reg,
	{
	    new_reg_list = append_reg_if_not_present(initial_reg_list,
				      region_dup(additional_reg));
	},
	additional_list);

    pips_debug(4,"end\n");

    return new_reg_list;
}


/* global variables IN: rest_list, other_lists
 * global variables modified: rest_list, other_lists
 */
static void
compare_other_list(region elem, list other_list)
{
    bool result = false;
    region other_elem;
    list rest_other_list;

    ifdebug(4)
	{
	    pips_debug(4,"begin for elem:\n");
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
	    print_region(elem);
	    reset_action_interpretation();	    
	    pips_debug(4,"and list:\n");
	    print_inout_regions(other_list);
	}

    if (other_list != NIL)
    {
	rest_other_list = other_list;
	do {
	    other_elem = EFFECT(CAR(rest_other_list));
	    rest_other_list = CDR(rest_other_list);

	    ifdebug(9)
		{
		    pips_debug(9,"compare elem:\n");
		    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
		    print_region(other_elem);
		    reset_action_interpretation();	    
		}

	    if ( effect_exact_p(other_elem) )
	    {
		pips_debug(9,"exact\n");

		if ( same_reg_ignore_action(elem,other_elem) )
		{
		    pips_debug(9,"same\n");

		    rest_list = append_all_not_present(rest_list,other_list);
		    result = true;
		}
	    }
	} while (result == false && rest_other_list != NIL);
	if (result == false)
	    other_lists = gen_nconc(other_lists,CONS(LIST,other_list,NIL));
    }
    pips_debug(4,"end\n");
}


/* global variables IN: rest_list, rest_other_lists, other_lists
 * global variables modified: rest_list, other_lists, rest_other_lists
 * ATTENTION: recursive function
 * compares "elem" (the current element from
 * the list currently being made into a class)
 * to each element of each list of "rest_other_lists"
 * (the other lists not yet made into classes)
 * if a match is found, "other_list"
 * (the other list containing the matching element "other_elem")
 * is appended to "rest_list" (the not yet treated elements from the list
 * currently being made into a class)
 * and "other_list" will no longer be a member of "other_lists"
 * if not, "other_list" is appended to "other_lists"
 */
static void
compare_rest_other_lists(region elem)
{
    list other_list;

    if (rest_other_lists != NIL)
    {
    ifdebug(4)
	{
	    pips_debug(4,"begin for:\n");
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
	    print_region(elem);
	    reset_action_interpretation();	    
	}

    other_list = LIST(CAR(rest_other_lists));
    rest_other_lists = CDR(rest_other_lists);
    compare_other_list(elem, other_list);

    if(rest_other_lists != NIL)
	compare_rest_other_lists(elem);

    pips_debug(4,"end\n");
    }
}


/* global variables IN: class, rest_other_lists, other_lists
 * global variables modified: class, other_lists, rest_other_lists, rest_list
 */
static void
make_class_from_list(list reg_list)
{
    region elem;

    ifdebug(4)
	{
	    pips_debug(4,"begin for:\n");
	    print_inout_regions(reg_list);
	}

    if (reg_list != NIL)
    {
	rest_list = reg_list;
	do {
	   elem = EFFECT(CAR(rest_list));
	   rest_list = CDR(rest_list);

	   ifdebug(9)
	       {
		   pips_debug(9,"elem:\n");
		   set_action_interpretation(ACTION_IN,ACTION_OUT);	    
		   print_region(elem);
		   reset_action_interpretation();	    
	       }

	   if ( effect_exact_p(elem) )
	   {
	       pips_debug(9,"exact\n");

	       rest_other_lists = other_lists;
	       other_lists = NIL;
	       compare_rest_other_lists(elem);
	   }
	   class = gen_nconc(class,CONS(EFFECT,elem,NIL));
	} while(rest_list != NIL);
    }

    pips_debug(4,"end\n");
}


/* global variables IN: other_lists
 * global variables modified:class, other_lists, rest_other_lists, rest_list
 * ATTENTION: recursive function
 */
static void
make_classes_from_lists()
{
    list next_list;

    if (other_lists != NIL)
    {
    pips_debug(4,"begin\n");

    next_list = LIST(CAR(other_lists));
/*    rest_other_lists = CDR(other_lists); */
/*	       other_lists = NIL; */
    other_lists = CDR(other_lists);
    class = NIL;

    make_class_from_list(next_list);
    l_alias_classes = gen_nconc(l_alias_classes,CONS(LIST,class,NIL));

    ifdebug(9)
	{
	    pips_debug(9,"class:\n");
	    print_inout_regions(class);		    
	    pips_debug(9,"other_lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);		    
		    pips_debug(9,"---\n");
		},
		    other_lists);
	}

    if (other_lists != NIL) make_classes_from_lists();

    pips_debug(4,"end\n");
    }
}


/* global variables IN: l_alias_lists
 * global variables modified: l_alias_lists
 * returns true if l_alias_lists modified,
 * i.e. if callee_class_elem is present
 * in one or more of the callers alias lists,
 * in which case,
 * the whole callee_alias_class is added to
 * each of these caller alias lists
 */
static bool
match_this_callee_class_elem(region callee_class_elem, list callee_alias_class)
{
    bool result = false;
    list rest_alias_lists, alias_list;
    region formal_reg_caller_list;

    ifdebug(4)
	{
	    pips_debug(4,"begin for elem:\n");
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
	    print_region(callee_class_elem);
	    reset_action_interpretation();	    
	}

    rest_alias_lists = l_alias_lists;
    if (l_alias_lists != NIL)
	do{
	    alias_list = LIST( CAR(rest_alias_lists) );
	    formal_reg_caller_list = EFFECT( CAR(alias_list) );

	    ifdebug(9)
		{
		    pips_debug(9,"compare with:\n");
		    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
		    print_region(formal_reg_caller_list);
		    reset_action_interpretation();	    
		}

	    if ( same_reg_ignore_action(formal_reg_caller_list,
					callee_class_elem) )
		    {
			pips_debug(9,"match\n");
	    ifdebug(9)
		{
		    pips_debug(9,"old alias list:\n");
		    print_inout_regions(alias_list);
		}

			result = true;
			alias_list =
			    append_all_not_present(alias_list,
						   callee_alias_class);

	    ifdebug(9)
		{
		    pips_debug(9,"new alias list:\n");
		    print_inout_regions(alias_list);
		}

		    }
	    rest_alias_lists = CDR(rest_alias_lists);
	}while (rest_alias_lists != NIL && result == false);

    pips_debug(4,"end\n");

    return result;
}


/* global variables IN: l_alias_lists
 * global variables modified: l_alias_lists
 */
static bool
add_callee_class_to_lists(list callee_alias_class )
{
    bool result = false;

    pips_debug(4,"begin\n");

    MAP(EFFECT,callee_class_elem,
	{
	    if ( match_this_callee_class_elem(callee_class_elem,
					      callee_alias_class) )
		result = true;
	},
    callee_alias_class);

    pips_debug(4,"end\n");

    return result;
}


/* global variables IN: other_lists
 * global variables modified: other_lists
 */
static void
save_callee_class(list callee_alias_class )
{
    pips_debug(4,"begin\n");

    other_lists =
	gen_nconc(other_lists,
		  CONS(LIST,regions_dup(callee_alias_class),NIL));

    pips_debug(4,"end\n");
}


/* global variables IN: l_alias_lists, other_lists
 * global variables modified: l_alias_lists, other_lists
 */
static void
add_classes_for_this_callee( string callee_name )
{
    list callee_alias_classes;

    pips_debug(4,"begin for callee %s\n",callee_name);

    callee_alias_classes = effects_to_list((effects)
				  db_get_memory_resource(DBR_ALIAS_CLASSES,
							 callee_name,
							 true));
    MAP(LIST,callee_alias_class,
	    {
		ifdebug(9)
		    {
			pips_debug(9,"add class:\n");
			print_inout_regions(callee_alias_class);
		    }

		if (!add_callee_class_to_lists(callee_alias_class))
		    save_callee_class(callee_alias_class);
	    },
    callee_alias_classes);

    pips_debug(4,"end\n");
}


static void
add_classes_callees(const char* module_name)
{
    callees all_callees;

    pips_debug(4,"begin\n");

    all_callees = (callees) db_get_memory_resource(DBR_CALLEES,
					       module_name,
					       true);

    MAP(STRING, callee_name,
	{
	    add_classes_for_this_callee(callee_name);
	},
	    callees_callees(all_callees));

    pips_debug(4,"end\n");
}


bool
alias_classes( const char* module_name )
{
    list alias_lists;
    entity module;

    debug_on("ALIAS_CLASSES_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);
    ifdebug(9)
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
	    /* and this to call print_region
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	 */    
	    /* that's it, but we musn't forget to reset everything below */
	}

    l_alias_lists = NIL;
    l_alias_classes = NIL;
    other_lists = NIL;

/*
    alias_lists =
	effects_to_list((effects)
			db_get_memory_resource(DBR_ALIAS_LISTS,
					       module_name,
					       true));
*/

    alias_lists =
	effects_classes_classes((effects_classes)
				db_get_memory_resource(DBR_ALIAS_LISTS,
						       module_name,
						       true));

    MAP(LIST,alias_list,
	{
	    l_alias_lists =
		gen_nconc(CONS(LIST,regions_dup(alias_list),NIL),
			  l_alias_lists);
	},
	    alias_lists);

    ifdebug(9)
	{
	    pips_debug(9,"alias lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);		    
		    pips_debug(9,"---\n");
		},
		    l_alias_lists);
	}

    add_classes_callees(module_name);

    ifdebug(9)
	{
	    pips_debug(9,"alias lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);
		    pips_debug(9,"---\n");
		},
		    l_alias_lists);
	    pips_debug(9,"callee classes:\n");
	    MAP(LIST,alias_class,
		{
		    print_inout_regions(alias_class);		    
		    pips_debug(9,"---\n");
		},
		    other_lists);
	}

    other_lists = gen_nconc(l_alias_lists,other_lists);

    make_classes_from_lists();

    DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_CLASSES, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_alias_classes));    

    ifdebug(9)
	{
/*	    reset_action_interpretation(); */
	    free_value_mappings();
	    reset_current_module_statement();
	    reset_cumulated_rw_effects();
	    reset_current_module_entity();
	}
    pips_debug(4,"end\n");
    debug_off();

    return true;
}

