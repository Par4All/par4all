/* $Id$
 */

#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
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
static list l_lists, new_class, rest_list, rest_lists;


/* add a copy of each element in additional_list
 * not already present in initial_reg_list
 * to the end of initial_reg_list
 */
static list
union_lists(list initial_reg_list, list additional_list)
{
    list new_reg_list = initial_reg_list;

    pips_debug(4,"begin\n");

    MAP(EFFECT,additional_reg,
	{
	    new_reg_list =
		append_reg_if_not_present(new_reg_list,additional_reg);
	},additional_list);

    pips_debug(4,"end\n");

    return new_reg_list;
}


/* global variables IN: rest_list, l_lists
 * global variables modified: rest_list, l_lists
 */
static void
compare_other_list(region elem, list other_list)
{
    bool result = FALSE;
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

		    rest_list = union_lists(rest_list,other_list);
		    result = TRUE;
		}
	    }
	} while (result == FALSE && rest_other_list != NIL);
	if (result == FALSE)
	    l_lists = CONS(LIST,other_list,l_lists);
    }
    pips_debug(4,"end\n");
}


/* global variables IN: rest_list, rest_lists, l_lists
 * global variables modified: rest_list, l_lists, rest_lists
 * compares "elem" (the current element from
 * the list currently being made into a class)
 * to each element of each list of "rest_lists"
 * (the other lists not yet made into classes)
 * if a match is found, "other_list"
 * (the other list containing the matching element "other_elem")
 * is appended to "rest_list" (the not yet treated elements from the list
 * currently being made into a class)
 * and "other_list" will no longer be a member of "l_lists"
 * if not, "other_list" is appended to "l_lists"
 */
static void
compare_rest_lists(region elem)
{
    list other_list;

    while (rest_lists != NIL)
    {
    ifdebug(4)
	{
	    pips_debug(4,"begin for:\n");
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
	    print_region(elem);
	    reset_action_interpretation();	    
	}

    other_list = LIST(CAR(rest_lists));
    rest_lists = CDR(rest_lists);
    compare_other_list(elem, other_list);
    }
    pips_debug(4,"end\n");
}


/* global variables IN: new_class, rest_lists, l_lists
 * global variables modified: new_class, l_lists, rest_lists, rest_list
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

    rest_list = reg_list;

    while (rest_list != NIL)
    {
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

	    rest_lists = l_lists;
	    l_lists = NIL;
	    compare_rest_lists(elem);
	}
	new_class = gen_nconc(new_class,CONS(EFFECT,elem,NIL));
    }

    pips_debug(4,"end\n");
}


/* global variables IN: l_lists, l_alias_classes
 * global variables modified:class, l_lists, rest_lists, rest_list,
 *                           l_alias_classes
 */
static void
unite_lists_containing_same_exact_region()
{
    list next_list;

    while (l_lists != NIL)
    {
    pips_debug(4,"begin\n");

    next_list = LIST(CAR(l_lists));
    l_lists = CDR(l_lists);
    new_class = NIL;

    make_class_from_list(next_list);
    l_alias_classes = CONS(EFFECTS,make_effects(new_class),l_alias_classes);

/*    ifdebug(9)
	{
	    pips_debug(9,"new_class:\n");
	    print_inout_regions(new_class);		    
	    pips_debug(9,"l_lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);		    
		    pips_debug(9,"---\n");
		},
		    l_lists);
	}*/

    if (l_lists != NIL) unite_lists_containing_same_exact_region();
    }
    pips_debug(4,"end\n");
}


/* global variables IN: l_alias_lists
 * global variables modified: l_alias_lists
 * compares "head" (the head of the current list)
 * to the head of each list of "rest_lists"
 * (the other lists not yet treated)
 * if a match is found, "other_list"
 * (the other list containing the matching head "other_head")
 * is appended to "new_list"
 * and "other_list" will no longer be a member of "l_alias_lists"
 * if not, "other_list" is appended to "l_alias_lists"
 */
static list
compare_heads_rest_lists(region head, list new_list)
{
    list rest_lists;
    region other_head;

    rest_lists = l_alias_lists;
    l_alias_lists = NIL;

    MAP(LIST, other_list,
    {
    ifdebug(4)
	{
	    pips_debug(4,"begin for:\n");
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
	    print_region(head);
	    reset_action_interpretation();	    
	}

    if (other_list != NIL)
    {
	    other_head = EFFECT(CAR(other_list));

	    ifdebug(9)
		{
		    pips_debug(9,"compare elem:\n");
		    set_action_interpretation(ACTION_IN,ACTION_OUT);	    
		    print_region(other_head);
		    reset_action_interpretation();	    
		}

	    if ( same_reg_ignore_action(head,other_head) )
	    {
		pips_debug(9,"same\n");

		new_list = union_lists(new_list,CDR(other_list));
	    }
	    else
	    {
	    l_alias_lists = CONS(LIST,other_list,l_alias_lists);
	    }
    }

    },rest_lists);

    pips_debug(4,"end\n");

    return new_list;
}


/* global variables IN: l_alias_lists, l_lists
 * global variables modified: l_alias_lists, l_lists
 */
static void
unite_lists_with_same_head()
{
    list next_list, new_list;
    region head;

    pips_debug(4,"begin\n");

    while (l_alias_lists != NIL)
    {

    next_list = LIST(CAR(l_alias_lists));
    l_alias_lists = CDR(l_alias_lists);
    new_list = next_list;

    if (next_list != NIL)
    {
	   head = EFFECT(CAR(next_list));

	   ifdebug(9)
	       {
		   pips_debug(9,"head:\n");
		   set_action_interpretation(ACTION_IN,ACTION_OUT);	    
		   print_region(head);
		   reset_action_interpretation();	    
	       }

	   new_list = compare_heads_rest_lists(head, new_list);
    }

    l_lists = CONS(LIST,new_list,l_lists);

/*    ifdebug(9)
	{
	    pips_debug(9,"new_list:\n");
	    print_inout_regions(new_list);		    
	    pips_debug(9,"l_alias_lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);		    
		    pips_debug(9,"---\n");
		},
		    l_alias_lists);
	}
	*/
    }

    pips_debug(4,"end\n");
}


/* global variables IN: l_alias_lists, other_lists
 * global variables modified: l_alias_lists, other_lists
 */
static void
add_lists_callee( string callee_name )
{
    list callee_alias_lists;

    pips_debug(4,"begin for callee %s\n",callee_name);

    callee_alias_lists =
	effects_classes_classes((effects_classes)
				db_get_memory_resource(DBR_ALIAS_LISTS,
						       callee_name,
						       TRUE));
    MAP(EFFECTS,callee_alias_list_effects,
	    {
		list callee_alias_list =
		    regions_dup(effects_effects(callee_alias_list_effects));

		ifdebug(9)
		    {
			pips_debug(9,"add list:\n");
			print_inout_regions(callee_alias_list);
		    }

		l_alias_lists = CONS(LIST,callee_alias_list,l_alias_lists);
	    },callee_alias_lists);

    pips_debug(4,"end\n");
}


static void
get_lists_callees(string module_name)
{
    callees all_callees;

    pips_debug(4,"begin\n");

    all_callees = (callees) db_get_memory_resource(DBR_CALLEES,
					       module_name,
					       TRUE);

    MAP(STRING, callee_name,
	{
	    add_lists_callee(callee_name);
	},
	    callees_callees(all_callees));

    pips_debug(4,"end\n");
}


bool
alias_classes( string module_name )
{
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
								 TRUE) );
	    set_cumulated_rw_effects((statement_effects)
				     db_get_memory_resource(
					 DBR_CUMULATED_EFFECTS,
					 module_name,
					 TRUE));
	    module_to_value_mappings(module);
	    /* and this to call print_region
	    set_action_interpretation(ACTION_IN,ACTION_OUT);	 */    
	    /* that's it, but we musn't forget to reset everything below */
	}

    l_alias_lists = NIL;
    l_alias_classes = NIL;
    l_lists = NIL;

    get_lists_callees(module_name);

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

    unite_lists_with_same_head();

    ifdebug(9)
	{
	    pips_debug(9,"new lists:\n");
	    MAP(LIST,alias_list,
		{
		    print_inout_regions(alias_list);		    
		    pips_debug(9,"---\n");
		},
		    l_lists);
	}

    unite_lists_containing_same_exact_region();

    ifdebug(9)
	{
	    pips_debug(9,"classes:\n");
	    MAP(EFFECTS,alias_class,
		{
		    print_inout_regions(effects_effects(alias_class));		    		   pips_debug(9,"---\n");
		},
		    l_alias_classes);
	}

    DB_PUT_MEMORY_RESOURCE(DBR_ALIAS_CLASSES, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_alias_classes));    

    ifdebug(9)
	{
	    free_value_mappings();
	    reset_current_module_statement();
	    reset_cumulated_rw_effects();
	    reset_current_module_entity();
	}
    pips_debug(4,"end\n");
    debug_off();

    return TRUE;
}

