/* $Id$ */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <sys/types.h>

#include "genC.h"
#include "database.h"
#include "makefile.h"
#include "linear.h"
#include "ri.h"
#include "properties.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"

#include "misc.h"

static void delete_derived_resources();

static list saved_active_phases = NIL;

void 
save_active_phases(void)
{
    makefile current_makefile = parse_makefile();
    
    if (saved_active_phases==NIL)
    {
	MAP(STRING, s, 
	    saved_active_phases = CONS(STRING, strdup(s), saved_active_phases),
	    makefile_active_phases(current_makefile));

	saved_active_phases = gen_nreverse(saved_active_phases);
    }
}

void
retrieve_active_phases(void)
{
    makefile current_makefile = parse_makefile();

    ifdebug(2) {
	puts("----- BEFORE RETREIVING -----");
	fprint_activated(stdout);
    }

    if (saved_active_phases != NIL)
    {
	gen_free_string_list(makefile_active_phases(current_makefile));
	makefile_active_phases(current_makefile) = saved_active_phases;
	saved_active_phases = NIL;

	ifdebug(2) {
	    puts("----- AFTER RETREIVING -----");
	    fprint_activated(stdout);
	}
    }
}

bool 
active_phase_p(string phase)
{
    makefile current_makefile = parse_makefile();

    MAP(STRING, s, 
	if (same_string_p(s, phase)) return TRUE,
	makefile_active_phases(current_makefile));

    return FALSE;
}


void 
fprint_activated(FILE *fd)
{
    makefile m = parse_makefile();
    MAP(STRING, s, fprintf(fd, "%s\n", s), makefile_active_phases(m));
}

/* return the phase which would be used to build a given resource.
 */
string active_phase_for_resource(string res)
{
    return rule_phase(find_rule_by_resource(res));
}

string activate(string phase)
{
    rule r;
    virtual_resource res;
    string vrn;
    string old_phase;
    makefile current_makefile = parse_makefile();
    bool rule_cyclic_p = TRUE;
    string status = phase;

    debug_on("PIPSMAKE_DEBUG_LEVEL");
    debug(1, "activate", "%s - requested\n", phase);

    pips_assert("open_module", db_get_current_workspace_name());

    /* find rule that describes phase */
    r = find_rule_by_phase(phase);
    if(r == rule_undefined) {
	user_error( "activate", "Rule `%s' undefined\n", phase);

    } else {

	/* complete simple cases */
	if (active_phase_p(phase)) {
	    user_warning ("activate", "Rule `%s' already active\n", phase);
	} else if (!gen_length(rule_produced(r))) {
	    user_error("activate", 
		       "Phase %s produces no resource\n", phase);
	} else {
	    /* GO: for many produced resources we loop over them
	       with the same 'old' code */
	    MAPL(pvrp, {
		bool require_produced_rule_p = FALSE;

		/* find resource res that is produced by phase */
		res = VIRTUAL_RESOURCE(CAR(pvrp));
		vrn = virtual_resource_name(res);

		MAPL(pvr, {
		    virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
		    string vrn2 = virtual_resource_name(vr);
		    owner vro = virtual_resource_owner(vr);

		    /* We do not check callers and callees
		     * I dropped select also, just in case... FC
		     */
		    if ( owner_callers_p(vro) || 
			 owner_callees_p(vro) || 
			 owner_select_p(vro)) {}
		    else if (same_string_p(vrn, vrn2))
			require_produced_rule_p = TRUE;

		}, (list) rule_required( r ) );

		/* If the current produced resource is not required
		   by the new rule */
		if (!require_produced_rule_p) {
		    rule_cyclic_p = FALSE;
		    /* find current active phase old_phase that produces res */
		    old_phase = rule_phase(find_rule_by_resource(vrn));

		    /* replace old_phase by phase in active phase list */
		    if (old_phase != NULL) {
			MAPL(pa, {
			    string s = STRING(CAR(pa));
	    
			    if (strcmp(s, old_phase) == 0) {
				free(STRING(CAR(pa)));
				STRING(CAR(pa)) = strdup(phase);
			    }
			}, makefile_active_phases(current_makefile));
		    }

		    /* this generates many warnings when called from select...
		     */
		    if (get_bool_property("ACTIVATE_DEL_DERIVED_RES"))
			delete_derived_resources (res);
		    else
			if (db_get_current_workspace_name()) {
			    /* remove resources with the same name as res 
			       to maintain consistency in the database */
			    db_unput_resources(vrn);
			}
		}
	    }, rule_produced(r));

	    if (rule_cyclic_p == TRUE) {
		user_error("activate",
			   "Phase %s is cyclic\n",
			   phase);
	    }
	}
    }
    debug_off();
    return (status);
}

/*
 * get the set of resources being derived from a given one 
 */
static void get_more_derived_resources (vrn, set_of_res)
string vrn;
set set_of_res;
{
    makefile m = parse_makefile();
    rule r;

    /* If the given resource is not in the set */
    if (set_belong_p (set_of_res, (char *) vrn))
	return;

    /* put it into the set  */
    set_add_element (set_of_res, set_of_res, (char *) vrn);

    debug(8, "get_more_derived_resources",
	  "got %s\n",
	  vrn);
	
    /* For all active  phases*/
    MAPL(pa, {

	r = find_rule_by_phase(STRING(CAR(pa)));
	
	if (rule_use_resource_produced(r) == TRUE)
	    debug(9, "get_more_derived_resources",
		  "Don't scan cycling phase %s\n",STRING(CAR(pa)));
	else
	{
	    debug(9, "get_more_derived_resources",
		  "Scan phase %s\n",STRING(CAR(pa)));

	    /* Search in the required rules */
	    MAPL(pvr, {
		virtual_resource res2 = VIRTUAL_RESOURCE(CAR(pvr));
		string vrn2 = virtual_resource_name(res2);

		/* If the resource names are equal */
		if (same_string_p(vrn, vrn2)) {
		
		    debug(9, "get_more_derived_resources",
			  "Resource %s is required by phase %s\n",
			  vrn, STRING(CAR(pa)));

		    /* make a recursion for all the produced rules */
		    MAPL(pvr3, {
			virtual_resource res3 = VIRTUAL_RESOURCE(CAR(pvr3));
			string vrn3 = virtual_resource_name(res3);
			/* Here, there is no infinite loop problem
			   with rule producing a ressource they require,
			   because the resource has already been had 
			   to the set */
			get_more_derived_resources(vrn3, set_of_res);
		    }, (list) rule_produced ( r ) );
		    break;
		}
	    }, (list) rule_required( r ) );
	}
    }, makefile_active_phases(m));
}

/* Test if a rule uses a resource it produces */
bool rule_use_resource_produced(r)
rule r;
{
    MAPL(pvrp, {
	/* find resource res that is produced by phase */
	virtual_resource res = VIRTUAL_RESOURCE(CAR(pvrp));
	string vrn = virtual_resource_name(res);

	MAPL(pvr, {
	    virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
	    string vrn2 = virtual_resource_name(vr);
	    /* owner vro = virtual_resource_owner(vr); */

	    /* We DO check callers and callees (DB,08/96) */
            /* if ( owner_callers_p(vro) || owner_callees_p(vro) ) {}
	    else
	    */ 
            if (same_string_p(vrn, vrn2))
		return TRUE;
	}, (list) rule_required( r ) );

    }, (list) rule_produced( r ) );

    return FALSE;
}

/*
 * Delete the resources derived from a given one 
 */
static void delete_derived_resources (virtual_resource res)
{
    set s = set_make (set_pointer);
    string vrn = virtual_resource_name(res);
    /* Get the set of virtual resource to destroy */
    get_more_derived_resources(vrn, s);
    SET_MAP(se, delete_named_resources((string) se), s);
    set_free(s);
}






