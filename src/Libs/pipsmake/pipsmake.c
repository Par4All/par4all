 /* pipsmake: call by need (make),
  * rule selection (activate),
  * explicit call (apply)
  *
  * Remi Triolet, Francois Irigoin, Pierre Jouvelot, Bruno Baron,
  * Arnauld Leservot, Guillaume Oget
  *
  * Notes: 

  *  - pismake uses some RI fields explicitly

  *  - see Bruno Baron's DEA thesis for more details

  *  - do not forget the difference between *virtual* resources like CALLERS.CODE
  *  and *real* resources like FOO.CODE; CALLERS is a variable (or a function) whose
  *  value depends on the current module; it is expanded into a list of real resources;
  *  the variables are CALLEES, CALLERS, ALL and MODULE (the current module itself);
  *  these variables are used to implement top-down and bottom-up traversals
  *  of the call tree; they make pipsmake different from make

  *  - memoization added to make() to speed-up a sequence of interprocedural 
  *  requests on real applications; a resource r is up-to-date if it already has been
  *  proved up-to-date, or if all its arguments have been proved up-to-date and
  *  all its arguments are in the database and all its arguments are
  *  older than the requested resource r; this scheme is correct as soon as activate()
  *  destroys the resources produced by the activated (and de-activated) rule

  *  - include of an automatically generated builder_map

  *  - explicit *recursive* destruction of obsolete resources by
  *  activate() but not by apply(); beware! You cannot assume that all
  *  resources in the database are consistent;
  *
  */
#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <setjmp.h>

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "builder_map.h"
#include "properties.h"
#include "pipsmake.h"

/* static functions 
 */
static void update_preserved_resources();
static bool rmake();
static bool apply_a_rule();
static bool apply_without_reseting_up_to_date_resources();
static bool make_pre_transformation();
static bool make_required();

static bool catch_user_error(bool (*f)(char *), string oname)
{
    jmp_buf pipsmake_jump_buffer;
    bool success = FALSE;

    if(setjmp(pipsmake_jump_buffer)) {
	success = FALSE;
    }
    else {
	push_pips_context (&pipsmake_jump_buffer);
	success = (*f)(oname);
    }
    pop_pips_context();
    return success;
}

/* Apply an instanciated rule with a given ressource owner 
*/

static bool apply_a_rule(oname, ru)
string oname;
rule ru;
{
    struct builder_map *pbm = builder_maps;
    bool first_time = TRUE;
    string run = rule_phase(ru);
    string rname;
    string rowner;
    bool is_required;
    bool print_timing_p = get_bool_property("LOG_TIMINGS");
    bool check_res_use_p = get_bool_property("CHECK_RESOURCE_USAGE");

    MAPL(prr, {
	rname = real_resource_resource_name(REAL_RESOURCE(CAR(prr)));
	rowner = real_resource_owner_name(REAL_RESOURCE(CAR(prr)));
	is_required = FALSE;

	MAPL (prrr, {
	    if ((same_string_p
		 (rname, real_resource_resource_name(REAL_RESOURCE(CAR(prrr))))) &&
		(same_string_p
		 (rowner, real_resource_owner_name(REAL_RESOURCE(CAR(prrr))))))
	    {
		is_required = TRUE;
		break;
	    }
	}, build_real_resources(oname,rule_required (ru)));

	user_log("  %-30.60s %8s   %s(%s)\n", 
		 first_time == TRUE ? (first_time = FALSE,run) : "",
		 is_required == TRUE ? "updating" : "building",
		 rname, rowner);

    }, build_real_resources(oname,rule_produced(ru)));

    for (pbm = builder_maps; pbm->builder_name != NULL; pbm++) {
	if (same_string_p(pbm->builder_name, run)) {
	    bool success_p = TRUE;
	    bool print_memory_usage_p = get_bool_property("LOG_MEMORY_USAGE");
	    double initial_memory_size = 0.;

	    if (check_res_use_p)
		init_resource_usage_check();

	    if (print_timing_p)
	    {
		init_log_timers();
	    }

	    if (print_memory_usage_p) 
		initial_memory_size = get_process_gross_heap_size();

	    success_p = catch_user_error(pbm->builder_func, oname);

	    if (print_timing_p) {
		string time_with_io,io_time;

		get_string_timers (&time_with_io, &io_time);

		user_log ("                                 time       ");
		user_log (time_with_io);
		user_log ("                                 IO time    ");
		user_log (io_time);
	    }

	    if (print_memory_usage_p) {
		double final_memory_size = get_process_gross_heap_size();
		user_log("\t\t\t\t memory size %10.3f, increase %10.3f\n",
			 final_memory_size,
			 final_memory_size-initial_memory_size);
	    }

	    if (check_res_use_p)
		do_resource_usage_check(oname, ru);

	    pips_malloc_debug();

	    update_preserved_resources(oname, ru);

	    if (run_pipsmake_callback() == FALSE)
		return FALSE;

	    if (interrupt_pipsmake_asap_p())
		return FALSE;

	    return success_p;
	}
    }

    pips_error("apply_a_rule", "could not find function %s\n", run);
    return FALSE;		/* should never be here ... */
}

/* FI: make is very slow when interprocedural analyzes have been selected;
 * some memorization has been added; we need to distinguish betweeen an
 * external make which initializes a set of up-to-date resources and
 * an internal recursive make which updates and exploits that set.
 *
 * This new functionality is extremely useful when old databases
 * are re-opened.
 *
 * apply(), which calls make many times, does not fully benefit from
 * this memoization scheme.
 */
static set up_to_date_resources = set_undefined;

void reset_make_cache()
{
    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;
}

static bool make(rname, oname)
string rname, oname;
{
    bool success_p = TRUE;

    debug_on("PIPSMAKE_DEBUG_LEVEL");
    debug(1, "make", "%s(%s) - requested\n", rname, oname);

    pips_assert("make", set_undefined_p(up_to_date_resources));
    up_to_date_resources = set_make(set_pointer);
    dont_interrupt_pipsmake_asap();
    save_active_phases();

    success_p = rmake(rname, oname);

    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;
    retrieve_active_phases();

    if (success_p)
	debug(1, "make", "%s(%s) - made\n", rname, oname);
    else
	debug(1, "make", "%s(%s) - could not be made\n", rname, oname);
    debug_off();

    return success_p;
}

static bool rmake(rname, oname)
string rname, oname;
{
    resource res;
    rule ru;

    debug(2, "rmake", "%s(%s) - requested\n", rname, oname);

    /* do we have this resource in our database ? */
    res = db_find_resource(rname, oname);

    /* is it up to date ? */
    if (res != resource_undefined) {
	if(set_belong_p(up_to_date_resources, (char *) res)) {
	    debug(5, "rmake", "resource %s(%s) found in up_to_date "
		      "with time stamp %d\n",
		      rname, oname, resource_time(res));
	    return TRUE;
	}
    }
    
    /* we look for the active rule to produce this resource */
    if ((ru = find_rule_by_resource(rname)) == rule_undefined) {
	pips_error("rmake", "could not find a rule for %s\n", rname);
    }

    /* we recursively make the pre transformations */
    if (!make_pre_transformation(oname, ru))
	return FALSE;

    /* we recursively make required resources */
    if (!make_required(oname, ru))
	return FALSE;

    if (check_resource_up_to_date (rname,oname)) {
	
	debug (8,"rmake",
	       "Resource %s(%s) becomes up-to-date after applying"
	       "pre-transformations and building required resources\n",
	       rname,oname);
    } else {
	bool success = FALSE;

	/* we build the resource */
	success = apply_a_rule(oname, ru);
	if (!success)
	    return FALSE;

	/* set up-to-date all the produced resources for that rule */
	MAPL(prr, {
	    real_resource rr = REAL_RESOURCE(CAR(prr));

	    string rron = real_resource_owner_name(rr);
	    string rrrn = real_resource_resource_name(rr);

	    res = db_find_resource(rrrn, rron);

	    if (res != resource_undefined) {
		debug(5, "rmake", "resource %s(%s) added to up_to_date "
		      "with time stamp %d\n",
		      rname, oname, resource_time(res));
		set_add_element(up_to_date_resources,
				up_to_date_resources, (char *) res);
	    }
	    else {
		pips_error("rmake", 
			   "resource %s(%s) just built is not found!\n",
			   rname,
			   oname);
	    }
	}, build_real_resources(oname, rule_produced(ru)));
    }
    return TRUE;
}

/*
 * Apply do NOT activate the rule applied. 
 * In the case of an interprocedural rule, the rules applied to the
 * callees of the main will be the default rules. For instance,
 * "apply PRINT_CALL_GRAPH_WITH_TRANSFORMERS" applies the rule
 * PRINT_CALL_GRAPH to all callees of the main, leading to a core
 * dump. 
 * A solution, suggested by FI, is to create an apply that checks the 
 * activation of the rule and exits if it is not activated 
 * else calls a "blind" apply -DB, 03-07-96 
 */
static bool apply(pname, oname)
string pname, oname;
{
    bool success_p = TRUE;

    debug_on("PIPSMAKE_DEBUG_LEVEL");
    debug(1, "apply", "%s.%s - requested\n", oname, pname);

    pips_assert("apply", set_undefined_p(up_to_date_resources));
    up_to_date_resources = set_make(set_pointer);
    dont_interrupt_pipsmake_asap();
    save_active_phases();

    success_p = apply_without_reseting_up_to_date_resources (pname,oname);

    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;
    retrieve_active_phases();

    debug(1, "apply", "%s.%s - done\n", oname, pname);
    debug_off();

    return success_p;
}

static bool apply_without_reseting_up_to_date_resources(pname, oname)
string pname, oname;
{
    rule ru;

    debug(2, "apply_without_reseting_up_to_date_resources",
	  "apply %s on %s\n", pname, oname);

    /* we look for the rule describing this phase */
    if ((ru = find_rule_by_phase(pname)) == rule_undefined)
	pips_error("apply", "could not find rule %s\n", pname);

    if (!make_pre_transformation(oname, ru))
	return FALSE;

    if (!make_required(oname, ru))
	return FALSE;

    return apply_a_rule (oname, ru);
}

/* this function returns the active rule to produce resource rname */
rule find_rule_by_resource(rname)
string rname;
{
    makefile m = parse_makefile();

    debug(5, "find_rule_by_resource", 
	  "searching rule for resource %s\n", rname);

    /* walking thru rules */
    MAPL(pr, {
	rule r = RULE(CAR(pr));
	bool resource_required_p = FALSE;


	/* walking thru resources required by this rule */
	MAPL(pvr, {
	    virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
	    string vrn = virtual_resource_name(vr);
	    owner vro = virtual_resource_owner(vr);

	    /* We do not check callers and callees */
	    if ( owner_callers_p(vro) || owner_callees_p(vro) ) {}
	    /* Is this resource required ?? */
	    else if (same_string_p(vrn, rname))
		resource_required_p = TRUE;

	}, rule_required(r));

	/* If this particular resource is not required */
	if (!resource_required_p) {
	    /* walking thru resources made by this particular rule */
	    MAPL(pvr, {
		virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
		string vrn = virtual_resource_name(vr);

		if (same_string_p(vrn, rname)) {

		    debug(5, "find_rule_by_resource", 
			  "made by phase %s\n", rule_phase(r));

		    /* is this phase an active one ? */
		    MAPL(pp, {
			if (same_string_p(STRING(CAR(pp)), rule_phase(r))) {
			    debug(5, "find_rule_by_resource",
				  "active phase\n");
			    return(r);
			}
		    }, makefile_active_phases(m));

		    debug(5, "find_rule_by_resource", "inactive phase\n");
		}
	    }, rule_produced(r));
	}
    }, makefile_rules(m));

    return(rule_undefined);
}

/* Translate and expand a list of virtual resources into a potentially 
 * much longer list of real resources
 *
 * In spite of the name, no resource is actually built.
 */
list build_real_resources(oname, lvr)
string oname;
list lvr;
{
    cons *pvr, *ps;
    list result = NIL;

    for (pvr = lvr; pvr != NIL; pvr = CDR(pvr)) {
	virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
	string vrn = virtual_resource_name(vr);
	tag vrt = owner_tag(virtual_resource_owner(vr));

	switch (vrt) {
	    /* FI: should be is_owner_workspace, but changing Newgen decl... */
	case is_owner_program:
	    /* FI: for  relocation of workspaces */
	    /*
	    result = gen_nconc(result, CONS(REAL_RESOURCE, 
					    make_real_resource(vrn, db_get_current_workspace_name()),
					    NIL));
					    */
	    result = gen_nconc(result, CONS(REAL_RESOURCE, 
					    make_real_resource(vrn, ""),
					    NIL));
	    break;

	case is_owner_module:
	    result = gen_nconc(result, 
			       CONS(REAL_RESOURCE, 
				    make_real_resource(vrn, oname),
				    NIL));
	    break;

	case is_owner_main:
	{
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
	    int number_of_main = 0;

	    db_get_module_list(&nmodules, module_list);
	    pips_assert("build_real_resources", nmodules>0);
	    for(i=0; i<nmodules; i++) {
		string on = module_list[i];

		if (entity_main_module_p
		    (local_name_to_top_level_entity(on)) == TRUE)
		{
		    if (number_of_main)
			pips_error("build_real_resources",
				   "More the one main\n");

		    number_of_main++;
		    debug(8, "build_real_resources", "Main is %s\n", on);
		    result = gen_nconc(result, 
				       CONS(REAL_RESOURCE, 
					    make_real_resource(vrn, on),
					    NIL));
		}
	    }
	    break;
	}
	case is_owner_callees:
	{
	    callees called_modules;
	    list lcallees;

	    if (!rmake(DBR_CALLEES, oname)) {
		/* FI: probably missing source code... */
		user_error ("build_real_resources",
			    "unable to build callees for %s\n%s\n",
			    oname, "Some source code probably is missing!");
	    }
	    
	    called_modules = (callees) 
		db_get_memory_resource(DBR_CALLEES, oname, TRUE);
	    lcallees = callees_callees(called_modules);

	    debug(8, "build_real_resources", "Callees of %s are:\n", oname);

	    for (ps = lcallees; ps != NIL; ps = CDR(ps)) {
		string on = STRING(CAR(ps));

		debug(8, "build_real_resources", "\t%s\n", on);

		result = gen_nconc(result, 
				   CONS(REAL_RESOURCE, 
					make_real_resource(vrn, on),
					NIL));
	    }
	    break;
	}
	case is_owner_callers:
	{
	    /* FI: the keyword callees was badly chosen; anyway, it's just
	       a list of strings... see ri.newgen */
	    callees caller_modules;
	    list lcallers;

	    if (!rmake(DBR_CALLERS, oname)) {
		user_error ("build_real_resources",
			    "unable to build callers for %s\n"
			    "Any missing source code?\n",
			    oname);
	    }

	    caller_modules = (callees) 
		db_get_memory_resource(DBR_CALLERS, oname, TRUE);
	    lcallers = callees_callees(caller_modules);

	    debug(8, "build_real_resources", "Callers of %s are:\n", oname);

	    for (ps = lcallers; ps != NIL; ps = CDR(ps)) {
		string on = STRING(CAR(ps));

		debug(8, "build_real_resources", "\t%s\n", on);

		result = gen_nconc(result, 
				   CONS(REAL_RESOURCE, 
					make_real_resource(vrn, on),
					NIL));
	    }
	    break;
	}
	case is_owner_all:
	{
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;

	    db_get_module_list(&nmodules, module_list);
	    pips_assert("build_real_resource", nmodules>0);
	    for(i=0; i<nmodules; i++) {
		string on = module_list[i];

		debug(8, "build_real_resources", "\t%s\n", on);

		result = gen_nconc(result, 
				   CONS(REAL_RESOURCE, 
					make_real_resource(vrn, on),
					NIL));
	    }
	    break;
	}
	case is_owner_select:
	{
	    /* do nothing ... */
	    break;
	}
	default:
	    pips_internal_error("unknown tag : %d\n", vrt);
	}
    }

    return(result);
}

/* compute all pre-transformations to apply a rule on an object 
 */
static bool make_pre_transformation(oname, ru)
rule ru;
string oname;
{
    list reals;
    bool success_p = TRUE;

    /* we select some resources */
    MAP(VIRTUAL_RESOURCE, vr, 
    {
	string vrn = virtual_resource_name(vr);
	tag vrt = owner_tag(virtual_resource_owner(vr));
	
	if (vrt == is_owner_select) {
	    
	    pips_debug(3, "rule %s : selecting phase %s\n",
		       rule_phase(ru), vrn);
	    
	    if (activate (vrn) == NULL) {
		success_p = FALSE;
		break;
	    }
	}
    }, rule_pre_transformation(ru));
    
    if (success_p) {
	/* we build the list of pre transformation real_resources */
	reals = build_real_resources(oname, rule_pre_transformation(ru));
	
	/* we recursively make the resources */
	MAPL(prr, {
	    real_resource rr = REAL_RESOURCE(CAR(prr));
	    
	    string rron = real_resource_owner_name(rr);
	    /* actually the resource name is a phase name !! */
	    string rrpn = real_resource_resource_name(rr);
	    
	    debug(3, "make_pre_transformation",
		  "rule %s : applying %s to %s - recursive call\n",
		  rule_phase(ru),
		  rrpn,
		  rron);
	    
	    if (!apply_without_reseting_up_to_date_resources (rrpn, rron))
		success_p = FALSE;
	    
	}, reals);
    }
    return TRUE;
}

/* compute all resources needed to apply a rule on an object */
static bool make_required(oname, ru)
rule ru;
string oname;
{
    list reals;
    bool success_p = TRUE;

    /* we build the list of required real_resources */
    reals = build_real_resources(oname, rule_required(ru));

    /* we recursively make required resources */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));
	
	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);
	
	debug(3, "make_required", "rule %s : %s(%s) - recursive call\n",
	      rule_phase(ru),
	      rrrn,
	      rron);
	
	if (!rmake(rrrn, rron)) {
	    success_p = FALSE;
	    /* Want to free the list ... */
	    break;
	}
	
	/* In french:
	   ici nous devons  tester si un des regles modified
	   fait partie des required. Dans ce cas on la fabrique
	   de suite. */
	
    }, reals);

    gen_free_list (reals);
    return success_p;
}

static void update_preserved_resources(oname, ru)
rule ru;
string oname;
{
    list reals;

    /* We increment the logical time (kept by pipsdbm) */
    db_inc_logical_time();

    /* we build the list of modified real_resources */
    reals = build_real_resources(oname, rule_modified(ru));

    /* we delete them from the uptodate set */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));

	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);

	/* is it up to date ? */
	if(set_belong_p(up_to_date_resources, (char *) rr))
	{
	    debug(3, "update_preserved_resources",
		  "resource %s(%s) deleted from up_to_date\n",
		  rrrn, rron);
	    set_del_element (up_to_date_resources,
			     up_to_date_resources,
			     (char *) rr);
	    /* GO 11/7/95: we need to del the resource from the data base
	       for a next call of pipsmake to find it unavailable */
	    db_unput_a_resource (rrrn, rron);
	}
    }, reals);

    gen_free_list (reals);
}

/* To be used at top-level
 * It creates a new up_to_date list to check is a resource is OK 
 */
bool real_resource_up_to_date_p(rname, oname)
string rname, oname;
{
    bool result;

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    pips_assert("real_resource_up_to_date_p",
		set_undefined_p(up_to_date_resources));

    up_to_date_resources = set_make(set_pointer);

    dont_interrupt_pipsmake_asap();

    result = check_resource_up_to_date(rname,oname);

    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;

    debug_off();
    
    return result;
}

/* To be used in a rule. use and update the up_to_dat list
 * created by makeapply 
 */
bool check_resource_up_to_date(rname, oname)
string rname, oname;
{
    resource res;
    list reals;
    rule ru;
    bool result = TRUE;

    res = db_find_resource(rname, oname);

    /* The resource should be in the data base */
    if (res == resource_undefined)
	return FALSE;

    /* Maybe is has already been proved true */
    if(set_belong_p(up_to_date_resources, (char *) res))
	return TRUE;

    /* We get the active rule to build this resource */
    if ((ru = find_rule_by_resource(rname)) == rule_undefined) {
	pips_error("check_resource_up_to_date",
		   "could not find a rule for %s\n", rname);
    }

    /* we build the list of required real_resources */
    /* Here we are sure (thanks to find_rule_by_resource) that the rule does not
       use a resource it produces */

    reals = build_real_resources(oname, rule_required(ru));

    /* we are going to check if the required resources are 
       - in the database or in the rule_modified list
       - proved up to date (recursively)
       - have timestamps older than the tested one
       */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));

	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);

	bool res_in_modified_list_p = FALSE;
	    
	/* we build the list of modified real_resources */
	list virtuals = rule_modified(ru);
	list reals2 = build_real_resources(oname, virtuals);

	MAPL(mod_prr, {
	    real_resource mod_rr = REAL_RESOURCE(CAR(mod_prr));
	    string mod_rron = real_resource_owner_name(mod_rr);
	    string mod_rrrn = real_resource_resource_name(mod_rr);

	    if ((same_string_p(mod_rron, rron)) &&
		(same_string_p(mod_rrrn, rrrn))) {
		/* we found it */
		res_in_modified_list_p = TRUE;
		debug(3, "check_resource_up_to_date",
		      "resource %s(%s) is in the rule_modified list",
		      rrrn, rron);
		break;
	    }
	}, reals2);

	gen_free_list (virtuals);
	gen_free_list (reals2);

	/* If the rule is in the modified list, then
	   don't check anything */
	if (res_in_modified_list_p == FALSE) {

	    resource resp = db_find_resource(rrrn, rron);

	    if (resp == resource_undefined) {
		debug(5, "check_resource_up_to_date",
		      "resource %s(%s) is not present and not in the rule_modified list",
		      rrrn, rron);
		result = FALSE;
		break;
	    } else {
		/* Check if this resource is up to date */
		if (check_resource_up_to_date(rrrn, rron)
		    == FALSE) {
		    debug(5, "check_resource_up_to_date",
			  "resource %s(%s) is not up to date", rrrn, rron);
		    result = FALSE;
		    break;
		}
		/* Check if the timestamp is OK */
		if (resource_time(res) < resource_time(resp)) {
		    debug(5, "check_resource_up_to_date",
			  "resource %s(%s) with time stamp %ld is newer "
			  "than resource %s(%s) with time stamp %ld\n",
			  rrrn, rron,
			  (long) resource_time(resp),
			  resource_name(res),
			  resource_owner_name(res),
			  (long) resource_time(res));
		    result = FALSE;
		    break;
		}
	    }
	}
    }, reals);

    gen_free_list (reals);

    /* If the resource is up to date then add it in the set */
    if (result == TRUE) {
	debug(5, "check_resource_up_to_date",
	      "resource %s(%s) added to up_to_date "
	      "with time stamp %d\n",
	      rname, oname, resource_time(res));
	set_add_element(up_to_date_resources,
			up_to_date_resources, (char *) res);
    }
    return result;
}


/* Delete from up_to_date all the resources of a given name */
void delete_named_resources (rn)
string rn;
{
    /* GO 29/6/95: many lines ...
       db_unput_resources_verbose (rn);*/
    db_unput_resources(rn);

    if (up_to_date_resources != set_undefined) {
	/* In this case we are called from a Pips phase 
	user_warning ("delete_named_resources",
		      "called within a phase (i.e. by activate())\n"); */
	SET_MAP(res, {
	    string res_rn = real_resource_resource_name((real_resource) res);
	    string res_on = real_resource_owner_name((real_resource) res);

	    if (same_string_p(rn, res_rn)) {
		debug(5, "delete_named_resources",
		      "resource %s(%s) deleted from up_to_date\n",
		      res_rn, res_on);
		set_del_element (up_to_date_resources,
				 up_to_date_resources,
				 (char *) res);
	    }
	}, up_to_date_resources);
    }
}

string get_first_main_module()
{

#define MAX__LENGTH 256

    static char name[MAX__LENGTH];
    char tmpfile[MAX__LENGTH];
    FILE *ftmp;
    int success_p;

    extern int system(char*);
    extern int unlink(char*);

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    strncpy (tmpfile,".seekfirstmainmoduleXXXXXX",MAX__LENGTH);

    mktemp (tmpfile);
    if (!(*tmpfile))
	pips_error("get_first_main_module",
		   "unable to make a temporary file\n");

    system(concatenate
	   ("sed -n -e 's;^[ 	t]*[pP][rR][oO][gG][rR][aA][mM][ 	]*\\([0-9a-zA-Z-_]*\\).*$;\\1;p' ",
	    db_get_current_workspace_directory(),
	    "/*.f > ",
	    tmpfile,
	    NULL));

    if ((ftmp = fopen (tmpfile,"r")) != NULL)
    {
	success_p = fscanf (ftmp,"%s\n", name);
	strupper(name,name);
	fclose (ftmp);
	unlink (tmpfile);
	if (success_p != 1)	/* bad item has been read */
	    *name = '\0';
    }

    debug_off ();

    if (*name == '\0')
	return string_undefined;
    return name;
}

/*
 * check the usage of resrouuuces 
 */
void do_resource_usage_check(string oname, rule ru)
{
    list reals;
    set res_read = set_undefined;
    set res_write = set_undefined;

    /* Get the dbm sets */
    get_logged_resources (&res_read, &res_write);

    /* build the real required resrouces */
    reals = build_real_resources(oname, rule_required (ru));

    /* Delete then from the set of read resources */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));
	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);
	string elem_name = strdup(concatenate(rron,".", rrrn, NULL));

	if (set_belong_p (res_read, elem_name)){
	    debug (5, "do_resource_usage_check",
		   "resource %s.%s has been read: ok\n",
		   rron, rrrn);
	    set_del_element(res_read, res_read, elem_name);
	} else
	    user_log ("resource %s.%s has not been read\n",
		      rron, rrrn);
    }, reals);

    /* Try to find an illegally read resrouce ... */
    SET_MAP(re,{
	user_log ("resource %s has been read\n", re);
    }, res_read);

    gen_free_list(reals);

    /* build the real produced resources */
    reals = build_real_resources(oname, rule_produced (ru));

    /* Delete then from the set of write resources */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));
	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);
	string elem_name = strdup(concatenate(rron,".", rrrn, NULL));

	if (set_belong_p (res_write, elem_name)){
	    debug (5, "do_resource_usage_check",
		   "resource %s.%s has been written: ok\n",
		   rron, rrrn);
	    set_del_element(res_write, res_write, elem_name);
	} else
	    user_log ("resource %s.%s has not been written\n",
		      rron, rrrn);
    }, reals);

    /* Try to find an illegally written resrouce ... */
    SET_MAP(re,{
	user_log ("resource %s has been written\n", re);
    }, res_write);

    gen_free_list(reals);

    set_clear(res_read);
    set_clear(res_write);
}

bool safe_make(res_n, module_n)
string res_n, module_n;
{
    jmp_buf long_jump_buffer;
    bool success = FALSE;
    bool print_timing_p = get_bool_property("LOG_TIMINGS");
    bool print_memory_usage_p = get_bool_property("LOG_MEMORY_USAGE");
    double initial_memory_size = 0.;

    if(find_rule_by_resource(res_n) == rule_undefined) {
	user_warning("safe_make", "Unkown resource \"%s\"\n", res_n);
	success = FALSE;
	return success;
    }

    if( setjmp(long_jump_buffer) ) {
	reset_make_cache();
	user_warning("safe_make",
		     "Request aborted in pipsmake: "
		     "build resource %s for module %s.\n", 
		     res_n, module_n);
	success = FALSE;
    }
    else {
	push_pips_context(&long_jump_buffer);
	user_log("Request: build resource %s for module %s.\n", 
		 res_n, module_n);

	if (print_timing_p) {
	    init_request_timers();
	}

	if (print_memory_usage_p) {
	    initial_memory_size = get_process_gross_heap_size();
	}

	pips_malloc_debug();

	success = make(res_n, module_n);
	if(success) {
	    user_log("%s made for %s.\n", res_n, module_n);

	    if (print_timing_p) {
		string request_time, phase_time, dbm_time;

		get_request_string_timers (&request_time, &phase_time,
					   &dbm_time);

		user_log ("                                 time       ");
		user_log (request_time);
		user_log ("                                 phase time ");
		user_log (phase_time);
		user_log ("                                 IO time    ");
		user_log (dbm_time);
	    }

	    if (print_memory_usage_p) {
		double final_memory_size = get_process_gross_heap_size();
		user_log("\t\t\t\t memory size %10.3f, increase %10.3f\n",
			 final_memory_size,
			 final_memory_size-initial_memory_size);
	    }
	}
	else {
	    user_warning("safe_make",
			 "Request aborted under pipsmake: "
			 "build resource %s for module %s.\n", 
			 res_n, module_n);
	}
    }
    pop_pips_context();

    return success;
}

bool safe_apply(phase_n, module_n)
string phase_n, module_n;
{
    jmp_buf long_jump_buffer;
    bool success = FALSE;
    bool print_timing_p = get_bool_property("LOG_TIMINGS");
    bool print_memory_usage_p = get_bool_property("LOG_MEMORY_USAGE");
    double initial_memory_size = 0.;
    rule r;

    if ((r = find_rule_by_phase(phase_n)) == rule_undefined) {
	user_warning("safe_apply", "Unkown phase/rule \"%s\"\n", phase_n);
	success = FALSE;
	return success;
    }

    if( setjmp(long_jump_buffer) ) {
	reset_make_cache();
	user_warning("safe_apply", 
		     "Request aborted in pipsmake: "
		     "perform rule %s on module %s.\n", 
		     phase_n, module_n);
	success = FALSE;
    }
    else {
    if (rule_use_resource_produced(r) && (! active_phase_p(phase_n))) {
        user_warning("safe_apply",
		     "Request aborted in pipsmake: "
		     "cyclic rule %s not activated.\n",
		     phase_n);
	success = FALSE;
    }
    else {
	push_pips_context(&long_jump_buffer);
	user_log("Request: perform rule %s on module %s.\n", 
		 phase_n, module_n);
	pips_malloc_debug();

	if (print_timing_p) {
	    init_request_timers();
	}

	if (print_memory_usage_p) {
	    initial_memory_size = get_process_gross_heap_size();
	}
     
	success = apply(phase_n, module_n);

	if (success) {
	    user_log("%s applied on %s.\n", phase_n, module_n);

	    if (print_timing_p) {
		string request_time, phase_time, dbm_time;

		get_request_string_timers (&request_time, &phase_time,
					   &dbm_time);

		user_log ("                                 time       ");
		user_log (request_time);
		user_log ("                                 phase time ");
		user_log (phase_time);
		user_log ("                                 IO time    ");
		user_log (dbm_time);
	    }

	    if (print_memory_usage_p) {
		double final_memory_size = get_process_gross_heap_size();
		user_log("\t\t\t\t memory size %10.3f, increase %10.3f\n",
			 final_memory_size,
			 final_memory_size-initial_memory_size);
	    }
	}
	else {
	    user_warning("safe_apply", 
			 "Request aborted under pipsmake: "
			 "perform rule %s on module %s.\n", 
			 phase_n, module_n);
	}
    }
    pop_pips_context();

    return success;
}






