/* $Id$
 * 
 * pipsmake: call by need (make),
 *
 * rule selection (activate),
 * explicit call (apply)
 *
 * Remi Triolet, Francois Irigoin, Pierre Jouvelot, Bruno Baron,
 * Arnauld Leservot, Guillaume Oget
 *
 * Notes: 
 *  - pismake uses some RI fields explicitly
 *  - see Bruno Baron's DEA thesis for more details
 *  - do not forget the difference between *virtual* resources like 
 *    CALLERS.CODE and *real* resources like FOO.CODE; CALLERS is a 
 *    variable (or a function) whose value depends on the current module; 
 *    it is expanded into a list of real resources;
 *    the variables are CALLEES, CALLERS, ALL and MODULE (the current module
 *    itself);
 *    these variables are used to implement top-down and bottom-up traversals
 *    of the call tree; they make pipsmake different from make
 *
 *  - memoization added to make() to speed-up a sequence of interprocedural 
 *  requests on real applications; a resource r is up-to-date if it already 
 *  has been
 *  proved up-to-date, or if all its arguments have been proved up-to-date and
 *  all its arguments are in the database and all its arguments are
 *  older than the requested resource r; this scheme is correct as soon as 
 *  activate()
 *  destroys the resources produced by the activated (and de-activated) rule
 
 *  - include of an automatically generated builder_map
 
 *  - explicit *recursive* destruction of obsolete resources by
 *  activate() but not by apply(); beware! You cannot assume that all
 *  resources in the database are consistent;
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

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

    MAP(REAL_RESOURCE, rr, 
    {
	rname = real_resource_resource_name(rr);
	rowner = real_resource_owner_name(rr);
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
 * apply(), which calls make() many times, does not fully benefit from
 * this memoization scheme.
 */
static set up_to_date_resources = set_undefined;

void reset_make_cache()
{
    pips_assert("set is defined", !set_undefined_p(up_to_date_resources));
    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;
}

void init_make_cache()
{
    pips_assert("not set", set_undefined_p(up_to_date_resources));
    up_to_date_resources = set_make(set_pointer);
}

void 
reinit_make_cache_if_necessary()
{
    if (!set_undefined_p(up_to_date_resources))
	reset_make_cache(), init_make_cache();
}

static bool 
rmake(string rname, string oname)
{
    rule ru;
    char * res = NULL;

    debug(2, "rmake", "%s(%s) - requested\n", rname, oname);

    /* is it up to date ? */
    if (db_resource_p(rname, oname)) {
	res = db_get_resource_id(rname, oname);
	if(set_belong_p(up_to_date_resources, (char *) res)) {
	    debug(5, "rmake", "resource %s(%s) found in up_to_date "
		      "with time stamp %d\n",
		      rname, oname, db_time_of_resource(rname, oname));
	    return TRUE; /* YES, IT IS! */
	}
	else
	    res = NULL; /* NO, IT IS NOT. */
    }
    
    /* we look for the active rule to produce this resource */
    if ((ru = find_rule_by_resource(rname)) == rule_undefined)
	pips_internal_error("could not find a rule for %s\n", rname);

    /* we recursively make the pre transformations */
    if (!make_pre_transformation(oname, ru))
	return FALSE;

    /* we recursively make required resources */
    if (!make_required(oname, ru))
	return FALSE;

    if (check_resource_up_to_date (rname, oname)) 
    {
	pips_debug (8, "Resource %s(%s) becomes up-to-date after applying"
		    "pre-transformations and building required resources\n",
		    rname,oname);
    } else {
	bool success = FALSE;

	/* we build the resource */
	success = apply_a_rule(oname, ru);
	if (!success) return FALSE;

	/* set up-to-date all the produced resources for that rule */
	MAP(REAL_RESOURCE, rr,
	{
	    string rron = real_resource_owner_name(rr);
	    string rrrn = real_resource_resource_name(rr);
	    
	    if (db_resource_p(rrrn, rron)) 
	    {
		res = db_get_resource_id(rrrn, rron);
		pips_debug(5, "resource %s(%s) added to up_to_date "
			   "with time stamp %d\n",
			   rname, oname, db_time_of_resource(rrrn, rron));
		set_add_element(up_to_date_resources, 
				up_to_date_resources, res);
	    }
	    else {
		pips_internal_error("resource %s(%s) just built not found!\n",
				    rname, oname);
	    }
	}, build_real_resources(oname, rule_produced(ru)));
    }
    return TRUE;
}

static bool 
make(string rname, string oname)
{
    bool success_p = TRUE;

    debug(1, "make", "%s(%s) - requested\n", rname, oname);

    init_make_cache();

    dont_interrupt_pipsmake_asap();
    save_active_phases();

    success_p = rmake(rname, oname);

    reset_make_cache();
    retrieve_active_phases();

    pips_debug(1, "%s(%s) - %smade\n", 
	       rname, oname, success_p? "": "could not be ");

    return success_p;
}

/* Apply do NOT activate the rule applied. 
 * In the case of an interprocedural rule, the rules applied to the
 * callees of the main will be the default rules. For instance,
 * "apply PRINT_CALL_GRAPH_WITH_TRANSFORMERS" applies the rule
 * PRINT_CALL_GRAPH to all callees of the main, leading to a core
 * dump. 
 * Safe apply checks if the rule applied is activated and produces ressources 
 * that it requires (no transitive closure) --DB 8/96
 */
static bool 
apply(string pname, string oname)
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

static bool 
apply_without_reseting_up_to_date_resources(string pname, string oname)
{
    rule ru;

    debug(2, "apply_without_reseting_up_to_date_resources",
	  "apply %s on %s\n", pname, oname);

    /* we look for the rule describing this phase */
    if ((ru = find_rule_by_phase(pname)) == rule_undefined) {
	user_warning("apply_without_reseting_up_to_date_resources",
		     "could not find rule %s\n", pname);
	return FALSE;
    }

    if (!make_pre_transformation(oname, ru))
	return FALSE;

    if (!make_required(oname, ru))
	return FALSE;

    return apply_a_rule (oname, ru);
}

/* this function returns the active rule to produce resource rname */
rule find_rule_by_resource(string rname)
{
    makefile m = parse_makefile();

    debug(5, "find_rule_by_resource", 
	  "searching rule for resource %s\n", rname);

    /* walking thru rules */
    MAP(RULE, r, {
	bool resource_required_p = FALSE;

	/* walking thru resources required by this rule */
	MAP(VIRTUAL_RESOURCE, vr,
	{
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
	    MAP(VIRTUAL_RESOURCE, vr, {
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
	    int i;
	    int number_of_main = 0;
	    gen_array_t a = db_get_module_list();
	    int nmodules = gen_array_nitems(a);

	    pips_assert("some modules...", nmodules>0);
	    for(i=0; i<nmodules; i++) {
		string on = gen_array_item(a, i);

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
				       make_real_resource(vrn, strdup(on)),
					    NIL));
		}
	    }

	    gen_array_full_free(a);
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
	    gen_array_t modules = db_get_module_list();
	    int nmodules = gen_array_nitems(modules), i;

	    pips_assert("some modules", nmodules>0);
	    for(i=0; i<nmodules; i++) {
		string on = gen_array_item(modules, i);
		pips_debug(8, "\t%s\n", on);
		result = gen_nconc(result, 
				   CONS(REAL_RESOURCE, 
					make_real_resource(vrn, strdup(on)),
					NIL));
	    }

	    gen_array_full_free(modules);
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
static bool 
make_pre_transformation(string oname, rule ru)
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
	MAP(REAL_RESOURCE, rr, {
	    string rron = real_resource_owner_name(rr);
	    /* actually the resource name is a phase name !! */
	    string rrpn = real_resource_resource_name(rr);
	    
	    pips_debug(3, "rule %s : applying %s to %s - recursive call\n",
		       rule_phase(ru), rrpn, rron);
	    
	    if (!apply_without_reseting_up_to_date_resources (rrpn, rron))
		success_p = FALSE;
	    
	    /* now we must drop the up_to_date cache.
	     * maybe not that often? Or one should perform the transforms
	     * Top-down to avoid recomputations, with ALL...
	     */
	    reset_make_cache();
	    init_make_cache();
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
    MAP(REAL_RESOURCE, rr, {
	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);
	
	pips_debug(3, "rule %s : %s(%s) - recursive call\n",
		   rule_phase(ru), rrrn, rron);
	
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

static void 
update_preserved_resources(string oname, rule ru)
{
    list reals;

    /* We increment the logical time (kept by pipsdbm) */
    db_inc_logical_time();

    /* we build the list of modified real_resources */
    reals = build_real_resources(oname, rule_modified(ru));

    /* we delete them from the uptodate set */
    MAP(REAL_RESOURCE, rr, {
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

/* returns whether resource is up to date.
 */
static bool 
check_physical_resource_up_to_date(string rname, string oname)
{
  list reals = NIL;
  rule ru = rule_undefined;
  bool result = TRUE;
  char * res = db_get_resource_id(rname, oname);

  /* Maybe is has already been proved true */
  if(set_belong_p(up_to_date_resources, res))
    return TRUE;

  /* We get the active rule to build this resource */
  if ((ru = find_rule_by_resource(rname)) == rule_undefined) {
      /* initial resources have no rule, but that does not matter... */
      if (same_string_p(rname, DBR_USER_FILE))
	  return TRUE;
      /* else */
      pips_internal_error("could not find a rule for %s\n", rname);
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
  MAP(REAL_RESOURCE, rr,
  {
      string rron = real_resource_owner_name(rr);
      string rrrn = real_resource_resource_name(rr);
      
      bool res_in_modified_list_p = FALSE;
      
      /* we build the list of modified real_resources */
      list virtuals = rule_modified(ru);
      list reals2 = build_real_resources(oname, virtuals);
      
      MAP(REAL_RESOURCE, mod_rr,
      {
	  string mod_rron = real_resource_owner_name(mod_rr);
	  string mod_rrrn = real_resource_resource_name(mod_rr);
	  
	  if ((same_string_p(mod_rron, rron)) &&
	      (same_string_p(mod_rrrn, rrrn))) {
	      /* we found it */
	      res_in_modified_list_p = TRUE;
	      pips_debug(3, "resource %s(%s) is in the rule_modified list",
			 rrrn, rron);
	      break;
	  }
      }, reals2);

    gen_free_list (virtuals);
    gen_free_list (reals2);

    /* If the rule is in the modified list, then
       don't check anything */
    if (res_in_modified_list_p == FALSE)
    {
	if (!db_resource_p(rrrn, rron)) {
	    pips_debug(5, "resource %s(%s) is not there "
		       "and not in the rule_modified list", rrrn, rron);
	    result = FALSE;
	    break;
	} else {
	    /* Check if this resource is up to date */
	    long rest;
	    long respt;
	    if (check_resource_up_to_date(rrrn, rron) == FALSE) {
		pips_debug(5, "resource %s(%s) is not up to date", rrrn, rron);
		result = FALSE;
		break;
	    }
	    rest = db_time_of_resource(rname, oname);
	    respt = db_time_of_resource(rrrn, rron);
	    /* Check if the timestamp is OK */
	    if (rest<respt)
	    {
		pips_debug(5, "resource %s(%s) with time stamp %ld is newer "
			   "than resource %s(%s) with time stamp %ld\n",
			   rrrn, rron, respt, rname, oname, rest);
		result = FALSE;
		break;
	    }
	}
    }
  }, reals);

  gen_free_list (reals);

  /* If the resource is up to date then add it in the set */
  if (result == TRUE) {
      pips_debug(5, "resource %s(%s) added to up_to_date "
		 "with time stamp %d\n",
		 rname, oname, db_time_of_resource(rname, oname));
      set_add_element(up_to_date_resources, up_to_date_resources, res);
  }
  return result;
}

/* this is quite ugly, but I wanted to put the enumeration down to pipsdbm.
 */
void
delete_some_resources(void)
{
    string what = get_string_property("PIPSDBM_RESOURCES_TO_DELETE");
    dont_interrupt_pipsmake_asap();

    user_log("Deletion of %s resources: ", what);

    if (same_string_p(what, "obsolete")) {
	int ndeleted;
	init_make_cache();
	ndeleted = 
	    db_delete_obsolete_resources(check_physical_resource_up_to_date); 
	reset_make_cache();
	if (ndeleted>0) user_log("%d destroyed.\n", ndeleted);
	else user_log("none destroyed.\n");
    } else if (same_string_p(what, "all")) {
	db_delete_all_resources();
	user_log("done.\n");
    } else
	pips_internal_error("unexpected delete request %s\n", what);
}


/* To be used in a rule. use and update the up_to_dat list
 * created by makeapply 
 */
bool check_resource_up_to_date(string rname, string oname)
{
    return db_resource_p(rname, oname)?
	check_physical_resource_up_to_date(rname, oname): FALSE;
}

/* Delete from up_to_date all the resources of a given name */
void delete_named_resources (string rn)
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
		pips_debug(5, "resource %s(%s) deleted from up_to_date\n",
			   res_rn, res_on);
		set_del_element (up_to_date_resources,
				 up_to_date_resources,
				 (char *) res);
	    }
	}, up_to_date_resources);
    }
}

void delete_all_resources(void)
{
    db_delete_all_resources();
    set_free(up_to_date_resources);
    up_to_date_resources = set_make(set_pointer);
}

string 
get_first_main_module(void)
{
    string name, tmp_file_name = strdup(".seekfirstmainmoduleXXXXXX");
    FILE * tmp_file;

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    if (!mktemp(tmp_file_name))
	pips_internal_error("unable to make a temporary file\n");

    system(concatenate
	   ("sed -n 's, ,,g;s,	,,g;s,^[pP][rR][oO][gG][rR][aA][mM]"
	    "\\([0-9a-zA-Z\\-_]*\\).*$,\\1,p' ",
	    db_get_current_workspace_directory(),
	    "/*/*.f_initial > ", /**/ tmp_file_name, 0));

    tmp_file = safe_fopen(tmp_file_name, "r");
    name = safe_readline(tmp_file);
    safe_fclose(tmp_file, tmp_file_name);
    unlink(tmp_file_name);
    free(tmp_file_name);

    if (name) strupper(name,name);
    else name=string_undefined;

    debug_off();
    return name;
}

/* check the usage of resources 
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
    MAP(REAL_RESOURCE, rr, {
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
    MAP(REAL_RESOURCE, rr, {
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

bool 
safe_make(string res_n, string module_n)
{
    jmp_buf long_jump_buffer;
    bool success = FALSE;
    bool print_timing_p = get_bool_property("LOG_TIMINGS");
    bool print_memory_usage_p = get_bool_property("LOG_MEMORY_USAGE");
    double initial_memory_size = 0.;

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    if(find_rule_by_resource(res_n) == rule_undefined) {
	user_warning("safe_make", "Unkown resource \"%s\"\n", res_n);
	success = FALSE;
	debug_off();
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

		user_log ("                                 stime      ");
		user_log (request_time);
		user_log ("                                 phase time ");
		user_log (phase_time);
		user_log ("                                 IO stime   ");
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
    debug_off();
    return success;
}

bool 
safe_apply(string phase_n, string module_n)
{
    jmp_buf long_jump_buffer;
    bool success = FALSE;
    bool print_timing_p = get_bool_property("LOG_TIMINGS");
    bool print_memory_usage_p = get_bool_property("LOG_MEMORY_USAGE");
    double initial_memory_size = 0.;
    rule r;

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    if ((r = find_rule_by_phase(phase_n)) == rule_undefined) {
	pips_user_warning("Unkown phase/rule \"%s\" for %s\n", 
			  phase_n, module_n);
	success = FALSE;
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
    debug_off();
    return success;
}
