 /* pipsmake: call by need
  *
  * Remi Triolet, Francois Irigoin, Pierre Jouvelot, Bruno Baron
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
  *  - memoization added to make() to try to speed-up a sequence of interprocedural 
  *  requests on real applications
  *  - include of an automatically generated builder_map
  */
#include <stdio.h>
extern int fprintf();
/* #include <stdlib.h> */
#include <sys/types.h>

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "pipsmake.h"

#include "pipsdbm.h"

#include "resources.h"
#include "phases.h"
#include "builder_map.h"

#include "pipsmake.h"

void build_resource(rname, oname, ru)
string rname, oname;
rule ru;
{
    struct builder_map *pbm = builder_maps;
    string run = rule_phase(ru);

    start_dotting(stdout, '.', 
		  "  %-20.20s building   %s(%s)", 
		  run, rname, oname);

    for (pbm = builder_maps; pbm->builder_name != NULL; pbm++) {
	if (strcmp(pbm->builder_name, run) == 0) {
	    (*pbm->builder_func)(oname);

	    pips_malloc_debug();

	    update_preserved_resources(oname, ru);

	    stop_dotting();

	    return;
	}
    }

    pips_error("build_resource", "could not find function %s\n", run);
}

/* FI: make is very slow when interprocedural analyzes have been selected;
 * some memoization should be added; we need to distinguish betweeen an
 * external make which initializes a set of up-to-date resources and
 * an internal recursive make which update and exploit that set.
 *
 * This new functionality would be extremely useful when old databases
 * are re-opened.
 *
 * apply(), which calls make many times, would not fully benefit from
 * this memoization scheme.
 */
static set up_to_date_resources = set_undefined;

void make(rname, oname)
string rname, oname;
{
    debug_on("PIPSMAKE_DEBUG_LEVEL");
    debug(1, "make", "%s(%s) - requested\n", rname, oname);

    up_to_date_resources = set_make(set_pointer);

    rmake(rname, oname);

    if ( signal_occured() ) {
	accounting_signal();
	make_close_program();
	exit(1);
    }

    set_free(up_to_date_resources);
    up_to_date_resources = set_undefined;

    debug(1, "make", "%s(%s) - made\n", rname, oname);
    debug_off();
}

void rmake(rname, oname)
string rname, oname;
{
    resource res;

    list reals;
    bool requptodate;

    rule ru;
    
    debug(1, "rmake", "%s(%s) - requested\n", rname, oname);

    /* do we have this resource in our database ? */
    res = db_find_resource(rname, oname);

    /* is it up to date ? */
    if (res != resource_undefined) {
	if(set_belong_p(up_to_date_resources, (char *) res)) {
	    debug(8, "rmake", "resource %s(%s) found in up_to_date\n",
			       rname, oname);
	    return;
	}
    }
    
    /* we look for the active rule to produce this resource */
    if ((ru = find_rule_by_resource(rname)) == rule_undefined) {
	pips_error("rmake", "could not find a rule for %s\n", rname);
    }

    /* we recursively make required resources */
    reals = make_required(oname, ru);

    /* do we have this resource in our database ? */
    res = db_find_resource(rname, oname);

    /* is it up to date ? */
    if (res != resource_undefined) {
	requptodate = TRUE;

	if(set_belong_p(up_to_date_resources, (char *) res)) {
	    debug(8, "rmake", "resource %s(%s) found in up_to_date\n",
			       rname, oname);
	}
	else {
	    debug(8, "rmake", "resource %s(%s) not found in up_to_date\n",
			       rname, oname);
	    MAPL(prr, {
		real_resource rr = REAL_RESOURCE(CAR(prr));

		string rron = real_resource_owner_name(rr);
		string rrrn = real_resource_resource_name(rr);

		resource resp = db_find_resource(rrrn, rron);

		if (resp == resource_undefined) {
		    pips_error("rmake", "resource %s(%s) should exist\n",
			       rrrn, rron);
		}

		if (resource_time(res) < resource_time(resp)) {
		    debug(1, "rmake", "resource %s(%s) is newer\n", rrrn, rron);
		    requptodate = FALSE;
		    break;
		}
	    }, reals);

	    if(requptodate==TRUE) {
		debug(8, "rmake", "resource %s(%s) added to up_to_date\n",
		      rname, oname);
		set_add_element(up_to_date_resources,
				up_to_date_resources, (char *) res);
	    }
	}
    }
    else {
	/* we could not find it */
	debug(1, "rmake", 
	      "%s(%s) - not found\n", rname, oname);
	requptodate = FALSE;
    }	    

    if (! requptodate) {
	/* we build the resource */
	build_resource(rname, oname, ru);
	res = db_find_resource(rname, oname);
	if (res != resource_undefined) {
	    debug(8, "rmake", "resource %s(%s) added to up_to_date\n",
			       rname, oname);
	    set_add_element(up_to_date_resources,
			    up_to_date_resources, (char *) res);
	}
	else {
	    pips_error("rmake", "resource %s(%s) just built is not found!\n", rname, oname );
	}
    }
    else {
	debug(1, "rmake", "%s(%s) is up to date\n", rname, oname);
    }
}

void apply(pname, oname)
string pname, oname;
{
    struct builder_map *pbm = builder_maps;
    rule ru;
    string rname;
    
    debug_on("PIPSMAKE_DEBUG_LEVEL");
    debug(1, "make", "apply %s on %s\n", pname, oname);

    up_to_date_resources = set_make(set_pointer);

    /* we look for the rule describing this phase */
    if ((ru = find_rule_by_phase(pname)) == rule_undefined) {
	pips_error("apply", "could not find rule %s\n", pname);
    }

    (void) make_required(oname, ru);

    rname = virtual_resource_name(VIRTUAL_RESOURCE(CAR(rule_produced(ru))));
    start_dotting(stdout, '.', 
		  "  %-20.20s building   %s(%s)", 
		  pname, rname, oname);

    for (pbm = builder_maps; pbm->builder_name != NULL; pbm++) {
	if (strcmp(pbm->builder_name, pname) == 0) {
	    (*pbm->builder_func)(oname);

	    stop_dotting();

	    set_free(up_to_date_resources);

	    debug_off();
	    return;
	}
    }

    update_preserved_resources(oname, ru);

    set_free(up_to_date_resources);

    pips_error("apply", "could not find function %s\n", pname);
    debug_off();
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

	/* walking thru resources made by this particular rule */
	MAPL(pvr, {
	    virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
	    string vrn = virtual_resource_name(vr);

	    if (same_string_p(vrn, rname)) {

		debug(7, "find_rule_by_resource", 
		      "made by phase %s\n", rule_phase(r));

		/* is this phase an active one ? */
		MAPL(pp, {
		    if (same_string_p(STRING(CAR(pp)), rule_phase(r))) {
			debug(5, "find_rule_by_resource", "active phase\n");
			return(r);
		    }
		}, makefile_active_phases(m));

		debug(1, "find_rule_by_resource", "inactive phase\n");
	    }
	}, rule_produced(r));
    }, makefile_rules(m));

    return(rule_undefined);
}

/* Translate and expand a list of virtual resources into a potentially much longer
 * list of real resources
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
	case is_owner_program:
	    result = gen_nconc(result, CONS(REAL_RESOURCE, 
					    make_real_resource(vrn, db_get_current_program_name()),
					    NIL));
	    break;

	case is_owner_module:
	    result = gen_nconc(result, 
			       CONS(REAL_RESOURCE, 
				    make_real_resource(vrn, oname),
				    NIL));
	    break;

	case is_owner_callees:
	{
	    callees called_modules;
	    list lcallees;

	    rmake(DBR_CALLEES, oname);
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

	    rmake(DBR_CALLERS, oname);
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
	default:
	    pips_error("build_real_resources", "unknown tag : %d\n", vrt);
	}
    }

    return(result);
}

/* compute all resources needed to apply a rule on an object */
list make_required(oname, ru)
rule ru;
string oname;
{
    list reals;

    /* we build the list of required real_resources */
    reals = build_real_resources(oname, rule_required(ru));

    /* we recursively make required resources */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));

	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);

	debug(1, "make_required", "%s(%s) - recursive call\n", rrrn, rron);

	(void) rmake(rrrn, rron);

    }, reals);

    return(reals);
}

void update_preserved_resources(oname, ru)
rule ru;
string oname;
{
    list reals;

    /* we build the list of preserved real_resources */
    reals = build_real_resources(oname, rule_preserved(ru));

    /* we update the timestamps of these resources */
    MAPL(prr, {
	real_resource rr = REAL_RESOURCE(CAR(prr));

	string rron = real_resource_owner_name(rr);
	string rrrn = real_resource_resource_name(rr);

	debug(1, "update_preserved_resources",
	      "%s(%s) is preserved\n", rrrn, rron);

	db_update_time(rrrn, rron);
	
    }, reals);
}
