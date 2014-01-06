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

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "genC.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "properties.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"

#include "misc.h"

static void delete_derived_resources();

static list saved_active_phases = NIL;

void save_active_phases(void)
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

void retrieve_active_phases(void)
{
    makefile current_makefile = parse_makefile();

    ifdebug(9) {
	puts("----- BEFORE RETRIEVING -----");
	fprint_activated(stdout);
    }

    if (saved_active_phases != NIL)
    {
	gen_free_string_list(makefile_active_phases(current_makefile));
	makefile_active_phases(current_makefile) = saved_active_phases;
	saved_active_phases = NIL;

	ifdebug(9) {
	    puts("----- AFTER RETREIVING -----");
	    fprint_activated(stdout);
	}
    }
}

bool active_phase_p(const char * phase)
{
    makefile current_makefile = parse_makefile();
    list apl = makefile_active_phases(current_makefile);

    FOREACH(STRING, s, apl)
      if (same_string_p(s, phase))
	return true; // new line for breakpoints

    return false;
}

/* Debugging function */
bool saved_active_phase_p(const char * phase)
{
  list sapl = saved_active_phases;

  if(ENDP(sapl)) {
    fprintf(stderr, "Active phases have not been saved\n");
  }
  else {
    FOREACH(STRING, s, sapl)
      if (same_string_p(s, phase))
	return true; // new line for breakpoints
  }

  return false;
}


void fprint_activated(FILE *fd)
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

const char * activate_phase(const char * phase)
{
  rule r;
  //virtual_resource res;
  string vrn;
  string old_phase;
  makefile current_makefile = parse_makefile();
  bool rule_cyclic_p = true;
  const char * status = phase;

  debug_on("PIPSMAKE_DEBUG_LEVEL");
  pips_debug(1, "%s - requested\n", phase);

  pips_assert("a current workspace is defined",
	      db_get_current_workspace_name());

  /* find rule that describes phase */
  r = find_rule_by_phase(phase);

  if(r == rule_undefined) {
    pips_user_warning("Rule `%s' undefined.\n"
		      "Check spelling and/or ACTIVE_PHASE property.\n",
		      phase);
    status = NULL;
  }
  else if (active_phase_p(phase)) {
    pips_user_warning ("Rule `%s' already active\n", phase);
  }
  else if (!gen_length(rule_produced(r))) {
    pips_user_warning("Phase %s produces no resource\n", phase);
    status = NULL;
  }
  else {
    /* GO: for many produced resources we loop over them
       with the same 'old' code */
    /* find resource res that is produced by phase */
    FOREACH(VIRTUAL_RESOURCE, res, rule_produced(r)) {
      bool require_produced_rule_p = false;

      vrn = virtual_resource_name(res);

      FOREACH(VIRTUAL_RESOURCE, vr, (list) rule_required(r)) {
	string vrn2 = virtual_resource_name(vr);
	owner vro = virtual_resource_owner(vr);

	/* We do not check callers and callees
	 * I dropped select also, just in case... FC
	 */
	if ( owner_callers_p(vro) ||
	     owner_callees_p(vro) ||
	     owner_select_p(vro)) {}
	else if (same_string_p(vrn, vrn2))
	  require_produced_rule_p = true;

      }

      /* If the current produced resource is not required
	 by the new rule */
      if (!require_produced_rule_p) {
	rule_cyclic_p = false;
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
	    /* remove resources with the same name as
	       res to maintain consistency in the
	       database */
	    db_unput_resources(vrn);
	  }
      }
    }

    if (rule_cyclic_p == true) {
      pips_user_warning("Phase %s is cyclic\n", phase);
      status = NULL;
    }
  }
  debug_off();
  return (status);
}

const char* activate(const char* phase)
{
  const char* r = activate_phase(phase);
  if(!r)
    pips_user_error("Phase activation error: check the phase names\n");
  return r;
}

/* Use property ACTIVE_PHASES to active the phases required by the
   user. */
bool activate_phases(void)
{
  string d = " ,\t\n";
  /* strtok breaks its first argument string */
  string ap = strdup(get_string_property("ACTIVE_PHASES"));
  string cap = strtok(ap, d);
  bool result = true;

  while(cap!=NULL) {
    pips_debug(1, "Phase to activate: %s\n", cap);
    if(!active_phase_p(cap)) {
     const char* r =  activate_phase(cap);
     result = r!=NULL;
    }
    cap = strtok(NULL, d);
  }
  free(ap);
  return result;
}

/* Choose the right combination of activate and setproperty for a
   given language.

   This is not really compatible with the litterate programming of
   pipsmake-rc.tex, where this information should be encoded.
 */
void activate_language(language l)
{
  if(language_fortran_p(l)) {
    /* Usual properties for Fortran */
    set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", true);
    set_bool_property("FOR_TO_WHILE_LOOP_IN_CONTROLIZER", false);
    set_bool_property("FOR_TO_DO_LOOP_IN_CONTROLIZER", false);

    if(!active_phase_p("PARSER"))
      activate("PARSER");
    if(!active_phase_p("FORTRAN_SYMBOL_TABLE"))
      activate("FORTRAN_SYMBOL_TABLE");
  } else if(language_fortran95_p(l)) {
    /* Usual properties for Fortran 90/95 */
    set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", false);
    set_bool_property("FOR_TO_WHILE_LOOP_IN_CONTROLIZER", false);
    set_bool_property("FOR_TO_DO_LOOP_IN_CONTROLIZER", false);

    // Temporary fix for autogenerated file
    if(!active_phase_p("PARSER"))
      activate("PARSER");
    if(!active_phase_p("FORTRAN_SYMBOL_TABLE"))
      activate("FORTRAN_SYMBOL_TABLE");
  } else if(language_c_p(l)) {
    /* Usual properties for C */
    set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", false);
    set_bool_property("FOR_TO_WHILE_LOOP_IN_CONTROLIZER", true);
    set_bool_property("FOR_TO_DO_LOOP_IN_CONTROLIZER", true);

    if(!active_phase_p("C_PARSER"))
      activate("C_PARSER");
    if(!active_phase_p("C_SYMBOL_TABLE"))
      activate("C_SYMBOL_TABLE");
  }
  else {
    /* The language is unknown*/
    pips_user_warning("Unknown language initialization\n");
  }
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

    pips_debug(8, "got %s\n", vrn);

    /* For all active  phases*/
    MAPL(pa, {

	r = find_rule_by_phase(STRING(CAR(pa)));

	if (rule_use_resource_produced(r) == true)
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
		return true;
	}, (list) rule_required( r ) );

    }, (list) rule_produced( r ) );

    return false;
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






