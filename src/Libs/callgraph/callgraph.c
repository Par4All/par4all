/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* callgraph.c

   Entry point:

   Pierre Berthomier, May 1990
   Lei Zhou, January 1991
   Guillaume Oget, June 1995
*/

/* To have asprintf(): */
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "constants.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "properties.h"
#include "ri-util.h"
#include "prettyprint.h"
#include "misc.h"
#include "database.h"     /* DB_PUT_FILE_RESOURCE is defined there */
#include "pipsdbm.h"
#include "resources.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "transformer.h"
#include "semantics.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "phases.h"

#include "callgraph.h"

/* get all the callees of the module module_name,return the no-empty
   list of string.
   if the module has callee(s),the first element of the return list
   is the mudule's last callee.
*/
list
string_to_callees(string module_name)
{
  callees cl = callees_undefined;

  if(FALSE && static_module_name_p(module_name))
    cl = (callees)db_get_memory_resource(DBR_CALLEES,module_name,TRUE);
  else {
    // Should be dealt with in ri-util
    // string ln = global_name_to_user_name(module_name);
    string ln = local_name(module_name);
    ln += strspn(ln, MAIN_PREFIX)
      + strspn(ln, BLOCKDATA_PREFIX)
      + strspn(ln, COMMON_PREFIX);
    cl = (callees)db_get_memory_resource(DBR_CALLEES,ln,TRUE);
  }
  return callees_callees(cl);
}


list
entity_to_callees(entity mod)
{
    list callees_list=NIL;
    string module_name = entity_name(mod);
    list rl = NIL;

    callees_list = string_to_callees(module_name);

    MAP(STRING, e,
	rl = CONS(ENTITY, module_name_to_entity(e), rl),
	callees_list);

    return rl;
}


/*
   callgraph_module_name(margin, module, fp)
*/
static void
callgraph_module_name(
    entity module,
    FILE * fp,
    int decor_type)
{
    string module_name = module_resource_name(module),
	dir = db_get_current_workspace_directory();
    text r = make_text(NIL);

    switch (decor_type) {
    case CG_DECOR_NONE:
	break;
    case CG_DECOR_COMPLEXITIES:
	MERGE_TEXTS(r,get_text_complexities(module_name));
	break;
    case CG_DECOR_TRANSFORMERS:
	MERGE_TEXTS(r,get_text_transformers(module_name));
	break;
    case CG_DECOR_PRECONDITIONS:
	MERGE_TEXTS(r,get_text_preconditions(module_name));
	break;
    case CG_DECOR_PROPER_EFFECTS:
	MERGE_TEXTS(r,get_text_proper_effects(module_name));
	break;
    case CG_DECOR_CUMULATED_EFFECTS:
	MERGE_TEXTS(r,get_text_cumulated_effects(module_name));
	break;
    case CG_DECOR_REGIONS:
	MERGE_TEXTS(r,get_text_regions(module_name));
	break;
    case CG_DECOR_IN_REGIONS:
	MERGE_TEXTS(r,get_text_in_regions(module_name));
	break;
    case CG_DECOR_OUT_REGIONS:
	MERGE_TEXTS(r,get_text_out_regions(module_name));
	break;
    default:
	pips_internal_error("unknown callgraph decoration for module %s\n",
			    module_name);
    }

    print_text(fp, r);
    fprintf(fp, " %s\n", module_name);

    MAP(ENTITY, e,
    {
	string n = module_resource_name(e);
	string f = db_get_memory_resource(DBR_CALLGRAPH_FILE, n, TRUE);
	string full = strdup(concatenate(dir, "/", f, NULL));

	safe_append(fp, full, CALLGRAPH_INDENT, TRUE);

	free(full);
    },
	entity_to_callees(module));

    free(dir);
}

bool
module_to_callgraph(
    entity module,
    int decor_type)
{
    string name, dir, local, full;
    FILE * fp;

    name = module_resource_name(module);
    local = db_build_file_resource_name(DBR_CALLGRAPH_FILE, name, ".cg");
    dir = db_get_current_workspace_directory();
    full = strdup(concatenate(dir, "/", local, NULL));
    free(dir);

    fp = safe_fopen(full, "w");
    callgraph_module_name(module, fp, decor_type);
    safe_fclose(fp, full);
    free(full);

    DB_PUT_FILE_RESOURCE(DBR_CALLGRAPH_FILE, name, local);
    return TRUE;
}



/************************************************************ UPDATE CALLEES */


/** Add a call to a function to a callee list
 */
static void
add_call_to_callees(const call c, callees *current_callees) {
  entity called = call_function(c);
  pips_assert("defined entity", !entity_undefined_p(called));

  if (type_functional_p(entity_type(called)) &&
      storage_rom_p(entity_storage(called)) &&
      (value_code_p(entity_initial(called)) ||
       value_unknown_p(entity_initial(called)))) {
    string name = entity_local_name(called);
    // Only add the callee if not already here:
    MAP(STRING, s,
	if (same_string_p(name, s))
	  return,
	callees_callees(*current_callees));
    callees_callees(*current_callees) =
      CONS(STRING, strdup(name), callees_callees(*current_callees));
  }
}


/** Recompute the callees of a module statement.

    @param stat is the module statement

    @return the callees of the module
*/
callees
compute_callees(const statement stat) {
  callees result;
  callees current_callees = make_callees(NIL);
  // 
  gen_context_recurse(stat, &current_callees,
		      call_domain, gen_true, add_call_to_callees);
  result = current_callees;
  current_callees = callees_undefined;
  return result;
}


/* Global computation of CALLERS, HEIGHT and DEPTH
 */


static void transitive_positions(set vertices,
				 hash_table arcs,
				 hash_table position)
{
  _int n = 1; /* Number of modules processed at the previous iteration */
  _int iter = 0;
  _int nvertices = 0;

  while(n>0) {
    n = 0;
    iter++;

    SET_MAP(v1, {
      _int cmp = 0; /* current module position */
      string source_module = (string) v1; /* gdb does not access v1 */
      callees c = (callees) hash_get(arcs, (void *) v1);
      list destinations = list_undefined;

      if(c == (callees) HASH_UNDEFINED_VALUE) {
	pips_internal_error("Arcs undefined for module %s\n", source_module);
      }
      else {
	destinations = callees_callees(c);
      }

      if(hash_get(position, v1) != HASH_UNDEFINED_VALUE)
	/* already processed */
	;
      else if(ENDP(destinations)) {
	hash_put(position, v1, (void *) cmp);
	n++;
	nvertices++;
      }
      else {
	MAP(STRING, v2, {
	  _int p = 0;

	  if((p = (_int) hash_get(position, v2)) == (_int) HASH_UNDEFINED_VALUE)
	    goto next;
	  cmp = p+1>cmp? p+1 : cmp;
	}, destinations);
	/* do not know when to perform a put or an update... */
	hash_put(position, v1, (void *) cmp);
	n++;
	nvertices++;
      next: ;
      }
    }, vertices);
    pips_debug(1, "Iteration %td completed with %td vertices processed\n"
	       "(total number of vertices processed: %td)\n",
	       iter, n, nvertices);
  }

  /* Check that all vertices are associated to a position... which only is
     true if no recursive cycle exists. */
  ifdebug(7) {
    SET_MAP(v1, {
      _int p = (_int)hash_get(position, (void *) v1);
      pips_assert("p is defined", p != (_int) HASH_UNDEFINED_VALUE);
    }, vertices);
  }

  n = 0;
  SET_MAP(v1, {
    if(hash_get(position, (void *) v1) == HASH_UNDEFINED_VALUE) {
      pips_user_warning("Module %s might be part of a recursive call cycle\n", (string) v1);
      n++;
    }
  }, vertices);
  if(n!=0)
      pips_user_warning("%td module could not be given a position in the call graph,"
			" probably because of a recursive call cycle.\n", n);
}


/* callgraph computes the caller list of each module, using the callees
 * list of them. As a side effect, it also computes their heights and
 * depths in the call graph and detects recursive call cycles.
 *
 * callgraph is not able to generate missing source code.
 *
 * Note: although callgraph is directly and internally used by pipsmake
 * which needs to know about the callers of a module to chain its rules,
 * it might be better to store it in a future (new) callgraph library (FI)
 *
 * Argument "name" is not used. It is instantiated as a specific module
 * by make() but this routine process the whole program.
 *
 */
bool callgraph(string name)
{
  gen_array_t module_array = db_get_module_list();
  int nmodules = gen_array_nitems(module_array);
  int i = 0;
  int n_new_modules = 0; /* Number of modules called whose source code is
                            missing but synthesized by PIPS */
  /* Should we deal with strings or with entities? */
  set modules = set_make(set_string);
  hash_table module_callers = hash_table_make(hash_string, 2*nmodules);
  hash_table module_callees =  hash_table_make(hash_string, 2*nmodules);
  hash_table module_depth =  hash_table_make(hash_string, 2*nmodules);
  hash_table module_height =  hash_table_make(hash_string, 2*nmodules);

  pips_assert("To silence gcc", name==name);
  pips_assert("The workspace contains at least one module", nmodules>0);

  /* Define the module_set and initialize the module callers, except for
     modules whose source code is missing. */
  for(i=0; i<nmodules; i++) {
    string module_name = gen_array_item(module_array, i);
    callees c = callees_undefined;

    set_add_element(modules, modules, module_name);

    c = make_callees(NIL);
    hash_put(module_callers, (void *) module_name, (void *) c);
  }

  /* Compute iteratively the callers from the callees. Synthesize missing
   * codes if necessary and if the corresponding property is set.
   *
   * The number of iterations depends on the order of the modules in the hash-tables.
   *
   * Simple but inefficient implementation for Cathar-2
   * */
  for(i=0; i<nmodules; i++) {
    string module_name = gen_array_item(module_array, i);
    callees called_modules = callees_undefined;
    list ccm = list_undefined;

    called_modules = (callees)
      db_get_memory_resource(DBR_CALLEES, module_name, TRUE);
    hash_put(module_callees, (void *) module_name, (void *) called_modules);

    for( ccm = callees_callees(called_modules);
	 !ENDP(ccm);
	 POP(ccm)) {
      string module_called = STRING(CAR(ccm));
      callees c = callees_undefined;
      bool found = set_belong_p(modules, module_called);

      /* Should not be an error as PIPS can synthesize missing code
	 and does it elsewhere... */
      if(!found) {
	extern bool rmake(string, string);

	pips_user_warning("no source file for module %s, let's try so synthesize code\n",
			  module_called);
	reset_pips_current_computation();
	/* set_pips_current_computation(DBR_CALLEES, module_called); */
	/* push_pips_current_computation(DBR_CALLEES, module_called); */
	if(rmake(DBR_CALLEES, module_called)) {
	  /* It has no callees to exploit anyway; it does not matter that
             it is not looped over by the main loop. module_callers is
             going to be updated and will be used to store the results. */
	  callees c = callees_undefined;

	  n_new_modules++;

	  set_add_element(modules, modules, module_called);

	  c = make_callees(NIL);
	  hash_put(module_callers, (void *) module_called, (void *) c);
	  c =  (callees)
	    db_get_memory_resource(DBR_CALLEES, module_called, TRUE);
	  hash_put(module_callees, (void *) module_called, (void *) c);
	}
	else {
	  /* You cannot call pips_user_error() again, as it has just been
             called by rmake via apply_a_rule()*/
	  /*
	    pips_user_error("Provide or let PIPS synthesize source code for module %s\n",
	    module_called);*/
	  set_pips_current_computation(BUILDER_CALLGRAPH, name);
	  return FALSE;
	}
	/* pop_pips_current_computation(DBR_CALLEES, module_called); */
	/* reset_pips_current_computation(); */
	set_pips_current_computation(BUILDER_CALLGRAPH, name);
      }

      c = (callees) hash_get(module_callers, (void *) module_called);
      pips_assert("callers are always initialized",
		  c != (callees) HASH_UNDEFINED_VALUE);
      callees_callees(c) =
	gen_nconc(callees_callees(c),
		  CONS(STRING, strdup(module_name), NIL));
    }
  }

  pips_debug(1, "Compute heights from callee arcs");
  transitive_positions(modules, module_callees, module_height);
  pips_debug(1, "Compute depths from caller arcs");
  transitive_positions(modules, module_callers, module_depth);

  ifdebug(7) {
    HASH_MAP(module_name, callers,
    {
      pips_debug(7, "adding %p as %s for module %s\n",
		 callers, DBR_CALLERS, (string) module_name);
      print_callees((callees) callers);
      fprintf(stderr, "HEIGHT = %td\n", (_int) hash_get(module_height, (string) module_name));
      fprintf(stderr, "DEPTH = %td\n", (_int) hash_get(module_depth, (string) module_name));
    }, module_callers);
  }

  /* Since Fabien is not available to help with pipsdbm... Let's move on with Cathare-2! */
  /*
  if(TRUE) {
    FILE * h_file = safe_fopen("height", "w");
    FILE * d_file = safe_fopen("depth", "w");
    HASH_MAP(module_name, callers,
    {
      fprintf(h_file,"%s\t%d\n", (string) module_name,
	      (int) hash_get(module_height, (string) module_name));
      fprintf(d_file,"%s\t%d\n", (string) module_name,
	      (int) hash_get(module_depth, (string) module_name));
    }, module_callers);
    safe_fclose(h_file, "height");
    safe_fclose(d_file, "depth");
  }
  */

  i=0;

  HASH_MAP(module_name, callers,
  {
    char *depth;
    char *height;

    DB_PUT_MEMORY_RESOURCE(DBR_CALLERS, (string) module_name, (void *) callers);

    asprintf(&depth,"%td", (_int) hash_get(module_depth, (string) module_name));
    DB_PUT_MEMORY_RESOURCE(DBR_DEPTH, (string) module_name, (void *) depth);

    asprintf(&height,"%td", (_int) hash_get(module_height, (string) module_name));
    DB_PUT_MEMORY_RESOURCE(DBR_HEIGHT, (string) module_name, (void *) height);
    i++;
  }, module_callers);

  pips_assert("The number of modules is unchanged", i==nmodules+n_new_modules);

  hash_table_free(module_callees);
  hash_table_free(module_callers);
  hash_table_free(module_height);
  hash_table_free(module_depth);
  set_free(modules);
  gen_array_full_free(module_array);
  return TRUE;
}
