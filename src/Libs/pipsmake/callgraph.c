/* Global computation of CALLERS, HEIGHT and DEPTH
 *
 * $Id$
 *
 * $Log: callgraph.c,v $
 * Revision 1.9  2003/06/27 12:10:08  irigoin
 * First version computing HEIGHT and DEPTh as well as CALLERS
 *
 *
 */


#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"

/* FI: just of call to rmake */
/* #include "pipsmake.h" */
#include "phases.h"

static void print_callees(callees c)
{
  list l = callees_callees(c);

  MAP(STRING, mn, {
    printf("%s\n", mn);
  }, l);
}

static void transitive_positions(set vertices,
				 hash_table arcs,
				 hash_table position)
{
  int n = 1; /* Number of modules processed at the previous iteration */
  int iter = 0;
  int nvertices = 0;

  while(n>0) {
    n = 0;
    iter++;

    SET_MAP(v1, {
      int cmp = 0; /* current module position */
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
	  int p = 0;

	  if((p = (int) hash_get(position, v2)) == (int) HASH_UNDEFINED_VALUE)
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
    pips_debug(1, "Iteration %d completed with %d vertices processed\n"
	       "(total number of vertices processed: %d)\n", iter, n, nvertices);
  }

  /* Check that all vertices are associated to a position... which only is
     true if no recursive cycle exists. */
  ifdebug(7) {
    SET_MAP(v1, {
      int p = (int) hash_get(position, (void *) v1);
      pips_assert("p is defined", p != (int) HASH_UNDEFINED_VALUE);
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
      pips_user_warning("%d module could not be given a position in the call graph,"
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
      fprintf(stderr, "HEIGHT = %d\n", (int) hash_get(module_height, (string) module_name));
      fprintf(stderr, "DEPTH = %d\n", (int) hash_get(module_depth, (string) module_name));
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
    char depth[13];
    char height[13];

    DB_PUT_MEMORY_RESOURCE(DBR_CALLERS, (string) module_name, (void *) callers);

    sprintf(depth,"%d", (int) hash_get(module_depth, (string) module_name));
    DB_PUT_MEMORY_RESOURCE(DBR_DEPTH, (string) module_name, (void *) strdup(depth));

    sprintf(height,"%d", (int) hash_get(module_height, (string) module_name));
    DB_PUT_MEMORY_RESOURCE(DBR_HEIGHT, (string) module_name, (void *) strdup(height));
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
