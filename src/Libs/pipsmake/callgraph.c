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
  int n = 1;
  int iter = 0;
  int nvertices = 0;

  while(n>0) {
    n = 0;
    iter++;

    SET_MAP(v1, {
      int cmp = 0; /* current module position */
      list destinations = callees_callees((callees) hash_get(arcs, (void *) v1));

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
    fprintf(stderr, "Iteration %d completed with %d vertices processed\n"
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
  SET_MAP(v1, {
    if(hash_get(position, (void *) v1) == HASH_UNDEFINED_VALUE)
      pips_user_warning("Module %s might be part of a recursive call cycle\n", (string) v1);
  }, vertices);
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
  /* Should we deal with strings or with entities? */
  set modules = set_make(set_string);
  hash_table module_callers = hash_table_make(hash_string, 2*nmodules);
  hash_table module_callees =  hash_table_make(hash_string, 2*nmodules);
  hash_table module_depth =  hash_table_make(hash_string, 2*nmodules);
  hash_table module_height =  hash_table_make(hash_string, 2*nmodules);

  pips_assert("To silence gcc", name==name);
  pips_assert("The workspace contains at least one module", nmodules>0);

  /* Define the module_set and initialize the module callers*/
  for(i=0; i<nmodules; i++) {
    string module_name = gen_array_item(module_array, i);
    callees c = callees_undefined;

    set_add_element(modules, modules, module_name);

    c = make_callees(NIL);
    hash_put(module_callers, (void *) module_name, (void *) c);
  }

  /* Compute iteratively the callers from the callees. */
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
      if(!found) pips_user_error("no source file for module %s\n", 
				 module_called);

      c = (callees) hash_get(module_callers, (void *) module_called);
      pips_assert("callers are always initialized",
		  c != (callees) HASH_UNDEFINED_VALUE);
      callees_callees(c) =
	gen_nconc(callees_callees(c), 
		  CONS(STRING, strdup(module_name), NIL));
    }
  }

  /*
  transitive_positions(modules, module_callees, module_height);
  transitive_positions(modules, module_callers, module_depth);
  */

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
    
  i=0;

  HASH_MAP(module_name, callers,
  {
    DB_PUT_MEMORY_RESOURCE(DBR_CALLERS, (string) module_name, (void *) callers);
    DB_PUT_MEMORY_RESOURCE(DBR_DEPTH, (string) module_name, (void *) strdup("0"));
    DB_PUT_MEMORY_RESOURCE(DBR_HEIGHT, (string) module_name, (void *) strdup("1"));
    i++;
  }, module_callers);

  pips_assert("The number of modules is unchanged", i==nmodules);

  hash_table_free(module_callees);
  hash_table_free(module_callers);
  hash_table_free(module_height);
  hash_table_free(module_depth);
  set_free(modules);
  gen_array_full_free(module_array);
  return TRUE;
}
