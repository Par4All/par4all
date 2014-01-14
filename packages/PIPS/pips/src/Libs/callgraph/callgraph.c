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
/* callgraph.c

   Entry point:

   Pierre Berthomier, May 1990
   Lei Zhou, January 1991
   Guillaume Oget, June 1995
*/

/* To have asprintf(): */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "ri.h"
#include "effects.h"
#include "text.h"
#include "text-util.h"
#include "constants.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "database.h"     /* DB_PUT_FILE_RESOURCE is defined there */
#include "pipsdbm.h"
#include "resources.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "semantics.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "phases.h"
#include "pipsmake.h"

#include "callgraph.h"

/* get all the callees of the module module_name,return the no-empty
   list of string.
   if the module has callee(s),the first element of the return list
   is the mudule's last callee.
*/
list
string_to_callees(const char* module_name)
{
  callees cl = callees_undefined;

  if(false && static_module_name_p(module_name))
    cl = (callees)db_get_memory_resource(DBR_CALLEES,module_name,true);
  else {
    // Should be dealt with in ri-util
    // string ln = global_name_to_user_name(module_name);
    const char* ln = local_name(module_name);
    ln += strspn(ln, MAIN_PREFIX)
      + strspn(ln, BLOCKDATA_PREFIX)
      + strspn(ln, COMMON_PREFIX);
    cl = (callees)db_get_memory_resource(DBR_CALLEES,ln,true);
  }
  return callees_callees(cl);
}


list
entity_to_callees(entity mod)
{
    list callees_list=NIL;
    const char* module_name = entity_name(mod);
    list rl = NIL;

    callees_list = string_to_callees(module_name);

    FOREACH(STRING, e,callees_list)
        rl = CONS(ENTITY, module_name_to_entity(e), rl);

    return rl;
}

typedef struct {
    list sites;
    entity m;
} gather_call_sites_t;

static void gather_call_sites(call c, gather_call_sites_t *p)
{
    if(same_entity_p(call_function(c), p->m))
        p->sites=CONS(CALL,c,p->sites);
}
static void gather_call_sites_in_block(statement s, gather_call_sites_t *p) {
    if(declaration_statement_p(s)) {
        FOREACH(ENTITY,e,statement_declarations(s)) {
            gen_context_recurse(entity_initial(e),p,call_domain,gen_true,gather_call_sites);
        }
    }
}

/**
 * given a list @p callers_statement of module statements
 * returns a list of calls to module @p called_module
 *
 */
list callers_to_call_sites(list callers_statement, entity called_module)
{
    gather_call_sites_t p ={ NIL,called_module };
    FOREACH(STATEMENT,caller_statement,callers_statement)
        gen_context_multi_recurse(caller_statement,&p,
                statement_domain,gen_true,gather_call_sites_in_block,
                call_domain,gen_true,gather_call_sites,0);
    return p.sites;
}

/**
 * given a list @p callers of module name calling module @p called module
 * return a list of their body
 */
list callers_to_statements(list callers)
{
    list statements = NIL;
    FOREACH(STRING,caller_name,callers)
    {
        statement caller_statement=(statement) db_get_memory_resource(DBR_CODE,caller_name,true);
        statements=CONS(STATEMENT,caller_statement,statements);
    }
    return gen_nreverse(statements);
}

/* change the parameter order for function @p module
 * using comparison function @p cmp
 * both compilation unit and callers are touched
 * SG: it may be put in ri-util,  but this would create a dependency from callgraph ...
 */
void sort_parameters(entity module, gen_cmp_func_t cmp) {
    /* retrieve the formal parameters */
    list fn = module_formal_parameters(module);
    /* order them */
    gen_sort_list(fn,cmp);
    /* update offset */
    intptr_t offset=0;
    int reordering[gen_length(fn)];/* holds correspondence between old and new offset */
    FOREACH(ENTITY,f,fn) {
        reordering[formal_offset(storage_formal(entity_storage(f)))-1] = offset;
        formal_offset(storage_formal(entity_storage(f)))=++offset;
    }
    /* update parameter list */
    list parameters = module_functional_parameters(module);
    list new_parameters = NIL;
    for(size_t i=0;i<gen_length(fn);i++) {
        new_parameters=CONS(PARAMETER,PARAMETER(gen_nth((int)reordering[i],parameters)),new_parameters);
    }
    new_parameters=gen_nreverse(new_parameters);
    module_functional_parameters(module)=new_parameters;
    /* change call sites */
    list callers = callees_callees((callees)db_get_memory_resource(DBR_CALLERS,get_current_module_name(), true));
    list callers_statement = callers_to_statements(callers);
    list call_sites = callers_to_call_sites(callers_statement,module);
    /* for each call site , reorder arguments according to table reordering */
    FOREACH(CALL,c,call_sites) {
        list args = call_arguments(c);
        list new_args = NIL;
        for(size_t i=0;i<gen_length(fn);i++) {
            new_args=CONS(EXPRESSION,EXPRESSION(gen_nth((int)reordering[i],args)),new_args);
        }
        new_args=gen_nreverse(new_args);
        gen_free_list(args);
        call_arguments(c)=new_args;
    }
    /* tell dbm of the update */
    for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter))
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, STRING(CAR(citer)),STATEMENT(CAR(siter)));
    db_touch_resource(DBR_CODE,compilation_unit_of_module(get_current_module_name()));

    /* yes! some people use free in pips ! */
    gen_free_list(call_sites);
    gen_free_list(callers_statement);

    gen_free_list(fn);
    gen_free_list(parameters);
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
    const char* module_name = module_resource_name(module);
	char *dir = db_get_current_workspace_directory();
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
	pips_internal_error("unknown callgraph decoration for module %s",
			    module_name);
    }

    print_text(fp, r);
    fprintf(fp, " %s\n", module_name);

    FOREACH(ENTITY, e,entity_to_callees(module))
    {
	const char* n = module_resource_name(e);
	string f = db_get_memory_resource(DBR_CALLGRAPH_FILE, n, true);
	string full = strdup(concatenate(dir, "/", f, NULL));

	safe_append(fp, full, CALLGRAPH_INDENT, true);

	free(full);
    }

    free(dir);
}

bool
module_to_callgraph(
    entity module,
    int decor_type)
{
    string dir, local, full;
    FILE * fp;

    const char *name = module_resource_name(module);
    local = db_build_file_resource_name(DBR_CALLGRAPH_FILE, name, ".cg");
    dir = db_get_current_workspace_directory();
    full = strdup(concatenate(dir, "/", local, NULL));
    free(dir);

    fp = safe_fopen(full, "w");
    callgraph_module_name(module, fp, decor_type);
    safe_fclose(fp, full);
    free(full);

    DB_PUT_FILE_RESOURCE(DBR_CALLGRAPH_FILE, name, local);
    return true;
}



/************************************************************ UPDATE CALLEES */


/** Add a call to a function to a callees list
 */
static void
add_call_to_callees(const call c, callees *current_callees) {
  entity called = call_function(c);
  pips_assert("defined entity", !entity_undefined_p(called));
  pips_debug(8,"considering: %s ->",  entity_local_name(called));
  if (type_functional_p(entity_type(called)) &&
      storage_rom_p(entity_storage(called)) &&
      (value_code_p(entity_initial(called)) ||
       value_unknown_p(entity_initial(called)))) {
    const char* name = entity_local_name(called);
    // Only add the callee if not already here:
    FOREACH(STRING, s, callees_callees(*current_callees))
        if (same_string_p(name, s))
            return;
    pips_debug(8,"adding: %s",  entity_local_name(called));
    callees_callees(*current_callees) =
      CONS(STRING, strdup(name), callees_callees(*current_callees));
  }
  pips_debug(8,"\n");
}

/**
   Add calls hidden in variable declarations to the current callees list
 */
static bool
declaration_statement_add_call_to_callees(const statement s, callees *current_callees)
{
  bool decl_p = declaration_statement_p(s);

  if (decl_p)
    {
      FOREACH(ENTITY, e, statement_declarations(s))
	{
	  if(type_variable_p(entity_type(e)))
	    {
	      value v_init = entity_initial(e);
	      if (value_expression_p(v_init))
		{
		  gen_context_recurse(v_init, current_callees,
				      call_domain, gen_true, add_call_to_callees);
		}
	    }
	}
    }
  return !decl_p; /* currently, declarations are attached to CONTINUE statements */
}

/** Recompute the callees of a module statement.

    @param stat is the module statement

    @return the callees of the module
*/
callees
compute_callees(const statement stat) {
  callees current_callees = make_callees(NIL);
  // Visit all the call site of the module:
  gen_context_multi_recurse(stat, &current_callees,
		      statement_domain, declaration_statement_add_call_to_callees, gen_null,
		      call_domain, gen_true, add_call_to_callees,
		      NULL);
  return current_callees;
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

    SET_FOREACH(string,v1,vertices) {
      _int cmp = 0; /* current module position */
      string source_module = (string) v1; /* gdb does not access v1 */
      callees c = (callees) hash_get(arcs, (void *) v1);
      list destinations = list_undefined;

      if(c == (callees) HASH_UNDEFINED_VALUE) {
	pips_internal_error("Arcs undefined for module %s", source_module);
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
	FOREACH(STRING, v2,destinations) {
	  _int p = 0;

	  if((p = (_int) hash_get(position, v2)) == (_int) HASH_UNDEFINED_VALUE)
	    goto next;
	  cmp = p+1>cmp? p+1 : cmp;
	}
	/* do not know when to perform a put or an update... */
	hash_put(position, v1, (void *) cmp);
	n++;
	nvertices++;
      next: ;
      }
    }
    pips_debug(1, "Iteration %td completed with %td vertices processed\n"
	       "(total number of vertices processed: %td)\n",
	       iter, n, nvertices);
  }

  /* Check that all vertices are associated to a position... which only is
     true if no recursive cycle exists. */
  ifdebug(7) {
    SET_FOREACH(string,v1,vertices) {
      _int p = (_int)hash_get(position,  v1);
      pips_assert("p is defined", p != (_int) HASH_UNDEFINED_VALUE);
    }
  }

  n = 0;
  SET_FOREACH(string,v1,vertices) {
    if(hash_get(position,  v1) == HASH_UNDEFINED_VALUE) {
      pips_user_warning("Module %s might be part of a recursive call cycle\n", v1);
      n++;
    }
  }
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
    const char* module_name = gen_array_item(module_array, i);
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
    const char* module_name = gen_array_item(module_array, i);
    callees called_modules = callees_undefined;
    list ccm = list_undefined;

    called_modules = (callees)
      db_get_memory_resource(DBR_CALLEES, module_name, true);
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

	pips_user_warning("no source file for module %s, let's try so synthesize code\n",
			  module_called);
	reset_current_phase_context();

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
	    db_get_memory_resource(DBR_CALLEES, module_called, true);
	  hash_put(module_callees, (void *) module_called, (void *) c);
      rmake(DBR_CALLERS, module_called);
	}
	else {
	  /* You cannot call pips_user_error() again, as it has just been
             called by rmake via apply_a_rule()*/
	  /*
	    pips_user_error("Provide or let PIPS synthesize source code for module %s\n",
	    module_called);*/
	  set_current_phase_context(BUILDER_CALLGRAPH, name);

	  return false;
	}
	/* pop_pips_current_computation(DBR_CALLEES, module_called); */
	/* reset_pips_current_computation(); */
	set_current_phase_context(BUILDER_CALLGRAPH, name);
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
  if(true) {
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
  return true;
}
