/* toposort.c

   Entry point:   list topologically_sorted_module_list(mod)

   returns the list of subroutines/functions called in the module mod 
   sorted as follow: we recursively build the graph of calls, taking
   mod for root, and we apply on it the topological sort.
   The first of the list is the module mod itself.

   Pierre Berthomier, May 1990
   Lei Zhou, January 1991
*/
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"

#include "constants.h"
#include "control.h"
#include "properties.h"
#include "ri-util.h"
#include "prettyprint.h"
#include "misc.h"


#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "transformer.h"
#include "semantics.h"

#include "icfg.h"

/* get all the callees of the module module_name,return the no-empty
   list of string.
   if the module has callee(s),the first element of the return list
   is the module's last callee.
*/

list module_name_to_callees(module_name)
string module_name;
{
    callees cl;
    static hash_table hash_table_to_callees_string;
    static bool hash_table_is_created = FALSE;
    list callees_list=NIL;

    cl = (callees)db_get_memory_resource(DBR_CALLEES,module_name,TRUE);

    if ( !hash_table_is_created ) {
	hash_table_to_callees_string = hash_table_make(hash_pointer, 0);
	hash_table_is_created = TRUE;
    }
    
    callees_list = (list)hash_get(hash_table_to_callees_string, module_name);

    if ( callees_list == (list)HASH_UNDEFINED_VALUE ) {
	callees_list = callees_callees(cl);
	hash_put(hash_table_to_callees_string, module_name, 
		 (char *)callees_list);
    }

    return(callees_list);
}

list module_to_callees(mod)
entity mod;
{
    list return_list = NIL;
    list callees_list = code_declarations(entity_code(mod));
    
    MAPL(ce,{
	entity e = ENTITY(CAR(ce));
	if ( type_functional_p(entity_type(e)) )
	    return_list = CONS(ENTITY, 
			       local_name_to_top_level_entity(entity_local_name(e)),
			       return_list);
    },callees_list);

    return(return_list);
}

void topological_number_assign_to_module(hash_module_to_depth, mod, n)
hash_table hash_module_to_depth;
entity mod;
int n;
{
    int depth = (int) hash_get(hash_module_to_depth, (char *) mod);
    list callees_list = module_to_callees(mod);

    if ((depth == (int) ICFG_NOT_FOUND) || (depth < n))
	hash_put(hash_module_to_depth, (char *) mod, (char *) n);
	      
    if ( callees_list != NIL ) {
    /* assigns depth n+1 to callees of current module */
	MAPL(pm,
            { entity e = ENTITY(CAR(pm));
	      topological_number_assign_to_module(hash_module_to_depth, e, n+1);
       	    },
	 callees_list);
    }
}

list module_list_sort(hash_module_to_depth, current_list, mod, n)
hash_table hash_module_to_depth;
list current_list;
entity mod;
int n;
{
    list callees_list = NIL;
    static list same_depth_list = NIL;
    int depth;
	
    /* create the callees list whose caller has the depth n */
    if ( same_depth_list == NIL )
	callees_list = module_to_callees(mod);
    else {
	MAPL(pm,{ entity e = ENTITY(CAR(pm));
		  callees_list = gen_nconc(callees_list,
					   gen_copy_seq(module_to_callees(e)));
		},
	     same_depth_list);
    }

    /* free the same_depth_list for later use */

    same_depth_list = NIL;

    /* create same_depth_list whose depth is n+1 */
    if ( callees_list != NIL ) {
	MAPL(pm,{ entity e = ENTITY(CAR(pm));
	          depth = (int) hash_get(hash_module_to_depth, (char *) e);
	          if ( depth == n+1 ) {
		      same_depth_list = gen_nconc(same_depth_list, 
					     CONS(ENTITY, e, NIL));
		      hash_put(hash_module_to_depth, 
			       (char *) e, (char *) -1);
		  }
	        },
	     callees_list);

	/* concatenate former current_list with same_depth_list */
	current_list = gen_nconc(current_list, same_depth_list);

	/* use module_list_sort recursively */
	current_list = module_list_sort(hash_module_to_depth, 
					   current_list,
					   ENTITY(CAR(same_depth_list)), 
					   n+1);
    }
    return (current_list);
}

list topologically_sorted_module_list(mod)
entity mod;
{
    /* "depth" of subroutine or function for topological sort */
    hash_table hash_module_to_depth = (hash_table) NULL;
    list sorted_list;
    
    hash_module_to_depth = hash_table_make(hash_pointer, 0);
    hash_dont_warn_on_redefinition();

    topological_number_assign_to_module(hash_module_to_depth, mod, 0);

    sorted_list = module_list_sort(hash_module_to_depth, NIL, mod, 0);

    pips_assert("free_hash_table",
		hash_module_to_depth != (hash_table) NULL);
    hash_table_free(hash_module_to_depth);
    hash_module_to_depth = (hash_table) NULL;

    return (sorted_list);
} 

void print_module_name_to_toposorts(module_name)
string module_name;
{
    list sorted_list = NIL;
    string filename;
    FILE *fp;
    entity mod = local_name_to_top_level_entity(module_name);

    pips_assert("print_module_name_to_toposorts", mod != entity_undefined &&
		entity_module_p(mod));

    fprintf(stderr, "topological-sorting callees for %s ...\n",
	    entity_name(mod));

    debug_on(ICFG_DEBUG_LEVEL);

    sorted_list = (list) topologically_sorted_module_list(mod);
    filename = strdup(concatenate(db_get_current_workspace_directory(), 
				  "/", module_name, ".topo",NULL));

    fp = safe_fopen(filename, "w");

    MAPL(pm,
         { fprintf(fp, "%s\n",entity_name(ENTITY(CAR(pm)))); },
	 sorted_list);

    safe_fclose(fp, filename);
    fprintf(stderr, "result written to %s\n", filename);

    debug_off();
}
