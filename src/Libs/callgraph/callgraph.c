/* callgraph.c

   Entry point:   

   Pierre Berthomier, May 1990
   Lei Zhou, January 1991
*/
#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"
#include "list.h"
#include "ri.h"
#include "text.h"
#include "constants.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "properties.h"
#include "prettyprint.h"
#include "ri-util.h"
#include "misc.h"
#include "database.h"     /* DB_PUT_FILE_RESOURCE is defined there */
#include "pipsdbm.h"
#include "resources.h"

#include "transformer.h"
#include "semantics.h"

#include "callgraph.h"

/* get all the callees of the module module_name,return the no-empty
   list of string.
   if the module has callee(s),the first element of the return list
   is the mudule's last callee.
*/

list string_to_callees(module_name)
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

list entity_to_callees(mod)
entity mod;
{
    list callees_list=NIL;
    string module_name = module_local_name(mod);
    list return_list = NIL;

    callees_list = string_to_callees(module_name);
    
    MAPL(ce,{string e = STRING(CAR(ce));
	     return_list = CONS(ENTITY, local_name_to_top_level_entity(e),
				return_list);
	 },callees_list);

    return(return_list);
}

/* 
   callgraph_module_name(margin, module, fp)
*/

void callgraph_module_name(margin, module, fp)
int margin;
entity module;
FILE *fp;
{
    extern int fprintf();

    (void) fprintf(fp,"%*s%s\n",margin,"", module_local_name(module));

    MAPL(pm,{ entity e = ENTITY(CAR(pm));
	      callgraph_module_name(margin + CALLGRAPH_INDENT, e, fp);
	  },
	 entity_to_callees(module) );
}
    
void module_to_callgraph(module)
entity module;
{
    string module_name = module_local_name(module);
    statement s = (statement)db_get_memory_resource(DBR_CODE, module_name, TRUE);
    string filename;
    FILE *fp;

    if ( s == statement_undefined ) {
	pips_error("module_to_callgraph","no statement for module %s\n",
		   module_name);
    }
    else {
	filename = strdup(concatenate(db_get_current_program_directory(), 
					  "/", module_name, ".cg",  NULL));

	fp = safe_fopen(filename, "w");

	callgraph_module_name(0, module, fp);

	safe_fclose(fp, filename);
	DB_PUT_FILE_RESOURCE(DBR_CALLGRAPH_FILE, 
			     strdup(module_name), filename);
    }
}
