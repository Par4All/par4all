/* callgraph.c

   Entry point:   

   Pierre Berthomier, May 1990
   Lei Zhou, January 1991
   Guillaume Oget, June 1995
*/
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "constants.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "properties.h"
#include "prettyprint.h"
#include "ri-util.h"
#include "misc.h"
#include "database.h"     /* DB_PUT_FILE_RESOURCE is defined there */
#include "pipsdbm.h"
#include "resources.h"
#include "effects.h"
#include "regions.h"
#include "transformer.h"
#include "semantics.h"
#include "complexity_ri.h"
#include "complexity.h"

#include "callgraph.h"

/* get all the callees of the module module_name,return the no-empty
   list of string.
   if the module has callee(s),the first element of the return list
   is the mudule's last callee.
*/

list string_to_callees(module_name)
string module_name;
{
    /* GO 6/7/95
       This code was fool. It used to store in a hash table
       things calculated every time ...
       
       Moreover if a ressource is not valid anymore a core dumped
       comes.
       
       
       callees cl;
       static hash_table hash_table_to_callees_string;
       static bool hash_table_is_created = FALSE;
       list callees_list=NIL;
       
       cl = (callees)db_get_memory_resource(DBR_CALLEES,module_name,TRUE);
       
       if ( !hash_table_is_created ) {
       hash_table_to_callees_string = hash_table_make(hash_pointer, 0);
       hash_table_is_created = TRUE;
       }
       
       callees_list=(list)hash_get(hash_table_to_callees_string,module_name);
       
       if ( callees_list == (list)HASH_UNDEFINED_VALUE ) {
       callees_list = callees_callees(cl);
       hash_put(hash_table_to_callees_string, module_name, 
       (char *)callees_list);
       }
       
       I prefer this with no useless Htable ...
       
       */

    callees cl;

    cl = (callees)db_get_memory_resource(DBR_CALLEES,module_name,TRUE);
       
    return callees_callees(cl);
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

void callgraph_module_name(margin, module, fp, decor_type)
int margin;
entity module;
FILE *fp;
int decor_type;
{
    string module_name = module_local_name(module);

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
	pips_error("callgraph_module_name",
		   "unknown callgraph decoration for module %s\n",
		   module_name);
    }

    print_text(fp, r);
    (void) fprintf(fp,"%*s%s\n",margin,"", module_name);

    MAPL(pm,{
	entity e = ENTITY(CAR(pm));
	callgraph_module_name(margin + CALLGRAPH_INDENT,
			      e,fp,decor_type);
    }, entity_to_callees(module) );
}
    
bool module_to_callgraph(module,decor_type)
entity module;
int decor_type;
{
    string module_name = module_local_name(module);
    statement s = (statement)db_get_memory_resource(DBR_CODE,
						    module_name, TRUE);

    if ( s == statement_undefined ) {
	pips_error("module_to_callgraph","no statement for module %s\n",
		   module_name);
    } else {
	FILE *fp;
	string localfilename = strdup(concatenate(module_name, ".cg",  NULL));
	string filename = strdup(concatenate(db_get_current_workspace_directory(), 
				      "/", localfilename,  NULL));

	fp = safe_fopen(filename, "w");

	callgraph_module_name(0, module, fp, decor_type);

	safe_fclose(fp, filename);
	DB_PUT_FILE_RESOURCE(DBR_CALLGRAPH_FILE, 
			     strdup(module_name), localfilename);
	free(filename);
    }
    return TRUE;
}
