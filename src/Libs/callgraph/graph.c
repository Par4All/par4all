/*
 * $Id$
 *
 * $Log: graph.c,v $
 * Revision 1.1  1998/03/19 17:09:21  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "misc.h"
#include "resources.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

/* build for module name a node and link to its successors.
 */
static bool 
node(FILE * out, string name)
{
    callees module_callees;
    bool first=TRUE;

    module_callees = (callees)
	db_get_memory_resource(DBR_CALLEES, name, TRUE);

    fprintf(out, "l(\"%s\",n(\"\",[],[\n", name);

    MAP(STRING, module_called, 
    {
	if (!first) fprintf(out, ",\n");
	first=FALSE;
	fprintf(out, "l(\"%s->%s\",e(\"\",[],r(\"%s\")))\n", 
		name, module_called, module_called);
    },
        callees_callees(module_callees));

    fprintf(out, "\n]))");
    return TRUE;
}

bool
graph_of_calls(string name)
{
    gen_array_t modules = db_get_module_list();
    int n = gen_array_nitems(modules), i;
    FILE * out;
    bool first = TRUE;
    string dir_name, file_name, full_name, module;

    dir_name = db_get_current_workspace_directory();

    file_name = db_build_file_resource_name
	(DBR_DVCG_FILE, WORKSPACE_PROGRAM_SPACE, ".daVinci");
    
    full_name = strdup(concatenate(dir_name, "/", file_name, 0));

    out = safe_fopen(full_name, "w");

    fprintf(out, "[\n");

    for (i=0; i<n; i++)
    {
	if (!first) fprintf(out, ",\n");
	first=FALSE;
	module = gen_array_item(modules, i);
	node(out, module);
    }

    fprintf(out, "]\n");
    safe_fclose(out, full_name);
    gen_array_free(modules);

    DB_PUT_FILE_RESOURCE(DBR_DVCG_FILE, PROGRAM_RESOURCE_OWNER, file_name);
    return TRUE;
}
