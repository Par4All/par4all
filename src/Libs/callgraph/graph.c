/*
 * $Id$
 *
 * Build a graph for the callgraph, instead of a tree.
 * Some other formats could be thought of, maybe?
 * Code by Corinne Ancourt and Fabien Coelho.
 *
 * $Log: graph.c,v $
 * Revision 1.2  1998/03/19 17:45:46  coelho
 * this version seems ok.
 *
 * Revision 1.1  1998/03/19 17:09:21  coelho
 * Initial revision
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "resources.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

#define DV_SUFFIX ".daVinci"

/* Build for module name a node and link to its successors.
 * It could be a per module resource, however the callgraph
 * is not expected to change often, and we can avoid to manipulate
 * the many small files.
 */
static void
node(FILE * out, string name)
{
    bool first=TRUE;
    callees module_callees;

    module_callees = (callees) db_get_memory_resource(DBR_CALLEES, name, TRUE);

    /* daVinci node prolog. */
    fprintf(out, "l(\"%s\",n(\"\",[a(\"OBJECT\",\"%s\")],[\n", name, name);

    /* one edge per callee */
    MAP(STRING, module_called, 
    {
	if (!first) fprintf(out, ",\n");
	first=FALSE;
	fprintf(out, "l(\"%s->%s\",e(\"\",[],r(\"%s\")))\n", 
		name, module_called, module_called);
    },
        callees_callees(module_callees));

    /* node epilog */
    fprintf(out, "\n]))");
}

/* To be called by pipsmake.
 * Generate a global resource, hence name is ignored.
 */
bool
graph_of_calls(string name)
{
    gen_array_t modules = db_get_module_list();
    int n = gen_array_nitems(modules), i;
    FILE * out;
    bool first = TRUE;
    string dir_name, file_name, full_name;

    /* build file name... and open it. */
    dir_name = db_get_current_workspace_directory();

    file_name = db_build_file_resource_name
	(DBR_DVCG_FILE, PROGRAM_RESOURCE_OWNER, DV_SUFFIX);
    
    full_name = strdup(concatenate(dir_name, "/", file_name, 0));

    out = safe_fopen(full_name, "w");

    /* prolog, per module stuff, epilog. */
    fprintf(out, "[\n");

    for (i=0; i<n; i++)
    {
	if (!first) fprintf(out, ",\n");
	first=FALSE;
	node(out, gen_array_item(modules, i));
    }

    fprintf(out, "]\n");

    /* close and clean... */
    safe_fclose(out, full_name);
    gen_array_free(modules);
    free(dir_name), dir_name=NULL;
    free(full_name), full_name=NULL;

    /* put resulting resource into pipsdbm. */
    DB_PUT_FILE_RESOURCE(DBR_DVCG_FILE, PROGRAM_RESOURCE_OWNER, file_name);

    return TRUE;
}
