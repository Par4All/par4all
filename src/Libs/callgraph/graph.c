/*
 * $Id$
 *
 * Build a graph for the callgraph, instead of a tree.
 * Some other formats could be thought of, maybe?
 * Code by Corinne Ancourt and Fabien Coelho.
 *
 * $Log: graph.c,v $
 * Revision 1.6  1998/11/24 11:55:26  coelho
 * lazy use of resource, not to core on resources that are not available.
 *
 * Revision 1.5  1998/04/14 19:01:52  coelho
 * linear.h
 *
 * Revision 1.4  1998/03/20 08:28:29  coelho
 * prettier, and more comments.
 *
 * Revision 1.3  1998/03/19 20:18:49  coelho
 * graph_of_calls moved to full_graph_of_calls.
 * new graph_of_calls deal with a module.
 * improve output.
 *
 * Revision 1.2  1998/03/19 17:45:46  coelho
 * this version seems ok.
 *
 * Revision 1.1  1998/03/19 17:09:21  coelho
 * Initial revision
 */

#include <stdio.h>
#include <string.h>

#include "linear.h"

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
    list lcallees = NIL;

    if (db_resource_p(DBR_CALLEES, name)) /* lazy callees. */
      lcallees = callees_callees((callees) 
	db_get_memory_resource(DBR_CALLEES, name, TRUE));

    /* daVinci node prolog. */
    fprintf(out, "l(\"%s\",n(\"\",[a(\"OBJECT\",\"%s\")],[", name, name);

    /* one edge per callee */
    MAP(STRING, module_called, 
    {
	if (!first) fprintf(out, ",\n");
	else { fprintf(out, "\n"); first=FALSE; }
	fprintf(out, " l(\"%s->%s\",e(\"\",[],r(\"%s\")))", 
		name, module_called, module_called);
    },
        lcallees);

    /* node epilog */
    fprintf(out, "]))");
}

/* static function to store whether a module has been seen during the 
 * recursive generation of the daVinci file.
 */
static hash_table seen = hash_table_undefined;
static bool first_seen = FALSE;
static void init_seen(void) { first_seen = FALSE;
    seen = hash_table_make(hash_string, 0); }
static void close_seen(void) { 
    hash_table_free(seen); seen = hash_table_undefined; }
static void set_as_seen(string m) { hash_put(seen, (char*) m, (char*) 1); }
static bool seen_p(string m){ return hash_defined_p(seen, (char*)m); }

/* generates into "out" a davinci node for module "name", 
 * and recurse to its not yet seen callees.
 */
static void
recursive_append(FILE* out, string name)
{
    if (seen_p(name)) return;
    /* else */
    if (first_seen) fprintf(out, ",\n");
    else first_seen = TRUE;
    node(out, name);
    set_as_seen(name);

    /* the resource may not have been defined, for instance if the
     * code was not parsed, because the %ALL dependence is limited.
     */
    if (db_resource_p(DBR_CALLEES, name))
    {
      callees l = (callees) db_get_memory_resource(DBR_CALLEES, name, TRUE);
      MAP(STRING, c, recursive_append(out, c), callees_callees(l));
    }
}

/* to be called by pipsmake.
 * builds the daVinci file for module "name".
 */
bool
graph_of_calls(string name)
{
    FILE * out;
    string dir_name, file_name, full_name;
    dir_name = db_get_current_workspace_directory();
    file_name = db_build_file_resource_name(DBR_DVCG_FILE, name, DV_SUFFIX);
    full_name = strdup(concatenate(dir_name, "/", file_name, 0));
    free(dir_name), dir_name = NULL;
    out = safe_fopen(full_name, "w");
    init_seen();
    
    /* do the job here. */
    fprintf(out, "[\n");
    recursive_append(out, name);
    fprintf(out, "]\n");
		     
    close_seen();
    safe_fclose(out, full_name);
    free(full_name), full_name = NULL;
    DB_PUT_FILE_RESOURCE(DBR_DVCG_FILE, name, file_name);

    return TRUE;
}

/* To be called by pipsmake.
 * Generate a global resource, hence name is ignored.
 */
bool full_graph_of_calls(string name)
{
    gen_array_t modules = db_get_module_list();
    int n = gen_array_nitems(modules), i;
    FILE * out;
    bool first = TRUE;
    string dir_name, file_name, full_name;

    pips_debug(7, "global call graph requested for %s (PROGRAM)\n", name);

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
