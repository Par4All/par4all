/*
 * $Id$
 *
 * regenerate user files. the modules are put in a file which has the
 * same name as the original user file. All regenerated files are stored
 * in the subdirectory Src of the database. If some includes are
 * generated, they should also be stored there. 
 * 
 * $Log: unsplit.c,v $
 * Revision 1.5  1997/10/28 14:33:05  keryell
 * Renamed basename and dirname as pips_basename and pips_dirname for OSF1.
 *
 * Revision 1.4  1997/10/27 09:52:00  coelho
 * basename moved to misc.
 *
 * Revision 1.3  1997/10/21 05:24:11  coelho
 * unsplit seems ok.
 *
 * Revision 1.2  1997/10/16 19:09:25  coelho
 * comment added.
 *
 * Revision 1.1  1997/10/16 18:59:07  coelho
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

/* initial user file -> generated user file
 */
static hash_table user_files = hash_table_undefined;

/* returns the new user file where to store user_file
 */
static string 
get_new_user_file(string dir_name, string user_file)
{
    string s = hash_get(user_files, user_file);
    if (s==HASH_UNDEFINED_VALUE) 
    {
	FILE * tmp;
	string name = pips_basename(user_file, NULL);
	s = strdup(concatenate(dir_name, "/", name, 0));
	hash_put(user_files, user_file, s);
	/* could check that the file does not exist...
	 * there could be homonymes...
	 */
	tmp = safe_fopen(s, "w");
	fprintf(tmp, "!!\n!! file for %s\n!!\n", name);
	safe_fclose(tmp, s);
	free(name);
    }
    return s;
}

/* unsplit > PROGRAM.user_file
 *         < ALL.user_file
 *         < ALL.printed_file
 */
bool
unsplit(string name)
{
    gen_array_t modules = db_get_module_list();
    int n = gen_array_nitems(modules), i;
    string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE),
	summary_name = db_build_file_resource_name
	(DBR_USER_FILE, PROGRAM_RESOURCE_OWNER, ".txt"), 
	summary_full_name, dir_name;
    FILE * summary;

    user_files = hash_table_make(hash_string, 2*n);

    dir_name = db_get_current_workspace_directory();
    summary_full_name = strdup(concatenate(dir_name, "/", summary_name, 0));

    summary = safe_fopen(summary_full_name, "w");
    fprintf(summary, "! module / file\n");

    /* each module PRINTED_FILE is appended to a new user file
     * depending on its initial user file.
     */
    for (i=0; i<n; i++) 
    {
	string module, user_file, new_user_file, printed_file, full;
	FILE * out, * in;

	module = gen_array_item(modules, i);
	user_file = db_get_memory_resource(DBR_USER_FILE, module, TRUE);
	new_user_file = get_new_user_file(src_dir, user_file);
	printed_file = db_get_memory_resource(DBR_PRINTED_FILE, module, TRUE);
	full = strdup(concatenate(dir_name, "/", printed_file, 0));

	out = safe_fopen(new_user_file, "a");
	in = safe_fopen(full, "r");

	safe_cat(out, in);

	safe_fclose(out, new_user_file);
	safe_fclose(in, full);

	fprintf(summary, "%s: %s\n", module, new_user_file);
	free(full);
    }

    /* clean 
     */
    safe_fclose(summary, summary_full_name);
    free(summary_full_name), free(src_dir), free(dir_name);
    gen_array_full_free(modules);
    HASH_MAP(k, v, free(v), user_files);
    hash_table_free(user_files);
    user_files = hash_table_undefined;

    /* kind of a pseudo resource...
     */
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, PROGRAM_RESOURCE_OWNER, summary_name);
    return TRUE;
}
