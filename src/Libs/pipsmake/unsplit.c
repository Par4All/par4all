/*
 * $Id$
 *
 * regenerate user files. the modules are put in a file which has the
 * same name as the original user file. All regenerated files are stored
 * in the subdirectory Src of the database. If some includes are
 * generated, they should also be stored there. 
 * 
 * $Log: unsplit.c,v $
 * Revision 1.1  1997/10/16 18:59:07  coelho
 * Initial revision
 *
 */

#include <stdio.h>
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
	int len = strlen(user_file)-1;
	while (len>=0 && user_file[len]!='/') len--;
	pips_assert("must be a / in user file", len>=0);
	s = strdup(concatenate(dir_name, &user_file[len], 0));
	/* could check that the file does not exist...
	 * there could be homonymes...
	 */
	hash_put(user_files, user_file, s);
    }
    return s;
}

/* unsplit > PROGRAM.USER_FILE
 *         < ALL.USER_FILE
 *         < ALL.PRINTED_FILE
 */
bool
unsplit(string name)
{
    gen_array_t modules = db_get_module_list();
    int n = gen_array_nitems(modules), i;
    string src_dir = db_get_directory_name_for_module("Src"),
	summary_name = db_build_file_resource_name(DBR_USER_FILE, "", ".txt");
    FILE * summary;

    user_files = hash_table_make(hash_string, 2*n);

    if (!purge_directory(src_dir) || !create_directory(src_dir))
	pips_internal_error("failure with directory %s\n", src_dir);

    summary = safe_fopen(summary_name, "w");
    fprintf(summary, "module / file\n");

    /* each module PRINTED_FILE is appended to a new user file
     * depending on its initial user file.
     */
    for (i=0; i<n; n++) 
    {
	string module, user_file, new_user_file, printed_file;
	FILE * out, * in;

	module = gen_array_item(modules, i);
	user_file = db_get_memory_resource(DBR_USER_FILE, module, TRUE);
	new_user_file = get_new_user_file(src_dir, user_file);
	printed_file = db_get_memory_resource(DBR_PRINTED_FILE, module, TRUE);

	out = safe_fopen(new_user_file, "a");
	in = safe_fopen(printed_file, "r");

	safe_cat(out, in);

	safe_fclose(out, new_user_file);
	safe_fclose(in, printed_file);

	fprintf(summary, "%s: %s\n", module, new_user_file);
    }

    /* clean 
     */
    safe_fclose(summary, summary_name);
    free(src_dir);
    gen_array_full_free(modules);
    HASH_MAP(k, v, free(v), user_files);
    hash_table_free(user_files);
    user_files = hash_table_undefined;

    /* kind of a pseudo resource...
     */
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, "", summary_name);
    return TRUE;
}
