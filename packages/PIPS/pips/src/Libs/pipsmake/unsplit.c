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
/*
 * regenerate user files. the modules are put in a file which has the
 * same name as the original user file. All regenerated files are stored
 * in the subdirectory Src of the database. If some includes are
 * generated, they should also be stored there.
 *
 */

#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "misc.h"
#include "resources.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "preprocessor.h"

/* initial user file -> generated user file
 */
static hash_table user_files = hash_table_undefined;

/* returns the new user file where to store user_file
 */
static string
get_new_user_file(string dir_name, string preprocessed_user_file)
{
  /* C or Fortran preprocessing may have or have not occured */
  string user_file = preprocessed_to_user_file(preprocessed_user_file);
  string s = hash_get(user_files, user_file);

  if (s==HASH_UNDEFINED_VALUE) {
    FILE * tmp;
    string name = pips_basename(user_file, NULL);

    pips_debug(1,
	       "It does not exist a file \"%p\"\n for user_file \"%s\"\n"
	       " and for preprocessed_user_file \"%s\".\n"
	       "in table %p\n",
	       s, user_file, preprocessed_user_file, user_files);
    s = strdup(concatenate(dir_name, "/", name, NULL));
    hash_put(user_files, user_file, s);
    /* could check that the file does not exist...
     * there could be homonymes...
     */
    if((tmp=fopen(s, "r"))!=NULL) {
      pips_internal_error("Rewriting existing file \"%s\" for user_file \"%s\""
			  " and for preprocessed_user_file \"%s\".\n",
			  s, user_file, preprocessed_user_file);
      fclose(tmp);
    }
    tmp = safe_fopen(s, "w");
    if(dot_f_file_p(user_file) || dot_f90_file_p(user_file)
        || dot_f95_file_p(user_file) ) {
      fprintf(tmp, "!!\n!! file for %s\n!!\n", name);
    }
    else if(dot_c_file_p(user_file)) {
      fprintf(tmp, "/*\n * file for %s\n */\n", name);
    }
    else {
      pips_internal_error("unexpected user file suffix: \"%s\"", user_file);
    }
    safe_fclose(tmp, s);
    free(name);
  }
  else {
    pips_debug(1,
	       "It does exist a file \"%s\"\n for user_file \"%s\"\n"
	       " and for preprocessed_user_file \"%s\".\n"
	       "in table %p\n",
	       s, user_file, preprocessed_user_file, user_files);
  }
  free(user_file);
  return s;
}

/* unsplit > PROGRAM.user_file
 *         < ALL.user_file
 *         < ALL.printed_file
 */
bool
unsplit(const char* name)
{
    gen_array_t modules = db_get_module_list_initial_order();
    int n = gen_array_nitems(modules), i;
    string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    string summary_name = db_build_file_resource_name
      (DBR_USER_FILE, PROGRAM_RESOURCE_OWNER, ".txt");
    string summary_full_name;
    string dir_name;
    FILE * summary;
    int nfiles = 0; /* Number of preexisting files in src_dir */

    pips_assert("unused argument", name==name);

    debug_on("UNSPLIT_DEBUG_LEVEL");

    user_files = hash_table_make(hash_string, 2*n);

    dir_name = db_get_current_workspace_directory();
    summary_full_name = strdup(concatenate(dir_name, "/", summary_name, NULL));

    /* Get rid of previous unsplitted files */

    if ((nfiles = safe_system_no_abort_no_warning
	 (concatenate("i=`ls  ", src_dir, " | wc -l`; export i; exit $i ", NULL)))>0) {
      int failure = 0;
      if ((failure = safe_system_no_abort(concatenate("/bin/rm ", src_dir, "/*",  NULL))))
	pips_user_warning("exit code for /bin/rm is %d\n", failure);
    }

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
	user_file = db_get_memory_resource(DBR_USER_FILE, module, true);
	new_user_file = get_new_user_file(src_dir, user_file);
	printed_file = db_get_memory_resource(DBR_PRINTED_FILE, module, true);
	full = strdup(concatenate(dir_name, "/", printed_file, NULL));
  pips_debug(1, "Module: \"%s\", user_file: \"%s\", new_user_file: \"%s\","
             "full: \"%s\"\n",
	           module, user_file, new_user_file, full);

	out = safe_fopen(new_user_file, "a");
	in = safe_fopen(full, "r");

	safe_cat(out, in);

	safe_fclose(out, new_user_file);
	safe_fclose(in, full);

	fprintf(summary, "%s: %s\n", module, new_user_file);
	free(full);
    }

    /* clean */
    safe_fclose(summary, summary_full_name);
    free(summary_full_name), free(src_dir), free(dir_name);
    gen_array_full_free(modules);
    HASH_MAP(k, v, free(v), user_files);
    hash_table_free(user_files);
    user_files = hash_table_undefined;

    debug_off();

    /* kind of a pseudo resource...
     */
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, PROGRAM_RESOURCE_OWNER, summary_name);

    return true;
}
