/* Copyright 2007-2012 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

bool step_install(__attribute__ ((unused)) const char* program_name)
{
  debug_on("STEP_INSTALL_DEBUG_LEVEL");

  /* Generation des fichiers sources dans workspace.database/Src/ */
  string dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  gen_array_t modules = db_get_module_list_initial_order();
  int n = gen_array_nitems(modules), i;
  hash_table user_files = hash_table_make(hash_string, 2*n);

  for (i=0; i<n; i++)
    {
      string module_name = gen_array_item(modules, i);
      string user_file = db_get_memory_resource(DBR_USER_FILE, module_name, true);
      string new_file = hash_get(user_files, user_file);
      if (new_file == HASH_UNDEFINED_VALUE)
	{
	  string base_name = pips_basename(user_file, NULL);
	  new_file = strdup(concatenate(dest_dir, "/", base_name, NULL));
	  hash_put(user_files, user_file, new_file);
	}

      string step_file = db_get_memory_resource(DBR_STEP_FILE, module_name, true);



      pips_debug(1, "Module: \"%s\"\n\tuser_file: \"%s\"\n\tinstall file : \"%s\"\n", module_name, user_file, new_file);
      FILE *out = safe_fopen(new_file, "a");
      FILE *in = safe_fopen(step_file, "r");

      safe_cat(out, in);

      safe_fclose(out, new_file);
      safe_fclose(in, step_file);
    }

  hash_table_free(user_files);
  free(dest_dir);

  /* Instalation des fichiers generes */
  dest_dir = strdup(get_string_property("STEP_INSTALL_PATH"));
  string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  if (empty_string_p(dest_dir))
    {
      free(dest_dir);
      dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    }

  safe_system(concatenate("step_install ", dest_dir, " ", src_dir,  NULL));

  free(src_dir);
  free(dest_dir);

  debug_off();
  return true;
}

