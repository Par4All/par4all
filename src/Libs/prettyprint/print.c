/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* Main C functions to print code, sequential or parallel
 *
 * $Id$
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "properties.h"
#include "top-level.h"
#include "text-util.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "prettyprint.h"

#include "constants.h"
#include "resources.h"

/******************************************************************** UTILS */

/* generate resource res_name for module mod_name with prefix file_ext
 * as the text provided. it should be made clear who is to free the
 * texte structure. currently it looks like a massive memory leak.
 */
bool
make_text_resource(
    string mod_name, /* module name */
    string res_name, /* resource name [DBR_...] */
    string file_ext, /* file extension */
    text texte       /* text to be printed as this resource */)
{
    string filename, localfilename, dir;
    FILE *fd;
   
    localfilename = db_build_file_resource_name(res_name, mod_name, file_ext);
    dir = db_get_current_workspace_directory();
    filename = strdup(concatenate(dir, "/", localfilename, NULL));
    free(dir);

    fd = safe_fopen(filename, "w");
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    print_text(fd, texte);
    debug_off();
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(res_name, mod_name, localfilename);
    write_an_attachment_file(filename);
    free(filename);
   
    return TRUE;
}

bool
make_text_resource_and_free(
    string mod_name,
    string res_name,
    string file_ext,
    text t)
{
    bool ok = make_text_resource(mod_name, res_name, file_ext, t);
    free_text(t);
    return ok;
}

static bool is_user_view;	/* print_code or print_source */

bool
user_view_p()
{
    return is_user_view;
}


/* Generic function to prettyprint some sequential or parallel code, or
   even user view for the given module. */
bool
print_code_or_source(string mod_name)
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module;
    statement mod_stat;
    string pp;
    string resource_name = strdup
	(get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	    DBR_GRAPH_PRINTED_FILE
		: (is_user_view ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE));
    string file_ext =
	strdup(concatenate
	       (is_user_view? PRETTYPRINT_FORTRAN_EXT : PREDICAT_FORTRAN_EXT,
		get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
		GRAPH_FILE_EXT : "",
		NULL));

    /* FI: This test could be moved up in pipsmake? */
    if(entity_undefined_p(module = module_name_to_entity(mod_name))) {
      /* FI: Should be a pips_internal_error() as pipsmake is here to
	 avoid this very problem... */
      pips_user_error("Module \"\%s\"\n not found", mod_name);
      return false;
    }

    set_current_module_entity(module);

    /* Since we want to prettyprint with a sequential syntax, save the
       PRETTYPRINT_PARALLEL property that may select the parallel output
       style before overriding it: */
    pp = strdup(get_string_property("PRETTYPRINT_PARALLEL"));
    /* Select the default prettyprint style for sequential prettyprint: */
    set_string_property("PRETTYPRINT_PARALLEL",
			get_string_property("PRETTYPRINT_SEQUENTIAL_STYLE"));

    mod_stat = (statement)
	db_get_memory_resource(is_user_view?
			       DBR_PARSED_CODE:DBR_CODE, mod_name, TRUE);

    set_current_module_statement(mod_stat);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    begin_attachment_prettyprint();
    init_prettyprint(empty_text);
    MERGE_TEXTS(r, text_module(module,mod_stat));
    success = make_text_resource(mod_name, resource_name, file_ext, r);
    end_attachment_prettyprint();

    debug_off();

    /* Restore the previous PRETTYPRINT_PARALLEL property for the next
       parallel code prettyprint: */
    set_string_property("PRETTYPRINT_PARALLEL", pp);
    free(pp);

    reset_current_module_entity();
    reset_current_module_statement();

    free_text(r);
    free(resource_name);
    free(file_ext);

    return success;
}


/* Build a textual resource for a parallel code using a string optional
   parallel style (dialect such as "f90", "doall", "hpf", "omp" */
static bool
print_parallelized_code_common(
    string mod_name,
    string style)
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module = module_name_to_entity(mod_name);
    statement mod_stat;
    string pp = string_undefined;

    if (style) {
	pp = strdup(get_string_property("PRETTYPRINT_PARALLEL"));
	set_string_property("PRETTYPRINT_PARALLEL", style);
    }

    begin_attachment_prettyprint();

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, TRUE);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    MERGE_TEXTS(r, text_module(module, mod_stat));
    debug_off();

    close_prettyprint();

    success = make_text_resource (mod_name, DBR_PARALLELPRINTED_FILE,
				  prettyprint_is_fortran? PARALLEL_FORTRAN_EXT : PARALLEL_C_EXT, r);

    end_attachment_prettyprint();

    if (style) {
	set_string_property("PRETTYPRINT_PARALLEL", pp);
	free(pp);
    }

    free_text(r);
    return success;
}


/************************************************************ PIPSMAKE HOOKS */

bool
print_code(string mod_name)
{
  is_user_view = FALSE;
  return print_code_or_source(mod_name);
}

bool
print_source(string mod_name)
{
  is_user_view = TRUE;
  return print_code_or_source(mod_name);
}

bool
print_parallelized_code(string mod_name)
{
    return print_parallelized_code_common(mod_name, NULL);
}

bool
print_parallelized90_code(string mod_name)
{
    return print_parallelized_code_common(mod_name, "f90");
}

bool
print_parallelized77_code(string mod_name)
{
    return print_parallelized_code_common(mod_name, "doall");
}

bool
print_parallelizedHPF_code(string module_name)
{
    return print_parallelized_code_common(module_name, "hpf");
}

#define all_priv "PRETTYPRINT_ALL_PRIVATE_VARIABLES"

bool
print_parallelizedOMP_code(string mod_name)
{
    if (get_bool_property(all_priv))
	pips_user_warning("avoid property " all_priv "=TRUE with OMP\n");

    return print_parallelized_code_common(mod_name, "omp");
}
