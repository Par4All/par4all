/* Main C functions to print code, sequential or parallel
 *
 * $Id$
 *
 * $Log: print.c,v $
 * Revision 1.21  1997/12/10 12:17:23  coelho
 * working free_text().
 *
 * Revision 1.20  1997/12/08 14:17:43  coelho
 * make_text_resource_and_free() added.
 *
 * Revision 1.19  1997/11/22 11:02:51  coelho
 * print_parallelizedOMP_code added.
 *
 * Revision 1.18  1997/11/21 13:17:58  coelho
 * cleaner headers.
 *
 * 
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
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

bool 
print_code_or_source(string mod_name)
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module = local_name_to_top_level_entity(mod_name);
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

    pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
    set_string_property(PRETTYPRINT_PARALLEL, "do");

    mod_stat = (statement)
	db_get_memory_resource(is_user_view?
			       DBR_PARSED_CODE:DBR_CODE, mod_name, TRUE);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    begin_attachment_prettyprint();
    init_prettyprint(empty_text);
    MERGE_TEXTS(r, text_module(module,mod_stat));
    success = make_text_resource(mod_name, resource_name, file_ext, r);
    end_attachment_prettyprint();

    debug_off();

    set_string_property(PRETTYPRINT_PARALLEL, pp); free(pp);

    free_text(r);
    free(resource_name);
    free(file_ext);
    return success;
}
    
static bool 
print_parallelized_code_common(
    string mod_name,
    string style)
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;
    string pp = string_undefined;

    if (style) {
	pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
	set_string_property(PRETTYPRINT_PARALLEL, style);
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
				  PARALLEL_FORTRAN_EXT, r);

    end_attachment_prettyprint();

    if (style) {
	set_string_property(PRETTYPRINT_PARALLEL, pp); 
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

bool 
print_parallelizedOMP_code(string mod_name)
{
    return print_parallelized_code_common(mod_name, "omp");
}
