 /* Main C functions to print code, sequential or parallel
  *
  * Modifications:
  *  - In order to make consistent with the ri-util, I modified some codes
  *    FROM
       ADD_SENTENCE_TO_TEXT(r, 
			   make_sentence(is_sentence_formatted, 
					 code_decls_text(entity_code(module))));
       MERGE_TEXTS(r, text_statement(module, 0, mod_stat));
       ADD_SENTENCE_TO_TEXT(r, sentence_tail(module));
  *    TO
       MERGE_TEXTS(r, text_module(module, mod_stat));
  *    23/10/91
  *    BTW, sentence_tail should have no argument at all.
  *
  *  - printparallelized_code should have properties: 
           set_bool_property("PRETTYPRINT_PARALLEL", TRUE);
	   set_bool_property("PRETTYPRINT_SEQUENTIAL", FALSE);
  *    Too. Not just print_parallelized90_code and print_parallelized77_code
  *    LZ, 17/01/92
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

static bool is_user_view;	/* print_code or print_source */

bool print_parallelized90_code(mod_name)
char *mod_name;
{
    bool success;
    bool f90_property = get_bool_property("PRETTYPRINT_FORTRAN90");

    set_bool_property("PRETTYPRINT_FORTRAN90", TRUE);
    success = print_parallelized_code(mod_name);
    set_bool_property("PRETTYPRINT_FORTRAN90", f90_property);

    return success;
}

bool print_parallelized77_code(mod_name)
char *mod_name;
{
    /* set_bool_property("PRETTYPRINT_FORTRAN90", FALSE); */
    return print_parallelized_code(mod_name);
}

bool print_parallelizedHPF_code(string module_name)
{
    bool ok, init;

    init = get_bool_property("PRETTYPRINT_HPF");
    set_bool_property("PRETTYPRINT_HPF", TRUE);
    ok = print_parallelized_code(module_name);
    set_bool_property("PRETTYPRINT_HPF", init);
    
    return ok;
}

bool print_parallelized_code(mod_name)
char *mod_name;
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;


    set_bool_property("PRETTYPRINT_PARALLEL", TRUE);
    set_bool_property("PRETTYPRINT_SEQUENTIAL", FALSE);

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, TRUE);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    MERGE_TEXTS(r, text_module(module, mod_stat));
    debug_off();

    close_prettyprint();

    success = make_text_resource (mod_name, DBR_PARALLELPRINTED_FILE,
				  PARALLEL_FORTRAN_EXT, r);

    return success;
}

bool print_code(mod_name)
char *mod_name;
{
  is_user_view = FALSE;
  return print_code_or_source(mod_name);
}

bool print_source(mod_name)
char *mod_name;
{
  is_user_view = TRUE;
  return print_code_or_source(mod_name);
}


/* Idem as print_code but add some emacs properties in the output.
   RK, 21/06/1995 */
bool
emacs_print_code(char *mod_name)
{
    bool success;
    /*
       is_emacs_prettyprint = TRUE;
       is_attachment_prettyprint = TRUE;
       */
    set_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES", TRUE);
  
    success = print_code(mod_name);

    set_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES", FALSE);
    /*
       is_emacs_prettyprint = FALSE;  
       is_attachment_prettyprint = FALSE;  
       */

    return success;
}


bool print_code_or_source(mod_name)
char *mod_name;
{
    bool success = FALSE;
    text r = make_text(NIL);
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;
    bool is_emacs_prettyprint =
	get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES");
  
    string resource_name = strdup
	(get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
	    DBR_GRAPH_PRINTED_FILE
		: is_emacs_prettyprint ? DBR_EMACS_PRINTED_FILE :
		    (is_user_view ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE));
    string file_ext =
	strdup(concatenate
	       (is_user_view? PRETTYPRINT_FORTRAN_EXT : PREDICAT_FORTRAN_EXT,
		is_emacs_prettyprint ? EMACS_FILE_EXT : "",
		get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? GRAPH_FILE_EXT : "",
		NULL));

    set_bool_property("PRETTYPRINT_PARALLEL", FALSE);
    set_bool_property("PRETTYPRINT_SEQUENTIAL", TRUE);

    mod_stat = (statement)
	db_get_memory_resource(is_user_view?DBR_PARSED_CODE:DBR_CODE, mod_name, TRUE);

    if (is_emacs_prettyprint) {	
	begin_attachment_prettyprint();
	name_almost_everything_in_a_module(mod_stat);
    }
    
    init_prettyprint(empty_text);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    MERGE_TEXTS(r, text_module(module,mod_stat));
    debug_off();
    success = make_text_resource (mod_name, resource_name, file_ext, r);

    if (is_emacs_prettyprint) {
	end_attachment_prettyprint();
	free_names_of_almost_everything_in_a_module();
    }
    
    free(resource_name);
    free(file_ext);
    return success;
}

bool make_text_resource(mod_name, res_name, file_ext, texte)
char *mod_name;
char *res_name;
char *file_ext;
text texte;
{
    char *filename;
    char *localfilename;
    FILE *fd;
    
    localfilename = strdup(concatenate(mod_name, file_ext, NULL));

    filename = strdup(concatenate(db_get_current_workspace_directory(), 
				  "/", localfilename, NULL));

    fd = safe_fopen(filename, "w");

    /* Add the attachment if necessary: */
    if (get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
	init_output_the_attachments_for_emacs(fd);
    
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    print_text(fd, texte);
    debug_off();

    /* Add the attachment if necessary: */
    if (get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
	output_the_attachments_for_emacs(fd);
    
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(strdup(res_name), strdup(mod_name), localfilename);
    free(filename);
    return TRUE;
}
