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

/* Are we working for emacs or not ? */
extern bool is_emacs_prettyprint;

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

bool print_parallelized_code(mod_name)
char *mod_name;
{
    text r = make_text(NIL);
    char *filename;
    FILE *fd;
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;

    set_bool_property("PRETTYPRINT_PARALLEL", TRUE);
    set_bool_property("PRETTYPRINT_SEQUENTIAL", FALSE);

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, TRUE);
    
    filename = strdup(concatenate(db_get_current_workspace_directory(), 
				  "/", mod_name, PARALLEL_FORTRAN_EXT, NULL));

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    MERGE_TEXTS(r, text_module(module, mod_stat));

    fd = safe_fopen(filename, "w");
    print_text(fd, r);
    debug_off();
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(DBR_PARALLELPRINTED_FILE, strdup(mod_name), 
			 filename);

    return TRUE;
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

  is_emacs_prettyprint = TRUE;
  is_attachment_prettyprint = TRUE;
  success = print_code(mod_name);
  is_emacs_prettyprint = FALSE;  
  is_attachment_prettyprint = FALSE;  

  return success;
}


bool print_code_or_source(mod_name)
char *mod_name;
{
    text r = make_text(NIL);
    char *filename;
    FILE *fd;
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;

    /*set_bool_property("PRETTYPRINT_FORTRAN90", FALSE);*/

    /* PRETTYPRINT_PARALLEL is not used */
    set_bool_property("PRETTYPRINT_PARALLEL", FALSE);
    /* PRETTYPRINT_SEQUENTIAL is not used */
    set_bool_property("PRETTYPRINT_SEQUENTIAL", TRUE);

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(is_user_view?DBR_PARSED_CODE:DBR_CODE, mod_name, TRUE);
  
    filename = strdup
	(concatenate(db_get_current_workspace_directory(), 
		     "/",
		     mod_name,
		     is_user_view? PRETTYPRINT_FORTRAN_EXT : PREDICAT_FORTRAN_EXT,
		     is_emacs_prettyprint ? EMACS_FILE_EXT : "",
		     get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? GRAPH_FILE_EXT : "",
		     NULL));

    MERGE_TEXTS(r, text_module(module,mod_stat));

    fd = safe_fopen(filename, "w");

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    print_text(fd, r);
    debug_off();

    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE
	(get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
	 DBR_GRAPH_PRINTED_FILE
	 : is_emacs_prettyprint ? DBR_EMACS_PRINTED_FILE :
	 (is_user_view ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE),
	 strdup(mod_name),
	 filename);

    return TRUE;
}

bool make_text_resource(mod_name, res_name, file_ext, texte)
char *mod_name;
char *res_name;
char *file_ext;
text texte;
{
    char *filename;
    FILE *fd;
    
    filename = strdup(concatenate(db_get_current_workspace_directory(), 
				  "/", mod_name, file_ext, NULL));

    fd = safe_fopen(filename, "w");

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    print_text(fd, texte);
    debug_off();

    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(strdup(res_name), strdup(mod_name), filename);

    return TRUE;
}
