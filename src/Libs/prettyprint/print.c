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

#include <stdio.h>
#include <string.h>
#include <values.h>

#include "genC.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "prettyprint.h"

#include "constants.h"
#include "resources.h"

static bool is_user_view;	/* print_code or print_source */

void print_parallelized90_code(mod_name)
char *mod_name;
{
    bool f90_property = get_bool_property("PRETTYPRINT_FORTRAN90");

    set_bool_property("PRETTYPRINT_FORTRAN90", TRUE);
    print_parallelized_code(mod_name);
    set_bool_property("PRETTYPRINT_FORTRAN90", f90_property);
}

void print_parallelized77_code(mod_name)
char *mod_name;
{
    /* set_bool_property("PRETTYPRINT_FORTRAN90", FALSE); */
    print_parallelized_code(mod_name);
}

void print_parallelized_code(mod_name)
char *mod_name;
{
    text r = make_text(NIL);
    char *filename;
    FILE *fd;
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    set_bool_property("PRETTYPRINT_PARALLEL", TRUE);
    set_bool_property("PRETTYPRINT_SEQUENTIAL", FALSE);

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, TRUE);
    
    filename = strdup(concatenate(db_get_current_program_directory(), 
				  "/", mod_name, PARALLEL_FORTRAN_EXT, NULL));

    MERGE_TEXTS(r, text_module(module, mod_stat));

    fd = safe_fopen(filename, "w");
    print_text(fd, r);
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(DBR_PARALLELPRINTED_FILE, strdup(mod_name), 
			 filename);

    debug_off();
}

void print_code(mod_name)
char *mod_name;
{
  is_user_view = FALSE;
  print_code_or_source(mod_name);
}

void print_source(mod_name)
char *mod_name;
{
  is_user_view = TRUE;
  print_code_or_source(mod_name);
}

void print_code_or_source(mod_name)
char *mod_name;
{
    text r = make_text(NIL);
    char *filename;
    FILE *fd;
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat;

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    /*set_bool_property("PRETTYPRINT_FORTRAN90", FALSE);*/

    /* PRETTYPRINT_PARALLEL is not used */
    set_bool_property("PRETTYPRINT_PARALLEL", FALSE);
    /* PRETTYPRINT_SEQUENTIAL is not used */
    set_bool_property("PRETTYPRINT_SEQUENTIAL", TRUE);

    init_prettyprint(empty_text);

    mod_stat = (statement)
	db_get_memory_resource(is_user_view?DBR_PARSED_CODE:DBR_CODE, mod_name, TRUE);
  
    filename = strdup(concatenate(db_get_current_program_directory(), 
				  "/",
				  mod_name,
				  is_user_view? PRETTYPRINT_FORTRAN_EXT : PREDICAT_FORTRAN_EXT,
				  NULL));

    MERGE_TEXTS(r, text_module(module,mod_stat));

    fd = safe_fopen(filename, "w");
    print_text(fd, r);
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(is_user_view? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE,
			 strdup(mod_name),
			 filename);
    debug_off();
}

void make_text_resource(mod_name, res_name, file_ext, texte)
char *mod_name;
char *res_name;
char *file_ext;
text texte;
{
    char *filename;
    FILE *fd;

    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    
    filename = strdup(concatenate(db_get_current_program_directory(), 
				  "/", mod_name, file_ext, NULL));

    fd = safe_fopen(filename, "w");
    print_text(fd, texte);
    safe_fclose(fd, filename);

    DB_PUT_FILE_RESOURCE(strdup(res_name), strdup(mod_name), filename);

    debug_off();
}
