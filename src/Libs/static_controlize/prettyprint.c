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
/* Name      :	prettyprint.c
 * package   :	static_controlize
 * Author    :	Arnauld LESERVOT
 * Date      :	May 93
 * Modified  :
 * Documents :	"Implementation of Array Data Flow Graph in Pips"
 * Comments  :
 */

/* Ansi includes	*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

/* Newgen includes	*/
#include "genC.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "ri.h"
#include "effects.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
typedef void * arc_label;
typedef void * vertex_label;
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "static_controlize.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "paf-util.h"

#define MAX_STATIC_CONTROL_LINE_NUMBER 2048
#define CODE_WITH_STATIC_CONTROLIZE_EXT ".stco"

/* Global variables */
static statement_mapping	Gsc_map;

/*=================================================================*/
/* void print_code_static_control((char*) module_name)		AL 05/93
 * Prettyprints a static_control mapping attached to a module_name.
 */
bool print_code_static_control(module_name)
const char* module_name;
{
    entity	module;
    statement	module_stat;
    text	txt = make_text(NIL);
    bool success;

    debug_on( "PRINT_STATIC_CONTROL_DEBUG_LEVEL" );

    ifdebug(1)
           user_log("\n\n *** PRINTING STATIC CONTROL for %s\n",
				 module_name);

    module = local_name_to_top_level_entity(module_name);
    module_stat = (statement)
	db_get_memory_resource(DBR_CODE, module_name, true);
    Gsc_map = (statement_mapping)
	db_get_memory_resource( DBR_STATIC_CONTROL, module_name, true);
    init_prettyprint(text_static_control);

/*
    filename = strdup(concatenate(db_get_current_workspace_directory(),
				  "/", module_name, ".stco", NULL));
    fd = safe_fopen(filename, "w");
*/

    MERGE_TEXTS(txt, text_module(module, module_stat));

/*
    print_text(fd, txt);
    safe_fclose(fd, filename);
    DB_PUT_FILE_RESOURCE(DBR_PRINTED_FILE, strdup(module_name),
			 	filename);
*/

    success = make_text_resource(module_name,
				 DBR_PRINTED_FILE,
				 CODE_WITH_STATIC_CONTROLIZE_EXT,
				 txt);

    close_prettyprint();

    debug_off();

    return(success);
}


/*=================================================================*/
/* text text_static_control((entity) module, (int) margin, (statement) stat)
 * Function hook used by package text-util to prettyprint a static_control.
 */
text text_static_control(entity module __attribute__ ((unused)),
			 int margin __attribute__ ((unused)),
			 statement stat)
{
    static_control sc = (static_control) GET_STATEMENT_MAPPING(Gsc_map, stat);

    return( store_sc_text_line( sc ));
}

/*=================================================================*/
/* text store_sc_text_line((static_control) sc)		AL 05/93
 * Stores a static_control prettyprinted.
 */
text store_sc_text_line( sc )
static_control sc;
{
    text sc_text = make_text(NIL);
    char *t = NULL;
    string comment;

    /* pips_assert("store_text_line", sefs_list != NIL); */
    if (get_prettyprint_language_tag() == is_language_c)
      comment = "//";
    else 
      comment = "C";


    t = concatenate(comment,"\t\t< is static >",
		(static_control_yes(sc)?" TRUE":" FALSE"), "\n", NULL);
        ADD_SENTENCE_TO_TEXT( sc_text,
			 make_sentence(is_sentence_formatted, strdup(t)));
	t = concatenate(comment,"\t\t< parameter >",
		 words_to_string(words_entity_list(static_control_params(sc))),
					 "\n", NULL);
	ADD_SENTENCE_TO_TEXT( sc_text,
			 make_sentence(is_sentence_formatted, strdup(t)));
	t = concatenate(comment,"\t\t<   loops   >",
		words_to_string(words_loop_list(static_control_loops(sc))),
					  NULL);
	ADD_SENTENCE_TO_TEXT( sc_text,
			make_sentence(is_sentence_formatted, strdup(t)));
	t = concatenate(comment,"\t\t<   tests   >",
		words_to_string(words_test_list(static_control_tests(sc))),
					  NULL);
	ADD_SENTENCE_TO_TEXT( sc_text,
			make_sentence(is_sentence_formatted, strdup(t)));

    return ( sc_text );
}

#define MAX_CHAR_NUMBER 39

/*=================================================================*/
/* cons *words_test_list((list) obj)		AL 05/93
 * Makes a list of strings from a list of test expressions.
 */
cons *words_test_list(obj)
list obj;
{
        cons*  ret_cons = NIL;
	string before_string = strdup(" ");
	string blank_string;
	if (get_prettyprint_language_tag() == is_language_c)
	  blank_string = strdup("//                            ");
	else
	  blank_string = strdup("C                             ");

        pips_debug(7, "doing \n");
        MAPL( exp_ptr, {
	    cons*	pc  = NIL;
                expression	exp = EXPRESSION(CAR(exp_ptr));

                pc = CHAIN_SWORD(pc, strdup( before_string ));
		pc = gen_nconc(pc, words_expression(exp, NIL));
                pc = CHAIN_SWORD(pc,"\n") ;
                ret_cons = gen_nconc(ret_cons, gen_copy_seq(pc));
		before_string = blank_string;
        }, obj);

	if( ret_cons == NIL ) ret_cons = CHAIN_SWORD(ret_cons, "\n");
        return( ret_cons );
}

/*=================================================================*/
/* cons *words_loop_list((list) obj)		AL 05/93
 * Returns a list of strings from a list of loops.
 */
cons *words_loop_list(obj)
list obj;
{
	cons*  ret_cons = NIL;
	string before_string = strdup(" ");
	string blank_string;
	if (get_prettyprint_language_tag() == is_language_c)
	  blank_string = strdup("//                            ");
	else
	  blank_string = strdup("C                             ");


	pips_debug(7, "doing \n");
	FOREACH(LOOP, l, obj) {
	  cons*	pc	= NIL;
	  entity	ind	= loop_index( l );
	  expression low  = range_lower(loop_range( l ));
	  expression up   = range_upper(loop_range( l ));
	  pc = CHAIN_SWORD(pc, strdup( before_string ));
	  pc = gen_nconc(pc, words_expression(low, NIL));
	  pc = CHAIN_SWORD(pc," <= ");
	  pc = CHAIN_SWORD(pc, entity_local_name(ind));
	  pc = CHAIN_SWORD(pc," <= ");
	  pc = gen_nconc(pc, words_expression(up, NIL));
	  pc = CHAIN_SWORD(pc,"\n") ;
	  ret_cons = gen_nconc(ret_cons, gen_copy_seq(pc));
	  before_string = blank_string;
	}

	if( ret_cons == NIL ) ret_cons = CHAIN_SWORD(ret_cons, "\n");
	return( ret_cons );
}

/*=================================================================*/
/* cons *words_entity_list((list) obj)		AL 05/93
 * Returns a list of strings from a list of entities.
 *
 * FI: Should be moved in ri-util/entity.c or entities.c when it exists
 */
cons *words_entity_list(obj)
list obj;
{
        list the_list = NIL;

        pips_debug(7, "doing \n");
        the_list = CHAIN_SWORD(the_list, " ");
        MAPL( ent_ptr, {
                string s = '\0';
                entity ent = ENTITY(CAR( ent_ptr ));
                s = strdup( concatenate(entity_local_name(ent), ", ", NULL) );
                ADD_ELEMENT_TO_LIST( the_list, STRING, s );
        }, obj);
	return( the_list );
}

/*=================================================================*/
