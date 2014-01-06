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
 * This File contains the routine of verification launching
 * 
 * Laurent Aniort & Fabien COELHO 1992
 * 
 * Modified      92 08 31 
 * Author        Arnauld Leservot 
 * Old Version   flint.c.old 
 * Object        Include this as a pips phase.
 */

/*************************************************************/

#include "local.h"

/* internal variables */
static FILE    *flint_messages_file = NULL;
static bool     no_message = true;
static int      number_of_messages = 0;
statement       flint_current_statement = statement_undefined;

/* name of module being flinted */
static char *flint_current_module_name;

/*************************************************************/
/* Routine of  global module verification                    */

bool
flinter(char * module_name)
{
    graph dependence_graph;
    entity module = local_name_to_top_level_entity(module_name);
    statement module_stat;
    string localfilename = NULL;
    string filename = NULL;
    
    /* user_error() is not used in flint, no need for an exception handler */

    debug_on("FLINT_DEBUG_LEVEL");

    flint_current_module_name = module_name;
    flint_current_statement = statement_undefined;
    number_of_messages = 0;
    no_message = true;

    debug(1, "flinter", "flinting module %s\n", module_name);
    
    /* Getting parsed code of module */
    /* the last parameter should be pure=true; the code should not be modified! */
    module_stat = (statement)
	db_get_memory_resource(DBR_CODE, module_name, true);

    /* Resource to trace uninitialized variables: */
    dependence_graph =
	(graph) db_get_memory_resource(DBR_CHAINS, module_name, true);
    set_ordering_to_statement(module_stat);

    set_proper_rw_effects((statement_effects)
			  db_get_memory_resource(DBR_PROPER_EFFECTS,
						 module_name,
						 true)); 
    set_current_module_statement(module_stat);
    set_current_module_entity(local_name_to_top_level_entity(module_name));

    localfilename = db_build_file_resource_name(DBR_FLINTED_FILE,
						module_name,
						".flinted");
    filename = strdup(concatenate(db_get_current_workspace_directory(), 
				  "/", localfilename, NULL));
    flint_messages_file = 
	(FILE *) safe_fopen(filename, "w");

    /* what is  done */
    pips_debug(3, "checking uninitialized variables\n");
    flint_uninitialized_variables(dependence_graph, module_stat);

    debug(3, "flinter", "checking commons\n");
    check_commons(module);	         /* checking commons */

    debug(3, "flinter", "checking statements\n");
    flint_statement(module_stat);	 /* checking syntax  */

    if (no_message)                      /* final message */
      fprintf(flint_messages_file,
	      "%s has been flinted : everything is ok.\n",
	      module_name);
    else
      fprintf(flint_messages_file,
	      "number of messages from flint for %s : %d\n",
	      module_name,
	      number_of_messages);
    
    safe_fclose(flint_messages_file, filename);
    DB_PUT_FILE_RESOURCE(DBR_FLINTED_FILE, strdup(module_name),
			 localfilename);
    free(filename);

    flint_current_module_name = NULL;
    flint_current_statement = statement_undefined;
    number_of_messages = 0;
    no_message = true;

    reset_proper_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();
	reset_ordering_to_statement();

    debug_off();

    /* Should have worked: */
    return true;
}


/*************************************************************/

/*
 * FLINT_MESSAGE(fonction, format [, arg] ... ) string fonction, format;
 */

void
flint_message(char *fun,
	      char *fmt,
	      ...) 
{
    va_list         args;
    int             order;

    va_start(args, fmt);

    /*
     * print name of function causing message, and in which module it
     * occured.
     */

    no_message = false;
    number_of_messages++;

    order = statement_ordering(flint_current_statement);

    (void) fprintf(flint_messages_file,
		   "flint message from %s, in module %s, in statement (%d.%d), number %td\n",
		   fun, flint_current_module_name,
		   ORDERING_NUMBER(order), ORDERING_STATEMENT(order),
		   statement_number(flint_current_statement));


    /* print out remainder of message */
    (void) vfprintf(flint_messages_file, fmt, args);

    va_end(args);

}
/*************************************************************/
/* Same as flint_message but without the function name       */

void
flint_message_2(char *fun,
		char *fmt,
		...) 
{
    va_list         args;

    va_start(args, fmt);

    no_message = false;
    number_of_messages++;

    (void) fprintf(flint_messages_file,
		   "flint message from %s, in module %s\n",
		   fun, flint_current_module_name);

    /* print out remainder of message */
    (void) vfprintf(flint_messages_file, fmt, args);

    va_end(args);

}


/*************************************************************/
/* Same as flint_message but a bare bones version            */
/* count is used to decide if we count tjis message or not.  */

void
raw_flint_message(bool count,
		  char *fmt,
		  ...) 
{
    va_list         args;

    va_start(args, fmt);

    no_message = false;
    if (count)
	number_of_messages++;

    (void) vfprintf(flint_messages_file, fmt, args);

    va_end(args);
}

/*************************************************************/
/* End of File */
