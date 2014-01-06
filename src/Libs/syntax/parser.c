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
/* Pre-parser for Fortran syntax idiosyncrasy
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "syntax.h"

#include "resources.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"

/* name of the current file */
char *CurrentFN = NULL;

/* the current function */
/* entity CurrentFunction; */

/* list of formal parameters of the current function  */
cons *FormalParameters = NIL;

/* the name of the current package, i.e. TOP-LEVEL or a module name? */
const char *CurrentPackage = NULL; 

/* Four areas used to allocate variables which are not stored in a
   common. These areas are just like commons, but the dynamic area is the
   only non-static area according to Fortran standard. The heap and the
   stack area are used to deal with ANSI extensions, pointers and
   allocatable arrays, and adjustable arrays. The dynamic area is stack
   allocated by most compilers but could be statically allocated since the
   array sizes are known. */
entity DynamicArea = entity_undefined;
entity StaticArea = entity_undefined;
entity HeapArea = entity_undefined;
entity StackArea = entity_undefined;
entity AllocatableArea = entity_undefined;

/* Indicates where the current instruction (in fact statement) starts and
   ends in the input file and gives its label. Temporary versions of these
   variables are used because of the pipeline existing between the reader
   and the actual parser. The names of the temporary variables are
   prefixed with "tmp_". The default and reset values of these variables
   and their temporary versions (declared in reader.c) must be
   consistent. */
int line_b_I, line_e_I, line_b_C, line_e_C;
char lab_I[6];

void reset_current_label_string()
{
    strcpy(lab_I, "");
}

string get_current_label_string()
{
    return lab_I;
}

void set_current_label_string(string ln)
{
    pips_assert("Label name is at most 5 characters long", strlen(ln)<=5);
    strcpy(lab_I, ln);
}

bool empty_current_label_string_p()
{
    bool empty_p = same_string_p(lab_I, "");

    return empty_p;
}
/* a string that will contain the value of the format in case of format
statement */
char FormatValue[FORMATLENGTH];


static bool parser_recursive_call = false;

/* Safety for recursive calls of parser required to process entries */
static void reset_parser_recursive_call()
{
    parser_recursive_call = false;
}

static void set_parser_recursive_call()
{
    parser_recursive_call = false;
}

/* Parser error handling */

bool InParserError = false;

bool ParserError(const char * f, char * m)
{
    entity mod = get_current_module_entity();

    /* Maybe a routine called by ParserError() may call ParserError()
     * e.g. AbortOfProcedure() thru remove_ghost_variables()
     */
    if(InParserError)
	return false;

    InParserError = true;

    uses_alternate_return(false);
    ResetReturnCodeVariable();
    SubstituteAlternateReturns("NO");

    syn_reset_lex();

    ResetBlockStack();

    /* Get rid of partly declared variables... */
    if(mod!=entity_undefined) {
	/* Callers may already have pointers towards this function.
	 * The prettyprinter core dumps if entity_initial is
	 * destroyed. Maybe, I should clean the declarations field
	 * in code, as well as decls_text.
	 */
	/*
	entity_type(mod) = type_undefined;
	entity_storage(mod) = storage_undefined;
	entity_initial(mod) = value_undefined;
	*/
	value v = entity_initial(mod);
	code c = value_code(v);
	code_declarations(c) = NIL;
	code_decls_text(c) = string_undefined;

	CleanLocalEntities(mod);
    }

    /* The error may occur before the current module entity is defined */
    error_reset_current_module_entity();
    safe_fclose(syn_in, CurrentFN);
    free(CurrentFN);
    CurrentFN = NULL;

    /* GetChar() will reinitialize its own buffer when called */

    /* Because of the strange behavior of BeginingOfParsing*/
    /* CurrentPackage = NULL; */
    CurrentPackage = TOP_LEVEL_MODULE_NAME;
    /* Too bad for memory leak... */
    DynamicArea = entity_undefined;
    StaticArea = entity_undefined;
    HeapArea = entity_undefined;
    reset_common_size_map_on_error();
    parser_reset_all_reader_buffers();
    parser_reset_StmtHeap_buffer();
    reset_parser_recursive_call();
    soft_reset_alternate_returns();
    ResetChains();
    AbortEntries();
    AbortOfProcedure();

    InParserError = false;

    /* FI: let catch_error() take care of this in pipsmake since debug_on()
       was not activated in ParserError */
    /* debug_off(); */
    user_error(f,"Parser error between lines %d and %d\n%s\n",
	       line_b_I,line_e_I,m);

    /* Should never be executed */
    return true;
}


/* this function is called for each new file (FI: once?)
 * FI: I do not understand how this works. It has an effect only once
 * during a pips process lifetime. The error handling routine resets
 * CurrentPackage to NULL, as it is when the pips process is started.
 *
 * Should I:
 *
 *  A modify the error handling routine to reset CurrentPackage to
 *    TOP_LEVEL_MODULE_NAME?
 *
 *  B reset CurrentPackage to TOP_LEVEL_MODULE_NAME each time the parser
 *    is entered?
 *
 * I choose A.
 */
void BeginingOfParsing()
{
    static bool called = false;

    if (called)
	return;

    /* the current package is initialized */
    CurrentPackage = TOP_LEVEL_MODULE_NAME;
    called = true;
}

/* parse "module.dbr_file"
 */

static bool the_actual_parser(string module,
			      string dbr_file)
{
    string dir;
    debug_on("SYNTAX_DEBUG_LEVEL");

    /* set up parser properties */
    init_parser_properties();
    parser_init_macros_support();

    /* parser is initialized */
    BeginingOfParsing();

    /* scanner is initialized */
    ScanNewFile();

    pips_assert("CurrentFN is NULL", CurrentFN==NULL);
    dir = db_get_current_workspace_directory();
    CurrentFN =
	strdup(concatenate(dir, "/",
			   db_get_file_resource(dbr_file, module, true), 0));
    free(dir);

    /* yacc parser is called */
    syn_in = safe_fopen(CurrentFN, "r");
    syn_parse();
    safe_fclose(syn_in, CurrentFN);
    free(CurrentFN);
    CurrentFN = NULL;

    /* Handle the special case for entries without looping forever */
    if(!parser_recursive_call) {
	if(!EmptyEntryListsP()) {
	    /* The requested parsed code may have been an entry code. Then it
	     * is not yet computed because the parsed code for the module was
	     * produced and only a file resource was produced for the entry code.
	     */
	    ResetEntries();
	    if(!db_resource_p(DBR_PARSED_CODE, module)) {
		set_parser_recursive_call();
		the_actual_parser(module, dbr_file);
		reset_parser_recursive_call();
	    }
	}
    }
    else {
	if(!EmptyEntryListsP()) {
	    pips_internal_error("Unexpected entry handling in parser recursive call");
	}
    }

    /* This debug_off() occurs too late since pipsdbm has been called
     * before. Initially, the parser was designed to parse more than
     * one subroutine/function/program at a time.  */
    debug_off();

    return true;
}

/* parser for HPFC.
 * just a different input file not to touch the original source file.
 * this parser should be selected/activated automatically.
 */
bool hpfc_parser(string module)
{
    return the_actual_parser(module, DBR_HPFC_FILTERED_FILE);
}

bool parser(string module)
{
    return the_actual_parser(module, DBR_SOURCE_FILE);
}

void init_parser_properties()
{
  init_parser_reader_properties();
}
