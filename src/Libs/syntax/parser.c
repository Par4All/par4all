#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "syntax.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "pipsdbm.h"

/*
 * declarations of extern variables
 */
/* name of the current file */
char *CurrentFN = NULL;

/* the current function */
/* entity CurrentFunction; */

/* list of formal parameters of the current function  */
cons *FormalParameters = NIL;

/* the name of the current package, i.e. TOP-LEVEL or a module name? */
char *CurrentPackage = NULL; 

/* two areas used to allocate variables which are not stored in 
   a common. These two areas are just like commons, but the dynamic 
   area is the only non-static area. */
entity DynamicArea = entity_undefined;
entity StaticArea = entity_undefined;

/* where the current instruction starts and ends. its label */
int line_b_I, line_e_I, line_b_C, line_e_C;
char lab_I[6];

/* a string that will contain the value of the format in case of format
statement */
char FormatValue[FORMATLENGTH];

extern void syn_reset_lex(void);

void 
ParserError(char * f, char * m)
{
    entity mod = get_current_module_entity();

    uses_alternate_return(FALSE);
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

    reset_current_module_entity();
    free(CurrentFN);
    CurrentFN = NULL;

    /* GetChar() will reinitialize its own buffer when called */

    /* Because of the strange behavior of BeginingOfParsing*/
    /* CurrentPackage = NULL; */
    CurrentPackage = TOP_LEVEL_MODULE_NAME;
    /* Too bad for memory leak... */
    DynamicArea = entity_undefined;
    StaticArea = entity_undefined;
    reset_common_size_map_on_error();
    parser_reset_all_reader_buffers();
    parser_reset_StmtHeap_buffer();
    safe_fclose(syn_in, CurrentFN);
    AbortOfProcedure();

    /* FI: let catch_error() take care of this in pipsmake since debug_on()
       was not activated in ParserError */
    /* debug_off(); */
    user_error(f,"Parser error between lines %d and %d\n%s\n",
	       line_b_I,line_e_I,m);
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
void 
BeginingOfParsing()
{
    static bool called = FALSE;

    if (called)
	return;

    /* the current package is initialized */
    CurrentPackage = TOP_LEVEL_MODULE_NAME;
    called = TRUE;
}

extern void syn_parse();

/* parse "module.dbr_file"
 */
static bool 
the_actual_parser(
    string module,
    string dbr_file)
{
    debug_on("SYNTAX_DEBUG_LEVEL");

    /* set up parser properties */
    init_parser_properties();
    parser_init_macros_support();

    /* parser is initialized */
    BeginingOfParsing();

    /* scanner is initialized */
    ScanNewFile();

    pips_assert("the_actual_parser", CurrentFN==NULL);
    CurrentFN = 
	strdup(concatenate(db_get_current_workspace_directory(),
			   "/",
			   db_get_file_resource(dbr_file, module, TRUE),
			   NULL));

    /* yacc parser is called */
    syn_in = safe_fopen(CurrentFN, "r");
    syn_parse();
    safe_fclose(syn_in, CurrentFN);
    free(CurrentFN);
    CurrentFN = NULL;

    /* This debug_off() occurs too late since pipsdbm has been called
     * before. Initially, the parser was designed to parse more than
     * one subroutine/function/program at a time.
     */
    debug_off();

    return TRUE;
}

/* parser for HPFC.
 * just a different input file not to touch the original source file.
 * this parser should be selected/activated automatically.
 */
bool 
hpfc_parser(string module)
{
    return the_actual_parser(module, DBR_HPFC_FILTERED_FILE);
}

bool 
parser(string module)
{
    return the_actual_parser(module, DBR_SOURCE_FILE);
}

void 
init_parser_properties()
{
  init_parser_reader_properties();
}
