/* $id$
   $Log: c_parser.c,v $
   Revision 1.7  2004/02/18 10:32:18  nguyen
   Remove c_parser_tmp

   Revision 1.6  2003/12/05 17:16:21  nguyen
   Initialize and free global stacks

   Revision 1.5  2003/09/05 14:18:25  nguyen
   Put keywords and typedefs hash table into DECLARATIONS resource of each
   compilation unit.

   Revision 1.4  2003/08/13 07:58:41  nguyen
   Add compilation_unit_parser

   Revision 1.3  2003/08/06 14:12:55  nguyen
   Upgraded version of C parser

   Revision 1.2  2003/08/04 14:20:08  nguyen
   Preliminary version of the C parser

   Revision 1.1  2003/06/24 07:23:20  nguyen
   Initial revision
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "transformations.h"

#include "c_syntax.h"
#include "cyacc.h"

#include "c_parser_private.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "pipsdbm.h"
#include "pipsmake.h"

/* To avoid warnings */
extern char *strdup(const char *s1);

string compilation_unit_name; 

list CalledModules = NIL; 

statement ModuleStatement = statement_undefined;
  
stack ContextStack = stack_undefined;
stack FunctionStack = stack_undefined;
stack FormalStack = stack_undefined;
stack OffsetStack = stack_undefined;
stack StructNameStack = stack_undefined;

/* Global counter */
int loop_counter = 1; 
int derived_counter = 1; 
 
hash_table keyword_typedef_table = hash_table_undefined;

void init_keyword_typedef_table()
{
  keyword_typedef_table = hash_table_make(hash_string,0);
  hash_put(keyword_typedef_table,"auto", (char *) TK_AUTO);
  hash_put(keyword_typedef_table,"break", (char *) TK_BREAK);
  hash_put(keyword_typedef_table,"case", (char *) TK_CASE);
  hash_put(keyword_typedef_table,"char", (char *) TK_CHAR);
  hash_put(keyword_typedef_table,"const", (char *) TK_CONST);
  hash_put(keyword_typedef_table,"continue", (char *) TK_CONTINUE);
  hash_put(keyword_typedef_table,"default", (char *) TK_DEFAULT);
  hash_put(keyword_typedef_table,"do", (char *) TK_DO);
  hash_put(keyword_typedef_table,"double", (char *) TK_DOUBLE);
  hash_put(keyword_typedef_table,"else", (char *) TK_ELSE);
  hash_put(keyword_typedef_table,"enum", (char *) TK_ENUM);
  hash_put(keyword_typedef_table,"extern", (char *) TK_EXTERN);
  hash_put(keyword_typedef_table,"float", (char *) TK_FLOAT);
  hash_put(keyword_typedef_table,"for", (char *) TK_FOR);
  hash_put(keyword_typedef_table,"goto", (char *) TK_GOTO);
  hash_put(keyword_typedef_table,"if", (char *) TK_IF);
  hash_put(keyword_typedef_table,"inline", (char *) TK_INLINE);
  hash_put(keyword_typedef_table,"int", (char *) TK_INT);
  hash_put(keyword_typedef_table,"long", (char *) TK_LONG);
  hash_put(keyword_typedef_table,"register", (char *) TK_REGISTER);
  hash_put(keyword_typedef_table,"restrict", (char *) TK_RESTRICT);
  hash_put(keyword_typedef_table,"return", (char *) TK_RETURN);
  hash_put(keyword_typedef_table,"short", (char *) TK_SHORT);
  hash_put(keyword_typedef_table,"signed", (char *) TK_SIGNED);
  hash_put(keyword_typedef_table,"sizeof", (char *) TK_SIZEOF);
  hash_put(keyword_typedef_table,"static", (char *) TK_STATIC);
  hash_put(keyword_typedef_table,"struct", (char *) TK_STRUCT);
  hash_put(keyword_typedef_table,"switch", (char *) TK_SWITCH);
  hash_put(keyword_typedef_table,"typedef", (char *) TK_TYPEDEF);
  hash_put(keyword_typedef_table,"union", (char *) TK_UNION);
  hash_put(keyword_typedef_table,"unsigned", (char *) TK_UNSIGNED);
  hash_put(keyword_typedef_table,"void", (char *) TK_VOID);
  hash_put(keyword_typedef_table,"volatile", (char *) TK_VOLATILE);
  hash_put(keyword_typedef_table,"while", (char *) TK_WHILE);

  /* typedef names are added lately */
}

void reset_keyword_typedef_table()
{
  hash_table_free(keyword_typedef_table);
}

/* This function checks if s is a C keyword/typedef name or not by using
   the hash table keyword_typedef_table.
   It returns an integer number corresponding to the keyword. 
   It returns 0 if s is not a keyword/typedef name */

int is_c_keyword_typedef(char * s)
{
  int i = (int) hash_get(keyword_typedef_table,s);
  return (char *) i == HASH_UNDEFINED_VALUE? 0: i;
}

/* parsing function generated by Bison, from cyacc.y*/
extern void c_parse();

void CParserError(char *msg)
{
  /* Reset the parser global variables ?*/
  pips_user_error(msg);
}

static bool actual_c_parser(string module_name, string dbr_file, bool is_compilation_unit_parser)
{
    string dir = db_get_current_workspace_directory();
    string file_name = strdup(concatenate(dir,"/",db_get_file_resource(dbr_file,module_name,TRUE),0));
    free(dir);

    if (is_compilation_unit_parser)
      {
	compilation_unit_name = module_name;
	init_keyword_typedef_table();
      }
    else
      {
	compilation_unit_name = compilation_unit_of_module(module_name);
	keyword_typedef_table = (hash_table) db_get_memory_resource(DBR_DECLARATIONS,compilation_unit_name,TRUE); 
      }

    ContextStack = stack_make(c_parser_context_domain,0,0);
    FunctionStack = stack_make(entity_domain,0,0);
    FormalStack = stack_make(basic_domain,0,0);
    OffsetStack = stack_make(basic_domain,0,0);
    StructNameStack = stack_make(code_domain,0,0);
    
    loop_counter = 1; 
    derived_counter = 1;
    CalledModules = NIL;
    
    debug_on("C_SYNTAX_DEBUG_LEVEL");
 
    if (compilation_unit_p(module_name))
      {
	/* Special case, set the compilation unit as the current module */
	MakeCurrentCompilationUnitEntity(module_name);
	/* I do not know to put this where to avoid repeated creations*/
	MakeTopLevelEntity(); 
      }

    /* yacc parser is called */
    c_in = safe_fopen(file_name, "r");
    c_parse();
    safe_fclose(c_in, file_name);

    pips_assert("Module statement is consistent",statement_consistent_p(ModuleStatement));

    ifdebug(2)
      {
	pips_debug(2,"Module statement: \n");
	print_statement(ModuleStatement);
	pips_debug(2,"and declarations: ");
	print_entities(statement_declarations(ModuleStatement));
	printf("\nList of callees:\n");
	MAP(STRING,s,
	{
	  printf("\t%s\n",s);
	},CalledModules);
      }

    if (compilation_unit_p(module_name))
      {
	ResetCurrentCompilationUnitEntity();	
      }
  
    if (is_compilation_unit_parser)
      {
	DB_PUT_MEMORY_RESOURCE(DBR_DECLARATIONS, 
			       module_name, 
			       (void *) keyword_typedef_table);   
      }
    else 
      {
	DB_PUT_MEMORY_RESOURCE(DBR_PARSED_CODE, 
			       module_name, 
			       (char *) ModuleStatement);
	DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, 
			       module_name, 
			       (char *) make_callees(CalledModules));
      }
    free(file_name);
    file_name = NULL;
    /*  reset_keyword_typedef_table();*/
    stack_free(&ContextStack);
    stack_free(&FunctionStack);
    stack_free(&FormalStack);
    stack_free(&OffsetStack);
    stack_free(&StructNameStack);
    debug_off();
    return TRUE;
}

bool c_parser(string module_name)
{
  return actual_c_parser(module_name,DBR_C_SOURCE_FILE,FALSE);
}

bool compilation_unit_parser(string module_name)
{
  return actual_c_parser(module_name,DBR_C_SOURCE_FILE,TRUE);
}





