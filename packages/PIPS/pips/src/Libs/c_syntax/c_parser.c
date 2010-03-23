/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

extern bool prettyprint_is_fortran;

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

// to store the mapping between the entity and its type stack

static hash_table entity_to_type_stack_table = hash_table_undefined;

void init_entity_type_storage_table()
{
  entity_to_type_stack_table = hash_table_make(hash_string,0);
  //put_stack_storage_table("test","T");
}

void put_to_entity_type_stack_table(entity key, stack value)
{
  if(stack_undefined_p(value))
    pips_internal_error("The stack must be defined");
  hash_put(entity_to_type_stack_table, entity_name(key),(void *) value);
}

stack get_from_entity_type_stack_table(entity key)
{
  void * p = hash_get(entity_to_type_stack_table, entity_name(key));

  if(p==HASH_UNDEFINED_VALUE)
    return stack_undefined;
  else
    return ((stack) p);
}

void remove_entity_type_stacks(list el)
{
  list ce = list_undefined;

  for(ce=el; !ENDP(ce); POP(ce)) {
    entity e = ENTITY(CAR(ce));
    //entity te = entity_undefined;
    void * p = hash_get(entity_to_type_stack_table, (void *) entity_name(e));
    //void * p = hash_delget(entity_to_type_stack_table, (void *) e, (void **) &te);

    pips_debug(8, "Remove type stack for \"%s\":", entity_name(e));
    //fprintf(stderr, "Remove type stack for \"%s\"\n", entity_name(e));
    //pips_debug(8, "get=%p, delget=%p\n", p1, p);
    if(p==HASH_UNDEFINED_VALUE) {
      ifdebug(8) {fprintf(stderr, "no associated stack\n");
      }
    }
    else {
      stack es = (stack) p;

      (void) hash_del(entity_to_type_stack_table, (void *) entity_name(e));
      if(!stack_undefined_p(es))
	stack_free(&es);
      ifdebug(8) fprintf(stderr, "done\n");
    }
  }
}

void reset_entity_type_stack_table()
{
  hash_table_free(entity_to_type_stack_table);
  entity_to_type_stack_table = hash_table_undefined;
}

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
  hash_put(keyword_typedef_table,"_Complex", (char *) TK_COMPLEX);
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
  hash_put(keyword_typedef_table,"__builtin_va_arg", (char *) TK_BUILTIN_VA_ARG);
  hash_put(keyword_typedef_table,"asm", (char *) TK_ASM);
  hash_put(keyword_typedef_table,"__asm__", (char *) TK_ASM);
  hash_put(keyword_typedef_table,"__volatile__", (char *) TK_VOLATILE);

  /* GNU predefined type(s), expecting no conflict with user named type */

  hash_put(keyword_typedef_table,"__builtin_va_list", (char *) TK_NAMED_TYPE);
  hash_put(keyword_typedef_table,"_Bool", (char *) TK_NAMED_TYPE);

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
_int
is_c_keyword_typedef(char * s)
{
  _int i = (_int) hash_get(keyword_typedef_table,s);
  return ((char *) i == HASH_UNDEFINED_VALUE) ? 0: i;
}

/* parsing function generated by Bison, from cyacc.y*/
extern void c_parse();

void CParserError(char *msg)
{
  entity mod = get_current_module_entity();
  string mod_name = entity_undefined_p(mod)? "entity_undefined":entity_user_name(mod);
  extern void c_reset_lex(void);
  extern int c_lineno;

  c_reset_lex();

  /* Reset the parser global variables ?*/

  pips_debug(4,"Reset current module entity %s\n", mod_name);

  /* The error may occur before the current module entity is defined */
  error_reset_current_module_entity();
  reset_current_dummy_parameter_number();

  // Get rid of partly declared variables
  if(mod!=entity_undefined) {
    /* Imported from syntax */
    extern void CleanLocalEntities(entity);
    value v = entity_initial(mod);
    code c = value_code(v);

    code_declarations(c) = NIL;
    code_decls_text(c) = string_undefined;
    CleanLocalEntities(mod);
  }

  // Free CalledModules?

  // Could not rebuild filename (A. Mensi)
  // c_in = safe_fopen(file_name, "r");
  // safe_fclose(c_in, file_name);

  /* Stacks are not allocated yet when dealing with external
     declarations. I assume that all stacks are declared
     simultaneously, hence a single test before freeing. */
  if(!entity_undefined_p(mod)) {
    reset_entity_type_stack_table();
    if(!stack_undefined_p(SwitchGotoStack)) {
      stack_free(&SwitchGotoStack);
      stack_free(&SwitchControllerStack);
      stack_free(&LoopStack);
      stack_free(&BlockStack);  
      /* Reset them to stack_undefined_p instead of STACK_NULL */
      SwitchGotoStack = stack_undefined;
      SwitchControllerStack = stack_undefined;
      LoopStack = stack_undefined;
      BlockStack = stack_undefined;

      stack_free(&ContextStack);
      stack_free(&FunctionStack);
      stack_free(&FormalStack);
      stack_free(&OffsetStack);
      stack_free(&StructNameStack);
      ContextStack = stack_undefined;
      FunctionStack = stack_undefined;
      FormalStack = stack_undefined;
      OffsetStack = stack_undefined;
      StructNameStack = stack_undefined;
    }
  }

  error_reset_current_C_line_number();
  /* get rid of all collected comments */
  reset_C_comment(TRUE);
  reset_expression_comment();

  pips_user_warning("\nRecovery from C parser failure not (fully) implemented yet.\n"
		    "C parser is likely to fail later if re-used.\n");
  pips_user_error("\n%s at line %d (%d)\n", msg, get_current_C_line_number(), c_lineno);
  debug_off();
}

static bool actual_c_parser(string module_name,
			    string dbr_file,
			    bool is_compilation_unit_parser)
{
    string dir = db_get_current_workspace_directory();
    string file_name =
      strdup(concatenate(dir, "/",
		     db_get_file_resource(dbr_file,module_name,TRUE), NULL));
    entity built_in_va_list = entity_undefined;
    entity built_in_bool = entity_undefined;
    entity built_in_complex = entity_undefined;
    entity built_in_va_start = entity_undefined;
    entity built_in_va_end = entity_undefined;
    entity built_in_va_copy = entity_undefined;

    free(dir);

    if (is_compilation_unit_parser)
      {
	compilation_unit_name = module_name;
	init_keyword_typedef_table();
      }
    else
      {
	compilation_unit_name = compilation_unit_of_module(module_name);
	keyword_typedef_table =
	  (hash_table) db_get_memory_resource(DBR_DECLARATIONS,
					      compilation_unit_name,TRUE);
      }

    ContextStack = stack_make(c_parser_context_domain,0,0);
    InitScope();
    FunctionStack = stack_make(entity_domain,0,0);
    FormalStack = stack_make(basic_domain,0,0);
    OffsetStack = stack_make(basic_domain,0,0);
    StructNameStack = stack_make(code_domain,0,0);

    loop_counter = 1;
    derived_counter = 1;
    CalledModules = NIL;

    debug_on("C_SYNTAX_DEBUG_LEVEL");
    pips_debug(1,"Module name: %s\n", module_name);
    pips_debug(1,"Compilation unit name: %s\n", compilation_unit_name);

    /* FI: not clean, but useful for debugging statement */
    ifdebug(1)
    {
      prettyprint_is_fortran = FALSE;
    }

    /* Predefined type(s): __builtin_va_list */
    built_in_va_list =
      find_or_create_entity(strdup(concatenate(compilation_unit_name,
					       MODULE_SEP_STRING,
					       TYPEDEF_PREFIX,
					       "__builtin_va_list",
					       NULL)));
    if(storage_undefined_p(entity_storage(built_in_va_list))) {
      entity_storage(built_in_va_list) = make_storage_rom();
      /* Let's lie about the real type */
      entity_type(built_in_va_list) =
	make_type(is_type_variable,
		  make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),
				NIL,
				NIL));
      entity_initial(built_in_va_list) = make_value_unknown();
    }
    built_in_bool =
      find_or_create_entity(strdup(concatenate(compilation_unit_name,
					       MODULE_SEP_STRING,
					       TYPEDEF_PREFIX,
					       "_Bool",
					       NULL)));
    if(storage_undefined_p(entity_storage(built_in_bool))) {
      entity_storage(built_in_bool) = make_storage_rom();
      entity_type(built_in_bool) =
	make_type(is_type_variable,
		  make_variable(make_basic_logical(DEFAULT_LOGICAL_TYPE_SIZE),
				NIL, NIL));
      entity_initial(built_in_bool) = make_value_unknown();
    }
    built_in_complex =
      find_or_create_entity(strdup(concatenate(compilation_unit_name,
					       MODULE_SEP_STRING,
					       TYPEDEF_PREFIX,
					       "_Complex",
					       NULL)));
    if(storage_undefined_p(entity_storage(built_in_complex))) {
      entity_storage(built_in_complex) = make_storage_rom();
      entity_type(built_in_complex) =
	make_type(is_type_variable,
		  make_variable(make_basic_complex(DEFAULT_COMPLEX_TYPE_SIZE),
				NIL, NIL));
      entity_initial(built_in_complex) = make_value_unknown();
    }

    /* Predefined functions(s): __builtin_va_end (va_arg() is parsed directly) */
    built_in_va_start =
      find_or_create_entity(strdup(concatenate(compilation_unit_name,
					       MODULE_SEP_STRING,
					       BUILTIN_VA_START,
					       NULL)));
    if(storage_undefined_p(entity_storage(built_in_va_start))) {
      basic va_list_b = make_basic(is_basic_typedef, built_in_va_list);
      type va_list_t =
	make_type(is_type_variable, make_variable(va_list_b, NIL, NIL));
      basic void_star_b = make_basic(is_basic_pointer, make_type_void(NIL));
      type void_start_t =
	make_type(is_type_variable, make_variable(void_star_b, NIL, NIL));
      entity_storage(built_in_va_start) = make_storage_rom();
      /* Let's lie about the real type... */
      entity_type(built_in_va_start) =
	make_type(is_type_functional,
		  make_functional(CONS(PARAMETER,
				       make_parameter(va_list_t,
						      make_mode(is_mode_value,
								UU),
						      make_dummy_unknown()),
				       CONS(PARAMETER,
					    make_parameter(void_start_t,
							   make_mode(is_mode_value, UU),
							   make_dummy_unknown()),
					    NIL)),
				  make_type(is_type_void,UU)));
      entity_initial(built_in_va_start) = make_value_intrinsic();
    }

    built_in_va_end = find_or_create_entity(strdup(concatenate(compilation_unit_name,
							    MODULE_SEP_STRING,
							    BUILTIN_VA_END,
							    NULL)));
    if(storage_undefined_p(entity_storage(built_in_va_end))) {
      basic va_list_b = make_basic(is_basic_typedef, built_in_va_list);
      type va_list_t =
	make_type(is_type_variable, make_variable(va_list_b, NIL, NIL));
      entity_storage(built_in_va_end) = make_storage_rom();
      /* Let's lie about the real type */
      entity_type(built_in_va_end) =
	make_type(is_type_functional,
		  make_functional(CONS(PARAMETER,
				       make_parameter(va_list_t,
						      make_mode(is_mode_value, UU),
						      make_dummy_unknown()),
				       NIL),
				  make_type(is_type_void,UU)));
      entity_initial(built_in_va_end) = make_value_intrinsic();
    }

    built_in_va_copy = find_or_create_entity(strdup(concatenate(compilation_unit_name,
							    MODULE_SEP_STRING,
							    BUILTIN_VA_COPY,
							    NULL)));
    if(storage_undefined_p(entity_storage(built_in_va_copy))) {
      basic va_list_b = make_basic(is_basic_typedef, built_in_va_list);
      type va_list_t =
	make_type(is_type_variable, make_variable(va_list_b, NIL, NIL));
      parameter va_list_p = make_parameter(va_list_t,
					   make_mode_value(),
					   make_dummy_unknown());
      entity_storage(built_in_va_copy) = make_storage_rom();
      /* Let's lie about the real type */
      entity_type(built_in_va_copy) =
	make_type(is_type_functional,
		  make_functional(CONS(PARAMETER,
				       va_list_p,
				       CONS(PARAMETER,
					    copy_parameter(va_list_p),
					    NIL)),
				  make_type(is_type_void,UU)));
      entity_initial(built_in_va_copy) = make_value_intrinsic();
    }

    if (compilation_unit_p(module_name))
      {
	/* Special case, set the compilation unit as the current module */
	MakeCurrentCompilationUnitEntity(module_name);
	/* I do not know to put this where to avoid repeated creations*/
	MakeTopLevelEntity();
      }

    /* discard_C_comment(); */
    set_current_C_line_number();
    init_C_comment();

    /* yacc parser is called */
    c_in = safe_fopen(file_name, "r");

    init_entity_type_storage_table();
    c_parse();

    safe_fclose(c_in, file_name);

    pips_assert("Module statement is consistent",
		statement_consistent_p(ModuleStatement));

    ifdebug(2)
      {
	pips_debug(2,"Module statement: \n");
	print_statement_of_module(ModuleStatement, module_name);

	pips_debug(2,"and declarations: ");
	print_entities(statement_declarations(ModuleStatement));
	FOREACH(ENTITY, e, statement_declarations(ModuleStatement)) {
	  pips_assert("e's type is defined", !type_undefined_p(entity_type(e)));
	  pips_assert("e's storage is defined", !storage_undefined_p(entity_storage(e)));
	  pips_assert("e's initial value is defined", !value_undefined_p(entity_initial(e)));
	  // Too strong because the normalize field of expressions in
	  //not initialized in the parser.
	  //pips_assert("e is fully defined", entity_defined_p(e));
	}
	if(!compilation_unit_p(module_name)) {
	  /* Even if you are not in a compilation unit, external
	     functions may be declared many times within one scope. */
	  pips_assert("Variables are declared once",
		      check_declaration_uniqueness_p(ModuleStatement));
	}
	else {
	  /* Variables allocated within the compilation unit should be declared only once, no? */
	  ; // no check yet
	}
	printf("\nList of callees:\n");
	MAP(STRING,s,
	{
	  printf("\t%s\n",s);
	},CalledModules);
      }

    if (compilation_unit_p(module_name))
      {
	ResetCurrentCompilationUnitEntity(is_compilation_unit_parser);
      }

    if (is_compilation_unit_parser)
      {
	/* Beware : the rule in pipsmake-rc.tex for compilation_unit_parser
	   does not include the production of parsed_code and callees.
	   This is not very clean, and is done to work around the way pipsmake
	   handles compilation units and modules.
	   There was no simple solution... BC.
	*/
	DB_PUT_MEMORY_RESOURCE(DBR_PARSED_CODE,
			       module_name,
			       (char *) ModuleStatement);
	DB_PUT_MEMORY_RESOURCE(DBR_DECLARATIONS,
			       module_name,
			       (void *) keyword_typedef_table);
	DB_PUT_MEMORY_RESOURCE(DBR_CALLEES,
			       module_name,
			       (char *) make_callees(NIL));
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
    reset_entity_type_stack_table(); /* Used to be done in ResetCurrentCompilationUnitEntity() */
    reset_current_C_line_number();
    reset_C_comment(compilation_unit_p(module_name));
    reset_current_dummy_parameter_number();
    reset_expression_comment();
    /*  reset_keyword_typedef_table();*/
    pips_assert("ContextStack is empty", stack_empty_p(ContextStack));
    stack_free(&ContextStack);
    stack_free(&FunctionStack);
    stack_free(&FormalStack);
    stack_free(&OffsetStack);
    stack_free(&StructNameStack);
    ContextStack = FunctionStack = FormalStack = OffsetStack = StructNameStack = stack_undefined;
    debug_off();
    return TRUE;
}

bool c_parser(string module_name)
{
  /* When the compilation_unit is parsed, it is parsed a second time
     and multiple declarations are certain to happen. */
   return actual_c_parser(module_name,DBR_C_SOURCE_FILE,FALSE);
}

bool compilation_unit_parser(string module_name)
{
  return actual_c_parser(module_name,DBR_C_SOURCE_FILE,TRUE);
}
