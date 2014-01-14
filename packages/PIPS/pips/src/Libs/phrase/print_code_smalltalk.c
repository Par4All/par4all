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
 *
 * This phase is used for PHRASE project. 
 *
 *
 * NB: The PHRASE project is an attempt to automatically (or semi-automatically)
 * transform high-level language for partial evaluation in reconfigurable logic 
 * (such as FPGAs or DataPaths). 
 *
 * This pass is used in context of PHRASE project for synthetisation of 
 * reconfigurable logic for a portion of initial code. This function can be 
 * viewed as a Smalltalk pretty-printer of a subset of Fortran. 
 *
 * alias print_code_smalltalk 'Smalltalk Pretty-Printer'
 *
 * print_code_smalltalk        > MODULE.smalltalk_code
 *       < PROGRAM.entities 
 *       < MODULE.code
 *
 * The Smalltalk code will be available in SMALLTALK_CODE_FILE 
 *
 * NB: This code is highly inspired from PRINT_C_CODE phase written by nguyen
 *
 */ 

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "text-util.h"


#define STPRETTY		".st"

#include "smalltalk-defs.h"

#define RESULT_NAME	"result"

#define current_module_is_a_function() \
  (entity_function_p(get_current_module_entity()))

static string st_statement(statement s);
static string st_sequence(sequence seq);
static string st_call(call c);
static string st_expression(expression);
static string st_reference(reference r);
static string st_test(test t);
static string st_loop(loop l);
static string st_whileloop(whileloop w);
static string st_forloop(forloop f);


/**
 * Return beautified string representing name for entity var
 */
static string st_entity_local_name(entity var)
{
  const char* name;

  pips_debug(6,"st_entity_local_name was : %s\n",entity_local_name(var));

  if (current_module_is_a_function() &&
      var != get_current_module_entity() &&
      same_string_p(entity_local_name(var), 
		    entity_local_name(get_current_module_entity()))) {
    name = RESULT_NAME;
  }
  else
    {
      name = entity_local_name(var);
  
      /* Delete all the prefixes */

      if (strstr(name,STRUCT_PREFIX) != NULL)
	name = strstr(name,STRUCT_PREFIX) + 1;
      if (strstr(name,UNION_PREFIX) != NULL)
	name = strstr(name,UNION_PREFIX) + 1;
      if (strstr(name,ENUM_PREFIX) != NULL)
	name = strstr(name,ENUM_PREFIX) + 1;      
      if (strstr(name,TYPEDEF_PREFIX) != NULL)
	name = strstr(name,TYPEDEF_PREFIX) + 1;
      if (strstr(name,MEMBER_SEP_STRING) != NULL)
	name = strstr(name,MEMBER_SEP_STRING) + 1;
      if (strstr(name,MAIN_PREFIX) != NULL)
	name = strstr(name,MAIN_PREFIX) + 1;
    }
  pips_debug(6,"st_entity_local_name is now : %s\n",name);
  return strdup(name);
}


/**
 * Return string representing expression enclosed by parenthesis
 */
static string st_brace_expression_as_string(expression exp)
{
  string result = OPENBRACKET;
  list args = call_arguments(syntax_call(expression_syntax(exp)));
  
  bool first = true;
  MAP(EXPRESSION,e,
  {
    if (brace_expression_p(e))
      result = strdup(concatenate(result,first?"":",",st_brace_expression_as_string(e),NULL));
    else
      result = strdup(concatenate(result,first?"":",",words_to_string(words_expression(e, NIL)),NULL));
    first = false;
  },args);
  result = strdup(concatenate(result,CLOSEBRACKET,NULL));
  return result;
}

/**
 * Return a string representing dimension reference for a 
 * dimension dim and an expression e
 * This function automatically convert bounds in fortran to bounds
 * starting from 0 by doing new_reference = old_reference - lower
 * This function  is valid even in case of non-directly-cumputable expressions
 */
static string st_dimension_reference_as_string (dimension dim, expression old_expression) {

  intptr_t low, old;
  string slow = NULL;
  string sold = NULL;
  bool low_given_by_expression = false;
  bool old_given_by_expression = false;
  string result = strdup(EMPTY);
  
  expression elow = dimension_lower(dim);
  expression eold = old_expression;
  
  if (expression_integer_value(elow, &low)) {
    low_given_by_expression = true;
  }
  else {
    low_given_by_expression = false;
    slow = st_expression(elow);
  }
  if (expression_integer_value(eold, &old)) {
    old_given_by_expression = true;
  }
  else {
    old_given_by_expression = false;
    sold = st_expression(eold);
  }

  if (low_given_by_expression) {
    if (old_given_by_expression) {
      pips_debug(5,"old=%"PRIdPTR" low=%"PRIdPTR"\n", old, low);
      string istr = i2a(old-low);
      result = strdup(concatenate(result, istr, NULL));
      free(istr);
    }
    else {
      pips_debug(5,"sold=%s low=%"PRIdPTR"\n", sold, low);
      string istr = i2a(low);
      result = strdup(concatenate(result, sold,"-",istr, NULL));
      free(istr);
    }
    }
  else {
    if (old_given_by_expression) {
      pips_debug(5,"old=%"PRIdPTR" slow=%s\n", old, slow);
      string istr = i2a(old);
      result = strdup(concatenate(result, istr,"-",
				  OPENBRACE,slow,CLOSEBRACE, NULL));
      free(istr);
    }
    else {
      pips_debug(5,"sold=%s slow=%s\n", sold, slow);
      result = strdup(concatenate(result, sold,"-",OPENBRACE,slow,
				  CLOSEBRACE, NULL));
    }
  }
  
  return result;
}

/**
 * Return a string representing dimension bounds of a dimension dim
 * This function automatically convert bounds in fortran to bounds
 * starting from 0 by doing upbound = (upper - lower + 1)
 * This function  is valid even in case of non-directly-cumputable expressions
 */
static string st_dimension_bound_as_string (dimension dim) {
  
  intptr_t low, up;
  string slow = NULL;
  string sup = NULL;
  bool low_given_by_expression = false;
  bool up_given_by_expression = false;
  string result = strdup(EMPTY);
  
  expression elow = dimension_lower(dim);
  expression eup = dimension_upper(dim);
  
  if (expression_integer_value(elow, &low)) {
    low_given_by_expression = true;
  }
  else {
    low_given_by_expression = false;
    slow = st_expression(elow);
  }
  if (expression_integer_value(eup, &up)) {
    up_given_by_expression = true;
  }
  else {
    up_given_by_expression = false;
    sup = st_expression(eup);
  }

  if (low_given_by_expression) {
    if (up_given_by_expression) {
      pips_debug(5,"up=%"PRIdPTR" low=%"PRIdPTR"\n", up, low);
      string istr = i2a(up-low+1);
      result = strdup(concatenate(result, istr, NULL));
      free(istr);
    }
    else {
      string istr = i2a(low-1);
      pips_debug(5,"sup=%s low=%"PRIdPTR"\n", sup, low);
      result = strdup(concatenate(result, sup,"-",istr, NULL));
      free(istr);
    }
    }
  else {
    if (up_given_by_expression) {
      pips_debug(5,"up=%"PRIdPTR" slow=%s\n", up, slow);
      string istr = i2a(up+1);
      result = strdup(concatenate(result, istr,"-",
				  OPENBRACE,slow,CLOSEBRACE, NULL));
      free(istr);
    }
    else {
      pips_debug(5,"sup=%s slow=%s\n", sup, slow);
      result = strdup(concatenate(result, sup,"-",OPENBRACE,slow,
				  CLOSEBRACE,"+1", NULL));
    }
  }
  
  return result;
}

/**
 * Return string representing array initialization 
 * for variable svar in SMALLTALK
 */
static string st_dim_string (string svar, list ldim)
{
  string result = "";
  int dimensions = 0;

  dimensions = gen_length(ldim);

  pips_debug(5,"Dimension : %d \n", dimensions);

  if (dimensions == 0) {
    return strdup(result);
  }

  else if (dimensions == 1) {

    dimension dim = DIMENSION(gen_nth(0,ldim));

    result = strdup(concatenate(svar, SPACE, SETVALUE, SPACE, 
				ARRAY, SPACE, 
				ARRAY_NEW, SPACE, 
				st_dimension_bound_as_string (dim), NULL));
    return result;
  }

  else if (dimensions == 2) {

    dimension dim = DIMENSION(gen_nth(0,ldim));
    dimension dim2 = DIMENSION(gen_nth(1,ldim));

    result = strdup(concatenate(svar, SPACE, SETVALUE, SPACE, 
				ARRAY2D, SPACE, 
				ARRAY2D_NEW1, SPACE, st_dimension_bound_as_string (dim), SPACE,
				ARRAY2D_NEW2, SPACE, st_dimension_bound_as_string (dim2),
				NULL));
    return result;
  }

  else {
    result = strdup("More than 2-dimensionals arrays not handled !");
    return result;
  }
}

/**
 * Return a string C-like representation of basic b
 */
static string c_basic_string(basic b)
{
  string result = "UNKNOWN_BASIC" SPACE;
  switch (basic_tag(b))
    {
    case is_basic_int:
      {
	pips_debug(2,"Basic int\n");
	switch (basic_int(b))
	  {
	  case 1: result = "char" SPACE; 
	    break;
	  case 2: result = "short" SPACE; 
	    break;
	  case 4: result = "int" SPACE; 
	    break;
	  case 6: result = "long" SPACE; 
	    break;
	  case 8: result = "long long" SPACE; 
	    break;
	  case 11: result = "unsigned char" SPACE;
	    break;
	  case 12: result = "unsigned short" SPACE;
	    break;
	  case 14: result = "unsigned int" SPACE;
	    break;
	  case 16: result = "unsigned long" SPACE;
	    break;
	  case 18: result = "unsigned long long" SPACE;
	    break;
	  case 21: result = "signed char" SPACE;
	    break;
	  case 22: result = "signed short" SPACE;
	    break;
	  case 24: result = "signed int" SPACE;
	    break;
	  case 26: result = "signed long" SPACE;
	    break;
	  case 28: result = "signed long long" SPACE;
	    break;
	  }
	break;
      }
    case is_basic_float:
      switch (basic_float(b))
	{
	case 4: result = "float" SPACE;
	  break;
	case 8: result = "double" SPACE;
	  break;
	}
      break;
    case is_basic_logical:
      result = "int" SPACE;
      break;
    case is_basic_string:
      result = "char" SPACE;
      break;
    case is_basic_bit:
      {
	result = "Basic bit not handled";
	break;
      }
    case is_basic_pointer:
      {
	result = "Basic pointer not handled";
	break;
      }
    case is_basic_derived:
      {
	result = "Basic derived not handled";
	break;
      }
    case is_basic_typedef:
      {
	result = "Basic typedef not handled";
	break;
      }  
    default:
      pips_internal_error("case not handled");
    }
  return strdup(result);
}

/**
 * Return a string representing Smalltalk declaration for
 * entity (constant or variable) var
 * NB: old function this_entity_cdeclaration(entity var)
 */
static string st_declaration(entity var)
{
  string result = "Undefined entity";
  const char* name = entity_local_name(var);
  type t = entity_type(var);
  storage s = entity_storage(var);

  switch (storage_tag(s)) {
  case is_storage_rom: 
    {
      string svar = st_entity_local_name(var);
      result = strdup(svar);
      free(svar);
    }
  default: 
    break;
  }
  switch (type_tag(t)) {
  case is_type_variable:
    {
      string svar;
      svar = st_entity_local_name(var);
      result = strdup(svar);
      free(svar);
      break;
    }
  case is_type_struct:
    {
      result = strdup(concatenate(name, ": undefined STRUCT in SMALLTALK", NULL));
      break;
    }
  case is_type_union:
    {
      result = strdup(concatenate(name, ": undefined UNION in SMALLTALK", NULL));
      break;
    }
  case is_type_enum:
    {
      result = strdup(concatenate(name, ": undefined ENUM in SMALLTALK", NULL));
      break;
    }
  default:
      break;
  }
 
  return result? result: strdup("");
}

/**
 * Return a string representing Smalltalk declaration 
 * initialisation for entity (constant or variable) var
 */
static string st_declaration_init(entity var)
{
  string result = NULL;
  type t = entity_type(var);
  storage s = entity_storage(var);

  pips_debug(2,"st_declaration_init for entity : %s\n",entity_name(var));
  
  switch (storage_tag(s)) {
  case is_storage_rom: 
    {
      /* This is a constant, we must initialize it */

      value va = entity_initial(var);
      
      if (!value_undefined_p(va))
	{
	  constant c = NULL;
	  pips_debug(4,"Constant with defined value\n");
	  if (value_constant_p(va))
	    c = value_constant(va);
	  else if (value_symbolic_p(va))
	    c = symbolic_constant(value_symbolic(va));
	  if (c)
	    {
	      if (constant_int_p(c))
		{
		  string sval = i2a(constant_int(c));
		  string svar = st_entity_local_name(var);
		  pips_debug(4,"Constant is an integer\n");
		  result = strdup(concatenate(svar, SPACE, SETVALUE
					      SPACE, sval, NULL));
		  
		  free(svar);
		  free(sval);
		  return result;
		}
	      else 
		{
		  string svar = st_entity_local_name(var);
		  pips_debug(4,"Type of constant not handled\n");
		  result = strdup(concatenate(svar, SPACE, SETVALUE
					      SPACE, "undefined", NULL));
		}
	    }
	}
      break;
    }
  default:
    break;
  }
  switch (type_tag(t)) {
  case is_type_variable:
    {
      int dimensions;
      variable v = type_variable(t);
      string svar;
      value val = entity_initial(var);

      svar = st_entity_local_name(var);

      dimensions = gen_length(variable_dimensions(v));
      pips_debug(4,"Dimensions: %zd\n", gen_length(variable_dimensions(v)));

      if (dimensions == 0) {

	if (!value_undefined_p(val)) {
	  if (value_expression_p(val)) {
	    /* This variable must be initialized
	     * Anyway, i don't know how to initialize a variable
	     * at declaration in Fortran !!! */
	    expression exp = value_expression(val);
	    if (brace_expression_p(exp))
	      result = strdup(concatenate(result,SETVALUE,st_brace_expression_as_string(exp),NULL));
	    else
	      result = strdup(concatenate(result,SETVALUE,words_to_string(words_expression(exp, NIL)),NULL));
	  }
	}
      }

      else if (dimensions < 3) {
	pips_debug(2,"Init for arrays \n");
	result = strdup(st_dim_string (svar, variable_dimensions(v)));
      }

      else {
	pips_debug(2,"Arrays dimension > 2 not handled\n");
      }

      free(svar);
      break;
    }
  case is_type_struct:
    {
      result = "undefined STRUCT in SMALLTALK";
      break;
    }
  case is_type_union:
    {
      result = "undefined UNION in SMALLTALK";
      break;
    }
  case is_type_enum:
    {
      result = "undefined ENUM in SMALLTALK";
      break;
    }
  default:
    break;
  }
 
  return result;
}

/**
 * Return a string representing Smalltalk declaration 
 * initialisation for entity (constant or variable) var
 */
static string st_declaration_comment(entity var)
{
  string comment = "Undefined entity";
  string svar = st_entity_local_name(var);
  type t = entity_type(var);

  pips_debug(2,"st_declaration_comment for entity : %s\n",entity_name(var));
  
  switch (type_tag(t)) {
  case is_type_variable:
    {
      int dimensions;
      variable v = type_variable(t);
      string st = c_basic_string(variable_basic(v));

      dimensions = gen_length(variable_dimensions(v));
      pips_debug(4,"Dimensions: %zd\n", gen_length(variable_dimensions(v)));

      if (dimensions == 0) {
	comment = strdup(concatenate(COMMENT, svar, ",", st, COMMENT, NULL));
      }
      
      else if (dimensions < 3) {
	
	if (dimensions == 1) {
	  comment = strdup(concatenate(COMMENT, svar, ",", st, ", 1 dimension", COMMENT, NULL));
	}
	else if (dimensions == 2) {
	  comment = strdup(concatenate(COMMENT, svar, ",", st, ", 2 dimensions", COMMENT, NULL));
	}
      }

      else {
	  comment = strdup(concatenate(COMMENT, svar, ",", st, ", Arrays dimension > 2 not handled", COMMENT, NULL));
      }

      break;
    }
  case is_type_struct:
    {
      comment = strdup(concatenate(COMMENT, svar, " : undefined STRUCT in SMALLTALK", COMMENT, NULL));
      break;
    }
  case is_type_union:
    {
      comment = strdup(concatenate(COMMENT, svar, " : undefined UNION in SMALLTALK", COMMENT, NULL));
      break;
    }
  case is_type_enum:
    {
      comment = strdup(concatenate(COMMENT, svar, " : undefined ENUM in SMALLTALK", COMMENT, NULL));
      break;
    }
  default:
      comment = strdup(concatenate(COMMENT, svar, " : undefined declaration in SMALLTALK", COMMENT, NULL));
  }
 
  free(svar);
  return comment;
}

/**
 * This function return a bool indicating if related entity e
 * represents a constant
 */
static bool constant_p(entity e)
{
  /* Constant variables */
  return storage_rom_p(entity_storage(e)) && 
    value_symbolic_p(entity_initial(e)) &&
    type_functional_p(entity_type(e));
}

/**
 * This function return a bool indicating if related entity e
 * represents a variable
 */
static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
}

/**
 * This function return a bool indicating if related entity e
 * represents an argument
 */
static bool argument_p(entity e)
{
  /* Formal variables */
  return type_variable_p(entity_type(e)) && 
    storage_formal_p(entity_storage(e));
}

/**
 * Return string representing arguments declaration
 * written in SmallTalk style
 */
static string 
st_arguments (entity module,
	      bool (*consider_this_entity)(entity),
	      string separator,
	      bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = true;

  /* Assert that entity represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "\n Prettyprinter declaration for argument :",st_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = strdup(concatenate("with:",st_entity_local_name(var), NULL));
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = false;
      }
  },code_declarations(c));
  return result;
}

/**
 * Return string representing variables or constants declaration
 * written in SmallTalk style
 */
static string 
st_declarations (entity module,
		 bool (*consider_this_entity)(entity),
		 string separator,
		 bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = true;

  /* Assert that entity represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "\n Prettyprinter declaration for variable :",st_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = st_declaration(var);
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = false;
      }
  },code_declarations(c));
  return result;
}

/**
 * Return string representing variables or constants declaration
 * initialisation written in SmallTalk style
 */
static string 
st_declarations_init(entity module,
		     bool (*consider_this_entity)(entity),
		     string separator,
		     bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = true;

  /* Assert that entity represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "Prettyprinter declaration initialisation for variable :",st_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = st_declaration_init(var);
	if (svar != NULL) {
	  result = strdup(concatenate(old, !first && !lastsep? separator: "",
				      svar, lastsep? separator: "", NULL));
	}
	else {
	  result = strdup(result);
	}
	free(old);
	free(svar);
	first = false;
      }
  },code_declarations(c));
  return result;
}

/**
 * Return string representing variables or constants declaration
 * initialisation written in SmallTalk style
 */
static string 
st_declarations_comment(entity module,
			bool (*consider_this_entity)(entity),
			string separator,
			bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = true;
  
  /* Assert that entity represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));
  
  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "Prettyprinter declaration initialisation for variable :",st_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = st_declaration_comment(var);
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = false;
      }
  },code_declarations(c));
  return result;
}

/**
 * Generate header for SMALLTALK module
 */
static string st_header(entity module)
{
  string result, svar, args;
  
  pips_assert("it is a function", type_functional_p(entity_type(module)));
  
  svar = st_entity_local_name(module);
  
  /* Generates the arguments declarations */
  args = st_arguments(module, 
		      argument_p, 
		      SPACE, 
		      true);

  result = strdup(concatenate(svar, SPACE, args, NL,
			      COMMENT, "Automatically generated with PIPS", COMMENT,
			      NL, NULL));
  
  return result;
}

/*********************************************************
 * Generate SMALLTALK code as String from module
 * root statement
 *********************************************************/

static string smalltalk_code_string(entity module, statement stat)
{
  string st_head, st_variables, st_constants;
  string st_variables_init, st_constants_init;
  string st_variables_comment;
  string st_body, result;

  ifdebug(2) {
    printf("Module statement: \n");
    print_statement(stat);
    printf("and declarations: \n");
    print_entities(statement_declarations(stat));
  }
  
  /* HEAD generates the header */
  st_head = st_header(module);
  ifdebug(3) {
    printf("HEAD: \n");
    printf("%s \n", st_head);
  }
  
  /* Generates the variables declarations */
  /* What about declarations associated to statements ??? */
  st_variables       = st_declarations(module, 
				       variable_p, 
				       SPACE, 
				       true);
  ifdebug(3) {
    printf("VARIABLES: \n");
    printf("%s \n", st_variables);
  }

  /* Generates the constant declarations */
  st_constants = st_declarations (module, 
				  constant_p, 
				  SPACE, 
				  true);
  ifdebug(3) {
    printf("CONSTANTS: \n");
    printf("%s \n", st_constants);
  }

  /* Generates the variables declarations initialisation */
  st_variables_init = st_declarations_init (module, 
					    variable_p, 
					    STSEMICOLON, 
					    true);
  ifdebug(3) {
    printf("VARIABLES INIT: \n");
    printf("%s \n", st_variables_init);
  }

  /* Generates the variables declarations comments */
  st_variables_comment = st_declarations_comment (module, 
						  variable_p, 
						  NL, 
						  true);
  ifdebug(3) {
    printf("VARIABLES COMMENT: \n");
    printf("%s \n", st_variables_comment);
  }

  /* Generates the constant declarations initialisation */
  st_constants_init = st_declarations_init (module, 
					    constant_p, 
					    STSEMICOLON, 
					    true);
  ifdebug(3) {
    printf("CONSTANTS INIT: \n");
    printf("%s \n", st_constants_init);
  }

  /* Generates the body */
  st_body = st_statement(stat);
  ifdebug(3) {
    printf("BODY: \n");
    printf("%s \n", st_body);
  }
  
  result = strdup(concatenate(st_head, NL, 
			      st_variables_comment, NL
			      BEGINTEMPVAR, st_constants, 
			      st_variables, ENDTEMPVAR, NL, NL,
			      st_constants_init, NL,
			      st_variables_init, NL,
			      st_body, NL, 
			      NULL));

  free(st_head);
  free(st_variables);
  free(st_constants);
  free(st_body);

  return result;
}

/*************************************************************** EXPRESSIONS */

typedef string (*prettyprinter)(string, list);

struct s_ppt
{
  char * intrinsic;
  char * c;
  prettyprinter ppt;
};

static bool expression_needs_parenthesis_p(expression);

/**
 * Return string representation for a list of expression le
 * representing an assignement, asserting that le is a list
 * of expressions containing exactly TWO expressions
 */
static string ppt_assignement (string in_smalltalk, list le)
{
  string result, svar;
  expression e1, e2;
  string s1, s2;
  bool p1, p2, pr1, pr2;
  syntax s;
  reference r;
  entity var;
  type t;
  variable v;
  list ldim;

  pips_assert("2 arguments to assignment call", gen_length(le)==2);

  e1 = EXPRESSION(CAR(le));
  s = expression_syntax(e1);
  pips_assert("assignment call: first expression is reference", 
	      syntax_tag(s) == is_syntax_reference);


  r = syntax_reference(s);
  var = reference_variable(r);
  t = entity_type(var);
  v = type_variable(t);
  ldim = variable_dimensions(v);

  svar = st_entity_local_name(var);

  e2 = EXPRESSION(CAR(CDR(le)));
  p2 = expression_needs_parenthesis_p(e2);
  s2 = st_expression(e2);


  if (gen_length(ldim) == 0) {
    /* This is a scalar variable */

    p1 = expression_needs_parenthesis_p(e1);
    s1 = st_reference(r);
    result = strdup(concatenate(p1? OPENPAREN: EMPTY, s1, p1? CLOSEPAREN: EMPTY,
				SPACE, in_smalltalk, SPACE, 
				p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
				NULL));
    free(s1);
  }
  
  else if (gen_length(ldim) == 1) {

    dimension dim = DIMENSION(gen_nth(0,ldim));
    expression e = EXPRESSION(gen_nth(0,reference_indices(r)));
    pr1 = expression_needs_parenthesis_p(e);

    dim = DIMENSION(gen_nth(0,ldim));
      
    result = strdup(concatenate(svar, SPACE, ARRAY_AT_PUT_1, SPACE,
				pr1? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim, e), 
				pr1? CLOSEPAREN: EMPTY, SPACE,
				ARRAY_AT_PUT_2, SPACE,
				p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
				NULL));
  }
  
  else if (gen_length(ldim) == 2) {
    
    dimension dim1 = DIMENSION(gen_nth(0,ldim));
    expression e1 = EXPRESSION(gen_nth(0,reference_indices(r)));
    dimension dim2 = DIMENSION(gen_nth(1,ldim));
    expression e2 = EXPRESSION(gen_nth(1,reference_indices(r)));
    pr1 = expression_needs_parenthesis_p(e1);
    pr2 = expression_needs_parenthesis_p(e2);
    
    result = strdup(concatenate(svar, SPACE, ARRAY2D_AT_AT_PUT_1, SPACE,
				pr1? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim1, e1), 
				pr1? CLOSEPAREN: EMPTY, SPACE,
				ARRAY2D_AT_AT_PUT_2, SPACE,
				pr2? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim2, e2), 
				pr2? CLOSEPAREN: EMPTY, SPACE,
				ARRAY2D_AT_AT_PUT_3, SPACE,
				p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
				NULL));
  }

  else {
    result = strdup("Arrays more than 2D are not handled !");
  }

  free(s2);

  return result;
}

/**
 * Return string representation for a list of expression le
 * representing a BINARY relation, asserting that le is a list
 * of expressions containing exactly TWO expressions
 */
static string ppt_binary(string in_smalltalk, list le)
{
  string result;
  expression e1, e2;
  string s1, s2;
  bool p1, p2;

  pips_assert("2 arguments to binary call", gen_length(le)==2);

  e1 = EXPRESSION(CAR(le));
  p1 = expression_needs_parenthesis_p(e1);
  s1 = st_expression(e1);

  e2 = EXPRESSION(CAR(CDR(le)));
  p2 = expression_needs_parenthesis_p(e2);
  s2 = st_expression(e2);

  result = strdup(concatenate(p1? OPENPAREN: EMPTY, s1, p1? CLOSEPAREN: EMPTY,
			      SPACE, in_smalltalk, SPACE, 
			      p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
			      NULL));

  free(s1);
  free(s2);

  return result;
}

/**
 * Return string representation for a list of expression le
 * representing a UNARY relation, asserting that le is a list
 * of expressions containing exactly ONE expression
 */
static string ppt_unary(string in_smalltalk, list le)
{
  string e, result;
  pips_assert("one arg to unary call", gen_length(le)==1);
  e = st_expression(EXPRESSION(CAR(le)));
  result = strdup(concatenate(in_smalltalk, SPACE, e, NULL));
  free(e);
  return result;
}

/**
 * Return string representation for a list of expression le
 * representing a UNARY POST relation, asserting that le is a list
 * of expressions containing exactly ONE expression
 */
static string ppt_unary_post(string in_smalltalk, list le)
{
  string e, result;
  pips_assert("one arg to unary post call", gen_length(le)==1);
  e = st_expression(EXPRESSION(CAR(le)));
  result = strdup(concatenate(e, SPACE, in_smalltalk, NULL));
  free(e);
  return result;
}

static string ppt_call(string in_smalltalk, list le)
{
  string scall, old;
  if (le == NIL)
    { 
      scall = strdup(concatenate(in_smalltalk, NULL));
    }
  else 
    {
      bool first = true;
      scall = strdup(concatenate(in_smalltalk, OPENPAREN, NULL));
     
      /* Attention: not like this for io statements*/
      MAP(EXPRESSION, e,
      {
	string arg = st_expression(e);
	old = scall;
	scall = strdup(concatenate(old, first? "": ", ", arg, NULL));
	free(arg);
	free(old);
	first = false;
      },le);

      old = scall;
      scall = strdup(concatenate(old, CLOSEPAREN, NULL));
      free(old);
    }
  return scall;
}

/**
 * This data structure encodes the differents intrinsic allowing
 * to convert fortran code to smalltalk code
 */
static struct s_ppt intrinsic_to_smalltalk[] = 
{
  { "+", "+", ppt_binary  },
  { "-", "-", ppt_binary },
  { "/", "/", ppt_binary },
  { "*", "*", ppt_binary },
  { "--", "-", ppt_unary },
  { "**", "**", ppt_binary },
  { "=", SETVALUE, ppt_assignement },
  { ".OR.", "||", ppt_binary },
  { ".AND.", "&&", ppt_binary },
  { ".NOT.", "!", ppt_unary },
  { ".LT.", "<", ppt_binary },
  { ".GT.", ">", ppt_binary },
  { ".LE.", "<=", ppt_binary },
  { ".GE.", ">=", ppt_binary },
  { ".EQ.", "==", ppt_binary },
  { ".NE.", "!=", ppt_binary },
  { ".EQV.", "==", ppt_binary },
  { ".NEQV.", "!=", ppt_binary },
  { ".", ".", ppt_binary },
  { "->", "->", ppt_binary},
  { "post++", "++", ppt_unary_post },
  {"post--", "--" , ppt_unary_post },
  {"++pre", "++" , ppt_unary },
  {"--pre", "--" , ppt_unary },
  {"&", "&" , ppt_unary },
  {"*indirection", "*" , ppt_unary },
  {"+unary", "+", ppt_unary },
  {"-unary", "-", ppt_unary },
  {"~", "~", ppt_unary },
  {"!", "!", ppt_unary },  
  {"%", "%" , ppt_binary },  
  {"+C", "+" , ppt_binary },
  {"-C", "-", ppt_binary }, 
  {"<<", "<<", ppt_binary },
  {">>", ">>", ppt_binary }, 
  {"<", "<" , ppt_binary },
  {">", ">" , ppt_binary },
  {"<=", "<=", ppt_binary },
  {">=", ">=", ppt_binary }, 
  {"==", "==", ppt_binary },
  {"!=", "!=", ppt_binary },  
  {"&bitand", "&", ppt_binary}, 
  {"^", "^", ppt_binary },
  {"|", "|", ppt_binary },
  {"&&", "&&", ppt_binary }, 
  {"||", "||", ppt_binary },  
  {"*=", "*=", ppt_binary },
  {"/=", "/=", ppt_binary },
  {"%=", "%=", ppt_binary },
  {"+=", "+=", ppt_binary },
  {"-=", "-=", ppt_binary },
  {"<<=", "<<=" , ppt_binary },
  {">>=", ">>=", ppt_binary },
  {"&=", "&=", ppt_binary },
  {"^=", "^=", ppt_binary },
  {"|=","|=" , ppt_binary },
  { NULL, NULL, ppt_call }
};

/**
 * Return the prettyprinter structure for SmallTalk
 */

static struct s_ppt * get_ppt(entity f)
{
  const char* called = entity_local_name(f);
  struct s_ppt * table = intrinsic_to_smalltalk;
  while (table->intrinsic && !same_string_p(called, table->intrinsic))
    table++;
  return table;
}

/**
 * Return bool indicating if expression e must be enclosed 
 * in parenthesis
 */
static bool expression_needs_parenthesis_p(expression e)
{
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      {
	struct s_ppt * p = get_ppt(call_function(syntax_call(s)));
	return p->ppt==ppt_binary;
      }
    case is_syntax_reference:
    case is_syntax_range:
    default:
      return false;
    }
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s
 */
static string st_statement(statement s)
{
  string result;
  instruction i = statement_instruction(s);
  list l = statement_declarations(s);
  ifdebug(3) {
    printf("\nCurrent statement : \n");
    print_statement(s);
  }
  switch (instruction_tag(i))
    {
    case is_instruction_test:
      {
	test t = instruction_test(i);
	pips_debug(2, "Instruction TEST\n");   
	result = st_test(t);
	break;
      }
    case is_instruction_sequence:
      {
	sequence seq = instruction_sequence(i);
	pips_debug(2, "Instruction SEQUENCE\n");   
	result = st_sequence(seq);
	break;
      }
    case is_instruction_loop:
      {
	loop l = instruction_loop(i);
	pips_debug(2, "Instruction LOOP\n");   
	result = st_loop(l);
	break;
      }
    case is_instruction_whileloop:
      {
	whileloop w = instruction_whileloop(i);
	pips_debug(2, "Instruction WHILELOOP\n");   
	result = st_whileloop(w);
	break;
      }
    case is_instruction_forloop:
      {
	forloop f = instruction_forloop(i);
	pips_debug(2, "Instruction FORLOOP\n");   
	result = st_forloop(f);
	break;
      }
    case is_instruction_call:
      {
	string scall = st_call(instruction_call(i));
	pips_debug(2, "Instruction CALL\n");   
	result = strdup(concatenate(scall, STSEMICOLON, NULL));
	break;
      }
    case is_instruction_unstructured:
      {
	/*unstructured u = instruction_unstructured(i);*/
	pips_debug(2, "Instruction UNSTRUTURED\n");   
	result = strdup(concatenate(COMMENT, 
				    "UNSTRUCTURED: Instruction not implementable in SMALLTALK", 
				    COMMENT, NL, NULL));
	break;
      }
    case is_instruction_goto:
      {
	/*statement g = instruction_goto(i);*/
	pips_debug(2, "Instruction GOTO\n");   
	result = strdup(concatenate(COMMENT, 
				    "GOTO: Instruction not implementable in SMALLTALK", 
				    COMMENT, NL, NULL));
	break;
      }
      /* add switch, forloop break, continue, return instructions here*/
    default:
      pips_user_warning("Instruction NOT IMPLEMENTED\n");   
      result = strdup(concatenate(COMMENT, " Instruction not implemented" NL, NULL));
      break;
    }

  if (!ENDP(l))
    {
      string decl = ""; 
      MAP(ENTITY, var,
      {
	string svar;
	debug(2, "\n In block declaration for variable :",st_entity_local_name(var));   
	svar = st_declaration(var);
	decl = strdup(concatenate(decl, svar, STSEMICOLON, NULL));
	free(svar);
      },l);
      result = strdup(concatenate(decl,result,NULL));
    }

  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a Test Statement (IF/THEN/ELSE)
 */
static string st_test(test t)
{
  string result;
  bool no_false;
  string cond, strue, sfalse;

  cond = st_expression(test_condition(t));
  strue = st_statement(test_true(t));
  no_false = empty_statement_p(test_false(t));
  
  sfalse = no_false? NULL: st_statement(test_false(t));
  
  if (no_false) {
    result = strdup(concatenate(OPENPAREN, cond, CLOSEPAREN, NL,
				ST_IFTRUE, SPACE, OPENBRACKET, NL, strue, 
				CLOSEBRACKET, STSEMICOLON,
				NULL));
  }
  else {
    result = strdup(concatenate(OPENPAREN, cond, CLOSEPAREN, NL,
				ST_IFTRUE, SPACE, OPENBRACKET, NL, strue, 
				CLOSEBRACKET, NL,  
				ST_IFFALSE, SPACE, OPENBRACKET, NL, sfalse, 
				CLOSEBRACKET, STSEMICOLON,
				NULL));
  }
  free(cond);
  free(strue);
  if (sfalse) free(sfalse);
  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a Sequence Statement 
 * (an ordered set of sequential statements)
 */
static string st_sequence(sequence seq)
{
  string result = strdup(EMPTY);
  MAP(STATEMENT, s,
  {
    string oldresult = result;
    string current = st_statement(s);
    if (current != NULL) {
      result = strdup(concatenate(oldresult, current, NULL));
    }
    else {
      result = strdup(oldresult);
    }
    free(current);
    free(oldresult);
  }, sequence_statements(seq));
  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a Loop Statement (DO...ENDDO)
 */
static string st_loop(loop l)
{
  string result, initialisation, loopbody, incrementation;
  string body = st_statement(loop_body(l));
  string index = st_entity_local_name(loop_index(l));
  range r = loop_range(l);
  string low = st_expression(range_lower(r));
  string up = st_expression(range_upper(r));
  string inc = st_expression(range_increment(r));
  intptr_t incasint;

  initialisation = strdup(concatenate(index, SPACE, SETVALUE, SPACE, low, STSEMICOLON, NULL));

  if (expression_integer_value(range_increment(r), &incasint)) {
      string istr;
    if (incasint >= 0) {
         istr = i2a(incasint);
      inc = strdup(concatenate(ST_PLUS, SPACE, 
			       istr, NULL));
    }
    else {
         istr = i2a(-incasint);
      inc = strdup(concatenate(ST_MINUS, SPACE, 
			       istr, NULL));
    }
    free(istr);
  }
  else {
    inc = strdup(concatenate(ST_PLUS, SPACE, 
			     st_expression(range_increment(r)), NULL));
  }

  incrementation = strdup(concatenate(index, SPACE, SETVALUE, SPACE, index, SPACE, inc, NULL));

  loopbody = strdup(concatenate(OPENBRACKET, index, SPACE, ST_LE, 
				SPACE, up, CLOSEBRACKET, SPACE,
				ST_WHILETRUE, SPACE, OPENBRACKET, NL,
				body, incrementation, 
				STSEMICOLON, CLOSEBRACKET, NULL));
				

  result = strdup(concatenate(initialisation, loopbody, STSEMICOLON, NULL));

  free(initialisation);
  free(incrementation);
  free(loopbody);
  free(body);
  free(low);
  free(up);
  free(index);
  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a While-Loop Statement (DO WHILE...ENDDO)
 */
static string st_whileloop(whileloop w)
{
  string result;
  string body = st_statement(whileloop_body(w));
  string cond = st_expression(whileloop_condition(w));
  /*evaluation eval = whileloop_evaluation(w);*/

  result = strdup(concatenate(OPENBRACKET, cond, CLOSEBRACKET, SPACE,
			      ST_WHILETRUE, SPACE, OPENBRACKET, NL,
			      body, 
			      CLOSEBRACKET, STSEMICOLON, NULL));
  
  free(cond);
  free(body);
  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a For-Loop Statement (I don't know how to 
 * specify in fortran !!!)
 */
static string st_forloop(forloop f)
{
  string result, loopbody;
  string body = st_statement(forloop_body(f));
  string init = st_expression(forloop_initialization(f));
  string cond = st_expression(forloop_condition(f));
  string inc = st_expression(forloop_increment(f));
  result = strdup(concatenate("for (", init, ";",cond,";",inc,") {" NL, 
			      body, "}" NL, NULL));
  
  loopbody = strdup(concatenate(OPENBRACKET, cond, CLOSEBRACKET, SPACE,
				ST_WHILETRUE, SPACE, OPENBRACKET, NL,
				body, inc, 
				STSEMICOLON, CLOSEBRACKET, NULL));
				

  result = strdup(concatenate(init, loopbody, STSEMICOLON, NULL));

  free(loopbody);
  free(inc);
  free(cond);
  free(init);
  free(body);
  return result;
}

/**
 * This method returns Smalltalk-like string representation (pretty-print)
 * for a statement s which is a Call Statement (a code line)
 */
static string st_call(call c)
{
  entity called = call_function(c);
  struct s_ppt * ppt = get_ppt(called);
  string result;

  /* special case... */
  if (same_string_p(entity_local_name(called), "STOP")) {
	result = NULL;
  }
  else if (same_string_p(entity_local_name(called), "CONTINUE")) {
	result = NULL;
  }
  else if (same_string_p(entity_local_name(called), "RETURN"))
    {
      if (entity_main_module_p(get_current_module_entity()))
	result = strdup(RETURNVALUE " 0");
      else if (current_module_is_a_function())
	result = strdup(RETURNVALUE SPACE RESULT_NAME);
      else
	result = strdup(RETURNVALUE);
    }
  else if (call_constant_p(c))
    {
      result = st_entity_local_name(called);
    }
  else
    {
      string s = st_entity_local_name(called);
      result = ppt->ppt(ppt->c? ppt->c: s, call_arguments(c));
      free(s);
    }

  return result;
}

/**
 * This function return a string representation of a reference r.
 * A reference is an array element, considering non-array variables
 * (scalar variables) are 0-dimension arrays elements. We must here
 * differently manage scalar, 1-D arrays (using SmallTalk Array 
 * class) and 2-D arrays (using SmallTalk Array2D).
 *
 * NB: in Fortran, the indexes are reversed
 */
static string st_reference(reference r)
{
  string result = strdup(EMPTY), svar;

  entity var = reference_variable(r);
  type t = entity_type(var);
  variable v = type_variable(t);
  list ldim = variable_dimensions(v);
  bool pr, pr1, pr2;

  svar = st_entity_local_name(reference_variable(r));

  if (gen_length(ldim) == 0) {
    
    /* This is a scalar variable, no need to manage array indices */
    result = strdup(st_entity_local_name(var));
    free(svar);
    return result;
  }

  else if (gen_length(ldim) == 1) {
    
    dimension dim = DIMENSION(gen_nth(0,ldim));
    expression e = EXPRESSION(gen_nth(0,reference_indices(r)));
    pr = expression_needs_parenthesis_p(e);

    dim = DIMENSION(gen_nth(0,ldim));
      
    result = strdup(concatenate(OPENPAREN, svar, SPACE, ARRAY_AT, SPACE,
				pr? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim, e), 
				pr? CLOSEPAREN: EMPTY,
				CLOSEPAREN, NULL));
  }
  
  else if (gen_length(ldim) == 2) {
    
    dimension dim1 = DIMENSION(gen_nth(0,ldim));
    expression e1 = EXPRESSION(gen_nth(0,reference_indices(r)));
    dimension dim2 = DIMENSION(gen_nth(1,ldim));
    expression e2 = EXPRESSION(gen_nth(1,reference_indices(r)));
    pr1 = expression_needs_parenthesis_p(e1);
    pr2 = expression_needs_parenthesis_p(e2);
    
    result = strdup(concatenate(OPENPAREN, svar,
				SPACE, ARRAY2D_AT_AT_1, SPACE, 
				pr1? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim1, e1),
				pr1? CLOSEPAREN: EMPTY,
				SPACE, ARRAY2D_AT_AT_2, SPACE,
				pr2? OPENPAREN: EMPTY,
				st_dimension_reference_as_string (dim2, e2),
				pr2? CLOSEPAREN: EMPTY,
				CLOSEPAREN, NULL));
  }
  
  else {
    
    result = strdup(concatenate(COMMENT, "Arrays more than 2D are not handled !", 
				COMMENT, NULL));
  }
  
  free(svar);
  return result;

}

static string st_expression(expression e)
{
  string result = NULL;
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      result = st_call(syntax_call(s));
      break;
    case is_syntax_range:
      result = strdup("range not implemented");
      break;
    case is_syntax_reference:
      result = st_reference(syntax_reference(s));
      break;
      /* add cast, sizeof here */
    default:
      pips_internal_error("unexpected syntax tag");
    }
  return result;
}

/*********************************************************
 * Phase main
 *********************************************************/

bool print_code_smalltalk(const char* module_name)
{
  FILE * out;
  string ppt, smalltalkcode, dir, filename;
  entity module;
  statement stat;

  /* We first build the future resource file, with a .st */
  smalltalkcode = db_build_file_resource_name(DBR_SMALLTALK_CODE_FILE, module_name, STPRETTY);
  module = module_name_to_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", smalltalkcode, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("SMALLTALK_PRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Begin SMALLTALK prettyprinter for %s\n", entity_name(module));
  ppt = smalltalk_code_string(module, stat);
  pips_debug(1, "End SMALLTALK prettyprinter for %s\n", entity_name(module));

  pips_debug(3, "What i got is \n%s\n", ppt);

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "/* SMALLTALK pretty print for module %s. */\n%s", module_name, ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_SMALLTALK_CODE_FILE, module_name, smalltalkcode);

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}
