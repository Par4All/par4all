/* 
   $Id$

   Try to prettyprint the RI in CLAIRE.
   Very basic at the time.

   print_claire_code        > MODULE.claire_printed_file
                            < PROGRAM.entities
                            < MODULE.code

   $Log: claire_prettyprinter.c,v $
   Revision 1.8  2004/04/02 14:55:40  hurbain
   Array generation works.

   Revision 1.7  2004/03/26 08:03:42  hurbain
   delock

   Revision 1.6  2004/03/25 14:38:18  hurbain
   Debug

   Revision 1.5  2004/03/25 14:35:11  hurbain
   *** empty log message ***

   Revision 1.4  2004/03/25 09:09:29  pips
   Removed pips error "not implemented yet." Hard to test with it ;)

   Revision 1.3  2004/03/24 16:02:03  hurbain
   First version for claire pretty printer. Currently only manages (maybe :p) arrays declarations.

   Revision 1.2  2004/03/11 15:09:43  irigoin
   function print_claire_code() declared for the link but not programmed yet.

*/

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "text-util.h"

#define COMMA         ","
#define EMPTY         ""
#define NL            "\n"
#define TAB           "    "
#define SEMICOLON     ";" NL
#define SPACE         " "

#define OPENBRACKET   "["
#define CLOSEBRACKET  "]"

#define OPENPAREN     "("
#define CLOSEPAREN    ")"

#define OPENBRACE     "{"
#define CLOSEBRACE    "}"

#define SHARPDEF      "#define"
#define COMMENT	      "//" SPACE

/* forward declaration. */
static string c_expression(expression);

/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
  (entity_function_p(get_current_module_entity()))


#define RESULT_NAME	"result"
static string claire_entity_local_name(entity var)
{
  string name;
  char * car;

  if (current_module_is_a_function() &&
      var != get_current_module_entity() &&
      same_string_p(entity_local_name(var), 
		    entity_local_name(get_current_module_entity())))
    name = RESULT_NAME;
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
    }

  /* switch to upper cases... */
  for (car = name; *car; car++)
    *car = (char) toupper(*car);
  
  return name;
}


/************************************************************** DECLARATIONS */

/* 
   integer a(n,m) -> int a[m][n];
   parameter (n=4) -> #define n 4
 */

static string 
int_to_string(int i)
{
  char buffer[50];
  sprintf(buffer, "%d", i);
  return strdup(buffer);
}

static string claire_dim_string(list ldim, string name)
{
  string result = "";
  int nbdim = 0;
  string origins = "origins = list<integer>(";
  string dimensions = "dimSizes = list<integer>(";
  string deuxpoints = " :: ";
  string data_array = "DATA_ARRAY(";
  string data_decl = "name = symbol!(";
  string dimstring = "dim = ";
  string datatype = "dataType = INTEGER)";

  if (ldim)
    {
      
      result = strdup(concatenate(name, deuxpoints, data_array, data_decl, name, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, dimstring, NULL));
      MAP(DIMENSION, dim, {
	expression elow = dimension_lower(dim);
	expression eup = dimension_upper(dim);
	int low;
	int up;
	nbdim++;
	if (expression_integer_value(elow, &low)){
	  if(nbdim != 1)
	    origins = strdup(concatenate(origins, COMMA ,int_to_string(low), NULL));
	  else
	    origins = strdup(concatenate(origins, int_to_string(low), NULL));
	}
	else pips_user_error("Array origins must be integer");

	if (expression_integer_value(eup, &up)){
	  if(nbdim != 1)
	    dimensions = strdup(concatenate(dimensions, COMMA ,int_to_string(up-low+1), NULL));
	  else
	    dimensions = strdup(concatenate(dimensions, int_to_string(up-low+1), NULL));
	}
	else pips_user_error("Array dimensions must be integer");
      }, ldim);
      result = strdup(concatenate(result, int_to_string(nbdim), COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, origins, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, dimensions, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, datatype, NL, NL, NULL));
    }
  printf("%s", result);
  return result;
}

static string this_entity_clairedeclaration(entity var)
{
  string result = NULL;
  string name = entity_local_name(var);
  type t = entity_type(var);
  storage s = entity_storage(var);
  pips_debug(2,"Entity name : %s\n",entity_name(var));
  /*  Many possible combinations */

  if (strstr(name,TYPEDEF_PREFIX) != NULL)
    /* This is a typedef name, what about typedef int myint[5] ???  */
    pips_user_error("Structs not supported");

  switch (type_tag(t)) {
  case is_type_variable:
    {
      variable v = type_variable(t);  
      string sd; // sd, svar, sq;
      //value val = entity_initial(var);
      //st = c_basic_string(variable_basic(v));
      sd = claire_dim_string(variable_dimensions(v), name);
      //sq = c_qualifier_string(variable_qualifiers(v));
      //svar = c_entity_local_name(var);
      
      
      result = strdup(concatenate(result, sd));
      /* problems with order !*/
      /*result = strdup(concatenate(sq, st, SPACE, svar, sd, NULL));
       */
      break;
    }
  case is_type_struct:
    {
      pips_user_error("Struct not allowed");
      break;
    }
  case is_type_union:
    {
      pips_user_error("Union not allowed");
      break;
    }
  case is_type_enum:
    {
      pips_user_error("Enum not allowed");
      break;
    }
  default:
  }
 
  return result? result: strdup("");
}

static string 
claire_declarations(entity module,
	       bool (*consider_this_entity)(entity),
	       string separator,
	       bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = TRUE;

  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "\n Prettyprinter declaration for variable :",claire_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = this_entity_clairedeclaration(var);
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = FALSE;
      }
  },code_declarations(c));
  return result;
}

static string claire_tasks(statement s)
{
  string result = "";
  instruction i = statement_instruction(s);
  list l = statement_declarations(s);
  printf("\nCurrent statement : \n");
  print_statement(s);
  printf("\nNo implementation of tasks at the moment\n");
  return result;
}

static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
}

static string claire_code_string(entity module, statement stat)
{
  string decls, tasks, result;

  /* What about declarations that are external a module scope ?
     Consider a source file as a module entity, put all declarations in it 
     (external static + TOP-LEVEL) */

  /* before_head only generates the constant declarations, such as #define*/
  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

  decls       = claire_declarations(module, variable_p, SEMICOLON, TRUE);
  tasks       = claire_tasks(stat);
  
  result = strdup(concatenate(decls, NL, tasks, NULL));

  free(decls);
  free(tasks);

  return result;
}


/******************************************************** PIPSMAKE INTERFACE */

#define CLAIREPRETTY    ".cl"

bool print_claire_code(string module_name)
{
  FILE * out;
  string ppt, claire, dir, filename;
  entity module;
  statement stat;

  claire = db_build_file_resource_name(DBR_CLAIRE_PRINTED_FILE, module_name, CLAIREPRETTY);

  module = local_name_to_top_level_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", claire, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("CLAIREPRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Begin Claire prettyprinter for %s\n", entity_name(module));
  ppt = claire_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();  

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "/* Claire pretty print for module %s. */\n%s", module_name, ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_CLAIRE_PRINTED_FILE, module_name, claire);

  reset_current_module_statement();
  reset_current_module_entity();

  return TRUE;
}
