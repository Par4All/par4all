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

#define DEBUG_XML 1

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "c_parser_private.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "text-util.h"

#include "effects-convex.h"
#include "effects-generic.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "semantics.h"

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

#define OPENANGLE     "<"
#define CLOSEANGLE    ">"
#define SLASH	      "/"

#define SHARPDEF      "#define"
#define COMMENT	      "//" SPACE
#define QUOTE         "\""


#define XML_TASK_PREFIX "T_"
#define XML_MOTIF_PREFIX "M_"
#define XML_ARRAY_PREFIX "A_"

#define XML_RL      NL,TAB,TAB

#define code_is_a_box 0
#define code_is_a_te 1
#define code_is_a_main 2
#define USER_MAIN "main"
/* array containing extern loop indices names */
static gen_array_t extern_indices_array;
/* array containing intern loop indices (name : "M_") */
static gen_array_t intern_indices_array;
/* array containing extern upperbounds */
static gen_array_t extern_upperbounds_array;
/* array containing intern upperbounds */
static gen_array_t intern_upperbounds_array;
/* array containing the tasks names*/
static gen_array_t tasks_names;

static const char* global_module_name;
#define BL             " "
#define TB1            "\t"

static hash_table hash_entity_def_to_task = hash_table_undefined;
static const char* global_module_name;
static int global_margin =0;

bool fortran_appli = true;
static bool box_in_statement_p=false;
static bool motif_in_statement_p=false;
static list cumulated_list=NIL;

#define MEM_PREFIX "-MEM:"
static string array_location_string;
static string array_mem_string;

char *cornerturn_info=NULL;

/**************************************************************** MISC UTILS */

#define current_module_is_a_function()			\
  (entity_function_p(get_current_module_entity()))

static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
}


#define RESULT_NAME	"result"
static string xml_entity_local_name(entity var)
{
  const char* name;

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
  char *rname=strupper(strdup(name),name);
  return rname;
}


/************************************************************** DECLARATIONS */

/*
  integer a(n,m) -> int a[m][n];
  parameter (n=4) -> #define n 4
*/



/* forward declaration */
static string xml_expression(expression);

/* Attention with Fortran: the indices are reversed. */
static string xml_reference_with_explicit_motif(reference r)
{
  string result = strdup(EMPTY), old, svar;
  MAP(EXPRESSION, e,
      {
	string s = strdup(xml_expression(e));
	old = result;
	result = strdup(concatenate(old, OPENBRACKET, s, CLOSEBRACKET, NULL));
	free(old);
	free(s);
      }, reference_indices(r));

  old = result;
  svar = xml_entity_local_name(reference_variable(r));
  result = strdup(concatenate(svar, old, NULL));
  free(old);
  free(svar);
  return result;
}

static string xml_expression(expression e)
{
  string result = string_undefined;
  syntax s = expression_syntax(e);

  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      result = xml_reference_with_explicit_motif(syntax_reference(s));
      break;
    case is_syntax_call: {
      value ev = EvalExpression(e);
      constant ec = value_constant(ev);
      int eiv = 0;

      if(!value_constant_p(ev)) {
	pips_user_error("Constant expected for XML loop bounds.\n");
      }
      if(!constant_int_p(ec)) {
	pips_user_error("Integer constant expected for XML loop bounds.\n");
      }
      eiv = constant_int(ec);
      result = strdup(itoa(eiv));
      break;
    }
    default:
      pips_internal_error("unexpected syntax tag");
    }
  return result;
}

static gen_array_t array_names;
static gen_array_t array_dims;

#define ITEM_NOT_IN_ARRAY -1

static int gen_array_index(gen_array_t ar, string item){
  int i;

  for(i = 0; i<(int) gen_array_nitems(ar); i++){
    if(gen_array_item(ar, i) != NULL){
      if(same_string_p(item, *((string *)(gen_array_item(ar, i))))){
	return i;
      }
    }
  }
  return ITEM_NOT_IN_ARRAY;
}

static string xml_dim_string(list ldim, string name)
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
  string name4p = name;
  string * namep = malloc(sizeof(string));
  int * nbdimptr = malloc(sizeof(int));
  *namep = name4p;
  if (ldim)
    {
      result = strdup(concatenate(name, deuxpoints, data_array, data_decl, QUOTE, name, QUOTE, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, dimstring, NULL));
      MAP(DIMENSION, dim, {
	  expression elow = dimension_lower(dim);
	  expression eup = dimension_upper(dim);

	  intptr_t low;
	  intptr_t up;
	  nbdim++;
	  if (expression_integer_value(elow, &low)){
	    if(nbdim != 1)
	      origins = strdup(concatenate(origins, COMMA ,i2a(low), NULL));
	    else
	      origins = strdup(concatenate(origins, i2a(low), NULL));
	  }
	  else pips_user_error("Array origins must be integer\n");

	  if (expression_integer_value(eup, &up)){
	    if(nbdim != 1)
	      dimensions = strdup(concatenate(dimensions, COMMA ,i2a(up-low+1), NULL));
	    else
	      dimensions = strdup(concatenate(dimensions, i2a(up-low+1), NULL));
	  }
	  else pips_user_error("Array dimensions must be integer\n");
	}, ldim);
      *nbdimptr = nbdim;
      gen_array_append(array_dims, nbdimptr);
      gen_array_append(array_names, namep);
      result = strdup(concatenate(result, i2a(nbdim), COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, origins, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, dimensions, CLOSEPAREN, COMMA, NL, NULL));
      result = strdup(concatenate(result, TAB, datatype, NL, NL, NULL));
    }
  return result;
}

static string this_entity_xmldeclaration(entity var)
{
  string result = strdup("");
  string name = strdup(concatenate("A_", entity_local_name(var), NULL));
  type t = entity_type(var);
  pips_debug(2,"Entity name : %s\n",entity_name(var));
  /*  Many possible combinations */

  if (strstr(name,TYPEDEF_PREFIX) != NULL)
    pips_user_error("Structs not supported\n");

  switch (type_tag(t)) {
  case is_type_variable:
    {
      variable v = type_variable(t);
      string sd;
      sd = strdup(xml_dim_string(variable_dimensions(v), name));
      result = strdup(concatenate(result, sd, NULL));
      break;
    }
  case is_type_struct:
    {
      pips_user_error("Struct not allowed\n");
      break;
    }
  case is_type_union:
    {
      pips_user_error("Union not allowed\n");
      break;
    }
  case is_type_enum:
    {
      pips_user_error("Enum not allowed\n");
      break;
    }
  default:
    pips_user_error("Something not allowed here\n");
  }

  return result;
}

static string
xml_declarations_with_explicit_motif(entity module,
				     bool (*consider_this_entity)(entity),
				     string separator,
				     bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = true;

  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
      {
	debug(2, "\n Prettyprinter declaration for variable :",xml_entity_local_name(var));
	if (consider_this_entity(var))
	  {
	    string old = strdup(result);
	    string svar = strdup(this_entity_xmldeclaration(var));
	    result = strdup(concatenate(old, !first && !lastsep? separator: "",
					svar, lastsep? separator: "", NULL));
	    free(old);
	    free(svar);
	    first = false;
	  }
      },code_declarations(c));
  return result;
}

static string xml_array_in_task(reference r, bool first, int task_number);

static string xml_call_from_assignation(call c, int task_number, bool * input_provided){
  /* All arguments of this call are in Rmode (inputs of the task) */
  /* This function is called recursively */
  list arguments = call_arguments(c);
  syntax syn;
  string result = "";

  MAP(EXPRESSION, expr, {
      syn = expression_syntax(expr);
      switch(syntax_tag(syn)){
      case is_syntax_call:{
	result = strdup(concatenate(result, xml_call_from_assignation(syntax_call(syn), task_number, input_provided), NULL));
	break;
      }
      case is_syntax_reference:{
	reference ref = syntax_reference(syn);
    string svar = xml_entity_local_name(reference_variable(ref));
	string varname = strdup(concatenate("A_", svar, NULL));
    free(svar);
	if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	  result = strdup(concatenate(result, xml_array_in_task(ref, false, task_number), NULL));
	  *input_provided = true;
	}
	break;
      }
      default:{
	pips_user_error("only call and references allowed here\n");
      }
      }
    }, arguments);
  return result;
}

static void xml_call_from_indice(call c, string * offset_array, string paving_array[], string fitting_array[]){
  entity called = call_function(c);
  string funname = xml_entity_local_name(called);
  list arguments = call_arguments(c);
  syntax args[2];
  int i = 0;
  int iterator_nr;
  if(gen_length(arguments)==2){
    if(same_string_p(funname, "+") || same_string_p(funname, "-") || same_string_p(funname, "*")){
      MAP(EXPRESSION, arg, {
	  args[i] = expression_syntax(arg);
	  i++;
	}, arguments);
      if(same_string_p(funname, "+")){
	if(syntax_tag(args[0]) == is_syntax_call){
	  xml_call_from_indice(syntax_call(args[0]), offset_array, paving_array, fitting_array);
	}
	if(syntax_tag(args[1]) == is_syntax_call){
	  xml_call_from_indice(syntax_call(args[1]), offset_array, paving_array, fitting_array);
	}
	if(syntax_tag(args[0]) == is_syntax_reference){
	  reference ref = syntax_reference(args[0]);
	  if((iterator_nr = gen_array_index(extern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    paving_array[iterator_nr] = strdup("1");
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    fitting_array[iterator_nr] = strdup("1");
	  }
	}
	if(syntax_tag(args[1]) == is_syntax_reference){
	  reference ref = syntax_reference(args[1]);
	  if((iterator_nr = gen_array_index(extern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    paving_array[iterator_nr] = strdup("1");
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    fitting_array[iterator_nr] = strdup("1");
	  }
	}
      }
      else if(same_string_p(funname, "-")){
	if(syntax_tag(args[1]) == is_syntax_call && gen_length(call_arguments(syntax_call(args[1])))==0){
	  if(syntax_tag(args[0]) == is_syntax_reference){
	    reference ref = syntax_reference(args[0]);
	    if((iterator_nr = gen_array_index(extern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      paving_array[iterator_nr] = strdup("1");
	    }
	    else if((iterator_nr = gen_array_index(intern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      fitting_array[iterator_nr] = strdup("1");
	    }
	  }
	  if(syntax_tag(args[0]) == is_syntax_call){
	    xml_call_from_indice(syntax_call(args[0]), offset_array, paving_array, fitting_array);
	  }
	  xml_call_from_indice(syntax_call(args[1]), offset_array, paving_array, fitting_array);
	}
	else {
	  pips_user_error("APOTRES doesn't allow negative coefficients in paving and fitting matrices\n");
	}
      }
      else if(same_string_p(funname, "*")){
	if(syntax_tag(args[0]) != is_syntax_call || syntax_tag(args[1]) != is_syntax_reference || gen_length(call_arguments(syntax_call(args[0])))!=0 ){
	  pips_user_error("Only scalar * reference are allowed here. Please develop expressions.\n");
	}
	else {
	  int intern_nr = gen_array_index(intern_indices_array, xml_entity_local_name(reference_variable(syntax_reference(args[1]))));
	  int extern_nr = gen_array_index(extern_indices_array, xml_entity_local_name(reference_variable(syntax_reference(args[1]))));
	  string mult =  strdup(xml_entity_local_name(call_function(syntax_call(args[0]))));
	  if(extern_nr != ITEM_NOT_IN_ARRAY){
	    paving_array[extern_nr] = mult;
	  }
	  else if(intern_nr != ITEM_NOT_IN_ARRAY){
	    fitting_array[intern_nr] = strdup(mult);
	  }
	}
      }
    }
    else{
      pips_user_error("only linear expression of indices allowed\n");
    }
  }
  else if(gen_length(arguments) == 0){
    *offset_array = funname;
  }
  else{
    pips_user_error("only +, -, * and constants allowed\n");
  }
}

#define XML_ARRAY_PREFIX "A_"

static string xml_array_in_task(reference r, bool first, int task_number){
  /* XML name of the referenced array */
  string varname = strdup(concatenate(XML_ARRAY_PREFIX,
				      xml_entity_local_name(reference_variable(r)),
				      NULL));
  /* iterator for dimensions of array */
  int indice_nr = 0;
  list indices = reference_indices(r);
  string result = "";
  /* number of external loops*/
  int extern_nb = gen_array_nitems(extern_indices_array);

  /* number of dimensions of referenced array */
  int index_of_array = gen_length(indices); /*((int *) (gen_array_item(array_dims, gen_array_index(array_names, varname))));*/

  /* number of internal loops*/
  int intern_nb = gen_array_nitems(intern_indices_array);

  /* list of offsets for XML code */
  string offset_array[index_of_array];
  /* paving matrix for XML code
     1st coeff: array dimension (row index)
     2nd coeff: iteration dimension (column index) */
  string paving_array[index_of_array][extern_nb];

  /* fitting matrix for XML code
     1st coeff: array dimension
     2nd coeff: iteration dimension*/
  string fitting_array[index_of_array][intern_nb];
  int i;
  int j;
  int depth = 0;

  bool null_fitting_p = true;
  string internal_index_declarations = strdup("");
  string fitting_declaration = strdup("");
  string fitting_declaration2 = strdup("");

  /* initialization of the arrays */
  for (i=0; i<index_of_array; i++)
    offset_array[i] = "0";

  for (i=0; i<index_of_array ; i++)
    for (j=0; j<extern_nb; j++)
      paving_array[i][j] = "0";

  for (i=0; i<index_of_array ; i++)
    for (j=0; j<intern_nb; j++)
      fitting_array[i][j] = "0";

  /* XML reference header */
  result = strdup(concatenate(result, "DATA(name = symbol!(\"", "T_", i2a(task_number),
			      "\" /+ \"", varname, "\"),", NL, TAB, TAB, NULL));

  result = strdup(concatenate(result, "darray = ", varname, "," NL, TAB, TAB, "accessMode = ", (first?"Wmode,":"Rmode,"),
			      NL, TAB, TAB, "offset = list<VARTYPE>(", NULL));

  /* Fill in paving, fitting and offset matrices from index expressions. */
  MAP(EXPRESSION, ind, {
      syntax sind = expression_syntax(ind);
      int iterator_nr;
      switch(syntax_tag(sind)){
      case is_syntax_reference:{
	reference ref = syntax_reference(sind);
	if((iterator_nr = gen_array_index(extern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	  paving_array[indice_nr][iterator_nr] = strdup("1");
	}
	else if((iterator_nr = gen_array_index(intern_indices_array, xml_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	  fitting_array[indice_nr][iterator_nr] = strdup("1");
	}

	break;
      }
      case is_syntax_call:{
	call c = syntax_call(sind);
	xml_call_from_indice(c, &(offset_array[indice_nr]), paving_array[indice_nr], fitting_array[indice_nr]);
	break;
      }
      default:{
	pips_user_error("Only call and reference allowed in indices.\n");
	break;
      }
      }
      indice_nr++;
    }, indices);


  /* generate offset list in XML code */
  for(i=0; i<index_of_array - 1; i++){
    result=strdup(concatenate(result, "vartype!(", offset_array[i],"), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", offset_array[i], "))," NL, NULL));

  /* fitting header */
  result = strdup(concatenate(result, TAB, TAB, "fitting = list<list[VARTYPE]>(", NULL));

  /* XML column-major storage of fitting matrix */
  for(i=0;i<intern_nb; i++){
    bool is_null_p = true;
    for(j = 0; j<index_of_array; j++){
      is_null_p = is_null_p && (same_string_p(fitting_array[j][i], "0"));
    }
    if(!is_null_p){
      null_fitting_p = false;
      fitting_declaration = strdup(concatenate(fitting_declaration, "list(", NULL));
      for(j = 0; j<index_of_array-1; j++){
	fitting_declaration = strdup(concatenate(fitting_declaration, "vartype!(", fitting_array[j][i], "), ", NULL));
      }
      fitting_declaration = strdup(concatenate(fitting_declaration,
					       "vartype!(",
					       fitting_array[j][i],
					       ")), ",
					       NULL));
    }
  }

  if(!null_fitting_p){
    fitting_declaration2 =
      strdup(concatenate(gen_strndup0(fitting_declaration,
				      strlen(fitting_declaration) - 2),
			 "),", NL, TAB, TAB, TAB, NULL));
    result = strdup(concatenate(result, fitting_declaration2, NULL));
  }

  if(null_fitting_p){
    result = strdup(concatenate(result, "list()),", NL, TAB, TAB, NULL));
  }

  null_fitting_p = true;
  /* Generation of paving XML code*/
  result = strdup(concatenate(result, TAB, TAB, "paving = list<list[VARTYPE]>(", NULL));

  for(i=0;i<extern_nb-1; i++){
    result = strdup(concatenate(result, "list(", NULL));
    for(j = 0; j<index_of_array-1; j++){
      result = strdup(concatenate(result, "vartype!(", paving_array[j][i], "), ", NULL));
    }
    result = strdup(concatenate(result, "vartype!(", paving_array[j][i], ")),", NL, TAB, TAB, TAB, NULL));
  }
  result = strdup(concatenate(result, "list(", NULL));
  for(j = 0; j<index_of_array-1; j++){
    result = strdup(concatenate(result, "vartype!(", paving_array[j][i], "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", paving_array[j][i], "))),", NL, TAB, TAB, NULL));

#define MONMAX(a, b) ((a<b)?b:a)

  /* Definition of the inner loop nest */
  /* FI->IH: if some columns are removed, the effective depth is unkown and must be computed here */
  /* result = strdup(concatenate(result, "inLoopNest = LOOPNEST(deep = ", i2a(MONMAX(gen_array_nitems(intern_indices_array), 1)), ",", NL, TAB, TAB, TAB, NULL)); */

  for (j = 0; j<intern_nb; j++){
    bool is_null_p = true;
    for(i = 0; i < index_of_array; i++){
      is_null_p = is_null_p && (same_string_p(fitting_array[i][j], "0"));
    }
    if(!is_null_p){
      depth++;
    }
  }
  if(depth==0) depth = 1; /* see comment just below about null fitting matrices. */
  result = strdup(concatenate(result, "inLoopNest = LOOPNEST(deep = ", itoa(depth), ",", NL, TAB, TAB, TAB, NULL));
  result = strdup(concatenate(result, "upperBound = list<VARTYPE>(", NULL));

  /* 3 cases :
     - the fitting matrix is null : must generate a (0,0) loop with dummy index
     - some fitting matrix column is null : do not generate anything
     - some fitting matrix column is not null : generate the corresponding loop bound and index name
  */

  for (j = 0; j<intern_nb; j++){
    bool is_null_p = true;
    for(i = 0; i < index_of_array; i++){
      is_null_p = is_null_p && (same_string_p(fitting_array[i][j], "0"));
    }
    if(!is_null_p){
      null_fitting_p = false;
      result = strdup(concatenate(result,
				  "vartype!(",
				  *((string *)(gen_array_item(intern_upperbounds_array, j))),
				  "), ",
				  NULL));
      internal_index_declarations =
	strdup(concatenate(internal_index_declarations,
			   QUOTE,
			   *((string *)(gen_array_item(intern_indices_array, j))),
			   QUOTE,
			   ", ",
			   NULL));
    }
  }
  if(!null_fitting_p)
    {
      result = strdup(concatenate(gen_strndup0(result, strlen(result) - 2),
				  "),", NULL));
      internal_index_declarations =
	strdup(concatenate(gen_strndup0(internal_index_declarations,
					strlen(internal_index_declarations) -2),
			   ")", NULL));
    }



  if(null_fitting_p){
    result = strdup(concatenate(result, "vartype!(1)),", NL, TAB, TAB, TAB, "names = list<string>(\"M_I\")", NULL));
  }
  else{
    result = strdup(concatenate(result, NL, TAB, "names = list<string>(", internal_index_declarations, NULL));
  }

  /* Complete XML reference */
  result = strdup(concatenate(result, "))", (first?")":","), NL, NULL));
  return result;
}

static string xml_call_from_loopnest(call c, int task_number){
  entity called = call_function(c);
  list arguments = call_arguments(c);
  syntax s;
  string result = "";
  string first_result = "";
  bool first = true;
  bool input_provided = false, output_provided = false;
  string name = strdup(xml_entity_local_name(called));

  if(!same_string_p(name, "="))
    pips_user_error("Only assignation allowed here.\n");

  FOREACH(EXPRESSION, e, arguments){
    s = expression_syntax(e);
    switch(syntax_tag(s)){
    case is_syntax_call:{
      if(first)
	pips_user_error("Call not allowed in left-hand side argument of assignation.\n");
      else
	result = strdup(concatenate(result, xml_call_from_assignation(syntax_call(s), task_number, &input_provided), NULL));
      break;
    }
    case is_syntax_reference:{

      reference r = syntax_reference(s);
      string varname = strdup(concatenate("A_", xml_entity_local_name(reference_variable(r)), NULL));
      if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	if(first){
	  first_result = xml_array_in_task(r, first, task_number);
	  output_provided = true;
	}
	else{
	  result = strdup(concatenate(result, xml_array_in_task(r, first, task_number), NULL));
	  input_provided = true;
	}
      }
    }
    default: pips_internal_error("unhandled case");
    }
    first = false;
  }

  if(!input_provided){
    result = strdup(concatenate("data = list<DATA>(dummyDATA, ", result, first_result, NULL));
  }
  else{
    result = strdup(concatenate("data = list<DATA>(", result, first_result, NULL));
  }
  if(!output_provided){
    result = strdup(concatenate(result, " dummyDATA)", NULL));
  }
  result = strdup(concatenate(result, TAB, ")", NL, NULL));
  return result;
}


static call sequence_call(sequence seq)
{
  call mc = call_undefined; /* meaningful call */
  int nc = 0; /* number of calls */

  MAP(STATEMENT, s, {
      if(continue_statement_p(s))
	;
      else if(statement_call_p(s)) {
	mc = instruction_call(statement_instruction(s));
	nc++;
      }
      else {
	nc = 0;
	break;
      }
    }, sequence_statements(seq));

  if(nc!=1)
    mc = call_undefined;

  return mc;
}

static loop sequence_loop(sequence seq)
{
  loop ml = loop_undefined; /* meaningful loop */
  int nl = 0; /* number of loops */

  MAP(STATEMENT, s, {
      if(continue_statement_p(s))
	;
      else if(statement_loop_p(s)) {
	ml = instruction_loop(statement_instruction(s));
	nl++;
      }
      else {
	nl = 0;
	break;
      }
    }, sequence_statements(seq));

  if(nl!=1)
    ml = loop_undefined;

  return ml;
}

static call xml_loop_from_loop(loop l, string * result, int task_number){

  string * up = malloc(sizeof(string));
  string * xml_name = malloc(sizeof(string));
  statement s = loop_body(l);
  instruction i = statement_instruction(s);
  int u, low;
  expression incr_e = range_increment(loop_range(l));
  syntax incr_s = expression_syntax(incr_e);

  if(!syntax_call_p(incr_s) ||
     strcmp( entity_local_name(call_function(syntax_call(incr_s))), "1") != 0 ) {
    pips_user_error("Loop increments must be constant \"1\".\n");
  }

  u = atoi(xml_expression(range_upper(loop_range(l))));
  low = atoi(xml_expression(range_lower(loop_range(l))));
  /*  printf("%i %i\n", u, low); */
  *up = strdup(i2a(u - low+1));
  //*up = xml_expression(range_upper(loop_range(l)) - range_lower(loop_range(l)) + 1);
  *xml_name = xml_entity_local_name(loop_index(l));
  if( (*xml_name)[0] == 'M'){
    gen_array_append(intern_indices_array, xml_name);
    gen_array_append(intern_upperbounds_array, up);
  }
  else{
    gen_array_append(extern_indices_array, xml_name);
    gen_array_append(extern_upperbounds_array, up);
  }

  switch(instruction_tag(i)){
  case is_instruction_loop:{
    loop l = instruction_loop(i);
    return xml_loop_from_loop(l, result, task_number);
    break;
  }
  case is_instruction_call: {
    call c = instruction_call(i);
    return c;
  }
  case is_instruction_sequence: {
    /* The sequence should contain only one meaningful call or one meaningful loop. */
    call c = sequence_call(instruction_sequence(i));
    loop l = sequence_loop(instruction_sequence(i));
    if(!call_undefined_p(c))
      return c;
    if(!loop_undefined_p(l))
      return xml_loop_from_loop(l, result, task_number);
  }
  default:
    pips_user_error("Only loops and calls allowed in a loop.\n");
  }
  return call_undefined;
}


/* We enter a loop nest. The first loop must be an extern loop. */
static string xml_loop_from_sequence(loop l, int task_number){
  statement s = loop_body(l);
  call c= call_undefined;
  int i;
  string * taskname = (string *)(malloc(sizeof(string)));
  expression incr_e = range_increment(loop_range(l));
  syntax incr_s = expression_syntax(incr_e);


  /* Initialize result string with the declaration of the task */
  string result;

  instruction ins = statement_instruction(s);
  string * name = malloc(sizeof(string));
  string * up = malloc(sizeof(string));
  int u, low;
  if(!syntax_call_p(incr_s) ||
     strcmp( entity_local_name(call_function(syntax_call(incr_s))), "1") != 0 ) {
    pips_user_error("Loop increments must be constant \"1\".\n");
  }


  *taskname = strdup(concatenate("T_", i2a(task_number), NULL));
  result = strdup(concatenate(*taskname,
			      " :: TASK(unitSpentTime = vartype!(1),"
			      NL, TAB, "exLoopNest = LOOPNEST(deep = ", NULL));
  gen_array_append(tasks_names, taskname);
  /* (re-)initialize task-scoped arrays*/
  extern_indices_array = gen_array_make(0);
  intern_indices_array = gen_array_make(0);
  extern_upperbounds_array = gen_array_make(0);
  intern_upperbounds_array = gen_array_make(0);

  *name = xml_entity_local_name(loop_index(l));
  u = atoi(xml_expression(range_upper(loop_range(l))));
  low = atoi(xml_expression(range_lower(loop_range(l))));
  *up = strdup(i2a(u - low+1));
  //*up = xml_expression(range_upper(loop_range(l)) - range_lower(loop_range(l)) + 1);

  if((*name)[0] == 'M'){
    pips_user_error("At least one extern loop is needed.\n");
  }
  else{
    gen_array_append(extern_indices_array, name);
    gen_array_append(extern_upperbounds_array, up);
  }


  switch(instruction_tag(ins)){
  case is_instruction_loop:{
    loop l = instruction_loop(ins);
    c = xml_loop_from_loop(l, &result, task_number);
    break;
  }
  case is_instruction_call:
    {
      c = instruction_call(ins);
    }
    break;
  case is_instruction_sequence:
    /* The sequence should contain only one meaningful call */
    if(!call_undefined_p(c=sequence_call(instruction_sequence(ins))))
      break;
    if(!loop_undefined_p(l=sequence_loop(instruction_sequence(ins)))) {
      c = xml_loop_from_loop(l, &result, task_number);
      break;
    }
    ;
  default:
    pips_user_error("Only loops and one significant call allowed in a loop.\n");
  }

  /* External loop nest depth */
  result = strdup(concatenate(result, i2a(gen_array_nitems(extern_upperbounds_array)), ",", NL, TAB, TAB, NULL));

  /* add external upperbounds */
  result = strdup(concatenate(result, "upperBound = list<VARTYPE>(", NULL));

  for(i=0; i< ((int) gen_array_nitems(extern_upperbounds_array)) - 1; i++){
    result = strdup(concatenate(result, "vartype!(", *((string *)(gen_array_item(extern_upperbounds_array, i))), "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(",*((string *)(gen_array_item(extern_upperbounds_array, i))), ")),",NL, TAB, TAB, NULL));

  /* add external indices names*/
  result = strdup(concatenate(result, "names = list<string>(", NULL));
  for(i=0; i<((int) gen_array_nitems(extern_indices_array)) - 1; i++){
    result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(extern_indices_array, i))), QUOTE ", ", NULL));
  }
  result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(extern_indices_array, i))), QUOTE, ")),", NL, TAB, NULL));

  result = strdup(concatenate(result, xml_call_from_loopnest(c, task_number), NULL));

  gen_array_free(extern_indices_array);
  gen_array_free(intern_indices_array);
  gen_array_free(extern_upperbounds_array);
  gen_array_free(intern_upperbounds_array);
  result = strdup(concatenate(result, NL, NULL));
  return result;
}

/* We are here at the highest level of statements. The statements are either
   loopnests or a RETURN instruction. Any other possibility pips_user_errors
   the prettyprinter.*/
static string xml_statement_from_sequence(statement s, int task_number){
  string result = "";
  instruction i = statement_instruction(s);

  switch(instruction_tag(i)){
  case is_instruction_loop:{
    loop l = instruction_loop(i);
    result = xml_loop_from_sequence(l, task_number);
    break;
  }
  case is_instruction_call:{
    /* RETURN should only be allowed as the last statement in the sequence */
    if(!return_statement_p(s) && !continue_statement_p(s))
      pips_user_error("Only RETURN and CONTINUE allowed here.\n");
    break;
  }
  default:{
    pips_user_error("Only loops and calls allowed here.\n");
  }
  }

  return result;
}

/* Concatentates each task to the final result.
   The validity of the task is not checked in this function but
   it is into xml_statementement_from_sequence and subsequent
   functions.*/
static string xml_sequence_from_task(sequence seq){
  string result = "";
  int task_number = 0;
  MAP(STATEMENT, s,
      {
	string oldresult = strdup(result);
	string current = strdup(xml_statement_from_sequence(s, task_number));

	if(strlen(current)==0) {
	  free(current);
	  result = oldresult;
	}
	else {
	  result = strdup(concatenate(oldresult, current, NULL));
	  free(current);
	  free(oldresult);
	  task_number++;
	}
      }, sequence_statements(seq));
  return result;
}

/* Manages tasks. The code is very defensive and hangs if sth not
   predicted happens. Here basically we begin the code in itself
   and thus $stat is obligatory a sequence. */
static string xml_tasks_with_motif(statement stat){
  int j;
  instruction i;
  string result = "tasks\n";
  if(statement_undefined_p(stat))
    {
      pips_internal_error("statement error");
    }
  i = statement_instruction(stat);
  tasks_names = gen_array_make(0);
  switch(instruction_tag(i)){
  case is_instruction_sequence:{
    sequence seq = instruction_sequence(i);
    result = xml_sequence_from_task(seq);
    break;
  }
  default:{
    pips_user_error("Only a sequence can be here");
  }
  }
  result = strdup(concatenate(result, NL, NL, "PRES:APPLICATION := APPLICATION(name = symbol!(", QUOTE, global_module_name, QUOTE, "), ", NL, TAB,NULL));
  result = strdup(concatenate(result, "tasks = list<TASK>(", NULL));
  for(j = 0; j<(int) gen_array_nitems(tasks_names) - 1; j++){
    result = strdup(concatenate(result, *((string *)(gen_array_item(tasks_names, j))), ", ", NULL));
  }
  result = strdup(concatenate(result, *((string *)(gen_array_item(tasks_names, j))), "))", NULL));

  return result;
}

/* Creates string for xml pretty printer.
   This string divides in declarations (array decl.) and
   tasks which are loopnest with an instruction at the core.
*/
static string xml_code_string(entity module, statement stat)
{
  string decls="", tasks="", result="";

  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

  decls       = xml_declarations_with_explicit_motif(module, variable_p, "", true);
  tasks       = xml_tasks_with_motif(stat);
  result = strdup(concatenate(decls, NL, tasks, NL, NULL));
  ifdebug(2)
    {
      printf("%s", result);
    }
  return result;
}


/******************************************************** PIPSMAKE INTERFACE */

#define XMLPRETTY    ".xml"

/* Initiates xml pretty print modules
 */
bool print_xml_code_with_explicit_motif(const char* module_name)
{
  FILE * out;
  string ppt, xml, dir, filename;
  entity module;
  statement stat;
  array_names = gen_array_make(0);
  array_dims = gen_array_make(0);
  xml = db_build_file_resource_name(DBR_XML_PRINTED_FILE, module_name, XMLPRETTY);

  global_module_name = module_name;
  module = module_name_to_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", xml, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  if(statement_undefined_p(stat))
    {
      pips_internal_error("No statement for module %s", module_name);
    }
  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("XMLPRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Begin Claire prettyprinter for %s\n", entity_name(module));
  ppt = xml_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "%s", ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_XML_PRINTED_FILE, module_name, xml);

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}


/* ======================================================= */

typedef struct
{
  stack  loops_for_call;
  stack loop_indices;
  stack current_stat;
  gen_array_t nested_loops;
  gen_array_t nested_loop_indices;
  gen_array_t nested_call;
}
  nest_context_t,
  * nest_context_p;


static void
xml_declarations(entity module, string_buffer result)
{
  list dim;
  int nb_dim =0;
  list ldecl = code_declarations(value_code(entity_initial(module)));
  list ld;

  entity var = entity_undefined;
  for(ld = ldecl; !ENDP(ld); ld = CDR(ld)){

    printf("inside for");
    var = ENTITY(CAR(ld));
    printf("Entity Name in the List : %s  ",entity_name(var));

    if (variable_p(var) && ( variable_entity_dimension(var) > 0))
      {
	nb_dim = variable_entity_dimension(var);
	printf("inside p, %d",nb_dim);

	string_buffer_append(result,
			     concatenate(TAB, OPENANGLE, "array name =", QUOTE, XML_ARRAY_PREFIX,entity_user_name(var),QUOTE,
					 " dataType = ", QUOTE,
					 "INTEGER", QUOTE,CLOSEANGLE
					 , XML_RL, NULL));
	string_buffer_append(result,
			     concatenate(OPENANGLE,"dimensions",CLOSEANGLE
					 , XML_RL,NULL));

	for (dim = variable_dimensions(type_variable(entity_type(var))); !ENDP(dim); dim = CDR(dim)) {

	  intptr_t low;
	  intptr_t  up;
	  expression elow = dimension_lower(DIMENSION(CAR(dim)));
	  expression eup = dimension_upper(DIMENSION(CAR(dim)));
	  if (expression_integer_value(elow, &low) && expression_integer_value(eup, &up)){
	    string_buffer_append(result,
				 concatenate( TAB,OPENANGLE, "range min =", QUOTE,  i2a(low), QUOTE, NULL));
	    string_buffer_append(result,
				 concatenate(" max =", QUOTE,  i2a(up-low+1),QUOTE, SLASH, CLOSEANGLE, XML_RL, NULL));
	  }
	  else pips_user_error("Array dimensions must be integer\n");

	}

	string_buffer_append(result,
			     concatenate(OPENANGLE, SLASH, "dimensions", CLOSEANGLE,NL ,NULL));
	string_buffer_append(result,
			     concatenate(TAB, OPENANGLE, SLASH, "array", CLOSEANGLE, NL, NULL));

      }
  }
}


static void
push_current_statement(statement s, nest_context_p nest)
{
  stack_push(s , nest->current_stat);
}

static void
pop_current_statement(statement s __attribute__ ((unused)),
		      nest_context_p nest)
{
  /*   if (debug) print_statement(s);*/
  stack_pop(nest->current_stat);
}

static void
push_loop(loop l, nest_context_p nest)
{
  /* on sauve le statement associe a la boucle courante */
  statement sl = (statement) stack_head(nest->current_stat);
  stack_push(sl , nest->loops_for_call);
  stack_push(loop_index(l) , nest->loop_indices);
}

static void
pop_loop(loop l __attribute__ ((unused)),
	 nest_context_p nest)
{
  stack_pop(nest->loops_for_call);
  stack_pop(nest->loop_indices);
}

static bool call_selection(call c, nest_context_p nest __attribute__ ((unused)))
{
  /* CA il faut implemeter  un choix judicieux ... distribution ou encapsulation*/
  /* pour le moment distribution systematique de tout call */
  /* il faut recuperer les appels de fonction value_code_p(entity_initial(f)*/
  entity f = call_function(c);
  if  (ENTITY_ASSIGN_P(f) || entity_subroutine_p(f)|| entity_function_p(f))
    {
      return true;
    }
  else return false;

  /*  statement s = (statement) stack_head(nest->current_stat);
      return ((!return_statement_p(s) && !continue_statement_p(s)));*/
}

static void store_call_context(call c  __attribute__ ((unused)),
			       nest_context_p nest)
{
  stack sl = stack_copy(nest->loops_for_call);
  stack si = stack_copy(nest->loop_indices);
  /* on sauve le statement associe au call */
  statement statc = (statement) stack_head(nest->current_stat) ;
  gen_array_append(nest->nested_loop_indices,si);
  gen_array_append(nest->nested_loops,sl);
  gen_array_append(nest->nested_call,statc);
}

static bool push_test(test t  __attribute__ ((unused)),
		      nest_context_p nest  __attribute__ ((unused)))
{
  /* encapsulation de l'ensemble des instructions appartenant au test*/
 
  return true;
}


static void pop_test(test t __attribute__ ((unused)),
		     nest_context_p nest __attribute__ ((unused)))
{
  /* encapsulation de l'ensemble des instructions appartenant au test*/
}


static void
search_nested_loops_and_calls(statement stmp, nest_context_p nest)
{
  gen_context_multi_recurse(stmp,nest, loop_domain,push_loop,pop_loop,
			    statement_domain, push_current_statement,pop_current_statement,
			    test_domain, push_test, pop_test,
			    call_domain,call_selection,store_call_context,
			    NULL);
}

static void __attribute__ ((unused)) print_call_selection(nest_context_p nest)
{
  int j;
  int numberOfTasks=(int) gen_array_nitems(nest->nested_call);
  for (j = 0; j<numberOfTasks; j++)
    {
      //statement s = gen_array_item(nest->nested_call,j);
      //stack st = gen_array_item(nest->nested_loops,j);
      /*   print_statement(s);
	   stack_map( st, print_statement);*/
    }
}


static expression expression_plusplus(expression e)
{
  expression new_e;
  if (expression_constant_p(e)) {
    new_e = int_to_expression(1+ expression_to_int(e));
  }
  else {
    entity add_ent = entity_intrinsic(PLUS_OPERATOR_NAME);
    new_e =  make_call_expression(add_ent,
				  CONS(EXPRESSION, e, CONS(EXPRESSION,  int_to_expression(1), NIL)));
  }
  return new_e;
}

static void xml_loop(stack st, string_buffer result)
{
  string_buffer_append(result,concatenate(TAB,SPACE, OPENANGLE, "outLoop", CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, OPENANGLE, "loopNest", CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, SPACE, OPENANGLE, "bounds", CLOSEANGLE, NL, NULL));

  STACK_MAP_X(s, statement,
	      {
		loop l = instruction_loop(statement_instruction(s));
		expression el =range_lower(loop_range(l));
		expression eu =range_upper(loop_range(l));
		expression new_eu= expression_plusplus(eu);

		string_buffer_append(result,
				     concatenate(TAB,SPACE,SPACE,SPACE,SPACE,OPENANGLE,"bound idx =",QUOTE,entity_user_name(loop_index(l)),QUOTE,NULL));
		string_buffer_append(result,
				     concatenate(SPACE, "lower =",QUOTE, words_to_string(words_expression(el,NIL)),QUOTE,NULL));
		string_buffer_append(result,
				     concatenate(SPACE, "upper =", QUOTE, words_to_string(words_expression(new_eu,NIL)),QUOTE, SLASH, CLOSEANGLE,NL,NULL));
	      },
	      st, 0);

  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, SPACE,  OPENANGLE,SLASH "bounds", CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, OPENANGLE,SLASH, "loopNest", CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE,  OPENANGLE, SLASH, "openLoop", CLOSEANGLE, NL, NULL));
}



static void xml_reference(int taskNumber __attribute__ ((unused)), reference r, bool wmode,
			  string_buffer result)
{

  const char* varname = entity_user_name(reference_variable(r));
  string_buffer_append
    (result,
     concatenate(SPACE, QUOTE, XML_ARRAY_PREFIX, varname, QUOTE, SPACE, "accessMode =", QUOTE,
		 (wmode?"W":"R"),QUOTE, CLOSEANGLE,NL,
		 NULL));
}

static void  find_motif(Psysteme ps, Pvecteur nested_indices, int dim, int nb_dim __attribute__ ((unused)), Pcontrainte *bound_inf, Pcontrainte *bound_sup, Pcontrainte *iterator, int *motif_up_bound, int *lowerbound, int *upperbound)
{
  Variable phi;
  Value	v;
  Pvecteur pi;
  Pcontrainte c, next, cl, cu, cl_dup, cu_dup,lind, lind_dup,
    list_cl=NULL,
    list_cu=NULL,
    list_ind=NULL;
  int lower =1;
  int upper =2;
  int ind =3;
  Pcontrainte bounds[4][4];
  int nb_bounds =0;
  int nb_lower = 0;
  int nb_upper = 0;
  int nb_indices=0;
  int i,j;
  Pbase vars_to_eliminate = BASE_NULLE;

  for (i=1; i<=3;i++)
    for (j=1; j<=3;j++)
      bounds[i][j]=CONTRAINTE_UNDEFINED;

  phi = (Variable) make_phi_entity(dim);
  /* elimination des variables autres de les phi et les indices de boucles englobants
     copie de la base + mise a zero des indices englobants + projection selon les elem de ce vecteur*/

  if (!SC_EMPTY_P(ps)) {
    vars_to_eliminate = vect_copy(ps->base);
    /* printf("Base des variables :\n");
       vect_print(vars_to_eliminate, entity_local_name);
    */
    vect_erase_var(&vars_to_eliminate, phi);
    for (pi = nested_indices; !VECTEUR_NUL_P(pi); pi = pi->succ)
      vect_erase_var(&vars_to_eliminate, var_of(pi));

    /* printf("Elimination des variables :\n");
       vect_print(vars_to_eliminate, entity_local_name);
    */

    sc_projection_along_variables_ofl_ctrl(&ps,vars_to_eliminate , NO_OFL_CTRL);

    for(c = sc_inegalites(ps), next=(c==NULL ? NULL : c->succ);
	c!=NULL;
	c=next, next=(c==NULL ? NULL : c->succ))
      {
	Pvecteur indices_in_vecteur = VECTEUR_NUL;
	/* printf("Tri de la contrainte :\n");
	   vect_print(c->vecteur, entity_local_name);
	*/
	v = vect_coeff(phi, c->vecteur);
	for (pi = nested_indices; !VECTEUR_NUL_P(pi); pi = pi->succ)
	  {
	    int coeff_index = vect_coeff(var_of(pi),c->vecteur);
	    if (coeff_index)
	      vect_add_elem(&indices_in_vecteur,var_of(pi), coeff_index);
	  }

	nb_indices=vect_size(indices_in_vecteur);
	nb_indices = (nb_indices >2) ? 2 : nb_indices;

	if (value_pos_p(v)) {
	  c->succ = bounds[upper][nb_indices+1];
	  bounds[upper][nb_indices+1] = c;
	  /* printf(" bornes inf avec indices de boucles englobants :\n");
	     vect_print(bounds[upper][nb_indices+1]->vecteur, entity_local_name); */
	  nb_upper ++;
	}
	else if (value_neg_p(v)) {
	  c->succ = bounds[lower][nb_indices+1];
	  bounds[lower][nb_indices+1] = c;
	  /* printf(" bornes inf avec indices de boucles englobants :\n");
	     vect_print(bounds[lower][nb_indices+1]->vecteur, entity_local_name);*/
	  lind = contrainte_make(indices_in_vecteur);
	  lind->succ = bounds[ind][nb_indices+1];
	  bounds[ind][nb_indices+1] = lind;
	  /* printf(" indices contenus dans la contrainte :\n");
	     vect_print(bounds[ind][nb_indices+1]->vecteur, entity_local_name); */
	  nb_lower ++;
	}
      }
    /* printf("Nb borne inf = %d, Nb borne sup = %d ;\n",nb_lower,nb_upper); */

    if  (!CONTRAINTE_UNDEFINED_P(bounds[lower][2])) {
      /* case with 1 loop index in the loop bound constraints */
      for(cl = bounds[lower][2], lind= bounds[ind][2]; cl !=NULL; cl=cl->succ,lind=lind->succ)  {
	for(cu = bounds[upper][2]; cu !=NULL; cu =cu->succ) {
	  /*  printf("Tests de la negation des  contraintes :\n");
	      vect_print(cl->vecteur, entity_local_name);
	      vect_print(cu->vecteur, entity_local_name); */
	  if (vect_opposite_except(cl->vecteur,cu->vecteur,TCST)){
	    cl_dup = contrainte_dup(cl);
	    cl_dup->succ = list_cl, list_cl=cl_dup;
	    cu_dup = contrainte_dup(cu);
	    cu_dup->succ = list_cu, list_cu=cu_dup;
	    lind_dup = contrainte_dup(lind);
	    lind_dup->succ = list_ind, list_ind = lind_dup;
	    nb_bounds ++;
	  }
	}
      }
      *bound_inf= list_cl;
      *bound_sup = list_cu;
      *iterator = list_ind;
      *motif_up_bound =- vect_coeff(TCST,list_cl->vecteur) - vect_coeff(TCST,list_cu->vecteur) +1;
      *lowerbound = vect_coeff(TCST,list_cl->vecteur);
      *upperbound = vect_coeff(TCST,list_cu->vecteur)+1;
    }
    else if (!CONTRAINTE_UNDEFINED_P(bounds[lower][1]) && !CONTRAINTE_UNDEFINED_P(bounds[upper][1])) {
      /* case where loop bounds are numeric */
      *bound_inf= bounds[lower][1];
      *bound_sup = bounds[upper][1];
      *iterator =  bounds[ind][1];
      *motif_up_bound = - vect_coeff(TCST,bounds[lower][1]->vecteur)
	- vect_coeff(TCST,bounds[upper][1]->vecteur)+1;
      *upperbound = vect_coeff(TCST,bounds[upper][1]->vecteur)+1;
      *lowerbound = vect_coeff(TCST,bounds[lower][1]->vecteur);
    } else {
      /* Only bounds with several loop indices */
      /* printf("PB - Only bounds with several loop indices\n"); */
      *bound_inf= CONTRAINTE_UNDEFINED;
      *bound_sup = CONTRAINTE_UNDEFINED;
      *iterator = CONTRAINTE_UNDEFINED;
      *motif_up_bound = 1;
      *upperbound = 1;
      *lowerbound = 0;
    }
  }
  vect_rm( vars_to_eliminate);
}


static void xml_tiling(int taskNumber, reference ref,  region reg, stack indices,  string_buffer result)
{
  fprintf(stderr,"XML");
  Psysteme ps_reg = sc_dup(region_system(reg));
  entity var = reference_variable(ref);
  int dim = (int) gen_length(variable_dimensions(type_variable(entity_type(var))));
  int i, j ;
  string_buffer buffer_bound = string_buffer_make(true);
  string_buffer buffer_offset = string_buffer_make(true);
  string_buffer buffer_fitting = string_buffer_make(true);
  string_buffer buffer_paving = string_buffer_make(true);
  string string_bound = "";
  string string_offset = "";
  string string_paving = "";
  string string_fitting =  "";
  Pvecteur iterat, pi= VECTEUR_NUL;
  Pcontrainte bound_inf = CONTRAINTE_UNDEFINED;
  Pcontrainte bound_up = CONTRAINTE_UNDEFINED;
  Pcontrainte iterator = CONTRAINTE_UNDEFINED;
  int motif_up_bound =0;
  int lowerbound = 0;
  int upperbound = 0;
  int dim_indices= stack_size(indices);
  int pav_matrix[10][10], fit_matrix[10][10];

  for (i=1; i<=9;i++)
    for (j=1;j<=9;j++)
      pav_matrix[i][j]=0, fit_matrix[i][j]=0;

  STACK_MAP_X(index,entity,
	      {
		vect_add_elem (&pi,(Variable) index ,VALUE_ONE);
	      }, indices,1);

  for(i=1; i<=dim ; i++) {
    Psysteme ps = sc_dup(ps_reg);
    sc_transform_eg_in_ineg(ps);

    find_motif(ps, pi, i,  dim, &bound_inf, &bound_up, &iterator,
	       &motif_up_bound, &lowerbound, &upperbound);
    string_buffer_append(buffer_offset,
			 concatenate(TAB,TAB,OPENANGLE,"offset val =", QUOTE,
				     (CONTRAINTE_UNDEFINED_P(bound_inf))? "0" :
				     i2a(vect_coeff(TCST,bound_inf->vecteur)),
				     QUOTE, SLASH, CLOSEANGLE,NL,NULL));
    if (!CONTRAINTE_UNDEFINED_P(iterator)) {
      for (iterat = pi, j=1; !VECTEUR_NUL_P(iterat); iterat = iterat->succ, j++)
	pav_matrix[i][j]= vect_coeff(var_of(iterat),iterator->vecteur);
    }
    if (!CONTRAINTE_UNDEFINED_P(bound_inf))
      fit_matrix[i][i]= (motif_up_bound >1) ? 1:0;

    string_buffer_append(buffer_bound,
			 concatenate(TAB,TAB, OPENANGLE, "bound idx=",
				     QUOTE, XML_MOTIF_PREFIX, i2a(taskNumber),"_",
				     entity_user_name(var), "_",i2a(i),QUOTE, SPACE,
				     "lower =" QUOTE,i2a(lowerbound),QUOTE,
				     SPACE, "upper =", QUOTE, i2a(upperbound),
				     QUOTE, SLASH, CLOSEANGLE,
				     NL,NULL));
  }

  for (j=1; j<=dim_indices ; j++){
    string_buffer_append(buffer_paving,concatenate(TAB,TAB, OPENANGLE,"row",
						   CLOSEANGLE,NULL));
    for(i=1; i<=dim ; i++)
      string_buffer_append(buffer_paving,
			   concatenate(OPENANGLE, "cell val =", QUOTE,
				       i2a( pav_matrix[i][j]),QUOTE, SLASH,
				       CLOSEANGLE,NULL));
    string_buffer_append(buffer_paving,concatenate(OPENANGLE,SLASH, "row",
						   CLOSEANGLE,NL,NULL));
  }
  for(i=1; i<=dim ; i++) {
    string_buffer_append(buffer_fitting,concatenate(TAB, TAB,OPENANGLE,"row",
						    CLOSEANGLE,NULL));
    for(j=1; j<=dim ; j++)
      string_buffer_append(buffer_fitting, concatenate(OPENANGLE, "cell val =", QUOTE,
						       i2a( fit_matrix[i][j]),
						       QUOTE, SLASH,
						       CLOSEANGLE,NULL));

    string_buffer_append(buffer_fitting,concatenate(OPENANGLE,SLASH, "row",
						    CLOSEANGLE,NL,NULL));
  }
  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,"origin",
					  CLOSEANGLE,NL,NULL));
  string_offset =string_buffer_to_string(buffer_offset);
  string_buffer_append(result,string_offset);
  free(string_offset);
  string_offset=NULL;
  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,SLASH,"origin",
					  CLOSEANGLE,NL,NULL));

  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,"fitting",
					  CLOSEANGLE,NL,NULL));
  string_fitting =string_buffer_to_string(buffer_fitting);
  string_buffer_append(result,string_fitting);
  free(string_fitting);
  string_fitting=NULL;
  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,SLASH,"fitting",
					  CLOSEANGLE,NL,NULL));
  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,"paving",
					  CLOSEANGLE,NL,NULL));
  string_paving =string_buffer_to_string(buffer_paving);
  string_buffer_append(result,string_paving);
  free(string_paving);
  string_paving=NULL;
  string_buffer_append(result,concatenate(TAB,SPACE,SPACE,OPENANGLE,SLASH,"paving",
					  CLOSEANGLE,NL,NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, OPENANGLE, "inLoop",
					  CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, OPENANGLE, "loopNest",
					  CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, SPACE, OPENANGLE, "bounds",
					  CLOSEANGLE, NL, NULL));
  string_bound =string_buffer_to_string(buffer_bound);
  string_buffer_append(result,string_bound);
  free(string_bound);
  string_bound=NULL;

  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, SPACE,OPENANGLE,
					  SLASH "bounds", CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE, SPACE, OPENANGLE,SLASH, "loopNest",
					  CLOSEANGLE, NL, NULL));
  string_buffer_append(result,concatenate(TAB,SPACE,  OPENANGLE, SLASH, "inLoop",
					  CLOSEANGLE, NL, NULL));


}

static void xml_references(int taskNumber, list l_regions, stack indices, string_buffer result)
{
  list lr;
  bool atleast_one_read_ref = false;
  bool atleast_one_written_ref = false;
  /*   Read array references first */
  for ( lr = l_regions; !ENDP(lr); lr = CDR(lr))
    {
      region re = REGION(CAR(lr));
      reference ref = effect_any_reference(re);
      if (array_entity_p(reference_variable(ref)) && region_read_p(re)) {
	atleast_one_read_ref = true;
	string_buffer_append(result,concatenate(TAB, SPACE, SPACE, OPENANGLE,  "data darray=",NULL));
	xml_reference(taskNumber, ref,  region_write_p(re), result);
	xml_tiling(taskNumber, ref,re, indices, result);
      }
    }
  if (!atleast_one_read_ref)
    string_buffer_append(result,concatenate(TAB,SPACE, SPACE, "dummyDATA",
					    NL,NULL));
  for ( lr = l_regions; !ENDP(lr); lr = CDR(lr))
    {
      region re = REGION(CAR(lr));
      reference ref = effect_any_reference(re);
      if (array_entity_p(reference_variable(ref)) && region_write_p(re)) {
	atleast_one_written_ref = true;
	string_buffer_append(result,concatenate(TAB, SPACE, SPACE, OPENANGLE, "data darray=",NULL));
	xml_reference(taskNumber, ref,  region_write_p(re), result);
	xml_tiling(taskNumber, ref,re, indices, result);
      }
    }
  if (!atleast_one_written_ref)
    string_buffer_append(result,concatenate(TAB,SPACE, SPACE,"dummyDATA",NL,NULL));
}

static void xml_data(int taskNumber,statement s, stack indices, string_buffer result )
{

  list  l_regions = regions_dup(load_statement_local_regions(s));
  string_buffer_append(result,concatenate(TAB,SPACE, OPENANGLE, "dataList", CLOSEANGLE, NL,NULL));
  /*
    ifdebug(2) {
    fprintf(stderr, "\n list of regions ");
    print_regions(l_regions);
    fprintf(stderr, "\n for the statement");
    print_statement(s);
    }
  */
  xml_references(taskNumber, l_regions, indices, result);
  string_buffer_append(result,concatenate(TAB,SPACE,OPENANGLE, SLASH, "dataList", CLOSEANGLE,NL,NULL));
  regions_free(l_regions);
}

static string task_complexity(statement s)
{
  complexity stat_comp = load_statement_complexity(s);
  string r;
  if(stat_comp != (complexity) HASH_UNDEFINED_VALUE && !complexity_zero_p(stat_comp)) {
    cons *pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, false,
						  true));
    r = words_to_string(pc);
  }
  else r = i2a(1);
  return  (r);
}
static void xml_task( int taskNumber, nest_context_p nest, string_buffer result)
{

  statement s = gen_array_item(nest->nested_call,taskNumber);
  stack st = gen_array_item(nest->nested_loops,taskNumber);
  stack sindices = gen_array_item(nest->nested_loop_indices,taskNumber);

  string_buffer_append(result, concatenate(NL,TAB,OPENANGLE,"task name =", QUOTE, XML_TASK_PREFIX,i2a(taskNumber),QUOTE, CLOSEANGLE, NL, NULL));
  string_buffer_append(result, concatenate(TAB, SPACE, OPENANGLE, "unitSpentTime", CLOSEANGLE,task_complexity(s),NULL));
  string_buffer_append(result, concatenate(OPENANGLE,SLASH,"unitSpentTime",CLOSEANGLE,NL,NULL));

  xml_loop(st, result);
  xml_data (taskNumber, s,sindices, result);
  string_buffer_append(result, concatenate(TAB,OPENANGLE,SLASH,"task",CLOSEANGLE, NL, NULL));
}

static void  xml_tasks(statement stat, string_buffer result){

  const char*  module_name = get_current_module_name();
  nest_context_t nest;
  int taskNumber =0;
  nest.loops_for_call = stack_make(statement_domain,0,0);
  nest.loop_indices = stack_make(entity_domain,0,0);
  nest.current_stat = stack_make(statement_domain,0,0);
  nest.nested_loops=  gen_array_make(0);
  nest.nested_loop_indices =  gen_array_make(0);
  nest.nested_call=  gen_array_make(0);

  string_buffer_append(result,concatenate(NL,SPACE,OPENANGLE,"tasks",CLOSEANGLE,NL, NULL));

  if(statement_undefined_p(stat)) {
    pips_internal_error("statement error");
  }

  search_nested_loops_and_calls(stat,&nest);
  /* ifdebug(2)  print_call_selection(&nest); */

  // string_buffer_append(result(concatenate(NL,OPENANGLE,"tasks",CLOSEANGLE,NL, NULL)));

  for (taskNumber = 0; taskNumber<(int) gen_array_nitems(nest.nested_call); taskNumber++)

    xml_task(taskNumber, &nest,result);
  string_buffer_append(result,concatenate(SPACE,OPENANGLE, SLASH, "tasks", CLOSEANGLE, NL,NULL));
  string_buffer_append(result,
		       concatenate(SPACE, OPENANGLE,
				   "application name = ",
				   QUOTE, module_name, QUOTE, CLOSEANGLE
				   NL, NULL));

  for(taskNumber = 0; taskNumber<(int) gen_array_nitems(nest.nested_call)-1; taskNumber++)
    string_buffer_append(result,concatenate(TAB, OPENANGLE, "taskref ref = ", QUOTE, XML_TASK_PREFIX,
					    i2a(taskNumber),QUOTE, SLASH,  CLOSEANGLE, NL, NULL));

  string_buffer_append(result,concatenate(TAB, OPENANGLE, "taskref ref = ", QUOTE, XML_TASK_PREFIX,
					  i2a(taskNumber),QUOTE, SLASH,  CLOSEANGLE, NL, NULL));

  string_buffer_append(result,concatenate(SPACE, OPENANGLE, SLASH, "application",CLOSEANGLE, NL, NULL));

  gen_array_free(nest.nested_loops);
  gen_array_free(nest.nested_loop_indices);
  gen_array_free(nest.nested_call);
  stack_free(&(nest.loops_for_call));
  stack_free(&(nest.loop_indices));
  stack_free(&(nest.current_stat));

}

/* Creates string for xml pretty printer.
   This string divides in declarations (array decl.) and
   tasks which are loopnests with an instruction at the core.
*/

static string xml_code(entity module, statement stat)
{
  string_buffer result=string_buffer_make(true);
  string result2;

  string_buffer_append(result,concatenate(OPENANGLE, "module name =", QUOTE,get_current_module_name(), QUOTE, CLOSEANGLE, NL, NULL ));
  xml_declarations(module,result);
  xml_tasks(stat,result);

  string_buffer_append(result,concatenate(OPENANGLE, SLASH, "module", CLOSEANGLE, NL, NULL ));
  result2=string_buffer_to_string(result);
  /* ifdebug(2)
     {
     printf("%s", result2);
     } */
  return result2;
}

static bool valid_specification_p(entity module __attribute__ ((unused)),
				  statement stat __attribute__ ((unused)))
{ return true;
}

/******************************************************** PIPSMAKE INTERFACE */

#define XMLPRETTY    ".xml"

/* Initiates xml pretty print modules
 */

bool print_xml_code(const char* module_name)
{
  FILE * out;
  string ppt;
  entity module = module_name_to_entity(module_name);
  string xml = db_build_file_resource_name(DBR_XML_PRINTED_FILE,
					   module_name, XMLPRETTY);
  string  dir = db_get_current_workspace_directory();
  string filename = strdup(concatenate(dir, "/", xml, NULL));
  statement stat=(statement) db_get_memory_resource(DBR_CODE,
						    module_name, true);

  init_cost_table();
  /* Get the READ and WRITE regions of the module */
  set_rw_effects((statement_effects)
		 db_get_memory_resource(DBR_REGIONS, module_name, true));

  set_complexity_map( (statement_mapping)
		      db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

  if(statement_undefined_p(stat))
    {
      pips_internal_error("No statement for module %s", module_name);
    }
  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("XMLPRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Spec validation before xml prettyprinter for %s\n",
	     entity_name(module));
  if (valid_specification_p(module,stat)){
    pips_debug(1, "Spec is valid\n");
    pips_debug(1, "Begin Xml prettyprinter for %s\n", entity_name(module));
    ppt = xml_code(module, stat);
    pips_debug(1, "end\n");
    debug_off();
    /* save to file */
    out = safe_fopen(filename, "w");
    fprintf(out,"%s",ppt);
    safe_fclose(out, filename);
    free(ppt);
  }

  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_XML_PRINTED_FILE, module_name, xml);

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}



//================== PRETTYPRINT TERAOPT DTD ============================


static string vect_to_string(Pvecteur pv) {
  return  words_to_string(words_syntax(expression_syntax(make_vecteur_expression(pv)),NIL));
}

bool vect_one_p(Pvecteur v) {
  return  (!VECTEUR_NUL_P(v) && vect_size(v) == 1 && vect_coeff(TCST, v) ==1);
}

bool vect_zero_p(Pvecteur v) {
  return  (VECTEUR_NUL_P(v) ||
	   (!VECTEUR_NUL_P(v) && vect_size(v) == 1 && value_zero_p(vect_coeff(TCST, v))));
}

static void type_and_size_of_var(entity var, const char ** datatype, int *size)
{
  // type t = ultimate_type(entity_type(var));
  type t = entity_type(var);
  if (type_variable_p(t)) {
    basic b = variable_basic(type_variable(t));
    int e = SizeOfElements(b);
    if (e==-1)
      *size = 9999;
    else
      *size = e;
    *datatype =basic_to_string(b);
  }
}


static void add_margin(int gm, string_buffer sb_result) {
  int i;
  for (i=1;i<=gm;i++)
    string_buffer_append(sb_result,TB1);
}

static void string_buffer_append_word(string str, string_buffer sb_result)
{
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   str,
				   CLOSEANGLE,
				   NL, NULL));
}

static void string_buffer_append_symbolic(string str, string_buffer sb_result){
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,"Symbolic",CLOSEANGLE,
				   str,
				   OPENANGLE,"/Symbolic",CLOSEANGLE,
				   NL, NULL));
}

static void string_buffer_append_numeric(string str, string_buffer sb_result){
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,"Numeric",CLOSEANGLE,
				   str,
				   OPENANGLE,"/Numeric",CLOSEANGLE,
				   NL, NULL));
}



static void
find_loops_and_calls_in_box(statement stmp, nest_context_p nest)
{
  gen_context_multi_recurse(stmp,nest, loop_domain,push_loop,pop_loop,
			    statement_domain, push_current_statement,pop_current_statement,
			    test_domain, push_test, pop_test,
			    call_domain,call_selection,store_call_context,
			    NULL);
}

// A completer
static void xml_Library(string_buffer sb_result)
{
  string_buffer_append_word("Library",sb_result);
  string_buffer_append_word("/Library",sb_result);
}
// A completer
static void xml_Returns(string_buffer sb_result __attribute__ ((unused)))
{
  //string_buffer_append_word("Returns",sb_result);
  //string_buffer_append_word("/Returns",sb_result);
}
// A completer
static void xml_Timecosts(string_buffer sb_result __attribute__ ((unused)))
{
  //string_buffer_append_word("Timecosts",sb_result);
  // string_buffer_append_word("/Timecosts",sb_result);
}

// A completer
static void  xml_AccessFunction(string_buffer sb_result __attribute__ ((unused)))
{
  //string_buffer_append_word("AccessFunction",sb_result);
  //string_buffer_append_word("/AccessFunction",sb_result);
}
// A completer
static void xml_Regions(string_buffer sb_result __attribute__ ((unused)))
{
  // string_buffer_append_word("Regions",sb_result);
  // string_buffer_append_word("/Regions",sb_result);
}
// A completer
static void xml_CodeSize(string_buffer sb_result)
{
  string_buffer_append_word("CodeSize",sb_result);
  string_buffer_append_word("/CodeSize",sb_result);
}

void insert_xml_string(const char* module_name, string s) {
  FILE * out;
  string dir = db_get_current_workspace_directory();
  string sm = db_build_file_resource_name(DBR_XML_PRINTED_FILE,
					  module_name, XMLPRETTY);
  string xml_module_name = strdup(concatenate(dir, "/", sm, NULL));
  out = safe_fopen(xml_module_name, "a");
  fprintf(out,"%s",s);
  safe_fclose(out, xml_module_name);
  free(xml_module_name);
}

/*
Pour traiter les cas ou il y a des patterns symboliques, sans preconditions
interprocedurales permettant d'evaluer un min sur les bornes:
Choix selon ordre de priorite:
 - borne numerique
 - borne symbolique avec un seul parametre et coeff ==1
 - borne symbolique avec un seul parametre
 - la premiere borne symbolique avec plusieurs parametres
*/

static Pcontrainte choose_pattern(Pcontrainte lpc)
{
  Pcontrainte cl= CONTRAINTE_UNDEFINED,
    cn= CONTRAINTE_UNDEFINED,
    cs1= CONTRAINTE_UNDEFINED,
    csx= CONTRAINTE_UNDEFINED,
    clv= CONTRAINTE_UNDEFINED;
  Pvecteur pv;
  for(cl = lpc; cl !=NULL; cl=cl->succ)  {
    if  (vect_size(cl->vecteur) ==1) {
      if (vect_dimension(cl->vecteur)==0) // borne constante numerique
	cn = cl;
      else  { // borne constante symbolique
	for (pv = cl->vecteur;pv!=NULL;pv=pv->succ) {
	  if (pv->var!= TCST) {
	    if (pv->val ==1)
	      cs1=cl;
	    else csx = cl;
	  }
	}
      }
    }
    else clv = cl;
  }
    if (cn!= CONTRAINTE_UNDEFINED) return cn;
    else if (cs1!= CONTRAINTE_UNDEFINED) return cs1;
    else if (csx!= CONTRAINTE_UNDEFINED) return csx;
    else return clv;
}

// A changer par une fonction qui detectera si la variable a ete definie
// dans un fichier de parametres ...
static bool entity_xml_parameter_p(entity e)
{
  const char* s = entity_local_name(e);
  bool b=false;
  if (strstr(s,"PARAM")!=NULL || strstr(s,"param")!=NULL) b = true;
  return (b);
}


static void  find_pattern(Psysteme ps, Pvecteur paving_indices, Pvecteur formal_parameters, int dim,  Pcontrainte *bound_inf, Pcontrainte *bound_sup, Pcontrainte *pattern_up_bound , Pcontrainte *iterator)
{
  Variable phi;
  Value	v;
  Pvecteur pi;
  Pcontrainte c, next, cl, cu, cl_dup, cu_dup,pattern_dup, lind, lind_dup,
    list_cl=NULL,
    list_cu=NULL,
    list_ind=NULL,
    list_pattern=NULL,
    ctmp=NULL,
    pattern = CONTRAINTE_UNDEFINED;
  int lower =1;
  int upper =2;
  int ind =3;
  Pcontrainte bounds[4][4];
  int nb_bounds =0;
  int nb_lower = 0;
  int nb_upper = 0;
  int nb_indices=0;
  int i,j;
  Pbase vars_to_eliminate = BASE_NULLE;
  Pvecteur vdiff = VECTEUR_NUL;

  for (i=1; i<=3;i++)
    for (j=1; j<=3;j++)
      bounds[i][j]=CONTRAINTE_UNDEFINED;

  phi = (Variable) make_phi_entity(dim);

  /* Liste des  variables a eliminer autres de les phi et les indices de boucles englobants
     et les parametres formels de la fonction.
     copie de la base + mise a zero des indices englobants
     + projection selon les elem de ce vecteur
  */
  if (!SC_EMPTY_P(ps)) {

    Psysteme ps_dup=sc_dup(ps);
    vars_to_eliminate = vect_copy(ps->base);
    vect_erase_var(&vars_to_eliminate, phi);

    //printf("paving_indices:\n");
    //vect_fprint(stdout,paving_indices,(char * (*)(Variable))  entity_local_name);

    for (pi = paving_indices; !VECTEUR_NUL_P(pi); pi = pi->succ)
      vect_erase_var(&vars_to_eliminate, var_of(pi));
    //printf("formal_parameters:\n");
    // vect_fprint(stdout,formal_parameters,(char * (*)(Variable)) entity_local_name);

    for (pi = formal_parameters; !VECTEUR_NUL_P(pi); pi = pi->succ)
      vect_erase_var(&vars_to_eliminate, var_of(pi));
    //printf("vars_to_eliminate:\n");
    //vect_fprint(stdout,vars_to_eliminate,(char * (*)(Variable)) entity_local_name);

    sc_projection_along_variables_ofl_ctrl(&ps_dup,vars_to_eliminate , NO_OFL_CTRL);

    //printf("Systeme a partir duquel on genere les contraintes  du motif:\n");
    //sc_fprint(stdout,ps,entity_local_name);

    for(c = sc_inegalites(ps_dup), next=(c==NULL ? NULL : c->succ);
	c!=NULL;
	c=next, next=(c==NULL ? NULL : c->succ))
      {
	Pvecteur indices_in_vecteur = VECTEUR_NUL;
	vdiff = VECTEUR_NUL;
	v = vect_coeff(phi, c->vecteur);
	for (pi = paving_indices; !VECTEUR_NUL_P(pi); pi = pi->succ)
	  {
	    int coeff_index = vect_coeff(var_of(pi),c->vecteur);
	    if (coeff_index)
	      vect_add_elem(&indices_in_vecteur,var_of(pi), coeff_index);
	  }
	/* on classe toutes les contraintes ayant plus de 2 indices de boucles
	   externes ensemble */
	nb_indices=vect_size(indices_in_vecteur);
	nb_indices = (nb_indices >2) ? 2 : nb_indices;

	if (value_pos_p(v)) {
	  c->succ = bounds[upper][nb_indices+1];
	  bounds[upper][nb_indices+1] = c;
	  /* printf(" bornes inf avec indices de boucles englobants :\n");
	     vect_print(bounds[upper][nb_indices+1]->vecteur, entity_local_name); */
	  nb_upper ++;
	}
	else if (value_neg_p(v)) {
	  c->succ = bounds[lower][nb_indices+1];
	  bounds[lower][nb_indices+1] = c;
	  /* printf(" bornes inf avec indices de boucles englobants :\n");
	     vect_print(bounds[lower][nb_indices+1]->vecteur, entity_local_name);*/
	  lind = contrainte_make(indices_in_vecteur);
	  lind->succ = bounds[ind][nb_indices+1];
	  bounds[ind][nb_indices+1] = lind;
	  /* printf(" indices contenus dans la contrainte :\n");
	     vect_print(bounds[ind][nb_indices+1]->vecteur, entity_local_name); */
	  nb_lower ++;
	}
      }
    // printf("Nb borne inf = %d, Nb borne sup = %d pour dimension %d;\n",nb_lower,nb_upper,dim);

    if  (!CONTRAINTE_UNDEFINED_P(bounds[lower][2])) {
      /* case with 1 loop index in the loop bound constraints */
      for(cl = bounds[lower][2], lind= bounds[ind][2]; cl !=NULL; cl=cl->succ,lind=lind->succ)  {
	for(cu = bounds[upper][2]; cu !=NULL; cu =cu->succ) {
	  int cv1 = vect_coeff(phi, cl->vecteur);
	  int cv2 = vect_coeff(phi, cu->vecteur);
	  if (value_abs(cv1)==value_abs(cv2))
	    vdiff = vect_add(cu->vecteur,cl->vecteur);
	  else  vdiff = vect_cl2(value_abs(cv1),cu->vecteur,value_abs(cv2),cl->vecteur);
	  vect_chg_sgn(vdiff);
	  vect_add_elem(&vdiff,TCST,1);
	  pattern = contrainte_make(vect_dup(vdiff));
	  for (pi = formal_parameters; !VECTEUR_NUL_P(pi); pi = pi->succ)
	    vect_erase_var(&vdiff, var_of(pi));
	  if (vect_dimension(vdiff)==0) {
	    cl_dup = contrainte_dup(cl);
	    cu_dup = contrainte_dup(cu);

	    for (pi = lind->vecteur; !VECTEUR_NUL_P(pi); pi = pi->succ)
	      {
		vect_erase_var(&(cl_dup->vecteur), var_of(pi));
		vect_erase_var(&(cu_dup->vecteur), var_of(pi));
	      }
	    vect_erase_var(&(cl_dup->vecteur), phi);
	    vect_erase_var(&(cu_dup->vecteur), phi);
	    cl_dup->succ = list_cl, list_cl=cl_dup;

	    vect_chg_sgn(cu_dup->vecteur);
	    cu_dup->succ = list_cu, list_cu=cu_dup;

	    lind_dup = contrainte_dup(lind);
	    lind_dup->succ = list_ind, list_ind = lind_dup;
	    pattern_dup = pattern;
	    pattern_dup->succ = list_pattern, list_pattern = pattern_dup;
	    nb_bounds ++;
	  }
	  vect_rm(vdiff);
	}
      }
      *bound_inf= list_cl;
      *bound_sup = list_cu;
      *pattern_up_bound = list_pattern;
      *iterator = list_ind;
    }
    else if (!CONTRAINTE_UNDEFINED_P(bounds[lower][1])
	     && !CONTRAINTE_UNDEFINED_P(bounds[upper][1])) {
      /* case where loop bounds are numeric */
      for(cl = bounds[lower][1]; cl !=NULL; cl=cl->succ)  {
	*bound_inf= cl;
	for(cu = bounds[upper][1]; cu !=NULL; cu =cu->succ) {
	  *bound_sup = cu;
	  int cv1 = vect_coeff(phi, cl->vecteur);
	  int cv2 = vect_coeff(phi, cu->vecteur);
	  for (pi = bounds[ind][1]->vecteur; !VECTEUR_NUL_P(pi); pi = pi->succ) {
	    vect_erase_var(&(cl->vecteur), var_of(pi));
	    vect_erase_var(&(cu->vecteur), var_of(pi));
	  }
	    if (value_abs(cv1)==value_abs(cv2))
	     vdiff = vect_add(cu->vecteur,cl->vecteur);
	    else  vdiff = vect_cl2(value_abs(cv1),cu->vecteur,value_abs(cv2),cl->vecteur);
	   vect_chg_sgn(vdiff);
	  vect_add_elem(&vdiff,TCST,1); 
	  ctmp = contrainte_make(vdiff);
	  ctmp->succ = *pattern_up_bound;
	  *pattern_up_bound = ctmp;
	  *iterator =  bounds[ind][1];
	}
      }
      //      printf("Liste des contraintes sur PHI-%d: \n",dim);
      // inegalites_fprint(stdout,*pattern_up_bound, (char * (*)(Variable)) entity_local_name);
       *pattern_up_bound = choose_pattern(*pattern_up_bound);
       }
    else {
      /* Only bounds with several loop indices */
      /* printf("PB - Only bounds with several loop indices\n"); */
      *bound_inf= CONTRAINTE_UNDEFINED;
      *bound_sup = CONTRAINTE_UNDEFINED;
      *iterator = CONTRAINTE_UNDEFINED;
    }
    base_rm( vars_to_eliminate);
    sc_rm(ps_dup);
  }
}

static void xml_Pattern_Paving( region reg,entity var, bool effet_read, Pvecteur formal_parameters, Pvecteur paving_indices, string_buffer sb_result)
{
  string_buffer buffer_pattern = string_buffer_make(true);
  string_buffer buffer_paving = string_buffer_make(true);
  string string_paving = "";
  string string_pattern =  "";
  Pvecteur voffset, vpattern_up_bound;

  if(reg != region_undefined) {
    reference ref = effect_any_reference(reg);
    entity vreg = reference_variable(ref);
    if ( array_entity_p(reference_variable(ref)) && same_entity_p(vreg,var) && region_read_p(reg)== effet_read) {
      Psysteme ps_reg = sc_dup(region_system(reg));
      Pvecteur iterat= VECTEUR_NUL;
      Pcontrainte bound_inf = CONTRAINTE_UNDEFINED;
      Pcontrainte bound_up = CONTRAINTE_UNDEFINED;
      Pcontrainte pattern_up_bound = CONTRAINTE_UNDEFINED;
      Pcontrainte iterator = CONTRAINTE_UNDEFINED;
      int dim = (int) gen_length(variable_dimensions(type_variable(entity_type(var))));
      int i ;
      int val =0;
      int inc =0;
      Variable phi;
      Value min, max;
      Psysteme ps1;
      bool  feasible;
      dimension vreg_dim;
      sc_transform_eg_in_ineg(ps_reg);
      string_buffer_append_word("Pattern",buffer_pattern);
      string_buffer_append_word("Pavage",buffer_paving);
      global_margin++;
      // pour chaque dimension : generer en meme temps le pattern et le pavage
      for(i=1; i<=dim; i++) {
	bound_inf = CONTRAINTE_UNDEFINED;
	bound_up = CONTRAINTE_UNDEFINED;
	pattern_up_bound = CONTRAINTE_UNDEFINED;
	voffset =vect_new(TCST,0);
	//printf("find pattern for entity %s [dim=%d]\n",entity_local_name(var),i);
	//sc_fprint(stdout,ps_reg,(char * (*)(Variable)) entity_local_name);
	find_pattern(ps_reg, paving_indices, formal_parameters, i, &bound_inf, &bound_up, &pattern_up_bound, &iterator);

		phi = (Variable) make_phi_entity(dim);
	if(base_contains_variable_p(sc_base(ps_reg),phi)) {
	  ps1 = sc_dup(ps_reg);
	  feasible = sc_minmax_of_variable(ps1, (Variable)phi, &min, &max);
	  if (feasible && min!=VALUE_MIN)
	    voffset = vect_new(TCST,min);
	}
	if (!CONTRAINTE_UNDEFINED_P(pattern_up_bound) &&   !VECTEUR_NUL_P(pattern_up_bound->vecteur))
	  vpattern_up_bound = pattern_up_bound->vecteur;
	else  { // if we cannot deduce pattern length from region, array dim size is taken
	  list ldim = variable_dimensions(type_variable(entity_type(vreg)));
	  vreg_dim = find_ith_dimension(ldim,i);
	  normalized ndim = NORMALIZE_EXPRESSION(dimension_upper(vreg_dim));
	  vpattern_up_bound = vect_dup((Pvecteur) normalized_linear(ndim));
	  if (vpattern_up_bound  != VECTEUR_NUL)
	    vect_add_elem(&vpattern_up_bound,TCST,1);
	}
	/* PRINT PATTERN and PAVING */
	if ( vect_zero_p(voffset)  && vect_one_p(vpattern_up_bound))
	  string_buffer_append_word("DimUnitPattern/",buffer_pattern);
	else {
	  add_margin(global_margin,buffer_pattern);
	  // A completer - choisir un indice pour le motif ?
	  string_buffer_append(buffer_pattern,
			       concatenate(OPENANGLE,
					   "DimPattern Index=",
					   QUOTE,QUOTE,CLOSEANGLE,NL,NULL));
	  global_margin++;
	  string_buffer_append_word("Offset",buffer_pattern);
	  string_buffer_append_symbolic(vect_to_string(voffset),
					buffer_pattern);
	  if (vect_dimension(voffset)==0)
	    string_buffer_append_numeric(vect_to_string(voffset),
					 buffer_pattern);
	  string_buffer_append_word("/Offset",buffer_pattern);

	  string_buffer_append_word("Length",buffer_pattern);
	  // The upper bound is not a complex expression
	  if (vpattern_up_bound  != VECTEUR_NUL)
	    string_buffer_append_symbolic(vect_to_string(vpattern_up_bound),
					  buffer_pattern);
	  else string_buffer_append_symbolic(
					     words_to_string(words_syntax(expression_syntax(dimension_upper(vreg_dim)),NIL)),
					     buffer_pattern);
	  if (vect_dimension(vpattern_up_bound)==0)
	    string_buffer_append_numeric(vect_to_string(vpattern_up_bound),
					 buffer_pattern);
	  string_buffer_append_word("/Length",buffer_pattern);

	  string_buffer_append_word("Stride",buffer_pattern);
	  val =1;
	  string_buffer_append_symbolic(i2a(val),buffer_pattern);
	  string_buffer_append_numeric(i2a(val),buffer_pattern);
	  string_buffer_append_word("/Stride",buffer_pattern);
	  global_margin--;
	  string_buffer_append_word("/DimPattern",buffer_pattern);
	}
	if (!CONTRAINTE_UNDEFINED_P(iterator) &&!CONTRAINTE_NULLE_P(iterator)) {
	  string_buffer_append_word("DimPavage",buffer_paving);
	  for (iterat = paving_indices; !VECTEUR_NUL_P(iterat); iterat = iterat->succ) {
	    if ((inc = vect_coeff(var_of(iterat),iterator->vecteur)) !=0) {
	      add_margin(global_margin,buffer_paving);
	      string_buffer_append(buffer_paving,
				   concatenate(OPENANGLE,
					       "RefLoopIndex Name=",
					       QUOTE,entity_user_name(var_of(iterat)),QUOTE, BL,
					       "Inc=", QUOTE,
					       i2a(inc),QUOTE,"/", CLOSEANGLE,NL,NULL));
	    }
	  }
	  string_buffer_append_word("/DimPavage",buffer_paving);
	}
	else
	  string_buffer_append_word("DimPavage/",buffer_paving);
      }
      global_margin--;
      string_buffer_append_word("/Pattern",buffer_pattern);
      string_buffer_append_word("/Pavage",buffer_paving);
      string_pattern = string_buffer_to_string(buffer_pattern);
      string_paving = string_buffer_to_string(buffer_paving);
      string_buffer_append(sb_result, string_pattern);
      free(string_pattern);
      string_pattern=NULL;
      string_buffer_append(sb_result, string_paving);
      free(string_paving);
      string_paving=NULL;
      xml_AccessFunction(sb_result);
      sc_free(ps_reg);
    }
  }
}
void vars_read_or_written(list effects_list, Pvecteur *vl)
{
  Pvecteur vin=VECTEUR_NUL;
  Pvecteur vout = VECTEUR_NUL;
  list pc;
  for (pc= effects_list;pc != NIL; pc = CDR(pc)){
    effect e = EFFECT(CAR(pc));
    reference r = effect_any_reference(e);
    if (array_entity_p(reference_variable(r))) {
      entity v = reference_variable(r);
      if (effect_read_p(e))
	vect_add_elem(&vin,v,1);
      else  vect_add_elem(&vout,v,1);
    }
  }
  vin = base_normalize(vin);
  vout = base_normalize(vout);
  *vl = vect_substract(vin,vout);
  *vl= base_normalize(*vl);
  vect_rm(vin);
  vect_rm(vout);
}

static void  xml_Region_Range(region reg, string_buffer sb_result)
{
  Variable phi;
  int i;
  Pcontrainte pc;
  if(reg != region_undefined) {
    reference ref = effect_any_reference(reg);
    entity vreg = reference_variable(ref);
    if (array_entity_p(reference_variable(ref))) {
      int dim = (int) gen_length(variable_dimensions(type_variable(entity_type(vreg))));
      Psysteme ps_reg = sc_dup(region_system(reg));
      /* on peut ameliorer les resultats en projetant
	 les egalites portant sur les variables autres que les PHI */
      /*
	Pvecteur vva = VECTEUR_NUL;
	Variable va;
	for(pc = sc_egalites(ps_reg); pc!=NULL;pc= pc->succ)
	{
	va = vect_one_coeff_if_any(pc->vecteur);
	if (va!= NULL && strcmp(variable_name(va),"PHI")==0 ) vect_add_elem(&vva,va,1);
	}
	sc_projection_along_variables_ofl_ctrl(&ps_reg,vva , NO_OFL_CTRL);
      */
      sc_transform_eg_in_ineg(ps_reg);
      for (i=1;i<=dim;i++) {
	string_buffer sbi_result=string_buffer_make(true);
	string_buffer sbu_result=string_buffer_make(true);
	string string_sbi, string_sbu;
	int fub = 0;
	int fib = 0;
	phi = (Variable) make_phi_entity(i);
	string_buffer_append(sbi_result,
			     concatenate("[", NULL));
	string_buffer_append(sbu_result,
			     concatenate(";",NULL));
	for(pc = sc_inegalites(ps_reg); pc!=NULL;pc= pc->succ)
	  {
	    int vc = vect_coeff(phi, pc->vecteur);
	    Pvecteur vvc = vect_dup(pc->vecteur);
	    string sb= NULL;
	    string scst;
	    if (value_pos_p(vc)) { // borne sup
	      vect_erase_var(&vvc,phi);
	      vect_chg_sgn(vvc);
	      sb = vect_to_string(vvc);
	      if (vc-1)
		scst =strdup(itoa(vc));
	      if (fub++)
		string_buffer_append(sbu_result,
				     concatenate(",",sb, NULL));
	      else
		string_buffer_append(sbu_result,
				     concatenate(sb, NULL));
	      if (vc-1)
		string_buffer_append(sbu_result,
				     concatenate("/",scst, NULL));
	    }
	    else if (value_neg_p(vc)) {
	      vect_erase_var(&vvc,phi);
	      sb = vect_to_string(vvc);
	      if (vc+1) scst =strdup(itoa(-1*vc));
	      if (fib++)
		string_buffer_append(sbi_result,
				     concatenate(",",sb, NULL));
	      else
		string_buffer_append(sbi_result,
				     concatenate(sb, NULL));
	      if (vc+1)
		string_buffer_append(sbi_result,
				     concatenate("/",scst, NULL));
	    }
	  }
	string_buffer_append(sbu_result,
			     concatenate("]",NULL));
	string_sbi = string_buffer_to_string(sbi_result);
	string_sbu = string_buffer_to_string(sbu_result);
	string_buffer_append(sb_result,string_sbi);
	string_buffer_append(sb_result,string_sbu);
      }
      sc_rm(ps_reg);
    }
  }
}

static bool string_in_list_p(string ts,list lr){
  bool trouve=false;
  MAPL(tt,
       {trouve= trouve || strcmp(STRING(CAR(tt)),ts) == 0;},
       lr);
  return trouve;
}


static void xml_Region_Parameter(list pattern_region, string_buffer sb_result)
{
  list lr;
  list lrr = NIL, lrw=NIL;
  bool effet_read = true;
  reference ref;
  region reg;
  entity v;

  string_buffer_append_word("ReferencedParameters",sb_result);
  global_margin++;

  for ( lr = pattern_region; !ENDP(lr); lr = CDR(lr))
    {
      reg = REGION(CAR(lr));
      ref = effect_any_reference(reg);
      v = reference_variable(ref);
      if (array_entity_p(reference_variable(ref))
	  && !(entity_static_variable_p(v) && !top_level_entity_p(v))) {
	string ts = strdup(entity_user_name(v));
	effet_read = region_read_p(reg);
	if ((effet_read && !string_in_list_p(ts,lrr)) || (!effet_read && !string_in_list_p(ts,lrw))) {
	  if (effet_read)
	    lrr = gen_nconc(lrr,CONS(STRING,ts,NIL));
	  else lrw = gen_nconc(lrw,CONS(STRING,ts,NIL));
	  add_margin(global_margin,sb_result);
	  string_buffer_append(sb_result,
			       concatenate(OPENANGLE,
					   "ReferencedParameter"," Name=",
					   QUOTE,entity_user_name(v),QUOTE, BL,
					   "Range=",QUOTE,NULL));
	  xml_Region_Range(reg, sb_result);

	  string_buffer_append(sb_result,
			       concatenate(QUOTE, BL, "Type=", QUOTE,(entity_xml_parameter_p(v))? "CONTROL":"DATA",QUOTE,BL,
					   "AccessMode=", QUOTE, (effet_read)? "USE":"DEF",QUOTE,BL,
					   "ArrayP=", QUOTE, (array_entity_p(v))?"TRUE":"FALSE",QUOTE, BL,
					   "Kind=", QUOTE,  "VARIABLE",QUOTE,"/",
					   CLOSEANGLE
					   NL, NULL));
	  global_margin++;
	  //	  xml_Pattern_Paving( reg,v, effet_read, formal_parameters,
	  //			     paving_indices,sb_result);

	  global_margin--;
	  //	  string_buffer_append_word("/ReferencedParameter",sb_result);
	}
      }
    }
  global_margin--;
  string_buffer_append_word("/ReferencedParameters",sb_result);
  gen_free_list(lrr);
   gen_free_list(lrw);
}

int find_effect_actions_for_entity(list leff, effect *effr, effect *effw, entity e)
{
  // return effet_rwb = 1 for Read, 2 for Write, 3 for Read/Write
  int effet_rwb=0;
  list lr = NIL;
  bool er = false;
  bool ew = false;

  for ( lr = leff; !ENDP(lr); lr = CDR(lr)) {
    effect eff=  EFFECT(CAR(lr));
    reference ref = effect_any_reference(eff);
    entity v = reference_variable(ref);
    if (same_entity_p(v,e)) {
      if (action_read_p(effect_action(eff)))
	er = true, *effr =eff;
      else ew = true, *effw=eff;
    }
  }
  effet_rwb = (er?1:0) +(ew?2:0);
  return (effet_rwb);
}

static void xml_ParameterUseToArrayBound(entity var, string_buffer sb_result)
{
  string sdim;
  entity FormalArrayName, mod = get_current_module_entity();
  int ith, FormalParameterNumber = (int) gen_length(module_formal_parameters(mod));
  global_margin++;
  for (ith=1;ith<=FormalParameterNumber;ith++) {
    FormalArrayName = find_ith_formal_parameter(mod,ith);
    if (type_variable_p(entity_type(FormalArrayName))
	&& ( variable_entity_dimension(FormalArrayName)>0)) {
      list ld, ldim = variable_dimensions(type_variable(entity_type(FormalArrayName)));
      int dim;
      for (ld = ldim, dim =1 ; !ENDP(ld); ld = CDR(ld), dim++) {
	expression elow = dimension_lower(DIMENSION(CAR(ld)));
	expression eup = dimension_upper(DIMENSION(CAR(ld)));
	const char * low= words_to_string(words_syntax(expression_syntax(elow),NIL));
	const char * up = words_to_string(words_syntax(expression_syntax(eup),NIL));
	const char * sv = entity_local_name(var);
	if ((strstr(low,sv)!=NULL) || (strstr(up,sv)!=NULL)) {
	  sdim= strdup(itoa(variable_entity_dimension(FormalArrayName)-dim+1));
	  add_margin(global_margin,sb_result);
	  string_buffer_append(sb_result,
			       concatenate(OPENANGLE,
					   "TaskParameterUsedFor"," ArrayName=",
					   QUOTE,entity_user_name(FormalArrayName),QUOTE, BL,
					   "Dim=", QUOTE,sdim,QUOTE,"/",
					   CLOSEANGLE,
					   NL, NULL));
	}
      }
    }
  }
  global_margin--;
}


static void xml_TaskParameter(bool assign_function,bool is_not_main_p, entity var, Pvecteur formal_parameters, list pattern_region, Pvecteur paving_indices, string_buffer sb_result)
{
  bool effet_read = true;
  region rwr = region_undefined;
  region rre = region_undefined;
  const char* datatype ="";
  int size=0;
  reference ref;
  region reg;
  entity v;
  bool  pavp=true;

  // rappel : les regions contiennent les effects sur les scalaires
  // pour les fonctions, on ecrit les parametres formels dans l'ordre
  // pour le MAIN, ordre des regions:
  if  (!is_not_main_p) {
    reg = REGION(CAR(pattern_region));
    ref = effect_any_reference(reg);
    v = reference_variable(ref);
    pavp =(vect_coeff(v,paving_indices) == 0);
    effet_read = region_read_p(reg);
  }
  else { // c'est une fonction -->  impression selon l'ordre des parametres formels
    // recherche des regions du parametre formel
    effet_read = (find_effect_actions_for_entity(pattern_region,&rre, &rwr,var)==1);
    // choix des regions write de preference
    reg = (rwr != region_undefined)? rwr : rre;
    if (reg != region_undefined) {
      ref = effect_any_reference(reg);
      v = reference_variable(ref);
    }
    else  // cas ou il n'y a pas d'effet sur le parametre formel,
      //  mais il fait partie de la liste des parametres de la fonction
      v = var, effet_read = true;
  }
  if (pavp) {
    type_and_size_of_var(v, &datatype,&size);
    add_margin(global_margin,sb_result);
    string_buffer_append(sb_result,
			 concatenate(OPENANGLE,
				     (assign_function)?"AssignParameter":"TaskParameter"," Name=",
				     QUOTE,entity_user_name(v),QUOTE, BL,
				     "Type=", QUOTE,(entity_xml_parameter_p(v))? "CONTROL":"DATA",QUOTE,BL,
				     "DataType=",QUOTE,datatype,QUOTE,BL,
				     "AccessMode=", QUOTE, (effet_read)? "USE":"DEF",QUOTE,BL,
				     "ArrayP=", QUOTE, (array_entity_p(v))?"TRUE":"FALSE",QUOTE, BL,
				     "Kind=", QUOTE,  "VARIABLE",QUOTE,
				     CLOSEANGLE,
				     NL, NULL));
    xml_ParameterUseToArrayBound(var,sb_result);
    global_margin++;
    xml_Pattern_Paving(reg,v, effet_read, formal_parameters,
		       paving_indices, sb_result);
    global_margin--;
    if (assign_function)
      string_buffer_append_word("/AssignParameter",sb_result);
    else
      string_buffer_append_word("/TaskParameter",sb_result);
  }

}



static void xml_TaskParameters(bool assign_function, int code_tag, entity module, list pattern_region, Pvecteur paving_indices, string_buffer sb_result)
{

  int ith;
  int FormalParameterNumber = (int) gen_length(module_formal_parameters(module));
  Pvecteur formal_parameters = VECTEUR_NUL;
  entity FormalArrayName = entity_undefined;
  list lr=NIL;

  if (assign_function)
    string_buffer_append_word("AssignParameters",sb_result);
  else
    string_buffer_append_word("TaskParameters",sb_result);
  global_margin++;

  // Creation liste des parametres formels
  for (ith=1;ith<=FormalParameterNumber;ith++) {
    FormalArrayName = find_ith_formal_parameter(module,ith);
    vect_add_elem (&formal_parameters,(Variable) FormalArrayName,VALUE_ONE);
  }

  if (code_tag != code_is_a_main) {
    // les parametres formels doivent etre introduits dans l'ordre.
    for (ith=1;ith<=FormalParameterNumber;ith++) {
      FormalArrayName = find_ith_formal_parameter(module,ith);
      xml_TaskParameter(assign_function,true,FormalArrayName,
			formal_parameters,pattern_region,paving_indices,sb_result);
    }
  }
  else   {
    for (lr = pattern_region; !ENDP(lr); lr = CDR(lr)) {
      xml_TaskParameter(assign_function,false,FormalArrayName,
			formal_parameters,lr,paving_indices,sb_result);
    }
  }
  global_margin--;
  if (assign_function)
    string_buffer_append_word("/AssignParameters",sb_result);
  else
    string_buffer_append_word("/TaskParameters",sb_result);
}


bool  eval_linear_expression(expression exp, Psysteme ps, int *val)
{
  normalized norm= normalized_undefined;
  Pvecteur   vexp,pv;
  bool result = true;
  *val = 0;

  // fprintf(stdout,"Expression a evaluer : %s",words_to_string(words_expression(exp,NIL)));

  if (expression_normalized(exp) == normalized_undefined)
    expression_normalized(exp)= NormalizeExpression(exp);
  norm = expression_normalized(exp);
  if (normalized_linear_p(norm) && !SC_UNDEFINED_P(ps)) {
    vexp = (Pvecteur)normalized_linear(norm);
    for (pv= vexp; pv != VECTEUR_NUL; pv=pv->succ) {
      Variable v = var_of(pv);
      if (v==TCST)
	*val += vect_coeff(TCST,vexp);
      else {
	if(base_contains_variable_p(sc_base(ps), v)) {
	  Value min, max;
	  Psysteme ps1 = sc_dup(ps);
	  bool  feasible = sc_minmax_of_variable2(ps1, (Variable)v, &min, &max);
	  if (feasible && value_eq(min,max)) {
	    *val += vect_coeff(v,vexp) *min;
	  }
	  else  // fprintf(stdout,"le systeme est non faisable\n");
	    result = false;
	} //  sc_free(ps1); le systeme est desalloue par sc_minmax_of_variable
	else
	  result = false;
      }
    }
  }
  else  // fprintf(stdout,"Ce n'est pas une expression lineaire\n");
    result = false;

  //fprintf(stdout,"Valeur trouvee : %d \n",*val);
  return result;
}

static void xml_Bounds(expression elow, expression eup,Psysteme prec, string_buffer sb_result)
{
  intptr_t low,up;
  int valr =0;
  /* Print XML Array LOWER BOUND */
  string_buffer_append_word("LowerBound",sb_result);
  global_margin++;
  string_buffer_append_symbolic(words_to_string(words_syntax(expression_syntax(elow),NIL)),
				sb_result);
  if (expression_integer_value(elow, &low))
    string_buffer_append_numeric(i2a(low),sb_result);
  else if (eval_linear_expression(elow,prec,&valr))
    string_buffer_append_numeric(i2a(valr),sb_result);

  global_margin--;
  string_buffer_append_word("/LowerBound",sb_result);

  /* Print XML Array UPPER BOUND */
  string_buffer_append_word("UpperBound",sb_result);
  global_margin++;
  string_buffer_append_symbolic(words_to_string(words_syntax(expression_syntax(eup),NIL)),
				sb_result);
  if (expression_integer_value(eup, &up))
    string_buffer_append_numeric(i2a(up),sb_result);
  else if (eval_linear_expression(eup,prec,&valr))
    string_buffer_append_numeric(i2a(valr),sb_result);

  global_margin--;
  string_buffer_append_word("/UpperBound",sb_result);

}
static void xml_Bounds_and_Stride(expression elow, expression eup, expression stride,
				  Psysteme prec, string_buffer sb_result)
{
  intptr_t inc;
  int  valr =0;
  xml_Bounds(elow, eup,prec,sb_result);
  string_buffer_append_word("Stride",sb_result);
  global_margin++;
  string_buffer_append_symbolic(words_to_string(words_syntax(expression_syntax(stride),NIL)),
				sb_result);
  if (expression_integer_value(stride, &inc))
    string_buffer_append_numeric(i2a(inc),sb_result);
  else if (eval_linear_expression(stride,prec,&valr))
    string_buffer_append_numeric(i2a(valr),sb_result);

  global_margin--;
  string_buffer_append_word("/Stride",sb_result);

}

static void find_memory_comment_on_array(statement s)
{
  string comm = statement_comments(s);
  string result = NULL;

  if (!string_undefined_p(comm)
      && (result=strstr(comm,array_mem_string))!=NULL)
    array_location_string = result;
}

static string memory_location_for_array_p(const char* sa)
{
  statement stat = get_current_module_statement();
  string result=NULL;
  int n=0;

  array_location_string = NULL;
  array_mem_string = (char *)malloc(strlen(sa)+5+1);
  (void)  strcpy(array_mem_string, sa);
  (void) strcat(array_mem_string,MEM_PREFIX);
  gen_recurse(stat, statement_domain, gen_true,find_memory_comment_on_array);
  if (array_location_string != NULL) {
    n= strcspn(array_location_string+strlen(sa)+5,":\n");
    result=(char *)malloc(n+1);
    strncpy(result,array_location_string+strlen(sa)+5,n);
    result[n]=(char) 0;
  }
  free(array_mem_string);
  return (result);
}


static void xml_Array(entity var,Psysteme prec,string_buffer sb_result)
{
  const char* datatype ="";
  list ld, ldim = variable_dimensions(type_variable(entity_type(var)));
  int i,j, size =0;
  int nb_dim = (int) gen_length(ldim);
  char* layout_up[nb_dim];
  char *layout_low[nb_dim];
  int no_dim=0;

  string spec_location = memory_location_for_array_p(entity_user_name(var));

  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Array Name=",
				   QUOTE,entity_user_name(var),QUOTE, BL,
				   "Type=", QUOTE,(entity_xml_parameter_p(var))? "CONTROL":"DATA",QUOTE,BL,
				   "Allocation=", QUOTE,
				   (heap_area_p(var) || stack_area_p(var)) ?
				   "DYNAMIC": "STATIC",
				   QUOTE,BL,
				   "Kind=", QUOTE,   "VARIABLE",QUOTE,
				   CLOSEANGLE,NL, NULL));

  /* Print XML Array DATA TYPE and DATA SIZE */
  type_and_size_of_var(var, &datatype,&size);
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "DataType Type=",QUOTE,datatype,QUOTE, BL,
				   "Size=",QUOTE, i2a(size), QUOTE, "/"
				   CLOSEANGLE,NL,NULL));
  /* Print XML Array Memory Location */
  if (spec_location!=NULL) {
    add_margin(global_margin,sb_result);
    string_buffer_append(sb_result,
			 concatenate(OPENANGLE,"ArrayLocation",CLOSEANGLE,
				     spec_location,OPENANGLE, "/ArrayLocation",CLOSEANGLE,
				     NL, NULL));
  }
  /* Print XML Array DIMENSION */

  if((int) gen_length(ldim) >0) {
    string_buffer_append_word("Dimensions",sb_result);

    for (ld = ldim ; !ENDP(ld); ld = CDR(ld)) {
      expression elow = dimension_lower(DIMENSION(CAR(ld)));
      expression eup = dimension_upper(DIMENSION(CAR(ld)));
      global_margin++;
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "Dimension",
				       CLOSEANGLE,NL, NULL));
      /* Print XML Array Bound */
      global_margin++;
      xml_Bounds(elow,eup,prec,sb_result);
      global_margin--;
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "/Dimension",
				       CLOSEANGLE,NL, NULL));
      global_margin--;
      layout_low[no_dim] = words_to_string(words_syntax(expression_syntax(elow),NIL));
      layout_up[no_dim] = words_to_string(words_syntax(expression_syntax(eup),NIL));
      no_dim++;
    }
  }

  string_buffer_append_word("/Dimensions",sb_result);

  /* Print XML Array LAYOUT */
  string_buffer_append_word("Layout",sb_result);
  global_margin++;
  for (i =0; i<= no_dim-1; i++) {
    string_buffer_append_word("DimLayout",sb_result);
    global_margin++;
    string_buffer_append_word("Symbolic",sb_result);
    if (i==no_dim-1) {
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,concatenate("1",NL,NULL));
    }
    else {
      add_margin(global_margin,sb_result);
      for (j=no_dim-1; j>=i+1; j--)
	{
	  if (strcmp(layout_low[j],"0")==0)
	    string_buffer_append(sb_result,
				 concatenate("(",layout_up[j],"+1)",NULL));
	  else
	    string_buffer_append(sb_result,
				 concatenate("(",layout_up[j],"-",
					     layout_low[j],"+1)",NULL));
	  if (j==i+1)
	    string_buffer_append(sb_result,NL);
	}
    }
    string_buffer_append_word("/Symbolic",sb_result);
    global_margin--;
    string_buffer_append_word("/DimLayout",sb_result);
  }
  global_margin--;
  string_buffer_append_word("/Layout",sb_result);
  string_buffer_append_word("/Array",sb_result);
}

static void cumul_effects_of_statement(statement s)
{
  cumulated_list = gen_append(cumulated_list,load_proper_rw_effects_list(s));
}

static void xml_GlobalArrays(Psysteme prec,string_buffer sb_result)
{
  list ldecl;
  int nb_dim=0;
  Pvecteur v_already_seen=VECTEUR_NUL;
  statement s = get_current_module_statement();
  cumulated_list = NIL;
  gen_recurse(s, statement_domain, gen_true, cumul_effects_of_statement);

  ldecl = statement_declarations(s);
  MAP(ENTITY,var, {
      vect_add_elem(&v_already_seen,var,1);
    },ldecl);
  if((int) gen_length( cumulated_list) >0) {
    string_buffer_append_word("GlobalArrays",sb_result);
    global_margin++;
    MAP(EFFECT,ef, {
	reference ref = effect_any_reference(ef);
	entity v = reference_variable(ref);
	if (type_variable_p(entity_type(v))
	    && ((nb_dim = variable_entity_dimension(v))>0)
	    && !io_entity_p(v)
	    &&  (storage_ram_p(entity_storage(v)))
	    && top_level_entity_p(v)
	    //  && entity_module_name(ram_section(storage_ram(entity_storage(v))))get_current_module_name()
	    && !storage_formal_p(entity_storage(v))
	    && !vect_coeff(v,v_already_seen)) {
	  vect_add_elem(&v_already_seen,v,1);
	  xml_Array(v,prec,sb_result);
	}
      }, cumulated_list);
    vect_rm(v_already_seen);
    global_margin--;
    string_buffer_append_word("/GlobalArrays",sb_result);
  }
  else
    string_buffer_append_word("GlobalArrays/",sb_result);
}



static void xml_LocalArrays(entity module, Psysteme prec,string_buffer sb_result)
{
  list ldecl;
  int nb_dim=0;

  if (fortran_appli)
    ldecl = code_declarations(value_code(entity_initial(module)));
  else {
    statement s = get_current_module_statement();
    ldecl = statement_declarations(s);
  }
  if((int) gen_length(ldecl) >0) {
    string_buffer_append_word("LocalArrays",sb_result);
    global_margin++;
    MAP(ENTITY,var, {
	if (type_variable_p(entity_type(var))
	    && ((nb_dim = variable_entity_dimension(var))>0)
	    &&  !(storage_formal_p(entity_storage(var)))) {
	  xml_Array(var,prec,sb_result);
	}
      },ldecl);

    global_margin--;
    string_buffer_append_word("/LocalArrays",sb_result);
  }
  else
    string_buffer_append_word("LocalArrays/",sb_result);
}

static void xml_FormalArrays(entity module, Psysteme prec,string_buffer sb_result)
{

  int FormalParameterNumber = (int) gen_length(module_formal_parameters(module));
  entity FormalArrayName = entity_undefined;
  int ith ;
  if(FormalParameterNumber >=1) {
    string_buffer_append_word("FormalArrays",sb_result);
    global_margin++;

    for (ith=1;ith<=FormalParameterNumber;ith++) {
      FormalArrayName = find_ith_formal_parameter(module,ith);
      if (type_variable_p(entity_type(FormalArrayName))
	  && ( variable_entity_dimension(FormalArrayName)>0)
	  &&  (storage_formal_p(entity_storage(FormalArrayName)))) {
	xml_Array(FormalArrayName,prec,sb_result);
      }
    }
    global_margin--;
    string_buffer_append_word("/FormalArrays",sb_result);
  }
  else string_buffer_append_word("FormalArrays/",sb_result);
}


static void insert_include_files(string_buffer sb_result)
{
  FILE *fichier;
  char  fname[16] = "tmp_include.txt";
  char name[100];
  fichier = fopen(fname, "r");
  if (fichier) {
    string_buffer_append_word("IncludedFiles",sb_result);
    global_margin++;
    while(fscanf(fichier, "%s ", name)!=EOF) {
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "File"," Name=",
				       QUOTE,strdup(name),QUOTE,"/",CLOSEANGLE,NL,NULL));
    }
    global_margin--;
    string_buffer_append_word("/IncludedFiles",sb_result);
    fclose(fichier);
  }
  else string_buffer_append_word("IncludedFiles/",sb_result);
}

static void motif_in_statement(statement s)
{
  string comm = statement_comments(s);
  if (!string_undefined_p(comm) && strstr(comm,"MOTIF")!=NULL)
    motif_in_statement_p= true;
}
static void box_in_statement(statement s)
{
  string comm = statement_comments(s);
  if (!string_undefined_p(comm) && strstr(comm,"BOX")!=NULL)
    box_in_statement_p= true;
}


static void xml_Loop(statement s, string_buffer sb_result)
{
  loop l = instruction_loop(statement_instruction(s));
  transformer t = load_statement_precondition(s);
  Psysteme prec = sc_dup((Psysteme) predicate_system(transformer_relation(t)));
  expression el =range_lower(loop_range(l));
  expression eu =range_upper(loop_range(l));
  expression stride = range_increment(loop_range(l));
  entity index =loop_index(l);

  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Loop Index=",QUOTE,
				   entity_user_name(index),QUOTE, BL,
				   "ExecutionMode=",QUOTE,
				   (execution_parallel_p(loop_execution(l)))? "PARALLEL":"SEQUENTIAL",
				   QUOTE,
				   CLOSEANGLE,NL,NULL));
  xml_Bounds_and_Stride(el,eu,stride,prec,sb_result);
  string_buffer_append_word("/Loop",sb_result);
  sc_free(prec);
}

static void xml_Loops(stack st,bool call_external_loop_p, list *pattern_region, Pvecteur *paving_indices, Pvecteur *pattern_indices, bool motif_in_te_p, string_buffer sb_result)
{
  bool in_motif_p=!call_external_loop_p && !motif_in_te_p;
  bool motif_on_loop_p=false;
  // Boucles externes a la TE
  if (call_external_loop_p)
    string_buffer_append_word("ExternalLoops",sb_result);
  else
    // Boucles externes au motif dans la TE
    string_buffer_append_word("Loops",sb_result);

  // if comment MOTIF is on the loop, the pattern_region is the loop region
  // if comment MOTIF is on a statement inside the sequence of the loop body
  // the cumulated regions of the loop body are taken
  // if comment MOTIF is outside the loop nest, the pattern_region is the call region
  global_margin++;
  if (st != STACK_NULL) {
    STACK_MAP_X(s, statement,
		{
		  loop l = instruction_loop(statement_instruction(s));
		  string comm = statement_comments(s);
		  entity index =loop_index(l);

		  if (!in_motif_p) {
		    // Test : Motif is in the loop body  or not
		    motif_in_statement_p=false;
		    gen_recurse(loop_body(l), statement_domain, gen_true,motif_in_statement);
		    motif_on_loop_p = !string_undefined_p(comm) && strstr(comm,"MOTIF")!=NULL;

		    if (motif_on_loop_p) {      // comment MOTIF  is on the Loop
		      *pattern_region = regions_dup(load_statement_local_regions(s));
		      vect_add_elem (pattern_indices,(Variable) index ,VALUE_ONE);
		    }
		    else if (motif_in_statement_p) {
		      // cumulated regions on the sequence of the loop body instructions are needed
		      *pattern_region = regions_dup(load_statement_local_regions(loop_body(l)));
		    }
		    in_motif_p =   (!call_external_loop_p && !motif_in_te_p) //Par default on englobe si TE
		      || in_motif_p        // on etait deja dans le motif
		      || motif_on_loop_p  // on vient de trouver un Motif sur la boucle
		      || (motif_in_te_p && !motif_on_loop_p && !motif_in_statement_p);
		    // motif externe au nid de boucles (cas des motif au milieu d'une sequence)
		  }
		  if (!in_motif_p) {
		    vect_add_elem (paving_indices,(Variable) index ,VALUE_ONE);
		    xml_Loop(s, sb_result);
		  }
		},
		st, 0);
  }
  global_margin--;
  if (call_external_loop_p)
    string_buffer_append_word("/ExternalLoops",sb_result);
  else
    string_buffer_append_word("/Loops",sb_result);
}

static Psysteme first_precondition_of_module(const char* module_name __attribute__ ((unused)))
{
  statement st1 = get_current_module_statement();
  statement fst = statement_undefined;
  instruction inst = statement_instruction(st1);
  transformer t;
  Psysteme prec= SC_UNDEFINED;
  switch instruction_tag(inst)
    {
    case is_instruction_sequence:{
      fst = STATEMENT(CAR(sequence_statements(instruction_sequence(inst))));
      break;
    }

    default:
      fst = st1;
    }
  t = (transformer)load_statement_precondition(fst);
  prec = sc_dup((Psysteme) predicate_system(transformer_relation(t)));
  return prec;
}

void matrix_init(Pmatrix mat, int n, int m)
{
  int i,j;
  for (i=1;i<=n;i++) {
    for (j=1;j<=m;j++) {
      MATRIX_ELEM(mat,i,j)=0;
    }
  }
}

static void xml_Matrix(Pmatrix mat, int n, int m, string_buffer sb_result)
{
  string srow, scolumn;
  int i,j;
  // cas des nids de boucles vides
  if (n==0 && m!=0) m=0;
  if (m==0 && n!=0) n=0;
  srow =strdup(itoa(n));
  scolumn=strdup(itoa(m));

  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Matrix NbRows=",
				   QUOTE,srow,QUOTE,BL,
				   "NbColumns=", QUOTE, scolumn,QUOTE,
				   BL,CLOSEANGLE,NL, NULL));
  for (i=1;i<=n;i++) {
    add_margin(global_margin,sb_result);
    string_buffer_append(sb_result,
			 concatenate(OPENANGLE,
				     "Row", CLOSEANGLE,BL, NULL));
    for (j=1;j<=m;j++) {
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,"c", CLOSEANGLE,
				       itoa(MATRIX_ELEM(mat,i,j)),
				       OPENANGLE, "/c", CLOSEANGLE,
				       BL, NULL));
    }
    string_buffer_append(sb_result,
			 concatenate(OPENANGLE,
				     "/Row", CLOSEANGLE, NL, NULL));
  }
  string_buffer_append_word("/Matrix",sb_result);
}

/*static void xml_Transposed_Matrix(reference rout, reference rin, list  ArrayInd1,list  ArrayInd2, int ArrayDim1, int ArrayDim2, string_buffer sb_result)
{
  Pmatrix mat;
  int i,j;
  Pvecteur pv1,pv2;
  i=1;
  j=1;
  string_buffer_append_word("Transposition",sb_result);
   mat = matrix_new(ArrayDim1,ArrayDim2);
  matrix_init(mat,ArrayDim1,ArrayDim2);
  global_margin++;
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,"TransposParameters ", "OUT=", QUOTE,
				   entity_user_name(reference_variable(rout)),QUOTE,BL,
				   "IN=", QUOTE,entity_user_name(reference_variable(rin)),QUOTE,"/",
				   CLOSEANGLE,NL,NULL));
  MAP(EXPRESSION, e1 , {
      if (expression_normalized(e1) == normalized_undefined)
	expression_normalized(e1)= NormalizeExpression(e1);
      pv1 = (Pvecteur)normalized_linear(expression_normalized(e1));
      j=1;
      MAP(EXPRESSION, e2 , {
	  if (expression_normalized(e2) == normalized_undefined)
	    expression_normalized(e2)= NormalizeExpression(e2);
	  pv2 = (Pvecteur)normalized_linear(expression_normalized(e2));
	  if (vect_equal(pv1,pv2))
	    MATRIX_ELEM(mat,i,j)=1;
	  j++;
	},
	ArrayInd2);
      i++;
    },
    ArrayInd1);
  
  xml_Matrix(mat,ArrayDim1,ArrayDim2,sb_result);
   global_margin--;
  string_buffer_append_word("/Transposition",sb_result);
}
*/

static void tri_abc(int tab_ind[4], int a, int b, int c)
{
  if (a==1) {
    tab_ind[3]=1;
    if (b<c) {tab_ind[1]=3;tab_ind[2]=2;}
    else {
      tab_ind[1]=2;tab_ind[2]=3;}
  }
  else if (b==1) {
    tab_ind[3]=2;
    if (a<c) {tab_ind[1]=3;tab_ind[2]=1;}
    else {tab_ind[1]=1;tab_ind[2]=3;}
  }
  else {
    tab_ind[3]=3;
    if (a<b) {tab_ind[1]=2;tab_ind[2]=1;}
    else {tab_ind[1]=1;tab_ind[2]=2;}
  }
}

static void xml_Transposed_Matrix2D( Pmatrix mat)
{
  MATRIX_ELEM(mat,1,1)=0;
  MATRIX_ELEM(mat,1,2)=1;
  MATRIX_ELEM(mat,2,1)=1;
  MATRIX_ELEM(mat,2,2)=0;
}

static void xml_Transposed_Matrix3D( Pmatrix mat,int a[7], int ArrayDim1, int ArrayDim2)
{

  int i,j,n;
  int tab_ind[3][4];
  tri_abc(tab_ind[1],a[1],a[2],a[3]);
  tri_abc(tab_ind[2],a[4],a[5],a[6]);
  for (i=1; i<= ArrayDim1; i++) {
    n=tab_ind[2][i];
    for (j=1;j<=3;j++) {
      if (tab_ind[1][j]==n)
	MATRIX_ELEM(mat,i,j)=1;
    }
  }
 }
static expression skip_field_and_cast_expression(expression arg)
{
 if (expression_field_p(arg))
      arg= EXPRESSION(CAR(call_arguments(expression_call(arg))));
    if (syntax_cast_p(expression_syntax(arg)))
      arg = cast_expression(syntax_cast(expression_syntax(arg))); 
    return arg;
}

// Only to deal with Opengpu cornerturns
static void  xml_Transposition(call c,int d,string_buffer sb_result)
{
  int tab[7];
  int i;
  expression arg1,arg2;
  value v;
  list args = call_arguments(c);
  Pmatrix mat;
  if (gen_length(args)>=3+3*d) {
    for (i=1; i<=d; i++) {
      arg1= EXPRESSION(CAR(args));
      POP(args);
    }
    for (i=1;i<=2*d;i++){
      arg2= EXPRESSION(CAR(args));
      v = EvalExpression(arg2);
      if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	tab[i] = constant_int(value_constant(v));
	POP(args);
      }
    }
    POP(args);
    arg1= EXPRESSION(CAR(args));
    POP(args);
    arg2= EXPRESSION(CAR(args));
    // case with array field reference
    arg1 = skip_field_and_cast_expression(arg1);
    arg2 = skip_field_and_cast_expression(arg2);
    if (array_argument_p(arg1) && array_argument_p(arg2)) {
      reference r1 = syntax_reference(expression_syntax(arg1));
      reference r2 = syntax_reference(expression_syntax(arg2));
      int ArrayDim1 = variable_entity_dimension(reference_variable(r1));
      int ArrayDim2 = variable_entity_dimension(reference_variable(r2));
      if (d==3) {
	mat = matrix_new(3,3);
	matrix_init(mat,mat->number_of_lines,mat->number_of_columns);
	xml_Transposed_Matrix3D(mat,tab, mat->number_of_lines,mat->number_of_columns) ;
      }
      if (d==2) {
	mat = matrix_new(2,2);
	matrix_init(mat,mat->number_of_lines,mat->number_of_columns);
	xml_Transposed_Matrix2D(mat);
      }
  
      string_buffer_append_word("Transposition",sb_result);
      global_margin++;
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,"TransposParameters ", "OUT=", QUOTE,
				       entity_user_name(reference_variable(r1)),QUOTE,BL,
				       "IN=", QUOTE,entity_user_name(reference_variable(r2)),QUOTE,"/",
				       CLOSEANGLE,NL,NULL));
      xml_Matrix(mat,mat->number_of_lines,mat->number_of_columns,sb_result);
      matrix_free(mat);
      global_margin--;
      string_buffer_append_word("/Transposition",sb_result);
    }

  }
}

static void  xml_Task(const char* module_name, int code_tag,string_buffer sb_result)
{
  nest_context_t task_loopnest;
  task_loopnest.loops_for_call = stack_make(statement_domain,0,0);
  task_loopnest.loop_indices = stack_make(entity_domain,0,0);
  task_loopnest.current_stat = stack_make(statement_domain,0,0);
  task_loopnest.nested_loops=  gen_array_make(0);
  task_loopnest.nested_loop_indices =  gen_array_make(0);
  task_loopnest.nested_call=  gen_array_make(0);
  stack nested_loops;
  list pattern_region =NIL;
  Pvecteur paving_indices = VECTEUR_NUL;
  Pvecteur pattern_indices = VECTEUR_NUL;
  bool motif_in_te_p = false;
  entity module = module_name_to_entity(module_name);
  Psysteme prec;
  string string_sb_result;
  statement stat_module=(statement) db_get_memory_resource(DBR_CODE,
							   module_name, true);
  reset_rw_effects();
  set_rw_effects
    ((statement_effects)
     db_get_memory_resource(DBR_REGIONS, module_name, true));

  push_current_module_statement(stat_module);
  prec = first_precondition_of_module(module_name);
  // printf("first_precondition_of_module %s\n",module_name);
  // sc_fprint(stdout,prec, (char * (*)(Variable)) entity_local_name);
  global_margin++;
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Task Name=",QUOTE,
				   module_name,QUOTE,CLOSEANGLE,NL, NULL));
  global_margin++;

  find_loops_and_calls_in_box(stat_module,&task_loopnest);
  pattern_region = regions_dup(load_statement_local_regions(stat_module));

  xml_Library(sb_result);
  xml_Returns(sb_result);
  xml_Timecosts(sb_result);
  xml_GlobalArrays(prec,sb_result);
  xml_LocalArrays(module, prec,sb_result);
  xml_FormalArrays(module,prec,sb_result);
  /*  On ne traite qu'une TE : un seul nid de boucles */
  nested_loops = gen_array_item(task_loopnest.nested_loops,0);
  // xml_Transposition(&task_loopnest,sb_result);
  xml_Region_Parameter(pattern_region, sb_result);
  gen_recurse(stat_module, statement_domain, gen_true,motif_in_statement);
  motif_in_te_p = motif_in_statement_p;
  xml_Loops(nested_loops,false,&pattern_region,&paving_indices, &pattern_indices, motif_in_te_p, sb_result);
  xml_TaskParameters(false,code_tag, module,pattern_region,paving_indices,sb_result);
  xml_Regions(sb_result);
  xml_CodeSize(sb_result);
  global_margin--;
  string_buffer_append_word("/Task",sb_result);
  global_margin--;

  string_sb_result=string_buffer_to_string(sb_result);
  insert_xml_string(module_name,string_sb_result);
  pop_current_module_statement();
  gen_array_free(task_loopnest.nested_loops);
  gen_array_free(task_loopnest.nested_loop_indices);
  gen_array_free(task_loopnest.nested_call);
  stack_free(&(task_loopnest.loops_for_call));
  stack_free(&(task_loopnest.loop_indices));
  stack_free(&(task_loopnest.current_stat));
  regions_free(pattern_region);
  sc_rm(prec);
}

// A completer
// ne traite que les cas ou tout est correctement aligne
// A traiter aussi le cas  ActualArrayDim = NIL
//
static void xml_Connection(list  ActualArrayInd,int ActualArrayDim, int FormalArrayDim, string_buffer sb_result)
{
  Pmatrix mat;
  int i,j;
  Pvecteur pv;
  string_buffer_append_word("Connection",sb_result);
  mat = matrix_new(ActualArrayDim,FormalArrayDim);
  matrix_init(mat,ActualArrayDim,FormalArrayDim);
  if (fortran_appli) {
    for (i=1;i<=ActualArrayDim && i<= FormalArrayDim;i++)
      MATRIX_ELEM(mat,i,i)=1;
  }
  else
    for (i=ActualArrayDim, j=FormalArrayDim;i>=1 && j>=1;i=i-1,j=j-1)
      MATRIX_ELEM(mat,i,j)=1;
  // Normalise les expressions lors du premier parcours
  // Utile aussi a  xml_LoopOffset
  MAP(EXPRESSION, e , {
      if (expression_normalized(e) == normalized_undefined)
	expression_normalized(e)= NormalizeExpression(e);
      pv = (Pvecteur)normalized_linear(expression_normalized(e));
      pv = pv;
    },
    ActualArrayInd);

  xml_Matrix(mat,ActualArrayDim,FormalArrayDim,sb_result);
  string_buffer_append_word("/Connection",sb_result);
}

static void xml_LoopOffset(list  ActualArrayInd,int ActualArrayDim, Pvecteur loop_indices,string_buffer sb_result)
{
  Pmatrix mat;
  Pvecteur loop_indices2 = vect_reversal(loop_indices);
  int nestloop_dim = vect_size(loop_indices2);
  Pvecteur pv, pv2;
  int i,j;

  string_buffer_append_word("LoopOffset",sb_result);
  mat = matrix_new(ActualArrayDim,nestloop_dim);
  matrix_init(mat,ActualArrayDim,nestloop_dim);
  i=1;
  MAP(EXPRESSION, e , {
      pv = (Pvecteur)normalized_linear(expression_normalized(e));
      for (pv2 = loop_indices2, j=1; !VECTEUR_NUL_P(pv2);pv2=pv2->succ,j++)
	MATRIX_ELEM(mat,i,j)=vect_coeff(pv2->var,pv);
      i++;
    },
    ActualArrayInd);

  xml_Matrix(mat,ActualArrayDim,nestloop_dim,sb_result);
  string_buffer_append_word("/LoopOffset",sb_result);
}

// A completer
// Ici, la constante vaut 0
static void xml_ConstOffset(int ActualArrayDim, string_buffer sb_result)
{
  Pmatrix mat;
  string_buffer_append_word("ConstOffset",sb_result);
  mat = matrix_new(ActualArrayDim,1);
  matrix_init(mat,ActualArrayDim,1);
  xml_Matrix(mat,ActualArrayDim,1,sb_result);
  string_buffer_append_word("/ConstOffset",sb_result);
}


static void list_of_arguments(call c, Pvecteur *vargs){
  list args = call_arguments(c);
  FOREACH(EXPRESSION,exp,args){
    if (expression_field_p(exp))
      exp= EXPRESSION(CAR(call_arguments(expression_call(exp))));
    if (expression_reference_p(exp)) {
      reference r1 = syntax_reference(expression_syntax(exp));
      vect_add_elem(vargs,reference_variable(r1), VALUE_ONE);
    }
  }

}

static void  xml_Arguments(statement s, entity function, Pvecteur loop_indices, Psysteme prec, string_buffer sb_result )
{
  call c = instruction_call(statement_instruction(s));
  list call_effect =load_proper_rw_effects_list(s);
  entity FormalArrayName, ActualArrayName = entity_undefined;
  reference ActualRef=reference_undefined;
  syntax sr;
  effect efr = effect_undefined, efw = effect_undefined;;
  intptr_t iexp,ith=0;
  int rw_ef=0;
  string aan ="";
  string quote_p="";
  int valr  ;
  string_buffer_append_word("Arguments",sb_result);
  global_margin++;
  FOREACH(EXPRESSION,exp,call_arguments(c)){
    ith ++;
    FormalArrayName = find_ith_formal_parameter(call_function(c),ith);
    exp = skip_field_and_cast_expression(exp);
    sr = expression_syntax(exp);
    if(syntax_call_p(sr)) {
      call cl = syntax_call(sr);
      const char* fun = entity_local_name(call_function(cl));
      if (strcmp(fun,ADDRESS_OF_OPERATOR_NAME) == 0) {
	exp = EXPRESSION(CAR(call_arguments(cl)));
	sr = expression_syntax(exp);
      }
    }
    if (syntax_reference_p(sr)) {
      ActualRef = syntax_reference(sr);
      ActualArrayName = reference_variable(ActualRef);
      aan = strdup(entity_user_name(ActualArrayName));
      rw_ef = find_effect_actions_for_entity(call_effect,&efr,&efw, ActualArrayName);
    }
    else {
      // Actual Parameter could be  an expression
      aan = words_to_string(words_syntax(sr,NIL));
      rw_ef = 1;
    }
    string_buffer_append_word("Argument",sb_result);
    if (!array_argument_p(exp)) { /* Scalar Argument */
      global_margin++;
      add_margin(global_margin,sb_result);
      if (strncmp(aan,"\"",1)==0)
	quote_p="";
      else quote_p=QUOTE;
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "ScalarArgument ActualName=",
				       quote_p,
				       aan,
				       quote_p,BL,
				       "FormalName=", QUOTE,entity_user_name(FormalArrayName), QUOTE,BL,
				       "AccessMode=",QUOTE,(rw_ef>=2)? "DEF": "USE", QUOTE,CLOSEANGLE,
				       NL, NULL));

      if (expression_integer_value(exp, &iexp))
	string_buffer_append_numeric(i2a(iexp),sb_result);
      else if (value_constant_p(EvalExpression(exp))) {
	string exps = words_to_string(words_expression(exp,NIL));
	string_buffer_append_numeric(exps,sb_result);
      }
      else if (eval_linear_expression(exp,prec,&valr))
	string_buffer_append_numeric(i2a(valr),sb_result);
      string_buffer_append_word("/ScalarArgument",sb_result);
      global_margin--;
    }
    else { /* Array Argument */
      int ActualArrayDim = variable_entity_dimension(ActualArrayName);
      char *  SActualArrayDim = strdup(itoa(ActualArrayDim));
      int FormalArrayDim = variable_entity_dimension(FormalArrayName);
      list ActualArrayInd = reference_indices(ActualRef);
      global_margin++;
      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "ArrayArgument ActualName=",
				       QUOTE,
				       entity_user_name(ActualArrayName),QUOTE,BL,
				       "ActualDim=", QUOTE,SActualArrayDim,QUOTE,BL,
				       "FormalName=", QUOTE,entity_user_name(FormalArrayName), QUOTE,BL,
				       "FormalDim=", QUOTE,itoa(FormalArrayDim),QUOTE,BL,
				       "AccessMode=",QUOTE,(rw_ef>=2)? "DEF":"USE",QUOTE,CLOSEANGLE,
				       NL, NULL));
      /* Save information to generate Task Graph */
      if (rw_ef>=2)
	hash_put(hash_entity_def_to_task, (char *)ActualArrayName,(char *)function);
      free(SActualArrayDim);

      xml_Connection(ActualArrayInd,ActualArrayDim,FormalArrayDim,sb_result);
      xml_LoopOffset(ActualArrayInd,ActualArrayDim, loop_indices,sb_result);
      xml_ConstOffset(ActualArrayDim,sb_result);
      global_margin--;
      string_buffer_append_word("/ArrayArgument",sb_result);
    }
    string_buffer_append_word("/Argument",sb_result);
  }
  global_margin--;
  string_buffer_append_word("/Arguments",sb_result);
}


// A completer
static void   xml_Dependances()
{
  // Not implemented Yet
}

bool array_in_effect_list_p(list effects_list)
{
  list pc;
  for (pc= effects_list;pc != NIL; pc = CDR(pc)){
    effect e = EFFECT(CAR(pc));
    reference r = effect_any_reference(e);
    entity v =  reference_variable(r);
    if (array_entity_p(v))
      return(true);
  }
  return(false);
}

static void xml_Call(entity module,  int code_tag,int taskNumber, nest_context_p nest, string_buffer sb_result)
{
  statement s = gen_array_item(nest->nested_call,taskNumber);
  stack st = gen_array_item(nest->nested_loops,taskNumber);
  call c = instruction_call(statement_instruction(s));
  entity func= call_function(c);
  list pattern_region=NIL;
  Pvecteur paving_indices = VECTEUR_NUL;
  Pvecteur pattern_indices = VECTEUR_NUL;
  bool motif_in_te_p=false;
  transformer t = load_statement_precondition(s);
  Psysteme prec = sc_dup((Psysteme) predicate_system(transformer_relation(t)));
  const char * strtmp = entity_user_name(func);

  pattern_region = regions_dup(load_statement_local_regions(s));
  if (!ENTITY_ASSIGN_P(func) || array_in_effect_list_p(pattern_region)) {
    add_margin(global_margin,sb_result);
    string_buffer_append(sb_result,
			 concatenate(OPENANGLE,
				     "Call Name=",
				     QUOTE,
				     ENTITY_ASSIGN_P(func) ?
				     "LocalAssignment" : entity_user_name(func),
				     QUOTE,
				     CLOSEANGLE,NL, NULL));
    global_margin++;
    //  only detect  Opengpu 2D and 3D cornerturns
      if (strstr(strtmp,"cornerturn_3D")!=NULL)
	xml_Transposition(c,3,sb_result);
      else
	if (strstr(strtmp,"cornerturn_2D")!=NULL)
	  xml_Transposition(c,2,sb_result);
    xml_Region_Parameter(pattern_region, sb_result);
    xml_Loops(st,true,&pattern_region,&paving_indices, &pattern_indices,motif_in_te_p, sb_result);

    if (ENTITY_ASSIGN_P(func))
      xml_TaskParameters(true,code_tag, module,pattern_region,paving_indices,sb_result);
    else
      xml_Arguments(s,func,paving_indices, prec,sb_result);
    xml_Dependances();
    global_margin--;
    string_buffer_append_word("/Call",sb_result);
  }
  regions_free(pattern_region);
  sc_free(prec);
}


void  xml_Compute_and_Need(entity func,list effects_list, Pvecteur vargs,string_buffer sb_result)
{
  string_buffer buffer_needs = string_buffer_make(true);
  string string_needs = "";
  Pvecteur vl = VECTEUR_NUL;
  list pc;
  Pvecteur va2 = vect_dup(vargs);

  vars_read_or_written(effects_list,&vl);

  for (pc= effects_list;pc != NIL; pc = CDR(pc)){
    effect e = EFFECT(CAR(pc));
    reference r = effect_any_reference(e);
    action ac = effect_action(e);
    entity v =  reference_variable(r);
    if ( array_entity_p(v) &&!io_entity_p(v) && !rand_effects_entity_p(v) 
	 // pour eviter les variables privees et  traiter les tableaux en R+W
	 &&  (vect_coeff(v,vl) ||  (vect_coeff(v,va2) && !action_read_p(ac))) 
	 && !(entity_static_variable_p(v) && !top_level_entity_p(v))) {
      if (action_read_p(ac)) {
	vect_chg_coeff(&vl,v,0);
	entity t =  hash_get(hash_entity_def_to_task,(char *) v);
	global_margin++;
	add_margin(global_margin,buffer_needs);
	string_buffer_append(buffer_needs,
			     concatenate(OPENANGLE,"Needs ArrayName=",
					 QUOTE,entity_user_name(v),QUOTE, BL,
					 "DefinedBy=",QUOTE,
					 (t!= HASH_UNDEFINED_VALUE) ? entity_local_name(t): "IN_VALUE",QUOTE,"/",
						   CLOSEANGLE,NL,NULL));
	global_margin--;
      }
      else {
	vect_chg_coeff(&vl,v,0);
	if (vect_coeff(v,va2)) vect_chg_coeff(&va2,v,0);
	global_margin++;
	add_margin(global_margin,sb_result);
	string_buffer_append(sb_result,
			     concatenate(OPENANGLE,"Computes ArrayName=",
					 QUOTE,entity_user_name(v),QUOTE,"/",
					 CLOSEANGLE,NL,
					 NULL));
	hash_put(hash_entity_def_to_task, (char *) v,(char *)func);
	global_margin--;
      }
    }
  }
  vect_rm(va2);
  string_needs =string_buffer_to_string(buffer_needs);
  string_buffer_append(sb_result,string_needs);
  free(string_needs);
  string_needs=NULL;
}
static void xml_BoxGraph(entity module, nest_context_p nest, string_buffer sb_result)
{
  int nb_call,callnumber;
  Pvecteur vargs = VECTEUR_NUL;
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "BoxGraph Name=",
				   QUOTE,entity_user_name(module),
				   QUOTE,
				   CLOSEANGLE,NL, NULL));

  nb_call = (int)gen_array_nitems(nest->nested_call);
  global_margin++;
  for (callnumber = 0; callnumber<nb_call; callnumber++) {
    statement s = gen_array_item(nest->nested_call,callnumber);
    call c = instruction_call(statement_instruction(s));
    entity func= call_function(c);
    bool assign_func = ENTITY_ASSIGN_P(func);
    const char* n= assign_func ? "LocalAssignment" : entity_user_name(func);
    list effects_list = regions_dup(load_statement_local_regions(s));
    if (!assign_func || array_in_effect_list_p(effects_list)) {

      add_margin(global_margin,sb_result);
      string_buffer_append(sb_result,
			   concatenate(OPENANGLE,
				       "TaskRef Name=",
				       QUOTE,n,QUOTE, CLOSEANGLE,NL,
				       NULL));
      list_of_arguments(c, &vargs);
      xml_Compute_and_Need(func,effects_list,vargs,sb_result);
      string_buffer_append_word("/TaskRef",sb_result);
    }
    regions_free(effects_list);
  }
  global_margin--;
  string_buffer_append_word("/BoxGraph",sb_result);
}
  bool entity_main_user_module_p(entity e)
{
  return (entity_module_p(e) && same_string_p(entity_local_name(e), USER_MAIN));
}

int find_code_status(const char* module_name)
{
  statement stat=(statement) db_get_memory_resource(DBR_CODE,
						    module_name, true);
  bool wmotif = false;
  bool wbox = false;

  motif_in_statement_p=false;
  gen_recurse(stat, statement_domain, gen_true,motif_in_statement);
  wmotif = motif_in_statement_p;
  gen_recurse(stat, statement_domain, gen_true,box_in_statement);
  wbox = box_in_statement_p;

  if (entity_main_user_module_p(module_name_to_entity(module_name)))
    return(code_is_a_main);
  else {
    if (wmotif && !wbox)
      return (code_is_a_te);
    else return(code_is_a_box);
  }
}

void insert_xml_callees(const char* module_name) {
  FILE * out, *ftest;
  string dir = db_get_current_workspace_directory();
  string sm = db_build_file_resource_name(DBR_XML_PRINTED_FILE,
					  module_name, XMLPRETTY);
  string xml_module_name = strdup(concatenate(dir, "/", sm, NULL));
  callees callers = (callees)db_get_memory_resource(DBR_CALLEES,module_name, true);
  out = safe_fopen(xml_module_name, "a");
  string xml_callee_name,xml_callee_name1,xml_callee_name2 ;
  char * stmp1 = (char *) malloc(strlen(dir)+7);
  char * stmp2 = strdup(concatenate("-Task.database/",NULL));
  int code_tag;

  strncpy(stmp1,dir,strlen(dir)-9);
  stmp1[strlen(dir)-9]='\0';
  strcat(stmp1,stmp2);
  MAP(STRING, callee_name, {
      string sc=(string) db_get_memory_resource(DBR_XML_PRINTED_FILE,
						callee_name, true);
      code_tag = find_code_status(callee_name);  
      xml_callee_name1 = strdup(concatenate(stmp1,sc, NULL));
      xml_callee_name2 = strdup(concatenate(dir, "/", sc, NULL));

      if ((code_tag == code_is_a_te) && strstr(dir,"Task") ==NULL 
	  && ((ftest = fopen(xml_callee_name1, "r")) != (FILE *) NULL)) {
	xml_callee_name=xml_callee_name1;
      }
      else xml_callee_name = xml_callee_name2;
      if((ftest = fopen(xml_callee_name, "r")) == (FILE *) NULL)
	fprintf(stdout,"fopen failed on file %s\n", xml_callee_name);
      else
	safe_append(out, xml_callee_name,0, true);
      free(xml_callee_name1);
      free(xml_callee_name2);
    },
    callees_callees(callers));
  safe_fclose(out, xml_module_name);
  free(stmp1);
  free(xml_module_name);
}

static void xml_Boxes(const char* module_name, int code_tag,string_buffer sb_result)
{
  entity module = module_name_to_entity(module_name);
  nest_context_t nest;
  int callnumber =0;
  statement stat = get_current_module_statement();
  string string_sb_result;
  Psysteme prec = first_precondition_of_module(module_name);
  nest.loops_for_call = stack_make(statement_domain,0,0);
  nest.loop_indices = stack_make(entity_domain,0,0);
  nest.current_stat = stack_make(statement_domain,0,0);
  nest.nested_loops=  gen_array_make(0);
  nest.nested_loop_indices =  gen_array_make(0);
  nest.nested_call=  gen_array_make(0);

  global_margin++;
  add_margin(global_margin,sb_result);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Box Name=",
				   QUOTE,module_name,QUOTE,
				   CLOSEANGLE,NL, NULL));
  global_margin++;

  if (!entity_main_module_p(module)) {
    xml_GlobalArrays(prec,sb_result);
    xml_LocalArrays(module,prec,sb_result);
    xml_FormalArrays(module,prec,sb_result);
  }
  else {
    string_buffer_append_word("GlobalArrays/",sb_result);
    xml_LocalArrays(module,prec,sb_result);
    //    string_buffer_append_word("LocalArrays/",sb_result);
    string_buffer_append_word("FormalArrays/",sb_result);
  }
  /* Search calls in Box */
  find_loops_and_calls_in_box(stat,&nest);

  for (callnumber = 0; callnumber<(int)gen_array_nitems(nest.nested_call); callnumber++)
    xml_Call(module, code_tag, callnumber, &nest,sb_result);

  xml_BoxGraph(module,&nest,sb_result);
  global_margin--;
  string_buffer_append_word("/Box",sb_result);
  string_sb_result=string_buffer_to_string(sb_result);
  insert_xml_string(module_name,string_sb_result);
  string_buffer_append_word("Tasks",sb_result);
  insert_xml_callees(module_name);
  string_buffer_append_word("/Tasks",sb_result);
  global_margin--;
  gen_array_free(nest.nested_loops);
  gen_array_free(nest.nested_loop_indices);
  gen_array_free(nest.nested_call);
  stack_free(&(nest.loops_for_call));
  stack_free(&(nest.loop_indices));
  stack_free(&(nest.current_stat));
  sc_free(prec);
}


static void xml_Application(const char* module_name, int code_tag,string_buffer sb_result)
{
  entity module = module_name_to_entity(module_name);
  string_buffer sb_ac = string_buffer_make(true);
  statement s = get_current_module_statement();
  string sr;
  global_margin = 0;
  Psysteme prec;
  Pvecteur vargs = VECTEUR_NUL;
  prec = first_precondition_of_module(module_name);
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "!DOCTYPE Application SYSTEM ",
				   QUOTE,
				   "APPLI_OPENGPU_v2.dtd",
				   QUOTE,CLOSEANGLE,NL, NULL));
  string_buffer_append(sb_result,
		       concatenate(OPENANGLE,
				   "Application Name=",
				   QUOTE,
				   get_current_module_name(),
				   QUOTE, BL,
				   "Language=",QUOTE,
				   (fortran_appli) ? "FORTRAN":"C",
				   QUOTE, BL,
				   "PassingMode=",
				   QUOTE,
				   (fortran_appli) ? "BYREFERENCE":"BYVALUE",
				   QUOTE,CLOSEANGLE,NL, NULL));
  global_margin ++;
  insert_include_files(sb_result);
  xml_GlobalArrays( prec,sb_result);
  add_margin(global_margin,sb_ac);
  string_buffer_append(sb_ac,
		       concatenate(OPENANGLE,
				   "ApplicationGraph Name=",
				   QUOTE,
				   module_name,
				   QUOTE,CLOSEANGLE,NL, NULL));
  global_margin ++;
  add_margin(global_margin,sb_ac);
  string_buffer_append(sb_ac,
		       concatenate(OPENANGLE,
				   "TaskRef Name=",
				   QUOTE,entity_user_name(module),
				   QUOTE,CLOSEANGLE,NL,NULL));
  global_margin -=2;
  xml_Boxes(module_name,code_tag,sb_result);
  global_margin +=2;

  cumulated_list = NIL;
  gen_recurse(s, statement_domain, gen_true, cumul_effects_of_statement);
  xml_Compute_and_Need(module,cumulated_list, vargs, sb_ac);
  string_buffer_append_word("/TaskRef",sb_ac);
  global_margin --;
  string_buffer_append_word("/ApplicationGraph",sb_ac);
  global_margin --;
  add_margin(global_margin,sb_ac);
  string_buffer_append(sb_ac,
		       concatenate(OPENANGLE,
				   "/Application",
				   CLOSEANGLE,NL, NULL));
  sr=string_buffer_to_string(sb_ac);

  insert_xml_string(module_name,sr);
  sc_free(prec);
}

/******************************************************** PIPSMAKE INTERFACE */

#define XMLPRETTY    ".xml"

bool print_xml_application(const char* module_name)
{
  FILE * out;
  entity module;
  string xml, dir, filename;
  statement stat;
  string_buffer sb_result=string_buffer_make(true);
  hash_entity_def_to_task = hash_table_make(hash_pointer,0);
  int code_tag;

  module = module_name_to_entity(module_name);
  xml = db_build_file_resource_name(DBR_XML_PRINTED_FILE,
				    module_name, XMLPRETTY);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", xml, NULL));
  stat=(statement) db_get_memory_resource(DBR_CODE,
					  module_name, true);

  set_current_module_entity(module);
  set_current_module_statement(stat);
  if(statement_undefined_p(stat)) {
      pips_internal_error("No statement for module %s", module_name);
    }
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS,
					       module_name,true));
  init_cost_table();
  // set_complexity_map( (statement_mapping)
  //		      db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

  /* Get the READ and WRITE regions of the module */
  set_rw_effects((statement_effects)
		 db_get_memory_resource(DBR_REGIONS, module_name, true));

  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,
					      module_name,
					      true));
  debug_on("XMLPRETTYPRINTER_DEBUG_LEVEL");
  out = safe_fopen(filename, "w");
  fprintf(out,"<!-- XML prettyprint for module %s. --> \n",module_name);
  safe_fclose(out, filename);

  fortran_appli = fortran_module_p(module);
  code_tag = find_code_status(module_name);
  switch (code_tag)
    {
    case code_is_a_box: {
      xml_Boxes(module_name,code_tag,sb_result);
      break;
    }
    case code_is_a_te:{
      xml_Task(module_name,code_tag,sb_result);
      break;
    }
    case code_is_a_main:{
      xml_Application(module_name, code_tag,sb_result);
      break;
    }
    default:
      pips_internal_error("unexpected kind of code for xml_prettyprinter");
    }

  pips_debug(1, "End Xml prettyprinter for %s\n", module_name);
  debug_off();
  string_buffer_free(&sb_result);
  hash_table_free(hash_entity_def_to_task);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_XML_PRINTED_FILE, module_name, xml);

  reset_current_module_statement();
  reset_current_module_entity();
  //  reset_complexity_map();
  reset_precondition_map();
  reset_rw_effects();
  reset_proper_rw_effects();
  return true;
}
