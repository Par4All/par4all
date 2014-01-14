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

   Try to prettyprint the RI in CLAIRE.
   Very basic at the time.

   print_claire_code        > MODULE.claire_printed_file
                            < PROGRAM.entities
                            < MODULE.code

*/

#define DEBUG_CLAIRE 1

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

#include "effects-convex.h"
#include "effects-generic.h"
#include "complexity_ri.h"
#include "complexity.h"

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
#define QUOTE         "\""


#define CLAIRE_TASK_PREFIX "T_"
#define CLAIRE_MOTIF_PREFIX "M_"
#define CLAIRE_ARRAY_PREFIX "A_"

#define CLAIRE_RL      NL,TAB,TAB

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

/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
  (entity_function_p(get_current_module_entity()))

static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
}


#define RESULT_NAME	"result"
static string claire_entity_local_name(entity var)
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
  char * rname = strupper(strdup(name),name);
  
  return rname;
}


/************************************************************** DECLARATIONS */

/* 
   integer a(n,m) -> int a[m][n];
   parameter (n=4) -> #define n 4
 */


/* Code duplicated from Newgen build.c */
/* forward declaration */
static string claire_expression(expression);

/* Attention with Fortran: the indices are reversed. */
static string claire_reference_with_explicit_motif(reference r)
{
  string result = strdup(EMPTY), old, svar;
  MAP(EXPRESSION, e, 
  {
    string s = strdup(claire_expression(e));
    
    old = result;
    result = strdup(concatenate(old, OPENBRACKET, s, CLOSEBRACKET, NULL));
    free(old);
    free(s);
  }, reference_indices(r));

  old = result;
  svar = claire_entity_local_name(reference_variable(r));
  result = strdup(concatenate(svar, old, NULL));
  free(old);
  free(svar);
  return result;
}

static string claire_expression(expression e)
{
  string result = string_undefined;
  syntax s = expression_syntax(e);

  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      result = claire_reference_with_explicit_motif(syntax_reference(s));
      break;
    case is_syntax_call: {
      value ev = EvalExpression(e);
      constant ec = value_constant(ev);
      int eiv = 0;

      if(!value_constant_p(ev)) {
	pips_user_error("Constant expected for CLAIRE loop bounds.\n");
      }
      if(!constant_int_p(ec)) {
	pips_user_error("Integer constant expected for CLAIRE loop bounds.\n");
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

  for(i = 0; i<gen_array_nitems(ar); i++){
    if(gen_array_item(ar, i) != NULL){
      if(same_string_p(item, *((string *)(gen_array_item(ar, i))))){
	return i;
      }
    }
  }
  return ITEM_NOT_IN_ARRAY;
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

static string this_entity_clairedeclaration(entity var)
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
      sd = strdup(claire_dim_string(variable_dimensions(v), name));
    
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
claire_declarations_with_explicit_motif(entity module,
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
    debug(2, "\n Prettyprinter declaration for variable :",claire_entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = strdup(result);
	string svar = strdup(this_entity_clairedeclaration(var));
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = false;
      }
  },code_declarations(c));
  return result;
}

static string claire_array_in_task(reference r, bool first, int task_number);

static string claire_call_from_assignation(call c, int task_number, bool * input_provided){
  /* All arguments of this call are in Rmode (inputs of the task) */
  /* This function is called recursively */
  list arguments = call_arguments(c);
  syntax syn;
  string result = "";
  
  MAP(EXPRESSION, expr, {
    syn = expression_syntax(expr);
    switch(syntax_tag(syn)){
    case is_syntax_call:{
      result = strdup(concatenate(result, claire_call_from_assignation(syntax_call(syn), task_number, input_provided), NULL));
      break;
    }
    case is_syntax_reference:{
      reference ref = syntax_reference(syn);
      string varname = strdup(concatenate("A_", claire_entity_local_name(reference_variable(ref)), NULL));
      if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	result = strdup(concatenate(result, claire_array_in_task(ref, false, task_number), NULL));
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

static void claire_call_from_indice(call c, string * offset_array, string paving_array[], string fitting_array[]){
  entity called = call_function(c);
  string funname = claire_entity_local_name(called);
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
	  claire_call_from_indice(syntax_call(args[0]), offset_array, paving_array, fitting_array);
	}
	if(syntax_tag(args[1]) == is_syntax_call){
	  claire_call_from_indice(syntax_call(args[1]), offset_array, paving_array, fitting_array);
	}
	if(syntax_tag(args[0]) == is_syntax_reference){
	  reference ref = syntax_reference(args[0]);
	  if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    paving_array[iterator_nr] = strdup("1");
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    fitting_array[iterator_nr] = strdup("1");
	  }
	}
	if(syntax_tag(args[1]) == is_syntax_reference){
	  reference ref = syntax_reference(args[1]);
	  if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	   paving_array[iterator_nr] = strdup("1");
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	   fitting_array[iterator_nr] = strdup("1");
	  }
	}
      }
      else if(same_string_p(funname, "-")){
	if(syntax_tag(args[1]) == is_syntax_call && gen_length(call_arguments(syntax_call(args[1])))==0){
	  if(syntax_tag(args[0]) == is_syntax_reference){
	    reference ref = syntax_reference(args[0]);
	    if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      paving_array[iterator_nr] = strdup("1");
	    }
	    else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      fitting_array[iterator_nr] = strdup("1");
	    }
	  }
	  if(syntax_tag(args[0]) == is_syntax_call){
	    claire_call_from_indice(syntax_call(args[0]), offset_array, paving_array, fitting_array);
	  }
	  claire_call_from_indice(syntax_call(args[1]), offset_array, paving_array, fitting_array);
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
	  int intern_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(syntax_reference(args[1]))));
	  int extern_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(syntax_reference(args[1]))));
	  string mult =  strdup(claire_entity_local_name(call_function(syntax_call(args[0])))); 
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

#define CLAIRE_ARRAY_PREFIX "A_"

static string claire_array_in_task(reference r, bool first, int task_number){
  /* CLAIRE name of the referenced array */
  string varname = strdup(concatenate(CLAIRE_ARRAY_PREFIX, 
				      claire_entity_local_name(reference_variable(r)), 
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

  /* list of offsets for CLAIRE code */
  string offset_array[index_of_array];
  /* paving matrix for CLAIRE code
   1st coeff: array dimension (row index)
   2nd coeff: iteration dimension (column index) */
  string paving_array[index_of_array][extern_nb];
  
  /* fitting matrix for CLAIRE code 
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

  /* CLAIRE reference header */
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
      if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	paving_array[indice_nr][iterator_nr] = strdup("1");
      }
      else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	fitting_array[indice_nr][iterator_nr] = strdup("1");
      }

      break;
    }
    case is_syntax_call:{
      call c = syntax_call(sind);
      claire_call_from_indice(c, &(offset_array[indice_nr]), paving_array[indice_nr], fitting_array[indice_nr]);
      break;
    }
    default:{
      pips_user_error("Only call and reference allowed in indices.\n");
      break;
    }
    }
    indice_nr++;
  }, indices);


  /* generate offset list in CLAIRE code */  
  for(i=0; i<index_of_array - 1; i++){
    result=strdup(concatenate(result, "vartype!(", offset_array[i],"), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", offset_array[i], "))," NL, NULL));

  /* fitting header */
  result = strdup(concatenate(result, TAB, TAB, "fitting = list<list[VARTYPE]>(", NULL));

  /* CLAIRE column-major storage of fitting matrix */
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
  /* Generation of paving CLAIRE code*/
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

  /* Complete CLAIRE reference */
  result = strdup(concatenate(result, "))", (first?")":","), NL, NULL)); 
  return result;
  
}

static string claire_call_from_loopnest(call c, int task_number){
  entity called = call_function(c);
  list arguments = call_arguments(c);
  
  syntax s;
  string result = "";
  string first_result = "";
  bool first = true;
  bool input_provided = false, output_provided = false;
  string name = strdup(claire_entity_local_name(called));

  if(!same_string_p(name, "="))
    pips_user_error("Only assignation allowed here.\n");
  
  FOREACH(EXPRESSION, e, arguments){
    s = expression_syntax(e);
    switch(syntax_tag(s)){
    case is_syntax_call:{
      if(first)
	pips_user_error("Call not allowed in left-hand side argument of assignation.\n");
      else
	result = strdup(concatenate(result, claire_call_from_assignation(syntax_call(s), task_number, &input_provided), NULL));
      break;
    }
    case is_syntax_reference:{
      
      reference r = syntax_reference(s);
      string varname = strdup(concatenate("A_", claire_entity_local_name(reference_variable(r)), NULL));
      if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	if(first){
	  first_result = claire_array_in_task(r, first, task_number);
	  output_provided = true;
	}
	else{
	  result = strdup(concatenate(result, claire_array_in_task(r, first, task_number), NULL));
	  input_provided = true;
	}
      }
    }
    default:pips_internal_error("unhandled case");
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

static call claire_loop_from_loop(loop l, string * result, int task_number){
  
  string * up = malloc(sizeof(string));
  string * claire_name = malloc(sizeof(string));
  statement s = loop_body(l);
  instruction i = statement_instruction(s);
  int u, low;
  expression incr_e = range_increment(loop_range(l));
  syntax incr_s = expression_syntax(incr_e);

  if(!syntax_call_p(incr_s) || 
     strcmp( entity_local_name(call_function(syntax_call(incr_s))), "1") != 0 ) {
    pips_user_error("Loop increments must be constant \"1\".\n");
  }

  u = atoi(claire_expression(range_upper(loop_range(l))));
  low = atoi(claire_expression(range_lower(loop_range(l))));
  /*  printf("%i %i\n", u, low); */
  *up = strdup(i2a(u - low+1));
	       //*up = claire_expression(range_upper(loop_range(l)) - range_lower(loop_range(l)) + 1);
  *claire_name = claire_entity_local_name(loop_index(l));
  if( (*claire_name)[0] == 'M'){
    gen_array_append(intern_indices_array, claire_name);
    gen_array_append(intern_upperbounds_array, up);
  }
  else{
    gen_array_append(extern_indices_array, claire_name);
    gen_array_append(extern_upperbounds_array, up);
  }

  switch(instruction_tag(i)){
  case is_instruction_loop:{
    loop l = instruction_loop(i);
    return claire_loop_from_loop(l, result, task_number);
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
      return claire_loop_from_loop(l, result, task_number);
    }
    default:
    pips_user_error("Only loops and calls allowed in a loop.\n");
  }
return call_undefined;
}


/* We enter a loop nest. The first loop must be an extern loop. */
static string claire_loop_from_sequence(loop l, int task_number){
  statement s = loop_body(l);
  call c;
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
  

  *name = claire_entity_local_name(loop_index(l));
  u = atoi(claire_expression(range_upper(loop_range(l))));
  low = atoi(claire_expression(range_lower(loop_range(l))));
  *up = strdup(i2a(u - low+1));
  //*up = claire_expression(range_upper(loop_range(l)) - range_lower(loop_range(l)) + 1);

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
    c = claire_loop_from_loop(l, &result, task_number);
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
      c = claire_loop_from_loop(l, &result, task_number);
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

  for(i=0; i<gen_array_nitems(extern_upperbounds_array) - 1; i++){
    result = strdup(concatenate(result, "vartype!(", *((string *)(gen_array_item(extern_upperbounds_array, i))), "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(",*((string *)(gen_array_item(extern_upperbounds_array, i))), ")),",NL, TAB, TAB, NULL));

  /* add external indices names*/
  result = strdup(concatenate(result, "names = list<string>(", NULL));
  for(i=0; i<gen_array_nitems(extern_indices_array) - 1; i++){
    result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(extern_indices_array, i))), QUOTE ", ", NULL));
  }
  result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(extern_indices_array, i))), QUOTE, ")),", NL, TAB, NULL));
  
  result = strdup(concatenate(result, claire_call_from_loopnest(c, task_number), NULL));

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
static string claire_statement_from_sequence(statement s, int task_number){
  string result = "";
  instruction i = statement_instruction(s);

  switch(instruction_tag(i)){
  case is_instruction_loop:{
    loop l = instruction_loop(i);
    result = claire_loop_from_sequence(l, task_number);
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
   it is into claire_statementement_from_sequence and subsequent
   functions.*/
static string claire_sequence_from_task(sequence seq){
  string result = "";
  int task_number = 0;
  MAP(STATEMENT, s,
  {
    string oldresult = strdup(result);
    string current = strdup(claire_statement_from_sequence(s, task_number));

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
static string claire_tasks_with_motif(statement stat){
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
    result = claire_sequence_from_task(seq);
    break;
  }
  default:{
    pips_user_error("Only a sequence can be here");
  }
  }
  result = strdup(concatenate(result, NL, NL, "PRES:APPLICATION := APPLICATION(name = symbol!(", QUOTE, global_module_name, QUOTE, "), ", NL, TAB,NULL));
  result = strdup(concatenate(result, "tasks = list<TASK>(", NULL));
  for(j = 0; j<gen_array_nitems(tasks_names) - 1; j++){
    result = strdup(concatenate(result, *((string *)(gen_array_item(tasks_names, j))), ", ", NULL));
  }
  result = strdup(concatenate(result, *((string *)(gen_array_item(tasks_names, j))), "))", NULL));

  return result;
}


/* Creates string for claire pretty printer.
   This string divides in declarations (array decl.) and 
   tasks which are loopnest with an instruction at the core.
*/
static string claire_code_string(entity module, statement stat)
{
  string decls="", tasks="", result="";

  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

  decls       = claire_declarations_with_explicit_motif(module, variable_p, "", true);
  tasks       = claire_tasks_with_motif(stat);
  
  result = strdup(concatenate(decls, NL, tasks, NL, NULL));
  ifdebug(2)
    {
      printf("%s", result);
    }
  return result;
}


/******************************************************** PIPSMAKE INTERFACE */

#define CLAIREPRETTY    ".cl"

/* Initiates claire pretty print modules
 */
bool print_claire_code_with_explicit_motif(const char* module_name)
{
  FILE * out;
  string ppt, claire, dir, filename;
  entity module;
  statement stat;
  array_names = gen_array_make(0);
  array_dims = gen_array_make(0);
  claire = db_build_file_resource_name(DBR_CLAIRE_PRINTED_FILE, module_name, CLAIREPRETTY);

  global_module_name = module_name;
  module = module_name_to_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", claire, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  if(statement_undefined_p(stat))
    {
      pips_internal_error("No statement for module %s", module_name);
    }
  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("CLAIREPRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Begin Claire prettyprinter for %s\n", entity_name(module));
  ppt = claire_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();  

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "// Claire pretty print for module %s. \n%s", module_name, ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_CLAIRE_PRINTED_FILE, module_name, claire);

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
claire_declarations(entity module, string_buffer result)
{ 
  bool comma = false;
  list dim;
  int nb_dim =0;
  string up_string ;
  MAP ( ENTITY, var,
  {
    if (variable_p(var) && ( variable_entity_dimension(var) >0))
      {
	string_buffer result_up = string_buffer_make(true);
	nb_dim = variable_entity_dimension(var);
	string_buffer_append(result,
			     concatenate(CLAIRE_ARRAY_PREFIX,entity_user_name(var), 
					 " :: DATA_ARRAY(name = symbol!(", QUOTE,
					 CLAIRE_ARRAY_PREFIX,entity_user_name(var), QUOTE,"),"
					 , CLAIRE_RL, NULL));
	string_buffer_append(result,
			     concatenate("dim = ",
					 i2a(nb_dim), ",", CLAIRE_RL,NULL));
	
	string_buffer_append(result, "origins = list<integer>(");
	comma = false; 
	for (dim = variable_dimensions(type_variable(entity_type(var))); !ENDP(dim); dim = CDR(dim)) {

	  intptr_t low;
	  intptr_t  up;
	  expression elow = dimension_lower(DIMENSION(CAR(dim)));
	  expression eup = dimension_upper(DIMENSION(CAR(dim)));
	 if (expression_integer_value(elow, &low) && expression_integer_value(eup, &up)){
	    string_buffer_append(result,
				 concatenate((comma)? ",":"", i2a(low),NULL));
	    string_buffer_append(result_up,
				 concatenate((comma)? ",":"", i2a(up-low+1),NULL));
	  }
	  else pips_user_error("Array dimensions must be integer\n");
	  comma = true;
	}

	string_buffer_append(result, concatenate("),", CLAIRE_RL,NULL));

	string_buffer_append(result, "dimSizes = list<integer>(");
	up_string=string_buffer_to_string(result_up);
	/*	string_buffer_free(&result_up,false);*/ // MEMORY LEAK???
	string_buffer_append(result,up_string);
	free(up_string);

	string_buffer_append(result,
			     concatenate("),", CLAIRE_RL,NULL));
	string_buffer_append(result,
			     concatenate("dataType = INTEGER)",
					 NL,NL,NULL));

      }
  },
	entity_declarations(module));


}


static void 
push_current_statement(statement s, nest_context_p nest)
{ 
    stack_push(s , nest->current_stat); 
}

static void 
pop_current_statement(statement s, nest_context_p nest)
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
pop_loop(loop l, nest_context_p nest)
{ 
 stack_pop(nest->loops_for_call); 
 stack_pop(nest->loop_indices); 
}
 
static bool call_selection(call c, nest_context_p nest) 
{ 

  /* CA il faut implemeter  un choix judicieux ... distribution ou encapsulation*/
  /* pour le moment distribution systematique de tout call */
  /* il faut recuperer les appels de fonction value_code_p(entity_initial(f)*/
  entity f = call_function(c); 
  if  (ENTITY_ASSIGN_P(f) || entity_subroutine_p(f))
    {  
      return true;
    }
  else return false;

  /*  statement s = (statement) stack_head(nest->current_stat);
      return ((!return_statement_p(s) && !continue_statement_p(s)));*/
}

static void store_call_context(call c, nest_context_p nest)
{
  stack sl = stack_copy(nest->loops_for_call);
  stack si = stack_copy(nest->loop_indices);
  /* on sauve le statement associe au call */ 
  statement statc = (statement) stack_head(nest->current_stat) ;
  gen_array_append(nest->nested_loop_indices,si);
  gen_array_append(nest->nested_loops,sl);
  gen_array_append(nest->nested_call,statc);
}

static bool push_test(test t,  nest_context_p nest)
{
  /* encapsulation de l'ensemble des instructions appartenant au test*/
  /* on ne fait rien pour le moment */
  return false;
}


static void pop_test(test t,  nest_context_p nest)
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
  int numberOfTasks=gen_array_nitems(nest->nested_call);
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

static void claire_loop(stack st, string_buffer result)
{
  bool comma_needed = false;
  string_buffer buffer_lower = string_buffer_make(true);
  string_buffer buffer_upper = string_buffer_make(true);
  string_buffer buffer_names = string_buffer_make(true);
  string lower_bounds = "";
  string upper_bounds = "";
  string name_bounds = "";

  string_buffer_append(result, "exLoopNest = LOOPNEST(deep = ");
  string_buffer_append(result, concatenate(i2a(stack_size(st)),",",NULL));

  STACK_MAP_X(s, statement,
  {
    loop l = instruction_loop(statement_instruction(s));
    expression el =range_lower(loop_range(l));
    expression eu =range_upper(loop_range(l));
    expression new_eu= expression_plusplus(eu);

  string_buffer_append(buffer_lower,
		       concatenate(comma_needed? ",": "",
				   "vartype!(",
				   words_to_string(words_expression(el,NIL)),
				   ")",NULL));
  string_buffer_append(buffer_upper,
		       concatenate(comma_needed? ",": "",
				   "vartype!(",
				   words_to_string(words_expression(new_eu,NIL)),
				   ")",NULL));
  string_buffer_append(buffer_names,
		       concatenate(comma_needed? ",": "",
				   QUOTE,entity_user_name(loop_index(l)),
				   QUOTE,NULL));
  comma_needed = true;
  },
	      st, 0);

  /* Lower bounds generation*/
  string_buffer_append(result,
		       concatenate(CLAIRE_RL,TAB, "lowerBound = list<VARTYPE>(", NULL));
  lower_bounds =string_buffer_to_string(buffer_lower);
  string_buffer_append(result,lower_bounds);
  free(lower_bounds), lower_bounds = NULL;
  string_buffer_append(result,"),");

  /* Upper bounds generation */
 string_buffer_append(result,
		      concatenate(CLAIRE_RL,TAB, "upperBound = list<VARTYPE>(", NULL));

  upper_bounds =string_buffer_to_string(buffer_upper);
  string_buffer_append(result,upper_bounds);
  free(upper_bounds), upper_bounds = NULL;
  string_buffer_append(result, "),");

  /* Loop Indices generation */
  string_buffer_append(result,
		       concatenate(CLAIRE_RL,TAB, "names = list<string>(", NULL));
  name_bounds =string_buffer_to_string(buffer_names);
  string_buffer_append(result,name_bounds);
  free(name_bounds), name_bounds = NULL;

  string_buffer_append(result, ")");

  string_buffer_append(result, concatenate("),",CLAIRE_RL,NULL));
}



static void claire_reference(int taskNumber, reference r, bool wmode,
		      string_buffer result)
{

 const char* varname = entity_user_name(reference_variable(r));
 string_buffer_append
   (result,
    concatenate("name = symbol!(\"",
		CLAIRE_TASK_PREFIX,i2a(taskNumber),
		"\" /+ \"", CLAIRE_ARRAY_PREFIX, varname, "\"),",
		CLAIRE_RL, TAB,
		"darray = ", CLAIRE_ARRAY_PREFIX, varname, ",",
		CLAIRE_RL,TAB,
		"accessMode = ",
		(wmode?"Wmode,":"Rmode,"), CLAIRE_RL,TAB,
		NULL));
}

static void  find_motif(Psysteme ps, Pvecteur nested_indices, int dim, int nb_dim, Pcontrainte *bound_inf, Pcontrainte *bound_sup, Pcontrainte *iterator, int *motif_up_bound)
{ 
  
  Variable phi;
  Value	v;
  Pvecteur pi;
  Pcontrainte c, next, cl, cu, cl_dup, cu_dup,lind, lind_dup,
    list_cl=NULL , 
    list_cu=NULL,
    list_ind=NULL;
  int lower =1;
  int upper =2;
  int ind =3;
  Pcontrainte bounds[3][3];
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
   }
   else if (!CONTRAINTE_UNDEFINED_P(bounds[lower][1]) && !CONTRAINTE_UNDEFINED_P(bounds[upper][1])) {
     /* case where loop bounds are numeric */
     *bound_inf= bounds[lower][1];
     *bound_sup = bounds[upper][1];
     *iterator =  bounds[ind][1];
     *motif_up_bound = - vect_coeff(TCST,bounds[lower][1]->vecteur) 
       - vect_coeff(TCST,bounds[upper][1]->vecteur)+1;
   } else {
     /* Only bounds with several loop indices */ 
     /* printf("PB - Only bounds with several loop indices\n"); */ 
    *bound_inf= CONTRAINTE_UNDEFINED;
    *bound_sup = CONTRAINTE_UNDEFINED;
    *iterator = CONTRAINTE_UNDEFINED;
    *motif_up_bound = 1;
    
   }
  
}


static void claire_tiling(int taskNumber, reference ref,  region reg, stack indices,  string_buffer result)
{
  Psysteme ps_reg = sc_dup(region_system(reg));
  
  entity var = reference_variable(ref);
  int dim = gen_length(variable_dimensions(type_variable(entity_type(var))));
  int i, j ;
  string_buffer buffer_lower = string_buffer_make(true);
  string_buffer buffer_upper = string_buffer_make(true);
  string_buffer buffer_names = string_buffer_make(true);
  string_buffer buffer_offset = string_buffer_make(true);
  string_buffer buffer_fitting = string_buffer_make(true);
  string_buffer buffer_paving = string_buffer_make(true);
 
  string string_lower = "";
  string string_upper = "";
  string string_names = "";
  string string_offset = "";
  string string_paving = "";
  string string_fitting =  "";
  bool comma = false;

  Pvecteur iterat, pi= VECTEUR_NUL;
  Pcontrainte bound_inf = CONTRAINTE_UNDEFINED;
  Pcontrainte bound_up = CONTRAINTE_UNDEFINED;
  Pcontrainte iterator = CONTRAINTE_UNDEFINED;
  int motif_up_bound =0;
  int dim_indices= stack_size(indices);
  int pav_matrix[10][10], fit_matrix[10][10];

  for (i=1; i<=9;i++)
    for (j=1;j<=9;j++)
      pav_matrix[i][j]=0, fit_matrix[i][j]=0;

  /* if (debug)  printf("matrix pavage dimension:[%d][%d]\n",dim,dim_indices); */
  
  STACK_MAP_X(index,entity,
  { 
    vect_add_elem (&pi,(Variable) index ,VALUE_ONE);
  }, indices,1);
  
  /* if (debug) 
    {
      printf("liste des indices de boucles englobants :\n");
      vect_print(pi, entity_local_name);
      }*/

  for(i=1; i<=dim ; i++)
    { 
      Psysteme ps = sc_dup(ps_reg); 
      sc_transform_eg_in_ineg(ps);
     
      find_motif(ps, pi, i,  dim, &bound_inf, &bound_up, &iterator, &motif_up_bound);  


	/*	extraction offset = terme / partie constante de lower*/
     	string_buffer_append(buffer_offset,
			     concatenate((comma)?",":"",
					 "vartype!(",
					 (CONTRAINTE_UNDEFINED_P(bound_inf))? "0" :
					 i2a(vect_coeff(TCST,bound_inf->vecteur)),
					 ")",NULL));

      /* paving = coef de l'indice de boucle */
	if (!CONTRAINTE_UNDEFINED_P(iterator)) {
	  for (iterat = pi, j=1; !VECTEUR_NUL_P(iterat); iterat = iterat->succ, j++)   
	      pav_matrix[i][j]= vect_coeff(var_of(iterat),iterator->vecteur);
	}
      
      /* fitting = 1 */
	if (!CONTRAINTE_UNDEFINED_P(bound_inf))
	  fit_matrix[i][i]= (motif_up_bound >1) ? 1:0;
	

	/* motif = boucle prof = dim */

	/* lower bound = 1 et upper bound = upper - lower */
      string_buffer_append(buffer_lower,
			   concatenate(comma? ",": "",
					      "vartype!(0)",NULL));
      string_buffer_append(buffer_upper,
			   concatenate(comma? ",": "",
				       "vartype!(",i2a(motif_up_bound),")",NULL));
      /* motif name = M_ numero = dim */

      string_buffer_append(buffer_names,
		       concatenate(comma? ",": "",
				   QUOTE,
				   CLAIRE_MOTIF_PREFIX, i2a(taskNumber),"_",
				   entity_user_name(var), "_",i2a(i),
				   QUOTE,NULL));
      comma = true;
    } 
  for (j=1; j<=dim_indices ; j++){
    if (j>1)  string_buffer_append(buffer_paving,strdup("),list("));
    for(i=1; i<=dim ; i++)
      string_buffer_append(buffer_paving,
			   concatenate((i>1)?",":"",
				       "vartype!(",
				       i2a( pav_matrix[i][j]),
				       ")",NULL));
  }
  for(i=1; i<=dim ; i++) { 
    if (i>1)  
      string_buffer_append(buffer_fitting, "),list(");
    for(j=1; j<=dim ; j++)
      string_buffer_append(buffer_fitting,
			   concatenate((j>1)?",":"",
				       "vartype!(",
				       i2a( fit_matrix[i][j]),
				       ")",NULL));
     }

  string_buffer_append(result, "offset = list<VARTYPE>(");
  string_offset =string_buffer_to_string(buffer_offset);
  string_buffer_append(result,string_offset);
  free(string_offset), string_offset = NULL;

  string_buffer_append(result,concatenate("),",CLAIRE_RL,TAB,NULL));
  string_buffer_append(result,"fitting = list<list[VARTYPE]>(list(");
  string_fitting =string_buffer_to_string(buffer_fitting);
  string_buffer_append(result,string_fitting);
  free(string_fitting), string_fitting = NULL;

  string_buffer_append(result, concatenate(")),",CLAIRE_RL,TAB,NULL));
  string_buffer_append(result, "paving = list<list[VARTYPE]>(list(");
  string_paving =string_buffer_to_string(buffer_paving);
  string_buffer_append(result,string_paving);
  free(string_paving), string_paving = NULL;

  string_buffer_append(result, concatenate(")),",CLAIRE_RL,TAB,NULL));
  string_buffer_append(result, "inLoopNest = LOOPNEST(deep = ");
  string_buffer_append(result, i2a(dim));
  string_buffer_append(result, concatenate(",",CLAIRE_RL,NULL));

  /* Motif Lower bounds generation*/
   string_buffer_append(result,
			concatenate(TAB,TAB,"lowerBound = list<VARTYPE>(", NULL));
  string_lower =string_buffer_to_string(buffer_lower);
  string_buffer_append(result,string_lower);
  free(string_lower), string_lower = NULL;
  string_buffer_append(result, "),");

  /*  Motif Upper bounds generation */
  string_buffer_append(result,
		       concatenate(CLAIRE_RL,TAB,TAB, "upperBound = list<VARTYPE>(", NULL));

  string_upper =string_buffer_to_string(buffer_upper);
  string_buffer_append(result,string_upper);
  free(string_upper), string_upper = NULL;
  string_buffer_append(result, "),");

  /*  Motif Loop Indices generation */
  string_buffer_append(result,
		       concatenate(CLAIRE_RL,TAB,TAB, "names = list<string>(", NULL));
  string_names =string_buffer_to_string(buffer_names);
  string_buffer_append(result, string_names);
  free(string_names), string_names = NULL;

  string_buffer_append(result, ")");
  string_buffer_append(result, ")");
}
static void claire_references(int taskNumber, list l_regions, stack indices, string_buffer result)
{ 
  list lr;
  bool atleast_one_read_ref = false;
  bool atleast_one_written_ref = false;
  bool comma = false;
/*   Read array references first */
   for ( lr = l_regions; !ENDP(lr); lr = CDR(lr))
     {
       region re = REGION(CAR(lr));
       reference ref = effect_any_reference(re);
       if (array_reference_p(ref) && region_read_p(re)) {
	 atleast_one_read_ref = true;
	 if (comma) string_buffer_append(result,concatenate(",",CLAIRE_RL,NULL));
	 string_buffer_append(result,concatenate(TAB,  "DATA(",NULL));
	 claire_reference(taskNumber, ref,  region_write_p(re), result);
	 /* fprintf(stderr, "\n la  region ");
	    print_regions(lr);*/

	 claire_tiling(taskNumber, ref,re, indices, result);
	 string_buffer_append(result,concatenate(TAB, ")", NULL));
	 comma = true;
       }
      }
   if (!atleast_one_read_ref)
     string_buffer_append(result,concatenate(TAB, "dummyDATA, ",
					     CLAIRE_RL,NULL));

   for ( lr = l_regions; !ENDP(lr); lr = CDR(lr))
     {
       region re = REGION(CAR(lr));
       reference ref = effect_any_reference(re);
       if (array_reference_p(ref) && region_write_p(re)) { 
	 atleast_one_written_ref = true;
	 if (comma) string_buffer_append(result, concatenate(",",CLAIRE_RL,NULL));
	 string_buffer_append(result, concatenate(TAB, "DATA(",NULL));
	 claire_reference(taskNumber, ref,  region_write_p(re), result);
	 /* fprintf(stderr, "\n la  region ");
	    print_regions(lr); */
	 claire_tiling(taskNumber, ref,re, indices, result);
	 string_buffer_append(result, ")");
	 comma = true;
    }
     }
   if (!atleast_one_written_ref) 
     string_buffer_append(result,concatenate(TAB,", dummyDATA ",NULL));
}

static void claire_data(int taskNumber,statement s, stack indices, string_buffer result )
{
  list  l_regions = regions_dup(load_statement_local_regions(s));
  string_buffer_append(result, concatenate("data = list<DATA>(",
					   CLAIRE_RL,NULL));
  /*
  ifdebug(2) {
    fprintf(stderr, "\n list of regions ");
    print_regions(l_regions);
    fprintf(stderr, "\n for the statement");
    print_statement(s);
  }
  */
   claire_references(taskNumber, l_regions, indices, result);

   /*
    claire_tiling();
    claire_motif();
   */
        string_buffer_append(result, concatenate(")",CLAIRE_RL,NULL));
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
static void claire_task( int taskNumber, nest_context_p nest, string_buffer result)
{
  statement s = gen_array_item(nest->nested_call,taskNumber);
  stack st = gen_array_item(nest->nested_loops,taskNumber);
  stack sindices = gen_array_item(nest->nested_loop_indices,taskNumber);

  string_buffer_append(result, CLAIRE_TASK_PREFIX);
  string_buffer_append(result, i2a(taskNumber));
  string_buffer_append(result, " :: TASK(unitSpentTime = vartype!(");
  string_buffer_append(result, task_complexity(s)); 
  string_buffer_append(result, concatenate("),",CLAIRE_RL,NULL));

  claire_loop(st, result);
  claire_data (taskNumber, s,sindices, result);
  string_buffer_append(result, concatenate(")",NL,NL,NULL));

}

static void  claire_tasks(statement stat, string_buffer result){

  const char*  module_name = get_current_module_name();
  nest_context_t nest;
  int taskNumber =0;
  nest.loops_for_call = stack_make(statement_domain,0,0);
  nest.loop_indices = stack_make(entity_domain,0,0);
  nest.current_stat = stack_make(statement_domain,0,0);
  nest.nested_loops=  gen_array_make(0);
  nest.nested_loop_indices =  gen_array_make(0);
  nest.nested_call=  gen_array_make(0);
  
  if(statement_undefined_p(stat)) {
    pips_internal_error("statement error");
  }
  
  search_nested_loops_and_calls(stat,&nest);
  /* ifdebug(2)  print_call_selection(&nest); */

  for (taskNumber = 0; taskNumber<gen_array_nitems(nest.nested_call); taskNumber++)
    
    claire_task(taskNumber, &nest,result);

  string_buffer_append(result,
		       concatenate(NL, NL,
				   "PRES:APPLICATION := APPLICATION(name = symbol!(",
				   QUOTE, module_name, QUOTE, "), ",
				   NL, TAB,NULL));
  string_buffer_append(result, "tasks = list<TASK>(");

  for(taskNumber = 0; taskNumber<gen_array_nitems(nest.nested_call)-1; taskNumber++)
   string_buffer_append(result, concatenate(CLAIRE_TASK_PREFIX,
					    i2a(taskNumber), ", ", NULL));
  string_buffer_append(result, concatenate(CLAIRE_TASK_PREFIX,
					   i2a(taskNumber) , "))", NL, NULL));

  gen_array_free(nest.nested_loops);
  gen_array_free(nest.nested_loop_indices);
  gen_array_free(nest.nested_call);
  stack_free(&(nest.loops_for_call));
  stack_free(&(nest.loop_indices));
  stack_free(&(nest.current_stat));

}


/* Creates string for claire pretty printer.
   This string divides in declarations (array decl.) and
   tasks which are loopnests with an instruction at the core.
*/
static string claire_code(entity module, statement stat)
{
  string_buffer result=string_buffer_make(true);
  string result2;

  claire_declarations(module,result);
  claire_tasks(stat,result);

  result2=string_buffer_to_string(result);
  /*  string_buffer_free(&result,true); */ // MEMORY LEAK ???
  /* ifdebug(2)
    {
      printf("%s", result2);
      } */
  return result2;
}

static bool valid_specification_p(entity module, statement stat)
{ return true;
}

/******************************************************** PIPSMAKE INTERFACE */

#define CLAIREPRETTY    ".cl"

/* Initiates claire pretty print modules
 */
bool print_claire_code(const char* module_name)
{
  FILE * out;
  string ppt;
 
  entity module = module_name_to_entity(module_name);
  string claire = db_build_file_resource_name(DBR_CLAIRE_PRINTED_FILE, 
					      module_name, CLAIREPRETTY);
  string  dir = db_get_current_workspace_directory();
  string filename = strdup(concatenate(dir, "/", claire, NULL));
  
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

  debug_on("CLAIREPRETTYPRINTER_DEBUG_LEVEL"); 
  pips_debug(1, "Spec validation before Claire prettyprinter for %s\n", 
	     entity_name(module));
  if (valid_specification_p(module,stat)){ 
    pips_debug(1, "Spec is valid\n");
    pips_debug(1, "Begin Claire prettyprinter for %s\n", entity_name(module));
    
    ppt = claire_code(module, stat);
    pips_debug(1, "end\n");
    debug_off();  
    
    /* save to file */
    out = safe_fopen(filename, "w");
    fprintf(out, "// Claire pretty print for module %s. \n%s", 
	    module_name, ppt);
    safe_fclose(out, filename);
    free(ppt);
  }

  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_CLAIRE_PRINTED_FILE, module_name, claire);

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}

 
