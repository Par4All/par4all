/* 
   $Id$

   Try to prettyprint the RI in CLAIRE.
   Very basic at the time.

   print_claire_code        > MODULE.claire_printed_file
                            < PROGRAM.entities
                            < MODULE.code

   $Log: claire_prettyprinter.c,v $
   Revision 1.18  2004/08/04 07:19:17  irigoin
   Better handling of comments. CONTINUE are ignored.

   Revision 1.17  2004/08/03 09:47:04  hurbain
   Bugs corrections

   Revision 1.16  2004/07/05 08:41:46  hurbain
   checkin pour install sur ciboure

   Revision 1.15  2004/06/24 14:12:13  hurbain
   Pre-holiday version.
   Stable, works (as far as I've tested ;) )

   Revision 1.14  2004/06/16 15:18:42  hurbain
   First definitive version.
   Some bugs left, but stable state.

   Revision 1.13  2004/06/16 09:37:49  hurbain
   Ça compile.
   Yapuka debugger.

   Revision 1.12  2004/06/16 09:00:00  hurbain
   Backup version. Is not compilable.

   Revision 1.11  2004/06/15 15:48:49  hurbain
   This is not a stable version, backup sync only.

   Revision 1.10  2004/06/15 15:33:47  hurbain
   Version pré-récupération de sauvegarde

   Revision 1.9  2004/06/10 14:00:16  hurbain
   *** empty log message ***

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
#define QUOTE         "\""


/* array containing extern loop indices names */
gen_array_t extern_indices_array;
/* array containing intern loop indices (name : "M_") */
gen_array_t intern_indices_array;
/* array containing extern upperbounds */
gen_array_t extern_upperbounds_array;
/* array containing intern upperbounds */
gen_array_t intern_upperbounds_array; 
/* array containing the tasks names*/
gen_array_t tasks_names;

string global_module_name;

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

static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
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

static string claire_reference(reference r);

static string claire_expression(expression e)
{
  string result = "";
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      result = claire_reference(syntax_reference(s));
      break;
    case is_syntax_call:
      result = claire_entity_local_name(call_function(syntax_call(s)));
      break;
    default:
      pips_internal_error("unexpected syntax tag");
    }
  return result;
}

/* Attention with Fortran: the indexes are reversed. */
static string claire_reference(reference r)
{
  string result = strdup(EMPTY), old, svar;
  MAP(EXPRESSION, e, 
  {
    string s = claire_expression(e);
    
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

gen_array_t array_names;
gen_array_t array_dims;

#define ITEM_NOT_IN_ARRAY -1

int gen_array_index(gen_array_t ar, string item){
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
	
	int low;
	int up;
	nbdim++;
	if (expression_integer_value(elow, &low)){
	  if(nbdim != 1)
	    origins = strdup(concatenate(origins, COMMA ,int_to_string(low), NULL));
	  else
	    origins = strdup(concatenate(origins, int_to_string(low), NULL));
	}
	else pips_user_error("Array origins must be integer\n");

	if (expression_integer_value(eup, &up)){
	  if(nbdim != 1)
	    dimensions = strdup(concatenate(dimensions, COMMA ,int_to_string(up-low+1), NULL));
	  else
	    dimensions = strdup(concatenate(dimensions, int_to_string(up-low+1), NULL));
	}
	else pips_user_error("Array dimensions must be integer\n");
      }, ldim);
      *nbdimptr = nbdim;
      gen_array_append(array_dims, nbdimptr);
      gen_array_append(array_names, namep);
      result = strdup(concatenate(result, int_to_string(nbdim), COMMA, NL, NULL));
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
  storage s = entity_storage(var);
  pips_debug(2,"Entity name : %s\n",entity_name(var));
  /*  Many possible combinations */

  if (strstr(name,TYPEDEF_PREFIX) != NULL)
    pips_user_error("Structs not supported\n");

  switch (type_tag(t)) {
  case is_type_variable:
    {
      variable v = type_variable(t);  
      string sd;
      sd = claire_dim_string(variable_dimensions(v), name);
    
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

static string claire_array_in_task(reference r, bool first, int task_number);

static string claire_call_from_assignation(call c, int task_number, bool * input_provided){
  /* All arguments of this call are in Rmode (inputs of the task) */
  /* This function is called recursively */
  entity called = call_function(c);
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
	result = strdup(concatenate(result, claire_array_in_task(ref, FALSE, task_number), NULL));
	*input_provided = TRUE;
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
	    paving_array[iterator_nr] = "1";
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	    fitting_array[iterator_nr] = "1";
	  }
	}
	if(syntax_tag(args[1]) == is_syntax_reference){
	  reference ref = syntax_reference(args[1]);
	  if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	   paving_array[iterator_nr] = "1";
	  }
	  else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	   fitting_array[iterator_nr] = "1";
	  }
	}
      }
      else if(same_string_p(funname, "-")){
	if(syntax_tag(args[1]) == is_syntax_call && gen_length(call_arguments(syntax_call(args[1])))==0){
	  if(syntax_tag(args[0]) == is_syntax_reference){
	    reference ref = syntax_reference(args[0]);
	    if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      paving_array[iterator_nr] = "1";
	    }
	    else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	      fitting_array[iterator_nr] = "1";
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
	  string mult =  claire_entity_local_name(call_function(syntax_call(args[0]))); 
	  if(extern_nr != ITEM_NOT_IN_ARRAY){
	    paving_array[extern_nr] = mult;
	  }
	  else if(intern_nr != ITEM_NOT_IN_ARRAY){
	    fitting_array[intern_nr] = mult;
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

static string claire_array_in_task(reference r, bool first, int task_number){
  string varname = strdup(concatenate("A_", claire_entity_local_name(reference_variable(r)), NULL));
  int indice_nr = 0;
  list indices = reference_indices(r);
  string result = "";
  int nb_loops = gen_array_nitems(extern_indices_array);
  
  int * index_of_array = (int *) (gen_array_item(array_dims, gen_array_index(array_names, varname)));
  string offset_array[*index_of_array];
  string paving_array[*index_of_array][gen_array_nitems(extern_indices_array)];
  string fitting_array[*index_of_array][gen_array_nitems(intern_indices_array)];
  int i;
  int j;
  
  for (i=0; i<*index_of_array; i++)
    offset_array[i] = "0";
  
  for (i=0; i<gen_array_nitems(extern_indices_array) ; i++)
    for (j=0; j<*index_of_array; j++)
      paving_array[i][j] = "0";

  for (i=0; i<gen_array_nitems(intern_indices_array) ; i++)
    for (j=0; j<*index_of_array; j++)
      fitting_array[i][j] = "0";
  result = strdup(concatenate(result, "DATA(name = symbol!(\"", "T_", int_to_string(task_number),
			      "\" /+ \"", varname, "\"),", NL, TAB, TAB, NULL));
  result = strdup(concatenate(result, "darray = ", varname, "," NL, TAB, TAB, "accessMode = ", (first?"Wmode,":"Rmode,"),
			      NL, TAB, TAB, "offset = list<VARTYPE>(", NULL));
  
  MAP(EXPRESSION, ind, {
    syntax sind = expression_syntax(ind);
    int iterator_nr;
    switch(syntax_tag(sind)){
    case is_syntax_reference:{
      reference ref = syntax_reference(sind);
      if((iterator_nr = gen_array_index(extern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	paving_array[indice_nr][iterator_nr] = "1";
      }
      else if((iterator_nr = gen_array_index(intern_indices_array, claire_entity_local_name(reference_variable(ref)))) != ITEM_NOT_IN_ARRAY){
	fitting_array[indice_nr][iterator_nr] = "1";
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
  for(i=0; i<*index_of_array - 1; i++){
    result=strdup(concatenate(result, "vartype!(", offset_array[i],"), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", offset_array[i], "))," NL, NULL));
  result = strdup(concatenate(result, TAB, TAB, "fitting = list<list[VARTYPE]>(", NULL));
  for(i=0;i<gen_array_nitems(intern_indices_array) - 1; i++){
    result = strdup(concatenate(result, "list(", NULL));
    for(j = 0; j<(*index_of_array)-1; j++){
      result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "), ", NULL));
    }
    result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], ")),", NL, TAB, TAB, TAB, NULL));
  }
  result = strdup(concatenate(result, "list(", NULL));
  if(gen_array_nitems(intern_indices_array) > 0){
    for(j = 0; j<(*index_of_array)-1; j++){
      result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "), ", NULL));
    }
    result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "))),", NL, TAB, TAB, NULL));
  }
  else {
    result = strdup(concatenate(result, ")),", NL, TAB, TAB, NULL));
  }
  result = strdup(concatenate(result, TAB, TAB, "paving = list<list[VARTYPE]>(", NULL));
  
 
  for(i=0;i<gen_array_nitems(extern_indices_array) - 1; i++){
    result = strdup(concatenate(result, "list(", NULL));
    for(j = 0; j<(*index_of_array)-1; j++){
      result = strdup(concatenate(result, "vartype!(", paving_array[i][j], "), ", NULL));
    }
    result = strdup(concatenate(result, "vartype!(", paving_array[i][j], ")),", NL, TAB, TAB, TAB, NULL));
  }
  result = strdup(concatenate(result, "list(", NULL));
  for(j = 0; j<(*index_of_array)-1; j++){
    result = strdup(concatenate(result, "vartype!(", paving_array[i][j], "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", paving_array[i][j], "))),", NL, TAB, TAB, NULL));
  
#define MONMAX(a, b) ((a<b)?b:a)

  result = strdup(concatenate(result, "inLoopNest = LOOPNEST(deep = ", int_to_string(MONMAX(gen_array_nitems(intern_indices_array), 1)), ",", NL, TAB, TAB, TAB, NULL));
  result = strdup(concatenate(result, "upperBound = list<VARTYPE>(", NULL));
  
  if(gen_array_nitems(intern_indices_array) > 0){
    for(i = 0; i<gen_array_nitems(intern_upperbounds_array) - 1; i++){
      result = strdup(concatenate(result, "vartype!(", *((string *)(gen_array_item(intern_upperbounds_array, i))), "), ", NULL));
    }
    
    result = strdup(concatenate(result, "vartype!(", *((string *)(gen_array_item(intern_upperbounds_array, i))), ")),", NULL));
    
    result = strdup(concatenate(result, NL, TAB, TAB, TAB, "names = list<string>(", NULL));
    
    for(i = 0; i<gen_array_nitems(intern_indices_array) - 1; i++){
      result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(intern_indices_array, i))), QUOTE, ", ", NULL));
    }

    result = strdup(concatenate(result, QUOTE, *((string *)(gen_array_item(intern_indices_array, i))), QUOTE, ")", NULL));
  }
  else{
    result = strdup(concatenate(result, "vartype!(1)),", NL, TAB, TAB, TAB, "names = list<string>(\"M_I\")", NULL));
  }
  result = strdup(concatenate(result, "))", (first?")":","), NL, NULL)); 
  return result;
  
}

static string claire_call_from_loopnest(call c, int task_number){
  entity called = call_function(c);
  list arguments = call_arguments(c);
  
  syntax s;
  string result = "";
  string first_result = "";
  bool first = TRUE;
  bool input_provided = FALSE, output_provided = FALSE;
  string name = claire_entity_local_name(called);

  if(!same_string_p(name, "="))
    pips_user_error("Only assignation allowed here.\n");
  
  MAP(EXPRESSION, e, {
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
	  output_provided = TRUE;
	}
	else{
	  result = strdup(concatenate(result, claire_array_in_task(r, first, task_number), NULL));
	  input_provided = TRUE;
	}
      }
    }
    }
    first = FALSE;
  }, arguments);

  if(!input_provided){
    result = strdup(concatenate("data = list<DATA>(dummyDATA, ", result, first_result, NULL));
  }
  else{
    result = strdup(concatenate("data = list<DATA>(", result, first_result, NULL));
  }
  if(!output_provided){
    result = strdup(concatenate(result, ", dummyDATA", NULL));
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
  call ml = loop_undefined; /* meaningful loop */
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

  u = atoi(claire_expression(range_upper(loop_range(l))));
  low = atoi(claire_expression(range_lower(loop_range(l))));
  *up = strdup(int_to_string(u - low +1));
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
  *taskname = strdup(concatenate("T_", int_to_string(task_number), NULL));
  gen_array_append(tasks_names, taskname);
  /* (re-)initialize task-scoped arrays*/
  extern_indices_array = gen_array_make(0);
  intern_indices_array = gen_array_make(0);
  extern_upperbounds_array = gen_array_make(0);
  intern_upperbounds_array = gen_array_make(0);
  
  /* Initialize result string with the declaration of the task */
  string result = strdup(concatenate(*taskname, 
			      " :: TASK(unitSpentTime = vartype!(1),"
			      NL, TAB, "exLoopNest = LOOPNEST(deep = ", NULL));

  instruction ins = statement_instruction(s);
  list li = statement_declarations(s);
  string * name = malloc(sizeof(string));
  string * up = malloc(sizeof(string));
  int u, low;
  *name = claire_entity_local_name(loop_index(l));
  u = atoi(claire_expression(range_upper(loop_range(l))));
  low = atoi(claire_expression(range_lower(loop_range(l))));
  *up = strdup(int_to_string(u - low +1));
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
  result = strdup(concatenate(result, int_to_string(gen_array_nitems(extern_indices_array)), ",", NL, TAB, TAB, NULL));
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
  int j;
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
    string oldresult = result;
    string current = claire_statement_from_sequence(s, task_number);

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
static string claire_tasks(statement stat){
  int j;
  string result = "tasks\n";
  instruction i = statement_instruction(stat);
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
  string decls, tasks, result;

  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

  decls       = claire_declarations(module, variable_p, "", TRUE);
  tasks       = claire_tasks(stat);
  
  result = strdup(concatenate(decls, NL, tasks, NL, NULL));

  printf("%s", result);

  return result;
}


/******************************************************** PIPSMAKE INTERFACE */

#define CLAIREPRETTY    ".cl"

/* Initiates claire pretty print modules
 */
bool print_claire_code(string module_name)
{
  FILE * out;
  string ppt, claire, dir, filename;
  entity module;
  statement stat;
  array_names = gen_array_make(0);
  array_dims = gen_array_make(0);
  claire = db_build_file_resource_name(DBR_CLAIRE_PRINTED_FILE, module_name, CLAIREPRETTY);

  global_module_name = module_name;
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
  fprintf(out, "// Claire pretty print for module %s. \n%s", module_name, ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_CLAIRE_PRINTED_FILE, module_name, claire);

  reset_current_module_statement();
  reset_current_module_entity();

  return TRUE;
}
