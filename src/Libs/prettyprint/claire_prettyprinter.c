/* 
   $Id$

   Try to prettyprint the RI in CLAIRE.
   Very basic at the time.

   print_claire_code        > MODULE.claire_printed_file
                            < PROGRAM.entities
                            < MODULE.code

   $Log: claire_prettyprinter.c,v $
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
      *nbdimptr = nbdim;
      gen_array_append(array_dims, nbdimptr);
      gen_array_append(array_names, namep);
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
      string sd;
      sd = claire_dim_string(variable_dimensions(v), name);
    
      result = strdup(concatenate(result, sd, NULL));
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
    pips_user_error("Something not allowed here");
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

static string claire_array_in_task(reference r, bool first, int task_number);

static string claire_call_from_assignation(call c, int task_number){
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
      result = strdup(concatenate(result, claire_call_from_assignation(syntax_call(syn), task_number), NULL));
      break;
    }
    case is_syntax_reference:{
      reference ref = syntax_reference(syn);
      string varname = claire_entity_local_name(reference_variable(ref));
      if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	result = strdup(concatenate(result, claire_array_in_task(ref, FALSE, task_number), NULL));
      }

     
      break;
    }
    default:{
      pips_user_error("only call and references allowed here");
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
  printf("Fonction : %s %i\n", funname, gen_length(arguments));
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
	  pips_user_error("APOTRES doesn't allow negative coefficients in paving and fitting matrices");
	}
      }
      else if(same_string_p(funname, "*")){
	if(syntax_tag(args[0]) != is_syntax_call || syntax_tag(args[1]) != is_syntax_reference || gen_length(call_arguments(syntax_call(args[0])))!=0 ){
	  pips_user_error("Only scalar * reference are allowed here. Please develop expressions.");
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
      pips_user_error("only linear expression of indices allowed");
    }
  }
  else if(gen_length(arguments) == 0){
    *offset_array = funname;
  }
  else{
    printf("%i\n", gen_length(arguments));
    pips_user_error("only +, -, * and constants allowed");
  }
}

static string claire_array_in_task(reference r, bool first, int task_number){
  string varname = claire_entity_local_name(reference_variable(r));
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
  
  result = strdup(concatenate(result, first?"":TAB, "DATA name = symbol!(\"", "T_", int_to_string(task_number),
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
      pips_user_error("Only call and reference allowed in indices");
      break;
    }
    }
    indice_nr++;
  }, indices);
  for(i=0; i<*index_of_array - 1; i++){
    result=strdup(concatenate(result, "vartype!(", offset_array[i],"), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", offset_array[i], "))," NL, NULL));
  result = strdup(concatenate(result, TAB, TAB, "fitting = list<list[VARTYPE]>(list(", NULL));
  for(i=0;i<gen_array_nitems(intern_indices_array) - 1; i++){
    result = strdup(concatenate(result, "list(", NULL));
    for(j = 0; j<(*index_of_array)-1; j++){
      result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "), ", NULL));
    }
    result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], ")),", NL, TAB, TAB, TAB, NULL));
  }
  result = strdup(concatenate(result, "list(", NULL));
  for(j = 0; j<(*index_of_array)-1; j++){
    result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", fitting_array[i][j], "))),", NL, TAB, TAB, NULL));

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
  
  result = strdup(concatenate(result, "inLoopNest = LOOPNEST(deep = ", int_to_string(gen_array_nitems(intern_indices_array)), NL, TAB, TAB, TAB, NULL));
  result = strdup(concatenate(result, "upperBound = list<VARTYPE>(", NULL));
  
  for(i = 0; i<gen_array_nitems(intern_upperbounds_array) - 1; i++){
    result = strdup(concatenate(result, "vartype!(", gen_array_item(intern_upperbounds_array, i), "), ", NULL));
  }

  result = strdup(concatenate(result, "vartype!(", gen_array_item(intern_upperbounds_array, i), ")),", NULL));

  result = strdup(concatenate(result, NL, TAB, TAB, TAB, "names = list<string>(", NULL));

  for(i = 0; i<gen_array_nitems(intern_indices_array) - 1; i++){
    result = strdup(concatenate(result, QUOTE, gen_array_item(intern_indices_array, i), QUOTE, ", ", NULL));
  }

  result = strdup(concatenate(result, QUOTE, gen_array_item(intern_indices_array, i), QUOTE, "),", NULL));

  result = strdup(concatenate(result, ")))", NL, NULL)); 
  return result;
  
}

static string claire_call_from_loopnest(call c, int task_number){
  entity called = call_function(c);
  list arguments = call_arguments(c);
  
  syntax s;
  string result = "";
  bool first = TRUE;
  string name = claire_entity_local_name(called);

  if(!same_string_p(name, "="))
    pips_user_error("only assignation allowed here");
  
  MAP(EXPRESSION, e, {
    s = expression_syntax(e);
    switch(syntax_tag(s)){
    case is_syntax_call:{
      if(first)
	pips_user_error("call not allowed in left-hand side argument of assignation");
      else
	result = strdup(concatenate(result, claire_call_from_assignation(syntax_call(s), task_number), NULL));
      break;
    }
    case is_syntax_reference:{
      
      reference r = syntax_reference(s);
      string varname = claire_entity_local_name(reference_variable(r));
      printf("%s\n", varname);
      if(gen_array_index(array_names, varname) != ITEM_NOT_IN_ARRAY){
	result = strdup(concatenate(result, claire_array_in_task(r, first, task_number), NULL));
      }
    }
    }
    first = FALSE;
  }, arguments);

  result = strdup(concatenate(result, TAB, ")", NL, NULL));
  return result;
}



static call claire_loop_from_loop(loop l, string * result, int task_number){
  
  string * up = malloc(sizeof(string));
  string * claire_name = malloc(sizeof(string));
  statement s = loop_body(l);
  instruction i = statement_instruction(s);

  *up = int_to_string(gen_array_nitems(extern_indices_array));
  *claire_name = claire_entity_local_name(loop_index(l));
  if( (*claire_name)[0] == 'M'){
    gen_array_append(intern_indices_array, claire_name);
  }
  else{
    gen_array_append(extern_indices_array, claire_name);
  }

  switch(instruction_tag(i)){
  case is_instruction_loop:{
    loop l = instruction_loop(i);
    return claire_loop_from_loop(l, result, task_number);
    break;
  }
  case is_instruction_call:{
    call c = instruction_call(i);
    *result = strdup(concatenate(*result, *up, ",", NL, TAB, TAB,NULL));
    return c;
  }
  default:
    pips_user_error("only loops and calls allowed in a loop");
  }
}


/* we enter a loopnest. the first loop is obligatory an extern loop */
static string claire_loop_from_sequence(loop l, int task_number){
  statement s = loop_body(l);
  call c;
  int i;
  /* (re-)initialize task-scoped arrays*/
  extern_indices_array = gen_array_make(0);
  intern_indices_array = gen_array_make(0);
  extern_upperbounds_array = gen_array_make(0);
  intern_upperbounds_array = gen_array_make(0);
  
  /* Initialize result string with the declaration of the task */
  string result = strdup(concatenate("T_", int_to_string(task_number), 
			      " :: TASK(unitSpentTime = vartype!(1),"
			      NL, TAB, "exLoopNest = LOOPNEST(deep = ", NULL));

  instruction ins = statement_instruction(s);
  list li = statement_declarations(s);

  string * name = malloc(sizeof(string));
  *name = claire_entity_local_name(loop_index(l));

  if((*name)[0] == 'M'){
    pips_user_error("At least one extern loop is needed");
  }
  else{
    string * up = malloc(sizeof(string));
    *up = claire_expression(range_upper(loop_range(l)));
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
  default:
    pips_user_error("only loops and calls allowed in a loop");
  }

  /* External loopnest depth*/
  result = strdup(concatenate(result, int_to_string(gen_array_nitems(extern_indices_array)), NL, TAB, TAB, NULL));
  /* add external upperbounds */
  result = strdup(concatenate(result, "upperBound = list<VARTYPE>(", NULL));
  for(i=0; i<gen_array_nitems(extern_upperbounds_array) - 1; i++){
    result = strdup(concatenate(result, "vartype!(", gen_array_item(extern_upperbounds_array, i), "), ", NULL));
  }
  result = strdup(concatenate(result, "vartype!(", gen_array_item(extern_upperbounds_array, i), ")),",NL, TAB, TAB, NULL));
  /* add external indices names*/
  result = strdup(concatenate(result, "names = list<string>(", NULL));
  for(i=0; i<gen_array_nitems(extern_indices_array) - 1; i++){
    result = strdup(concatenate(result, QUOTE, gen_array_item(extern_indices_array, i), QUOTE ", ", NULL));
  }
  result = strdup(concatenate(result, QUOTE, gen_array_item(extern_indices_array, i), QUOTE, ")),", NL, TAB, NULL));
  
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
    call c = instruction_call(i);
    entity called = call_function(c);
    string name = claire_entity_local_name(called);
    if(!same_string_p(name, "RETURN"))
      pips_user_error("Only RETURN call allowed here");
    break;
  }
  default:{
    pips_user_error("Only loops and calls allowed here");
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
    string current = claire_statement_from_sequence(s, task_number++);
    result = strdup(concatenate(oldresult, current, NULL));
    free(current);
    free(oldresult);
  }, sequence_statements(seq));
  return result;
}

/* Manages tasks. The code is very defensive and hangs if sth not
   predicted happens. Here basically we begin the code in itself
   and thus $stat is obligatory a sequence. */
static string claire_tasks(statement stat){
  string result = "tasks\n";
  instruction i = statement_instruction(stat);
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
  free(decls);
  free(tasks);

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
