/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "defines-local.h"
#include "step_private.h"
#include "preprocessor.h"
#include "makefile.h"
#include "pipsmake.h"
#include <strings.h>
#include <string.h>


static string saved_srcpath=string_undefined;

static step_status step_status_g = (step_status) NULL;

GENERIC_GLOBAL_FUNCTION(step_analyse_map,analyse_map);
GENERIC_GLOBAL_FUNCTION(step_mpi_module_map,step_entity_map);
GENERIC_GLOBAL_FUNCTION(step_omp_module_map,step_entity_map);

static void step_init_status()
{
  init_step_analyse_map();
  init_step_mpi_module_map();
  init_step_omp_module_map();
  step_status_g = make_step_status(get_step_analyse_map(),get_step_mpi_module_map(),get_step_omp_module_map());
}

/*les libérations mémoires sont faites par pipsdbm... */
static void step_reset_status()
{
  reset_step_analyse_map();
  reset_step_mpi_module_map();
  reset_step_omp_module_map();
  step_status_g = NULL;
}

void step_save_status()
{
  pips_assert("some current status",step_status_g);
  step_status_analyses(step_status_g) = get_step_analyse_map();
  step_status_mpi_module(step_status_g) = get_step_mpi_module_map();
  step_status_omp_module(step_status_g) = get_step_omp_module_map();
  
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_STATUS, "", step_status_g);
  step_reset_status();
}

void step_load_status()
{
  step_status_g = (step_status)db_get_memory_resource(DBR_STEP_STATUS, "", TRUE);
  set_step_analyse_map(step_status_analyses(step_status_g));
  set_step_mpi_module_map(step_status_mpi_module(step_status_g));
  set_step_omp_module_map(step_status_omp_module(step_status_g));
}

bool step_init(string program_name)
{
  string srcpath;

  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "program_name = %s\n", program_name);
  
  set_bool_property("PARSER_WARN_FOR_COLUMNS_73_80", FALSE);
  // set_bool_property("PRETTYPRINT_IO_EFFECTS", FALSE);
  
  step_init_status();
  step_save_status();

  srcpath=strdup(concatenate(getenv("PIPS_ROOT"),"/",STEP_DEFAULT_RT_H,NULL));
  saved_srcpath=pips_srcpath_append(srcpath);
  free(srcpath);

  debug_off();
  return TRUE;
}


bool step_install(string program_name)
{
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering %s\n", program_name);
  
  string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  string dest_dir = get_string_property("STEP_INSTALL_PATH");

  if (empty_string_p(dest_dir))
    dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);

  // concatenation des PRINTED_FILE selon les USER_FILE
  unsplit(NULL);

  // filtrage des 'include "STEP.h"' et copie des fichiers sources
  safe_system(concatenate("step_install ", dest_dir, " ", src_dir , "/*",  NULL));
  
  // mise en place de STEP.f et STEP.h
  safe_system(concatenate("cp $PIPS_ROOT/share/STEP.[fh] ", dest_dir, NULL));

  free(src_dir);
  free(dest_dir);
  
  pips_debug(1, "fin step_install\n");
  debug_off();
  return TRUE;
}

void step_print_code(FILE* file, entity module, statement statmt)
{
    text t;
    pips_debug(1, "statmt = %p   module = %p\n", statmt, module);
   
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    t = text_module(module, statmt);
    print_text(file, t);
    free_text(t);
    
    debug_off();
}


entity expr_to_entity(e)
expression e;
{
    syntax s = expression_syntax(e);
    
    switch (syntax_tag(s))
    {
    case is_syntax_call:
	return call_function(syntax_call(s));
    case is_syntax_reference:
	return reference_variable(syntax_reference(s));
    case is_syntax_range:
    default: 
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
	return entity_undefined; 
    }
}

expression entity_to_expr(e)
     entity e;
{
  type t=entity_type(e);
  syntax s;
  expression exp;
  switch (type_tag(t)){
  case is_type_variable:
    s=make_syntax(is_syntax_reference,make_reference(e,NIL));
    exp=make_expression(s,normalized_undefined);
    break;
  case is_type_functional:
    s=make_syntax(is_syntax_call,make_call(e,NIL));
    exp=make_expression(s,normalized_undefined);
    break;
  default:
    pips_internal_error("unexpected entity tag: %d\n", type_tag(t));
    return expression_undefined; 
  }
  return exp;
}


/* non utilise */
void step_instruction_print(instruction instr)
{

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      printf("%s: %s: instruction sequence\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_test:
      printf("%s: %s: instruction test\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_loop:
      printf("%s: %s: instruction loop\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_whileloop:
      printf("%s: %s: instruction whileloop\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_goto:
      printf("%s: %s: instruction goto\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_call:
      printf("%s: %s: instruction call\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_unstructured:
      printf("%s: %s: instruction unstructured\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_multitest:
      printf("%s: %s: instruction multitest\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_forloop:
      printf("%s: %s: instruction forloop\n",__FILE__, __FUNCTION__);
      break;
    case is_instruction_expression:
      printf("%s: %s: instruction expression\n",__FILE__, __FUNCTION__);
      break;
    }
}

string step_remove_quote(string s)
{
  char *d=index(s,'\'');
  char *f=rindex(s,'\'');
  int n=f-d;
  char *dest=malloc(n*sizeof(char));
  strncpy(dest,d+1,n-1);
  dest[n-1]=0;
  string result=strdup(dest);
  free(dest);
  return result;
}

string step_upperise(string s)
{
  int i,len=strlen(s);
  for (i=0;i<len;i++)
    s[i]=toupper(s[i]);
  return s;
}
