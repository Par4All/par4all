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
#include "pipsmake.h"
#include <strings.h>
#include <string.h>


bool step_init(string program_name)
{
  string srcpath;

  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "program_name = %s\n", program_name);
  
  set_bool_property("PARSER_WARN_FOR_COLUMNS_73_80", FALSE);
  //  set_bool_property("PRETTYPRINT_IO_EFFECTS", FALSE);
  
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_STATUS, "", make_step_status(NIL,NIL));
  global_step_analyse_init();

  srcpath=strdup(concatenate(getenv("PIPS_ROOT"),"/",STEP_DEFAULT_RT_H,NULL));
  string old_path=pips_srcpath_append(srcpath);
  free(old_path);
  free(srcpath);

  debug_off();
  return TRUE;
}


bool step_install(string program_name)
{
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering %s\n", program_name);
  
  string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  string dest_dir = strdup(get_string_property("STEP_INSTALL_PATH"));
  string runtime = strdup(get_string_property("STEP_RUNTIME"));

  if (empty_string_p(dest_dir))
    {
      free(dest_dir);
      dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    }

  // concatenation des PRINTED_FILE selon les USER_FILE
  unsplit(NULL);

  // suppression du fichier regroupant les modules directives
  string directives_file_name = step_directives_USER_FILE_name();
  safe_system(concatenate("rm ", src_dir, "/$(basename ",directives_file_name ,")",  NULL));
  free(directives_file_name);

  // installation des fichiers générés
  safe_system(concatenate("step_install ", runtime ," ", dest_dir, " ", src_dir,  NULL));

  free(src_dir);
  free(dest_dir);
  free(runtime);
  
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
