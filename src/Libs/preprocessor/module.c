/* 
 * $Id$
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"

#include "text-util.h"
#include "misc.h"
#include "properties.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "resources.h"
#include "phases.h"
#include "preprocessor.h"

/* High-level functions about modules, using pipsdbm and ri-util and some global variables assumed properly set */

/* Retrieve all declarations linked to a module, but the local
   variables private to loops. Allocate and build a new list which
   will have to be freed by the caller. */

static list current_module_declaration_list = list_undefined;

static void add_local_statement_declarations(statement s)
{
  current_module_declaration_list = gen_nconc(current_module_declaration_list,
					      gen_copy_seq(statement_declarations(s)));
}

list current_module_declarations()
{
  entity m = get_current_module_entity();
  statement s = get_current_module_statement();
  list dl = gen_copy_seq(code_declarations(value_code(entity_initial(m))));

  current_module_declaration_list = NIL;

    gen_multi_recurse
      (s, 
       statement_domain, add_local_statement_declarations, gen_null,
       NULL);

  dl = gen_nconc(dl, current_module_declaration_list);

  /* FI: maybe we should also look up the declarations in the compilation unit... */

  ifdebug(8) {
    pips_debug(8, "Current module declarations:\n");
    print_entities(dl);
    fprintf(stderr, "\n");
  }

  return dl;
}

entity module_entity_to_compilation_unit_entity(entity m)
{
  entity cu = entity_undefined;

  if(compilation_unit_entity_p(m)) 
    cu = m;
  else {
    string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), TRUE);
    string lufn = strrchr(aufn, '/')+1;

    if(lufn!=NULL) {
      string n = strstr(lufn, ".cpp_processed.c");
      int l = n-lufn;
      string cun = strndup(lufn, l);

      if(static_module_name_p(cun)) {
	string end = strrchr(cun, FILE_SEP_CHAR);
	*(end+1) = '\0';
	cu = local_name_to_top_level_entity(cun);
      }
      else {
	string ncun = strdup(concatenate(cun, FILE_SEP_STRING, NULL));
	cu = local_name_to_top_level_entity(ncun);
	free(ncun);
      }
      free(cun);
    }
    else
      pips_internal_error("Not implemented yet\n");
  }
  pips_assert("cu is a compilation unit", compilation_unit_entity_p(cu));
  return cu;
}

bool c_module_p(entity m)
{
  bool c_p = FALSE;

  if(entity_module_p(m)) {
    string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), TRUE);
    string n = strstr(aufn, ".cpp_processed.c");

    c_p = (n!=NULL);
  }
  return c_p;
}
