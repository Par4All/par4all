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
