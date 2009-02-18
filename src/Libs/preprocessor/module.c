/*
 * $Id$
 */
#include <stdio.h>

// strndup are GNU extensions...
#define _GNU_SOURCE
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

/* High-level functions about modules, using pipsdbm and ri-util and
   some global variables assumed properly set
 */

static list current_module_declaration_list = list_undefined;

static void add_local_statement_declarations(statement s)
{
  current_module_declaration_list = gen_nconc(current_module_declaration_list,
					      gen_copy_seq(statement_declarations(s)));
}

/* Retrieve all declarations linked to a module, but the local
   variables private to loops. Allocate and build a new list which
   will have to be freed by the caller.

   This function has been implemented twice.

   It may be useless because code_declarations() is supposed to
   contain all module declarations, regardless of where the
   declarations happens.
 */
list module_declarations(entity m)
{
  statement s = get_current_module_statement();
  list dl = gen_copy_seq(code_declarations(value_code(entity_initial(m))));

  current_module_declaration_list = NIL;

    gen_multi_recurse
      (s,
       statement_domain, add_local_statement_declarations, gen_null,
       NULL);

  dl = gen_nconc(dl, current_module_declaration_list);

  /* FI: maybe we should also look up the declarations in the compilation unit... */

  ifdebug(9) {
    pips_debug(8, "Current module declarations:\n");
    print_entities(dl);
    fprintf(stderr, "\n");
  }

  return dl;
}

list current_module_declarations()
{
  entity m = get_current_module_entity();
  return module_declarations(m);
}

/* Retrieve the compilation unit containing a module definition.

   The implementation is clumsy.

   It would be nice to memoize the informtion as with
   get_current_module_entity().
*/
entity module_entity_to_compilation_unit_entity(entity m)
{
  entity cu = entity_undefined;

  if(compilation_unit_entity_p(m))
    cu = m;
  else {
    // string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), TRUE);
    string aufn = db_get_memory_resource(DBR_USER_FILE, module_local_name(m), TRUE);
    string lufn = strrchr(aufn, '/')+1;

    if(lufn!=NULL) {
      string n = strstr(lufn, PP_C_ED);
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

bool language_module_p(entity m, string lid)
{
  bool c_p = FALSE;

  if(entity_module_p(m)) {
    /* FI: does not work with static functions */
    //string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), TRUE);
    string aufn = db_get_memory_resource(DBR_USER_FILE, module_local_name(m), TRUE);
    string n = strstr(aufn, lid);

    c_p = (n!=NULL);
  }
  return c_p;
}

bool c_module_p(entity m)
{
  return language_module_p(m, PP_C_ED);
}

bool fortran_module_p(entity m)
{
  return language_module_p(m, FORTRAN_FILE_SUFFIX);
}

/* Return a list of all variables and functions accessible somewhere in a module. */
list module_entities(entity m)
{
  entity cu = module_entity_to_compilation_unit_entity(m);
  list cudl = gen_copy_seq(code_declarations(value_code(entity_initial(cu))));
  list mdl = module_declarations(m);

  cudl = gen_nconc(cudl, mdl);
  return cudl;
}

/* Based on entity_user_name, but preserves scoping information when
   needed to avoid ambiguity between two or more variables with the
   same name.

   A variable or function name can be used for an external or static
   entity in the compilation unit, it can be used as function
   parameter, and then declared within the function at any scope
   level.

   This function cannot directly be used in an interprocedural setting
   as it is only related to one module.

   It is assumed that no new string is allocated, but a pointer to an
   existing string is returned.

   The current implementation is extremely ineffective, especially for
   a very unlikely, although mystifying, problem.
*/
string entity_module_unambiguous_user_name(entity e, entity m)
{
  string uan = entity_user_name(e);

  /* No problem in Fortran */
  if(c_module_p(m)) {
    list conflicts = module_entities(m);

    MAP(ENTITY, v, {
	if(v!=e) {
	  string vn = entity_user_name(v);
	  if(same_string_p(uan,vn)) {
	    uan = entity_local_name(e);
	    break;
	  }
	}
      }, conflicts);
    gen_free_list(conflicts);
  }

  return (uan);
}

string entity_unambiguous_user_name(entity e)
{
  entity m = get_current_module_entity();

  return entity_module_unambiguous_user_name(e, m);
}

/* In interprocedural context, returns the shortest non-ambiguous name
   for a variable. If it is local to the current module, use the user
   name. If not return entity_name(), which is not fully satisfying
   for C variables because it includes scope information.

   Note also that this function assumes the existence of a current module.

   This function does not allocate a new string, which implies to keep
   the scope information in the name of extraprocedural variables and
   functions.
*/
string
entity_minimal_name(entity e)
{
  entity m = get_current_module_entity();
  string mln = module_local_name(m);
  string emn = string_undefined;

  pips_assert("some current entity", !entity_undefined_p(m));

  if (strcmp(mln, entity_module_name(e)) == 0) {
    //return global_name_to_user_name(entity_name(e));
    emn = entity_unambiguous_user_name(e);
  }
  else {
    //entity om = local_name_to_top_level_entity(mln);
    //emn = entity_interprocedural_unambiguous_user_name(e, om);
    emn = entity_name(e);
  }

  return emn;
}
