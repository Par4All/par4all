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

// strndup are GNU extensions...
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "constants.h"

#include "text-util.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "resources.h"
#include "phases.h"
#include "control.h"
#include "c_syntax.h"
#include "preprocessor.h"


/* High-level functions about modules, using pipsdbm and ri-util and
   some global variables assumed properly set
 */

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
  list dl = get_current_module_declarations();
  if (list_undefined_p(dl))
    {
      dl = code_declarations(value_code(entity_initial(m)));
      set_current_module_declarations(dl);
    }  

  /* FI: maybe we should also look up the declarations in the compilation unit... */

  ifdebug(9) {
    pips_debug(8, "Current module declarations:\n");
    print_entities(dl);
    fprintf(stderr, "\n");
  }

  return gen_copy_seq(dl);
}

list current_module_declarations()
{
  entity m = get_current_module_entity();
  return module_declarations(m);
}
/* Return a list of all variables and functions accessible somewhere in a module. */
list module_entities(entity m)
{
  entity cu = module_entity_to_compilation_unit_entity(m);
  list cudl = gen_copy_seq(code_declarations(value_code(entity_initial(cu))));
  list mdl = module_declarations(m);

  pips_assert("compilation unit is an entity list.",
	      entity_list_p(code_declarations(value_code(entity_initial(cu)))));
  pips_assert("initial cudl is an entity list.", entity_list_p(cudl));
  pips_assert("mdl is an entity list.", entity_list_p(mdl));

  cudl = gen_nconc(cudl, mdl);

  /* Make sure you only have entities in list cudl */
  pips_assert("Final cudl is an entity list.", entity_list_p(cudl));

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
const char* entity_module_unambiguous_user_name(entity e, entity m)
{
  const char* uan = entity_user_name(e);

  /* No problem in Fortran */
  if(c_module_p(m)) {
    list conflicts = module_entities(m);

    FOREACH(ENTITY, v, conflicts){
	if(v!=e) {
	  const char* vn = entity_user_name(v);
	  if(same_string_p(uan,vn)) {
	    uan = entity_local_name(e);
	    break;
	  }
	}
      }
    gen_free_list(conflicts);
  }

  return (uan);
}

const char* entity_unambiguous_user_name(entity e)
{
  entity m = get_current_module_entity();

  return entity_module_unambiguous_user_name(e, m);
}

/* In interprocedural context, returns the shortest non-ambiguous name
   for a variable. If it is local to the current module, use the user
   name. If not return entity_name(), which is not fully satisfying
   for C variables because it includes scope information.

   Note also that this function assumes the existence of a current module.

   FI: why is this function in preprocessor and not in ri-util?
*/
static const char* entity_more_or_less_minimal_name(entity e, bool strict_p)
{
  entity m = get_current_module_entity();
  const char* mln = module_local_name(m);
  const char* emn = string_undefined;
  string cun = string_undefined; // compilation unit name
  entity cu = entity_undefined; // compilation unit
  list cudl = list_undefined; // compilation unit declaration list
  list mdl = code_declarations(value_code(entity_initial(m)));

  if(c_module_p(m) && !compilation_unit_p(mln)) {
    /* in pipsmake library...*/
    string compilation_unit_of_module(const char*);
    cun = compilation_unit_of_module(mln);
    cu = FindEntity(TOP_LEVEL_MODULE_NAME, cun);
    cudl = code_declarations(value_code(entity_initial(cu)));
    free(cun);
  }

  pips_assert("some current entity", !entity_undefined_p(m));

  /* gen_in_list_p() would/should be sufficient... */
  if (strcmp(mln, entity_module_name(e)) == 0
      || gen_in_list_p(e, mdl)) {
    /* The variable is declared in the current module */
    //return global_name_to_user_name(entity_name(e));
    if(strict_p)
      emn = entity_unambiguous_user_name(e);
    else
      emn = entity_user_name(e);
  }
  else if (!list_undefined_p(cudl) && gen_in_list_p(e, cudl)) {
    /* The variable is declared in the compilation unit of the current
       module */
    //return global_name_to_user_name(entity_name(e));
    if(strict_p)
      emn = entity_unambiguous_user_name(e);
    else
      emn = entity_user_name(e);
  }
  else if (entity_field_p(e)) {
    /* The variable is a union or struct field. No need to
       disambiguate. */
    if(strict_p)
      emn = entity_unambiguous_user_name(e);
    else
      emn = entity_user_name(e);
  }
  else if (entity_formal_p(e)) {
    /* Formal parameters should always be unambiguous? */
    if(strict_p)
      emn = entity_unambiguous_user_name(e);
    else
      emn = entity_user_name(e);
  }
  else if (strstr(entity_module_name(e), DUMMY_PARAMETER_PREFIX)) {
    /* The variable is a dummy parameter entity, used in a function
       declaration */
    if(strict_p)
      emn = entity_local_name(e);
    else {
      /* In analysis results, let's know when dummy parameters are
	 used... */
      //emn = strdup(entity_local_name(e));
      emn = strdup(entity_name(e));
    }
  }
  else if (strcmp(TOP_LEVEL_MODULE_NAME, entity_module_name(e)) == 0) {
    /* The variable is a ??? */
    if(strict_p)
      emn = entity_local_name(e);
    else
      emn = strdup(entity_local_name(e));
  }
  else if (strcmp(REGIONS_MODULE_NAME, entity_module_name(e)) == 0) {
    /* The variable is a PHI entity */
    if(strict_p)
      emn = entity_local_name(e);
    else
      emn = strdup(entity_local_name(e));
  }
  else if (strcmp(POINTS_TO_MODULE_NAME, entity_module_name(e)) == 0) {
    /* The variable is a stub entity for formal and global pointers */
    if(strict_p)
      emn = entity_local_name(e);
    else
      emn = strdup(entity_local_name(e));
  }
  else {
    /* must be used to prettyprint interprocedural information... */
    //entity om = local_name_to_top_level_entity(mln);
    //emn = entity_interprocedural_unambiguous_user_name(e, om);
    if(strict_p)
      emn = entity_name(e);
    else
      emn = strdup(entity_name(e));
  }

  return emn;
}

/* Do preserve scope informations

   This function does not allocate a new string, which implies to keep
   the scope information in the name of extraprocedural variables and
   functions.
 */
const char* entity_minimal_name(entity e)
{
  return entity_more_or_less_minimal_name(e, true);
}

/* Do not preserve scope information

   A new string is allocated.
 */
const char* entity_minimal_user_name(entity e)
{
  return entity_more_or_less_minimal_name(e, false);
}

/* Retrieve the compilation unit containing a module definition.

   The implementation is clumsy.

   It would be nice to memoize the information as with
   get_current_module_entity().
*/
entity module_entity_to_compilation_unit_entity(entity m)
{
  entity cu = entity_undefined;

  if(compilation_unit_entity_p(m))
    cu = m;
  else {
    // string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), true);
    string aufn = db_get_memory_resource(DBR_USER_FILE, module_local_name(m), true);
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
      pips_internal_error("Not implemented yet");
  }
  pips_assert("cu is a compilation unit", compilation_unit_entity_p(cu));
  return cu;
}
bool language_module_p(entity m, string lid)
{
  bool c_p = false;

  if(entity_module_p(m)) {
    /* FI: does not work with static functions */
    //string aufn = db_get_memory_resource(DBR_USER_FILE, entity_user_name(m), true);
    /* SG: must check if the ressource exist (not always the case) */
    const char* lname= module_local_name(m);
    if( db_resource_p(DBR_USER_FILE,lname) )
    {
        string aufn = db_get_memory_resource(DBR_USER_FILE, module_local_name(m), true);
        string n = strstr(aufn, lid);

        c_p = (n!=NULL);
    }
    else
        c_p = true; /* SG: be positive ! (needed for Hpfc)*/
  }
  return c_p;
}

/** Add an entity to the current's module compilation unit declarations
 * we have to generate its statement if none cerated before
 * due to limitation of pipsmake, it is not always possible to make sure from pipsmake
 * that this ressource is created
 * for example in INLINING (!) we would like to tell pipsmake
 * we need the CODE resource from all module callers
 *
 * @param[in] e is the entity to add
 * @param[in] cu is the compilation unit
 */
void AddEntityToCompilationUnit(entity e, entity cu ) {
    statement s = statement_undefined;
    const char* cum = module_local_name(cu);
    if( c_module_p(cu) ) {
        if(!db_resource_required_or_available_p(DBR_PARSED_CODE,cum))
        {
            bool compilation_unit_parser(const char*);
            entity tmp = get_current_module_entity();
            statement stmt = get_current_module_statement();
            reset_current_module_entity();
            reset_current_module_statement();
            compilation_unit_parser(cum);
            if(!entity_undefined_p(tmp))
                set_current_module_entity(tmp);
            if(!statement_undefined_p(stmt))
                set_current_module_statement(stmt);
        }
        if(!db_resource_required_or_available_p(DBR_CODE,cum))
        {
            bool controlizer(const char*);
            entity tmp = get_current_module_entity();
            statement stmt = get_current_module_statement();
            reset_current_module_entity();
            reset_current_module_statement();
            controlizer(cum);
            if(!entity_undefined_p(tmp))
                set_current_module_entity(tmp);
            if(!statement_undefined_p(stmt))
                set_current_module_statement(stmt);
        }
        s=(statement)db_get_memory_resource(DBR_CODE,cum,true);
    }
    /* SG: when adding a new entity to compilation unit,
     * one should check the entity is not already present
     * but an entity with the same name may already be defined there
     * so check this with a very costly test*/
    list cu_entities = entity_declarations(cu);
    FOREACH(ENTITY,cue,cu_entities)
        if(same_string_p(entity_user_name(e),entity_user_name(cue)))
            return;
    AddLocalEntityToDeclarations(e,cu,s);
    if( c_module_p(cu) ) {
        module_reorder(s);
        db_put_or_update_memory_resource(DBR_CODE,cum,s,true);
        db_touch_resource(DBR_CODE,cum);
        if( typedef_entity_p(e) ) {
            keyword_typedef_table = (hash_table)db_get_memory_resource(DBR_DECLARATIONS, cum, true);
            put_new_typedef(entity_user_name(e));
            //SG: we have to do this behind the back of pipsmake. Not Good for serialization, but otherwise it forces the recompilation of the parsed_code of the associated modules, not good :(
            //DB_PUT_MEMORY_RESOURCE(DBR_DECLARATIONS, cum, keyword_typedef_table);
        }

    }
}

/** Remove an entity from the current's module compilation unit declarations
 *
 * @param[in] e is the entity to remove
 * @param[in] cu is the compilation unit
 */
void RemoveEntityFromCompilationUnit(entity e, entity cu ) {
    statement s = statement_undefined;
    const char* cum = module_local_name(cu);
    if( c_module_p(cu) ) {
        if(!db_resource_required_or_available_p(DBR_CODE,cum))
        {
            bool controlizer(const char*);
            entity tmp = get_current_module_entity();
            statement stmt = get_current_module_statement();
            reset_current_module_entity();
            reset_current_module_statement();
            controlizer(cum);
            if(!entity_undefined_p(tmp))
                set_current_module_entity(tmp);
            if(!statement_undefined_p(stmt))
                set_current_module_statement(stmt);
        }
        s=(statement)db_get_memory_resource(DBR_CODE,cum,true);
    }

    // Remove entity from global declaration lists
    gen_remove(&entity_declarations(cu),e);
    // FIXME : s is only defined for c_module !!
    gen_remove(&statement_declarations(s),e);

    remove_declaration_statement(s, e);
    if( c_module_p(cu) ) {
        module_reorder(s);
        db_put_or_update_memory_resource(DBR_CODE,cum,s,true);
        db_touch_resource(DBR_CODE,cum);
    }
}



void
AddEntityToModuleCompilationUnit(entity e, entity module)
{
    list tse = type_supporting_entities(NIL,entity_type(e));
    entity cu = module_entity_to_compilation_unit_entity(module);
    FOREACH(ENTITY,se,tse) {
        const char * eln = entity_local_name(se);
        if(strstr(eln, DUMMY_STRUCT_PREFIX) || strstr(eln, DUMMY_UNION_PREFIX) ) {
            continue;
        }
        AddEntityToCompilationUnit(se,cu);
    }
    gen_free_list(tse);
    AddEntityToCompilationUnit(e,cu);
}

static void do_recompile_module(entity module, statement module_statement) {

    /* build and register textual representation */
    text t = text_module(module, module_statement);
    //add_new_module_from_text(module,t,fortran_module_p(modified_module),compilation_unit_of_module(module));
    string dirname = db_get_current_workspace_directory();
    string res = fortran_module_p(module)? DBR_INITIAL_FILE : DBR_C_SOURCE_FILE;
    string filename = db_get_file_resource(res,module_local_name(module),true);
    string fullname = strdup(concatenate(dirname, "/",filename, NULL));
    FILE* f = safe_fopen(fullname,"w");
    free(fullname);
    print_text(f,t);
    fclose(f);
    free_text(t);

    DB_PUT_FILE_RESOURCE(res,module_local_name(module),filename);

    /* the module will be reparsed, so fix(=delete) all its previous entites */
#if 0
    {
        list p = NIL;
        FOREACH(ENTITY, e, entity_declarations(modified_module))
        {
            if( recompile_module_removable_entity_p((gen_chunkp)e))
                gen_clear_tabulated_element((gen_chunk*)e);
            else
                p = CONS(ENTITY,e,p);
        }
        entity_declarations(modified_module) = gen_nreverse(p);
        code_initializations(value_code(entity_initial(modified_module)))=make_sequence(NIL);
    }
#endif
}


/* build a textual representation of the modified module and update db
 */
bool
recompile_module(const char* module)
{
    entity modified_module = module_name_to_entity(module);
    statement modified_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module, true);
    do_recompile_module(modified_module,modified_module_statement);
    return true;


}
