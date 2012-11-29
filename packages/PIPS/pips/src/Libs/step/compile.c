/* Copyright 2007-2012 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

#include "pipsmake.h" // for compilation_unit_of_module



#define STEP_GENERATED_SUFFIX_F ".step_generated.f"
#define STEP_GENERATED_SUFFIX_C ".step_generated.c"

extern statement compile_mpi(statement stmt, string new_module_name, step_directive drt, int transformation);


/*
 *
 compile_omp
 *
 */

statement compile_omp(statement stmt, step_directive d)
{
  pips_debug(1, "begin\n");

  string begin_txt, end_txt;
  bool is_fortran = fortran_module_p(get_current_module_entity());
  bool is_block_construct = step_directive_to_strings(d, is_fortran, &begin_txt, &end_txt);

  if(!string_undefined_p(begin_txt))
    {
      if(!is_block_construct)
	{
	  if(ENDP(statement_block(stmt)))
	    insert_statement(stmt, make_plain_continue_statement(), false);
	  stmt = STATEMENT(CAR(statement_block(stmt)));
	}
      add_pragma_str_to_statement(stmt, begin_txt, false);
    }

  if(!string_undefined_p(end_txt))
    {
      insert_statement(stmt, make_plain_continue_statement(), false);
      stmt = last_statement(stmt);
      add_pragma_str_to_statement(stmt, end_txt, false);
    }

  pips_debug(1, "end\n");
  return stmt;
}

void add_omp_guard(statement *block)
{
  pips_assert("block", block && !statement_undefined_p(*block) && statement_block_p(*block));
  pips_debug(1, "begin\n");

  if (!empty_statement_or_continue_p(*block))
    {
      statement barrier_stmt = make_empty_block_statement();
      step_directive barrier_guard = make_step_directive(STEP_BARRIER, statement_undefined, NIL);
      step_directive master_guard = make_step_directive(STEP_MASTER, statement_undefined, NIL);
      compile_omp(barrier_stmt, barrier_guard);
      compile_omp(*block, master_guard);
      free_step_directive(barrier_guard);
      free_step_directive(master_guard);

      *block = make_block_statement(CONS(STATEMENT, *block, CONS(STATEMENT, barrier_stmt, NIL)));
    }

  pips_debug(1, "end\n");
}


/*
  Suppression du pragma string "STEP" ajoute par step_directive
*/
static statement remove_STEP_pragma(statement stmt)
{
  list ext_l = NIL;

  pips_debug(1, "begin\n");

  FOREACH(EXTENSION, ext, extensions_extension(statement_extensions(stmt)))
    {
      pragma p = extension_pragma(ext);
      if(pragma_string_p(p) && strncmp(pragma_string(p), STEP_SENTINELLE, strlen(STEP_SENTINELLE))==0)
	{
	  pips_debug(2,"drop pragma : %s\n", pragma_string(p));
	  free_extension(ext);
	}
      else
	ext_l = CONS(EXTENSION, ext, ext_l);
    }
  gen_free_list(extensions_extension(statement_extensions(stmt)));
  extensions_extension(statement_extensions(stmt)) = gen_nreverse(ext_l);

  pips_debug(1, "end\n");
  return stmt;
}

/*
  Compute the new module name
  Create the new module entity (empty body)

  No module is created if OpenMP transformation only
*/


static void get_module_name_directive_suffix(step_directive drt, string *directive_txt)
{

  pips_debug(1, "begin\n");
  bool is_fortran = fortran_module_p(get_current_module_entity());

  switch(step_directive_type(drt))
    {
    case STEP_PARALLEL:
      *directive_txt = strdup("PAR");
      break;
    case STEP_DO:
      *directive_txt = strdup(is_fortran?"DO":"FOR");
      break;
    case STEP_PARALLEL_DO:
      *directive_txt = strdup(is_fortran?"PARDO":"PARFOR");
      break;
    case STEP_MASTER:
      *directive_txt = strdup("MASTER");
      break;
    case STEP_SINGLE:
      *directive_txt = strdup("SINGLE");
      break;
    case STEP_BARRIER:
      *directive_txt = strdup("BARRIER");
      break;
    default: assert(0);
    }

  pips_debug(1, "end *directive_txt = %s\n", *directive_txt);
}

static int get_directive_transformation_type(step_directive drt)
{
  int transformation_type = -1;
  pips_debug(2, "begin\n");

  FOREACH(STEP_CLAUSE, c, step_directive_clauses(drt))
    {
      /* que se passe t-il si plusieurs clauses de transformation ?*/

      if(step_clause_transformation_p(c))
	transformation_type = step_clause_transformation(c);
    }
  
  pips_debug(2,"end transformation_type : %d\n", transformation_type);
  
  return transformation_type;
}

static int get_directive_transformation(step_directive drt, string *transformation_txt)
{
  int transformation = -1;
  pips_debug(1, "begin\n");

  transformation = get_directive_transformation_type(drt);
  
   switch (transformation)
    {
    case STEP_TRANSFORMATION_MPI:
      *transformation_txt = strdup("MPI");
      break;
    case STEP_TRANSFORMATION_HYBRID:
      *transformation_txt = strdup("HYBRID");
      break;
    case STEP_TRANSFORMATION_SEQ:
    case STEP_TRANSFORMATION_OMP:
      *transformation_txt = strdup("");
      break;
    default:
      assert(0);
    }

   pips_debug(1, "end transformation = %d\n", transformation);
   return transformation;
}

static entity create_new_module_entity (list *last_module_name, string directive_txt, string transformation_txt)
{
  string previous_name, new_module_name;
  string prefix;
  entity new_module;

  previous_name = STRING(CAR(*last_module_name));

  assert(asprintf(&prefix,"%s_%s_%s", previous_name, directive_txt, transformation_txt)>=0);
  new_module_name = build_new_top_level_module_name(prefix, true);
  new_module = make_empty_subroutine(new_module_name, copy_language(module_language(get_current_module_entity())));

  free(prefix);

  *last_module_name = CONS(STRING, new_module_name, *last_module_name);
  pips_debug(1, "new_module %p : %s\n", new_module, new_module_name);
  return new_module;
}

static bool compile_filter(statement stmt, list *last_module_name)
{
  string directive_txt, transformation_txt;
  step_directive drt;
  int transformation_type;

  if(!step_directives_bound_p(stmt))
    return true;

  pips_debug(1, "begin\n");

  drt = step_directives_load(stmt);
  get_module_name_directive_suffix(drt, &directive_txt);
  transformation_type = get_directive_transformation(drt, &transformation_txt);

  if (transformation_type == STEP_TRANSFORMATION_MPI || transformation_type == STEP_TRANSFORMATION_HYBRID)
    {
      entity new_module = create_new_module_entity(last_module_name, directive_txt, transformation_txt);
      pips_debug(1, "New entity module created : %s\n", entity_name(new_module));
    }


  free(directive_txt);
  free(transformation_txt);

  pips_debug(1, "end\n");
  return true;
}


static void compile_rewrite(statement stmt, list *last_module_name)
{

  int transformation;
  string new_module_name;
  step_directive drt;

  pips_debug(1, "begin\n");
  
  if(!step_directives_bound_p(stmt))
    return;
  
  new_module_name = STRING(CAR(*last_module_name));
  pips_debug(1, "stack_name current name = %s\n", new_module_name);

  drt = step_directives_load(stmt);
  ifdebug(3)
    step_directive_print(drt);
  
  remove_STEP_pragma(stmt);

  transformation = get_directive_transformation_type(drt);

  switch( transformation )
    {
    case STEP_TRANSFORMATION_SEQ:
      break;
    case STEP_TRANSFORMATION_OMP:
      compile_omp(stmt, drt);
      break;
    case STEP_TRANSFORMATION_MPI:
    case STEP_TRANSFORMATION_HYBRID:
      compile_mpi(stmt, new_module_name, drt, transformation);
      POP(*last_module_name);
      break;
    default:
      assert(0);
    }

  pips_debug(1, "end\n");
}


void step_compile_analysed_module(const char* module_name, string finit_name)
{
  pips_debug(1, "begin\n");

  entity module = local_name_to_top_level_entity(module_name);
  statement stmt = (statement)db_get_memory_resource(DBR_CODE, module_name, false);

  set_current_module_entity(module);
  set_current_module_statement(stmt);

  step_directives_init(0);
  load_step_comm();

  statement_effects rw_effects = (statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, false);
  statement_effects cummulated_rw_effects = (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, false);
  set_rw_effects(rw_effects);
  set_cumulated_rw_effects(cummulated_rw_effects);

  /* Code transformation */
  list last_module_name = CONS(STRING, (string)module_name, NIL);

  gen_context_recurse(stmt, &last_module_name, statement_domain, compile_filter, compile_rewrite);

  if (entity_main_module_p(module))
    {
      string init_subroutine_name;
      init_subroutine_name = fortran_module_p(get_current_module_entity())?
	RT_STEP_init_fortran_order:
	RT_STEP_init_c_order;
      statement init_stmt = call_STEP_subroutine2(init_subroutine_name, NULL);
      statement finalize_stmt = call_STEP_subroutine2(RT_STEP_finalize, NULL);

      insert_statement(stmt, init_stmt, true);
      insert_statement(last_statement(stmt), finalize_stmt, true);
    }

  reset_cumulated_rw_effects();
  reset_rw_effects();
  free_statement_effects(cummulated_rw_effects);
  free_statement_effects(rw_effects);
  
  reset_step_comm();
  step_directives_reset();
  reset_current_module_statement();
  reset_current_module_entity();
  
  /* File generation */
  text code_txt = text_named_module(module, module, stmt);
  bool saved_b1 = get_bool_property("PRETTYPRINT_ALL_DECLARATIONS");
  bool saved_b2 = get_bool_property("PRETTYPRINT_STATEMENT_NUMBER");
  set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", true);
  set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", false);
  
  FILE *f = safe_fopen(finit_name, "w");
  print_text(f, code_txt);
  safe_fclose(f, finit_name);
  
  set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", saved_b1);
  set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", saved_b2);
  free_text(code_txt);
  free_statement(stmt);
  pips_debug(1, "end\n");
}


/* generated source: no analyse and no compilation necessary. Keep the source as it is. */
void step_compile_generated_module(const char* module_name, string finit_name)
{
  pips_debug(1, "begin\n");
  
  string source_file, fsource_file;
  entity module = local_name_to_top_level_entity(module_name);
  bool is_fortran = fortran_module_p(module);

  source_file = db_get_memory_resource(is_fortran?DBR_INITIAL_FILE:DBR_C_SOURCE_FILE, module_name, true);

  assert(asprintf(&fsource_file,"%s/%s" , db_get_current_workspace_directory(), source_file)>=0);
  pips_debug(1, "Copie from: %s\n", fsource_file);
  safe_copy(fsource_file, finit_name);
  
  if(compilation_unit_p(module_name))
    {
      safe_system(concatenate("echo '#include \"step_api.h\"' >>",finit_name,  NULL));
      pips_debug(1, "\tAdd prototype for STEP generated modules\n");
      gen_array_t modules = db_get_module_list_initial_order();
      int n = gen_array_nitems(modules), i;
      FILE *out = safe_fopen(finit_name, "a");
      for (i=0; i<n; i++)
	{
	  string name = gen_array_item(modules, i);
	  if (same_string_p((const char*)module_name, (const char*)compilation_unit_of_module(name)) &&
	      !same_string_p(module_name, name) &&
	      !step_analysed_module_p(name))
	    {
	      statement stmt = make_plain_continue_statement();
	      statement_declarations(stmt) = CONS(ENTITY, module_name_to_entity(name), NIL);
	      text txt = text_statement(entity_undefined, 0, stmt, NIL);
	      print_text(out, txt);
	      free_text(txt);
	      free_statement(stmt);
	    }
	}
      safe_fclose(out, finit_name);
    }
  pips_debug(1, "end\n");
}

bool step_compile(const char* module_name)
{
  debug_on("STEP_COMPILE_DEBUG_LEVEL");
  pips_debug(1, "Begin considering module_name = %s\n", module_name);

  entity module = local_name_to_top_level_entity(module_name);

  bool is_fortran = fortran_module_p(module);
  string init_name, finit_name;
  
  init_name = db_build_file_resource_name(DBR_STEP_FILE, module_name, is_fortran?  STEP_GENERATED_SUFFIX_F :  STEP_GENERATED_SUFFIX_C);
  assert(asprintf(&finit_name,"%s/%s" , db_get_current_workspace_directory(), init_name)>=0);

  if(step_analysed_module_p(module_name))
    {
      /* analysed source : let's do the transformations */
      step_compile_analysed_module(module_name, finit_name);
    }
  else
    {
      /* generated module: nothing to do but copy the generated source */
      step_compile_generated_module(module_name, finit_name);
    }

  DB_PUT_FILE_RESOURCE(DBR_STEP_FILE, module_name, finit_name);

  pips_debug(1, "End\n");
  debug_off();
  return true;
}
