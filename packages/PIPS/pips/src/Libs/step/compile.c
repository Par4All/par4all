/* Copyright 2007, 2008, 2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

/*
  Genere et met en place la fonction contenant le code MPI
  et un appel a la fonction precedemment outlinee

  IN: les resultats du module analyse c'est a dire la liste de regions SEND 


*/
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include "defines-local.h"
#include "effects-convex.h"
#include "alias_private.h"
#include "preprocessor.h"

#define LOCAL_DEBUG 2
#define MAX_NB_CRITICALS 10

static step_status status = (step_status) NULL;
entity current_module_save;
statement body_critical[MAX_NB_CRITICALS] ;
const char* name_critical[MAX_NB_CRITICALS];
int nb_critical;
/*les liberations memoires sont faites par pipsdbm... */
static void save_status()
{
  pips_assert("some current status",status);
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_STATUS, "", status);
  status = NULL;
}

static void load_status()
{
  status = (step_status)db_get_memory_resource(DBR_STEP_STATUS, "", true);
}

/*############################################################################################*/

/*
  Duplication du prototype

  Parcours de la liste de declarations passee en parametre
  Creation d'une nouvelle entite pour chaque declaration

  Les nouvelles entites sont ajoutees dans la liste des variables
  formelles de la nouvelle fonction
*/
void step_add_formal_copy(entity mpi_module, list declaration)
{
  int ith;
  functional ft;
  list formal=NIL;
  pips_debug(1, "mpi_module = %p, declaration = %p\n", mpi_module, declaration);
  
  FOREACH(ENTITY, e,declaration) 
    {
      if(entity_formal_p(e))
	{
	   entity new=FindOrCreateEntity(entity_user_name(mpi_module),entity_user_name(e));
	   entity_type(new)=copy_type(entity_type(e));
	   entity_initial(new)=copy_value(entity_initial(e));
	   formal=gen_once(new,formal);
	}
      if (entity_function_p(e))
	{
	   AddEntityToDeclarations(e,mpi_module);
	}
    }
  
  ith = gen_length(formal);
  ft = type_functional(entity_type(mpi_module));

  FOREACH(ENTITY, v,formal) 
  {
      entity_storage(v)=make_storage(is_storage_formal, make_formal(mpi_module,ith--));
      pips_debug(8,"variable %s #%i\n",entity_name(v),ith+1);
      AddEntityToDeclarations(v,mpi_module);
      functional_parameters(ft) =
	CONS(PARAMETER,make_parameter(entity_type(v), 
				      MakeModeReference(),make_dummy_identifier(v)), functional_parameters(ft));
    }

  gen_free_list(formal);

  pips_debug(1, "End\n");
}

/*############################################################################################*/
string step_find_new_module_name(entity original, int step_transformation)
{
  int id=0;
  entity e;
  const char* original_name=entity_user_name(original);
  string suffix,newname;
  pips_debug(1,"original_name = %s\n", original_name);

  switch(step_transformation)
    {
    case STEP_TRANSFORMATION_OMP:
      suffix = STEP_OMP_SUFFIX;
      break;
    case STEP_TRANSFORMATION_MPI:
      suffix = STEP_MPI_SUFFIX;
      break;
    case STEP_TRANSFORMATION_HYBRID :
      suffix = STEP_HYB_SUFFIX;
      break;
    case STEP_TRANSFORMATION_SEQ :
      suffix = "";
    default:
      pips_internal_error("unexpected transformation :%d",step_transformation);
    }
  newname = strdup(concatenate(original_name,suffix,NULL));
  
  e = gen_find_tabulated(newname, entity_domain);
  while(!entity_undefined_p(e))
    {
      free(newname);
      id++; 
      newname=strdup(concatenate(original_name,"_",i2a(id),NULL));
      e = gen_find_tabulated(newname, entity_domain);
    }
  
  pips_debug(1,"newname = %s\n",newname);
  return newname;
}

/*############################################################################################*/
static string step_head_hook(entity __attribute__ ((unused)) e) 
{
  pips_debug(1, "step_head_hook\n");
  
  return strdup(concatenate
		("      implicit none\n",
		 "      include \"STEP.h\"\n", NULL));
}

statement step_guard_hybride(statement s)
{
  if (!empty_statement_p(s))
    {
      statement s_begin = make_continue_statement(entity_empty_label());
      statement_comments(s_begin) = strdup("\n!$omp master\n");
      statement s_end = make_continue_statement(entity_empty_label());
      statement_comments(s_end) = strdup("\n!$omp end master\n!$omp barrier\n\n");
      return make_block_statement(CONS(STATEMENT,s_begin,
				       CONS(STATEMENT,s,
					    CONS(STATEMENT,s_end,NIL))));
    }
  else
    return s;
}

static entity compile_module(entity directive_module, int step_transformation)
{
  directive d;
  int i,stop;
  entity current_module_save;
  string new_name;
  entity new_module;
  statement body;
  pips_debug(1, "directive_module = %p, step_transformation = %d\n", directive_module, step_transformation);

  pips_assert("current module defined",!entity_undefined_p(directive_module));
  pips_assert("global_directives loaded",!global_directives_undefined_p());
  pips_assert("first compilation",!gen_in_list_p(directive_module,step_status_compiled(status)));

  d = load_global_directives(directive_module);
  pips_debug(1,"Directive module %s : %s\n", entity_local_name(directive_module), directive_txt(d));
  pips_debug(2, "transformation %i\n", step_transformation);
  printf("Directive module  : %s\n", entity_local_name(directive_module));
  current_module_save = get_current_module_entity();
  if(!entity_undefined_p(current_module_save))
    reset_current_module_entity();
  set_current_module_entity(directive_module);

  new_name = step_find_new_module_name(directive_module,step_transformation);
  new_module = make_empty_subroutine(new_name,copy_language(module_language(get_current_module_entity())));
  step_add_formal_copy(new_module,code_declarations(value_code(entity_initial(directive_module))));

  /* initialisation du futur body par le code du module de directive */
  body = copy_statement((statement)db_get_memory_resource(DBR_CODE, entity_local_name(directive_module), true));
  { /* elimination du statement "return" */
    statement statmt=find_last_statement(body);
    if (!statement_undefined_p(statmt))
      {
	free_instruction(statement_instruction(statmt));
	statement_instruction(statmt)=make_continue_instruction();
	statement_label(statmt)=entity_empty_label();
      }
  }

  /* modification du body selon la transformation (OMP, MPI, HYBRID) */
  global_step_analyses_load();
  switch(type_directive_tag(directive_type(d)))
    {
    case is_type_directive_omp_parallel:
      stop=0;i=0;
      statement b_critical;	
      for(i=0;i<nb_critical;i++)
	{
	  if(strstr(name_critical[i],entity_local_name(directive_module))!=NULL)
	    {
		if(i==0)
			b_critical= body_critical[i];
		else
			b_critical = make_block_statement ( CONS(STATEMENT, b_critical, CONS(STATEMENT, body_critical[i],NIL)));
		stop=1;
	    }	
	
	}
      if(stop==0 && i==nb_critical)
	body = step_compile_parallel(step_transformation, new_module, body,NULL);
      else
	body = step_compile_parallel(step_transformation, new_module, body,b_critical);
		
      break;
    case is_type_directive_omp_parallel_do:
    case is_type_directive_omp_do:
      stop=0;i=0;
      statement b_critical_do;
      for(i=0;i<nb_critical;i++)
	{
	  if(strstr(name_critical[i],entity_local_name(directive_module))!=NULL)
	     {
		if(i==0)
			b_critical_do= body_critical[i];
		else
			b_critical_do = make_block_statement ( CONS(STATEMENT, b_critical_do, CONS(STATEMENT, body_critical[i],NIL)));
		stop=1;
	    }	
	 
	}
      if(stop==0 && i==nb_critical)
	body = step_compile_do(step_transformation, new_module, body,NULL);
      else
	body = step_compile_do(step_transformation, new_module, body,b_critical_do);
	
      break;
    case is_type_directive_omp_barrier:
      body = step_compile_barrier(step_transformation, new_module, body);
      break;
    case is_type_directive_omp_master:
      body = step_compile_master(step_transformation, new_module, body);
      break;
    case is_type_directive_omp_critical:
      body = step_compile_critical(step_transformation, new_module, body);
      break;	
    default:
      pips_user_warning("Directive %s : not yet implemented\n", directive_txt(d));
    }
  
  global_step_analyses_save();
  
  // ajout du statement "return"
  body = make_block_statement(CONS(STATEMENT, body,
				   CONS(STATEMENT,make_return_statement(new_module),NIL)));

  step_RT_set_local_declarations(new_module,body);

  bool is_fortran=fortran_module_p(get_current_module_entity());
  if (is_fortran)
    { 
      set_prettyprint_language_tag(is_language_fortran);
    }
  else 
    {
      set_prettyprint_language_tag(is_language_c);
    }
  //generation du fichier source 
  if(step_transformation == STEP_TRANSFORMATION_HYBRID ||
     step_transformation == STEP_TRANSFORMATION_MPI )
    {
      set_prettyprinter_head_hook(step_head_hook);
    }

  add_new_module(new_name, new_module, body, is_fortran);
  reset_prettyprinter_head_hook();

  // mise à jour du status
  step_status_generated(status)=CONS(ENTITY, new_module, step_status_generated(status));
  step_status_compiled(status)=CONS(ENTITY, directive_module, step_status_compiled(status));
  
  free_statement(body);
  free(new_name);
  
  reset_current_module_entity();
  set_current_module_entity(current_module_save);

  pips_debug(1, "new_module = %p\n", new_module);
  
  return new_module;
}

static void compile_main(statement *body)
{
  pips_debug(1, "Main module : %p\n",body);
  list arglist=NIL;
  string RT_STEP_Init;
  
  if(fortran_module_p(get_current_module_entity()))
    RT_STEP_Init=RT_STEP_Init_Fortran_Order;
  else
    RT_STEP_Init=RT_STEP_Init_C_Order;

  // STEP_Init and STEP_Finalize call insertion

  if(count_critical==0)
  	sequence_statements(instruction_sequence(statement_instruction(*body))) =
    	gen_insert_before(call_STEP_subroutine(RT_STEP_Finalize,NIL,type_undefined), // insert STEP_Finalize before ...
			      find_last_statement(*body),// ... return statement and ...
			      CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Init,NIL,type_undefined), // ... insert STEP_Init at ...
				   statement_block(*body)) // ... body begining
			      );
  else
	sequence_statements(instruction_sequence(statement_instruction(*body))) =
    	gen_insert_before(call_STEP_subroutine(RT_STEP_Finalize,NIL,type_undefined), // insert STEP_Finalize before ...
			      find_last_statement(*body),// ... return statement and ...
			      CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Init,NIL,type_undefined), // ... insert STEP_Init at ...	
			      CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Critical_Spawn,arglist,type_undefined), // ... insert STEP_Spawn ...
				   statement_block(*body))) // ... body begining
			      );
	
}

static bool compile_directive_filter(call c)
{
  entity called  = call_function(c);
  pips_debug(1, "called = %s\n", entity_name(called));

  if (bound_global_directives_p(called))
    {
      int step_transformation;
      directive d=load_global_directives(called);
      pips_debug(2, "substitution %s -> %s\n", entity_name(call_function(c)), directive_txt(d));

      if (directive_transformation_p(d,&step_transformation))
	call_function(c) = compile_module(called,step_transformation);

    }
  return false;
}
static void compile_module_critical(entity directive_module, int step_transformation)
{
  directive d;
  entity current_module_save;
  string new_name;
  entity new_module;
  
  pips_assert("current module defined",!entity_undefined_p(directive_module));
  pips_assert("global_directives loaded",!global_directives_undefined_p());
  pips_assert("first compilation",!gen_in_list_p(directive_module,step_status_compiled(status)));

  d = load_global_directives(directive_module);
  current_module_save = get_current_module_entity();
  if(!entity_undefined_p(current_module_save))
    reset_current_module_entity();
  set_current_module_entity(directive_module);

  new_name = step_find_new_module_name(directive_module,step_transformation);
  new_module = make_empty_subroutine(new_name,copy_language(module_language(get_current_module_entity())));
  
  global_step_analyses_load(); 
  //en cas de la directive critical, rajouter la dernière mise à jour  
  if (type_directive_tag(directive_type(d))== is_type_directive_omp_critical)
    { 
    name_critical[nb_critical] = entity_local_name(directive_module);
    body_critical[nb_critical] = step_compile_critical_update(step_transformation, new_module);//il contient le code de la dérnière reception des données 
    nb_critical++;	
    }
  global_step_analyses_save();
  
  reset_current_module_entity();
  set_current_module_entity(current_module_save);

  return ;//new_module;
}

static bool compile_directive_critical_filter(call c)
{
  entity called  = call_function(c);
  pips_debug(1, "called = %s\n", entity_name(called));

  if (bound_global_directives_p(called))
    {
      int step_transformation;
      directive d=load_global_directives(called);
      pips_debug(2, "substitution %s -> %s\n", entity_name(call_function(c)), directive_txt(d));

      if (directive_transformation_p(d,&step_transformation))
	/*call_function(c) =*/ compile_module_critical(called,step_transformation);

    }
  return false;
}

bool step_compile(const char* module_name)
{ 
  entity module;
  statement body;
  
  debug_on("STEP_DEBUG_LEVEL");

  module = local_name_to_top_level_entity(module_name);
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_COMPILE_DEBUG_LEVEL");

  set_current_module_entity(module);

  load_status();
  global_directives_load();

  body = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  if (!gen_in_list_p(module,step_status_generated(status))) // on ne compile pas un module genere par STEP
    {
      if (entity_main_module_p(module))
	{
	  pips_debug(2, "Main module\n");
	  compile_main(&body);
	}
        if(count_critical > 0)
        {//nb_critical=0;
		gen_recurse(body, call_domain, compile_directive_critical_filter, gen_null);
		reset_current_module_entity();
      		module = local_name_to_top_level_entity(module_name);
      		set_current_module_entity(module);
	}
      /* substitution des modules directives par les modules compiles*/
      gen_recurse(body, call_domain, compile_directive_filter, gen_null);
      
      module_reorder(body);
      if(ordering_to_statement_initialized_p())
	reset_ordering_to_statement();
      
      DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(body));
      DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, body);
    }

  global_directives_save();
  save_status();
  reset_current_module_entity();
  pips_debug(1, "End\n");
  debug_off(); 
  debug_off();
  return true;
}
