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
#include "instrumentation.h"
#include "preprocessor.h"

#define LOCAL_DEBUG 2

static step_status status = (step_status) NULL;

/*les libérations mémoires sont faites par pipsdbm... */
static void save_status()
{
  pips_assert("some current status",status);
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_STATUS, "", status);
  status = NULL;
}

static void load_status()
{
  status = (step_status)db_get_memory_resource(DBR_STEP_STATUS, "", TRUE);
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
  entity module=get_current_module_entity();
  pips_debug(1, "mpi_module = %p, declaration = %p\n", mpi_module, declaration);
  
  FOREACH(ENTITY, e,declaration) 
    {
      if(!type_area_p(entity_type(e)) && !intrinsic_entity_p(e))
	{
	  if (local_entity_of_module_p(e,module))
	    {
	      entity new=FindOrCreateEntity(entity_user_name(mpi_module),entity_local_name(e));
	      entity_type(new)=copy_type(entity_type(e));
	      entity_initial(new)=copy_value(entity_initial(e));
	  
	      formal=CONS(ENTITY,new,formal);
	    }
	  else
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
				      MakeModeReference(),make_dummy_unknown()), functional_parameters(ft));
    }

  gen_free_list(formal);

  pips_debug(1, "End\n");
}


/*############################################################################################*/
/*
  Declaration des tableaux INITIAL et BUFFER pour gerer le cas d'entrelacement des regions SEND, 
*/
static expression step_local_interlaced(entity mpi_module,string name_,entity array)
{
  string name=strdup(name_);
  entity e = FindOrCreateEntity(entity_user_name(mpi_module),name);
  pips_assert("not null",e!=NULL);
  pips_assert("entity defined",!entity_undefined_p(e));

  if (type_undefined_p(entity_type(e)))
    {
      entity area = FindOrCreateEntity(entity_user_name(mpi_module), DYNAMIC_AREA_LOCAL_NAME);
      entity_type(e)=copy_type(entity_type(array));
      area_layout(type_area(entity_type(area))) = gen_nconc(area_layout(type_area(entity_type(area))), CONS(ENTITY, e, NIL));
      entity_storage(e) = make_storage_ram(make_ram(mpi_module,area,UNKNOWN_RAM_OFFSET,NIL));
      code_declarations(EntityCode(mpi_module))=gen_nconc(code_declarations(EntityCode(mpi_module)),CONS(ENTITY,e,NIL));
    }
  pips_assert("variable",entity_variable_p(e));
  free(name);
  return entity_to_expression(e);
}

static expression step_sizevariable_in_expression(entity e)
{
  list dim_l,dim_ll;
  dimension d;
  expression p;
  expression s;
  variable v=type_variable(entity_type(e));
  pips_debug(1, "v : %p\n", v);

  dim_l = gen_copy_seq(variable_dimensions(v));
  d = dimension_undefined;
  p = make_expression_1();
  s = expression_undefined;

  for (dim_ll=dim_l;dim_ll;)
    {
      d=DIMENSION(CAR(dim_ll));POP(dim_ll);
      s=binary_intrinsic_expression(MINUS_OPERATOR_NAME,copy_expression(dimension_upper(d)),copy_expression(dimension_lower(d)));
      s=binary_intrinsic_expression(PLUS_OPERATOR_NAME,s,make_expression_1());
      p=binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,p,s);
    }
  gen_free_list(dim_l);
  pips_debug(1, "p = %p\n", p);
  return p;
}

statement build_call_STEP_InitInterlaced(entity mpi_module, entity interlaced)
{
  /* cas d'entrelacement des regions SEND: besoin de 3 tableaux
     - le tableau des donnees
     - avec 2 tableaux supplementaires pour faire la comparaison
        * tableau de valeurs initiales appele initial
        * tableau de valeurs modifiees appele buffer
     
     creation des 2 tableaux supplementaires:
     - ajout des declarations dans la nouvelle fonction MPI
  */
  entity array=gen_find_tabulated(concatenate(entity_user_name(mpi_module), MODULE_SEP_STRING,
						   entity_user_name(interlaced),NULL),
				       entity_domain);
  pips_assert("defined array",!entity_undefined_p(array));
  expression initialArray = step_local_interlaced(mpi_module,STEP_INITIAL_NAME(array),array);
  expression bufferArray = step_local_interlaced(mpi_module,STEP_BUFFER_NAME(array),array);
  expression nb_elements = step_sizevariable_in_expression(array);

  return call_STEP_subroutine(RT_STEP_InitInterlaced,
			      CONS(EXPRESSION,nb_elements,
				   CONS(EXPRESSION,entity_to_expression(array),
					CONS(EXPRESSION,initialArray,
					     CONS(EXPRESSION,bufferArray,NIL)))),
			      entity_type(interlaced));
}

statement build_call_STEP_AlltoAllRegion(entity mpi_module, int nb_communication_max, boolean merge,entity array, expression expr_nb_region)
{
  /*
    subroutine STEP_AlltoAllRegion_I(dim,
     &     nb_regions,regions,
     &     size,array,
   ( &     initial,buffer )
     &     STEP_TAG_DEFAULT,
     &     max_nb_request,requests,nb_request,STEP_NONBLOCKING)
  */ 
  expression expr_origine, expr_dim;
  expression expr_region, expr_size;
  expression expr_array,expr_tag;
  expression expr_max_nb_request, expr_requests, expr_nb_request;
  expression expr_algorithm;
  list arglist, arglist_merge;
  entity array_region = step_local_SR(mpi_module,array, step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module));
  string RT_name;
  if(merge)
    {
      RT_name=RT_STEP_AlltoAllRegion_Merge;
      arglist_merge = CONS(EXPRESSION,step_local_interlaced(mpi_module,STEP_INITIAL_NAME(array),array),
			   CONS(EXPRESSION,step_local_interlaced(mpi_module,STEP_BUFFER_NAME(array),array),NIL));
    }
  else
    {
      RT_name=RT_STEP_AlltoAllRegion;
      arglist_merge = NIL;
    }

  expr_origine = reference_to_expression(make_reference(array_region,
							CONS(EXPRESSION, step_symbolic(STEP_INDEX_SLICE_LOW_NAME, mpi_module),
							     CONS(EXPRESSION, make_expression_1(),
								  CONS(EXPRESSION, int_to_expression(0), NIL)))));
  
  expr_dim = copy_expression(dimension_upper(DIMENSION(gen_nth(1,variable_dimensions(type_variable(entity_type(array_region)))))));
  
  expr_region = entity_to_expression(array_region);
  expr_size = step_function(RT_STEP_SizeRegion,CONS(EXPRESSION,copy_expression(expr_dim),
						    CONS(EXPRESSION,expr_origine,NIL)));
  
  expr_array = entity_to_expression(array);
  expr_tag = step_symbolic(STEP_TAG_DEFAULT_NAME, mpi_module);
  expr_max_nb_request = step_parameter_max_nb_request(mpi_module, int_expr(nb_communication_max));
  expr_requests = step_local_requests_array(mpi_module, int_expr(nb_communication_max));
  expr_nb_request = step_local_nb_request(mpi_module);
  expr_algorithm = step_symbolic(STEP_NONBLOCKING_NAME, mpi_module);
  
  arglist = CONS(EXPRESSION,expr_dim,
		 CONS(EXPRESSION,expr_nb_region,
		      CONS(EXPRESSION,expr_region,
			   CONS(EXPRESSION,expr_size,
				CONS(EXPRESSION,expr_array,
				     gen_nconc(arglist_merge,
					       CONS(EXPRESSION,expr_tag,
						    CONS(EXPRESSION,expr_max_nb_request,
							 CONS(EXPRESSION,expr_requests,
							      CONS(EXPRESSION,expr_nb_request,
								   CONS(EXPRESSION,expr_algorithm,NIL)))))))))));
		 
  return call_STEP_subroutine(RT_name, arglist, entity_type(array));
}

static statement build_call_STEP_WaitALL(entity mpi_module, int nb_communication_max)
{      
  list arglist;
  statement stmt;
  arglist = CONS(EXPRESSION,step_local_nb_request(mpi_module),
		 CONS(EXPRESSION,step_local_requests_array(mpi_module,int_to_expression(nb_communication_max)),NIL));
  
  stmt=call_STEP_subroutine(RT_STEP_WaitAll, arglist, type_undefined);
  statement_comments(stmt)=strdup("C     If STEP_Nb_Request equals 0, STEP_WAITALL does nothing\n");
  return stmt;
}

statement step_handle_comm_requests(entity module,list comm_stmt, int nb_communication_max)
{
  list block=comm_stmt;

  if(!ENDP(block))
    {
      statement s = make_assign_statement(step_local_nb_request(module),int_expr(0));
      statement_comments(s)=strdup(concatenate("\nC     Communicating data to other nodes",
					       "\nC     3 communication shemes for all-to-all personalized broadcast :",
					       "\nC     STEP_NONBLOCKING, STEP_BLOCKING1 and STEP_BLOCKING2.",
					       "\nC     A nonblocking algo increment STEP_Nb_Request.\n",NULL));
      //nb_request=0
      block = CONS(STATEMENT,s,block);

      //call waitALL
      block=gen_nconc(block,CONS(STATEMENT, build_call_STEP_WaitALL(module, nb_communication_max),NIL));
    }
  return make_block_statement(block);
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
      pips_internal_error("unexpected transformation :%d\n",step_transformation);
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
  pips_assert("current module defined",!entity_undefined_p(directive_module));
  pips_assert("global_directives loaded",!global_directives_undefined_p());
  pips_assert("first compilation",!gen_in_list_p(directive_module,step_status_compiled(status)));

  directive d=load_global_directives(directive_module);
  pips_debug(1,"Directive module %s : %s\n", entity_local_name(directive_module), directive_txt(d));
  pips_debug(2, "transformation %i\n", step_transformation);

  entity current_module_save=get_current_module_entity();
  if(!entity_undefined_p(current_module_save))
    reset_current_module_entity();
  set_current_module_entity(directive_module);

  string new_name = step_find_new_module_name(directive_module,step_transformation);
  entity new_module = make_empty_subroutine(new_name,copy_language(module_language(get_current_module_entity())));
  step_add_formal_copy(new_module,code_declarations(value_code(entity_initial(directive_module))));

  // initialisation du futur body par le code du module de directive
  statement body = copy_statement((statement)db_get_memory_resource(DBR_CODE, entity_local_name(directive_module), TRUE));
  { // elimination du statement "return"
    statement s=find_last_statement(body);
    free_instruction(statement_instruction(s));
    statement_instruction(s)=make_continue_instruction();
    statement_label(s)=entity_empty_label();
  }

  // modification du body selon la transformation (OMP, MPI, HYBRID)
  global_step_analyses_load();
  switch(type_directive_tag(directive_type(d)))
    {
    case is_type_directive_omp_parallel:
      body = step_compile_parallel(step_transformation, new_module, body);
      break;
    case is_type_directive_omp_parallel_do:
    case is_type_directive_omp_do:
      body = step_compile_do(step_transformation, new_module, body);
      break;
    case is_type_directive_omp_barrier:
      body = step_compile_barrier(step_transformation, new_module, body);
      break;
    case is_type_directive_omp_master:
      body = step_compile_master(step_transformation, new_module, body);
      break;
    default:
      pips_user_warning("Directive %s : not yet implemented\n", directive_txt(d));
    }
  global_step_analyses_save();

  // ajout du statement "return"
  body = make_block_statement(CONS(STATEMENT, body,
				   CONS(STATEMENT,make_return_statement(new_module),NIL)));
  
  //generation du fichier source 
  if(step_transformation == STEP_TRANSFORMATION_HYBRID ||
     step_transformation == STEP_TRANSFORMATION_MPI )
    set_prettyprinter_head_hook(step_head_hook);
  add_new_module(new_name, new_module, body, TRUE);
  reset_prettyprinter_head_hook();

  // mise à jour du status
  step_status_generated(status)=CONS(ENTITY, new_module, step_status_generated(status));
  step_status_compiled(status)=CONS(ENTITY, directive_module, step_status_compiled(status));

  free_statement(body);
  free(new_name);
  
  reset_current_module_entity();
  set_current_module_entity(current_module_save);
  return new_module;
}

static void compile_main(statement *body)
{
  pips_debug(1, "Main module : %p\n",body);
  
  // STEP_Init and STEP_Finalize call insertion
  sequence_statements(instruction_sequence(statement_instruction(*body))) =
    gen_insert_before(call_STEP_subroutine(RT_STEP_Finalize,NIL,type_undefined), // insert STEP_Finalize before ...
		      find_last_statement(*body),// ... return statement and ...
		      CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Init_Fortran_Order,NIL,type_undefined), // ... insert STEP_Init at ...
			   statement_block(*body)) // ... body begining
		      );
}

static bool module_directive_filter(call c)
{
  pips_debug(1, "c = %p\n", c);
  entity called=call_function(c);
  if (bound_global_directives_p(called))
    {
      int step_transformation;
      directive d=load_global_directives(called);
      pips_debug(2, "substitution %s -> %s", entity_name(call_function(c)), directive_txt(d));

      if (directive_transformation_p(d,&step_transformation))
	call_function(c) = compile_module(called,step_transformation);

    }
  return FALSE;
}

bool step_compile(string module_name)
{ 
  entity module = local_name_to_top_level_entity(module_name);
  statement body;
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_COMPILE_DEBUG_LEVEL");

  set_current_module_entity(module);

  load_status();
  global_directives_load();

  body = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  if (!gen_in_list_p(module,step_status_generated(status))) // on ne compile pas un module genere par STEP
    {
      if (entity_main_module_p(module))
	compile_main(&body);
      
      /* substitution des modules directives par les modules compiles*/
      gen_recurse(body, call_domain, module_directive_filter, gen_null);
      
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
  return TRUE;
}
