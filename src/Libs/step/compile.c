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
#include "instrumentation.h"
#include "preprocessor.h"


#define LOCAL_DEBUG 2

//  liste des statements constituant le corps de la nouvelle fonction (module) MPI
list step_seqlist=list_undefined;

// constante symbolique definie dans mpi_module
entity step_sizeRegion=entity_undefined;
entity step_i_slice_low=entity_undefined;
entity step_i_slice_up=entity_undefined;
entity step_max_nb_loopslices=entity_undefined;
entity step_max_nb_request=entity_undefined;
entity step_nonblocking=entity_undefined;
entity step_requests=entity_undefined;
entity step_nb_request=entity_undefined;

GENERIC_GLOBAL_FUNCTION(send_region_entities, step_entity_map)


/*############################################################################################*/
// utilise pour mapper les entites (entre arguments formels et entites du module courant)
GENERIC_GLOBAL_FUNCTION(old_entities,step_entity_map)
GENERIC_GLOBAL_FUNCTION(new_entities,step_entity_map)
void step_store_new_entity(entity new, entity old)
{
    pips_assert("defined", !entity_undefined_p(new)&&!entity_undefined_p(old));
    pips_debug(5,"%s old %s -> %s new %s\n",
	       entity_label_p(old)?"label":"reference",entity_name(old),
	       entity_label_p(new)?"label":"reference",entity_name(new));
    store_or_update_new_entities(old, new);
    store_or_update_old_entities(new, old);
}

void step_reference_update(reference ref)
{
  entity *pe=&reference_variable(ref);
  pips_debug(1,"entity_name(reference_variable(ref)) = %s\n",entity_name(reference_variable(ref)));
  
  if (bound_new_entities_p(*pe)) 
    {
      entity ne = load_new_entities(*pe);
      pips_debug(5, "%s -> %s\n", entity_name(*pe), entity_name(ne));
      *pe = ne;
    }

  pips_debug(1,"Fin\n");
}

/*############################################################################################*/
/*
  Declaration des constantes symboliques I_SLICE_LOW, I_SLICE_UP, MAX_NB_LOOPSLICES
  et de variables diverses: STEP_sizeRegion, mpi_status_size, STEP_Status

  creation des entites et ajout aux declarations de la nouvelle fonction MPI
*/

void step_add_parameter(entity module)
{
  type t_integer;
  int max_slice;
  entity mpi_status_size;
  list dimlist;
  dimension dim;
  entity area;
  int offset;
  entity step_com_status;

  t_integer=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), NIL);
  // ajout de lower=1
  step_i_slice_low=FindOrCreateEntity(entity_user_name(module),strdup(STEP_INDEX_SLICE_LOW_NAME));
  entity_type(step_i_slice_low)=copy_type(t_integer);
  step_i_slice_low=MakeParameter(step_i_slice_low,int_expr(1));

  // ajout de upper=2
  step_i_slice_up=FindOrCreateEntity(entity_user_name(module),strdup(STEP_INDEX_SLICE_UP_NAME));
  entity_type(step_i_slice_up)=copy_type(t_integer);
  step_i_slice_up=MakeParameter(step_i_slice_up,int_expr(2));
  
  // ajout de max_nb_loopslices: declare dans properties-rc.tex
  /* attention valeur mise en dur dans STEP.h */
  max_slice=16;
  //  max_slice=get_int_property("STEP_PROPERTY_MAX_NB_LOOPSLICES");
  step_max_nb_loopslices=FindOrCreateEntity(entity_user_name(module),strdup(STEP_MAX_NB_LOOPSLICES_NAME));
  entity_type(step_max_nb_loopslices)=t_integer;
  step_max_nb_loopslices=MakeParameter(step_max_nb_loopslices,int_expr(max_slice));
  
  // ajout de STEP_sizeRegion resultat de la fonction du meme nom

  step_sizeRegion = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,RT_STEP_SizeRegion,NULL), entity_domain);
  
  // creation de MPI_STATUS_SIZE
  mpi_status_size=FindOrCreateEntity(entity_user_name(module),strdup(STEP_MPI_STATUS_SIZE_NAME));
  entity_type(mpi_status_size)=t_integer;
  mpi_status_size=MakeParameter(mpi_status_size,int_expr(4));
  
  // creation de STEP_NONBLOCKING
  step_nonblocking=FindOrCreateEntity(entity_user_name(module),strdup(STEP_NONBLOCKING_NAME));
  entity_type(step_nonblocking)=copy_type(t_integer);
  step_nonblocking=MakeParameter(step_nonblocking,int_expr(0));

  // creation de STEP_status

  step_com_status=FindOrCreateEntity(entity_user_name(module),strdup(STEP_STATUS_NAME));
  dim=make_dimension(make_expression_1(),
		     entity_to_expression(mpi_status_size));

  dimlist = CONS(DIMENSION,dim,NIL);
  entity_type(step_com_status)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), dimlist);

  area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);
  offset = add_variable_to_area(area, step_com_status);
  entity_storage(step_com_status) = make_storage(is_storage_ram,make_ram(module,area,offset,NIL));
  entity_initial(step_com_status) = make_value_unknown();
}

/*
  Duplication du prototype

  Parcours de la liste de declarations passee en parametre
  Creation d'une nouvelle entite pour chaque declaration

  Les nouvelles entites sont ajoutees dans la liste des variables
  formelles de la nouvelle fonction

  Mise a jour d'une table d'association pour faire la correspondance
  entre les anciennes et les nouvelles entites notamment pour
  recuperer les analyses PIPS faites sur les anciennes entites
*/
void step_add_formal_copy(entity mpi_module, list declaration)
{
  int ith;
  functional ft;
  list formal=NIL;

  pips_debug(1, "mpi_module = %p, declaration = %p\n", mpi_module, declaration);

  MAP(ENTITY, e, {
      if(!type_area_p(entity_type(e))){
	entity new=FindOrCreateEntity(entity_user_name(mpi_module),entity_local_name(e));
	
	type t=copy_type(entity_type(e));
	gen_recurse(t,reference_domain, gen_true, step_reference_update);
	
	entity_type(new)=t;
	entity_initial(new)=copy_value(entity_initial(e));

	formal=CONS(ENTITY,new,formal);

	step_store_new_entity(new,e);
      }
    },declaration);
  
  ith = gen_length(formal);
  ft = type_functional(entity_type(mpi_module));

  MAP(ENTITY, v, {
      entity_storage(v)=make_storage(is_storage_formal, make_formal(mpi_module,ith--));
      pips_debug(8,"variable %s #%i\n",entity_name(v),ith+1);
      AddEntityToDeclarations(v,mpi_module);
      functional_parameters(ft) =
	CONS(PARAMETER,make_parameter(entity_type(v), 
				      MakeModeReference(),make_dummy_unknown()), functional_parameters(ft));
    },formal); 

  gen_free_list(formal);

  pips_debug(1, "End\n");
}

/* A partir d'un module, d'un nom et d'un tableau, creation d'un autre
   tableau stockant pour chaque noeud les bornes d'indices de tableau traites
*/
entity step_create_region_array(entity module,string name,entity array,bool slice)
{
  entity ne;
  dimension dim_d, bounds, slices;
  list d=NIL;
  entity area;
  int offset;

  pips_debug(1, "module = %p, name = %s, array = %p\n", module, name, array);

  ne = FindOrCreateEntity(entity_user_name(module),name);

  dim_d = make_dimension(make_expression_1(),
			     int_expr(gen_length(variable_dimensions(type_variable(entity_type(array))))));

  bounds = make_dimension(make_call_expression(step_i_slice_low,NIL),
				  make_call_expression(step_i_slice_up,NIL));
  if(slice)
    {
      slices = make_dimension(int_to_expression(0),make_call_expression(step_max_nb_loopslices,NIL));
      d=CONS(DIMENSION,slices,d);
    }

  d = CONS(DIMENSION,bounds,CONS(DIMENSION,dim_d,d));

  entity_type(ne)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), d);

  area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);
  offset=add_variable_to_area(area, ne);
  entity_storage(ne)=make_storage(is_storage_ram,make_ram(module,area,offset,NIL));
  
  entity_initial(ne)=make_value_unknown();
  
  AddEntityToDeclarations(ne,module);

  pips_debug(1, "ne = %p\n", ne);

  return ne;
}

/*
  Placement des bornes d'indice d'un tableau dans son tableau de regions
*/
statement build_assigne_region0(int nb,region reg,entity array_region)
{
  statement body;
  reference ref;
  type t;
  list slices=NIL;
  int d;

  pips_debug(1, "reg = %p, array_region = %p\n", reg, array_region);

  body = make_block_statement(NIL);
  ref = reference_undefined;
  t = entity_type(region_entity(reg));
  pips_assert("variable",type_variable_p(t));


  for(d=0; d<nb; d++)
    {
      slices=CONS(EXPRESSION,int_to_expression(0),slices);
    }
  
  d=0; // nb: gen_nth numerote les elements a partir de 0 d'ou l'initialisation a 0
  MAP(EXPRESSION, exp,
      {
	dimension dim_d;
	
	dim_d = DIMENSION(gen_nth(d,variable_dimensions(type_variable(t))));
	d++;

	//affectation "lower"
	ref=make_reference(array_region,
			   CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
				CONS(EXPRESSION,int_to_expression(d),slices)));
	insert_statement(body,
			 make_assign_statement(make_expression(make_syntax_reference(ref),normalized_undefined),
					       copy_expression(dimension_lower(dim_d))),
			 FALSE);
	
	//affectation "upper"
	ref=make_reference(array_region,
			   CONS(EXPRESSION,entity_to_expression(step_i_slice_up),
				CONS(EXPRESSION,int_to_expression(d),slices)));
	insert_statement(body,
			 make_assign_statement(make_expression(make_syntax_reference(ref),normalized_undefined),
					       copy_expression(dimension_upper(dim_d))),
			 FALSE);
      },
      reference_indices(effect_any_reference(reg)));

  pips_debug(1, "body = %p\n", body);
  return body;
}

/*############################################################################################*/
/*
  Creation de l'entite requests pour contenir les requetes dans le cadre des communications non blocantes
*/
entity step_declare_requests_array(entity module,list comm)
{
  list dimlist; 
  entity area;
  int offset;
  pips_assert("requests_array",entity_undefined_p(step_requests));
  pips_assert("nb_requests",entity_undefined_p(step_nb_request));
  if(!ENDP(comm))
    {
      step_nb_request = make_scalar_integer_entity(STEP_NBREQUEST_NAME, entity_user_name(module));

      // creation de la constante symbolique définissant la taille du tableau
      step_max_nb_request=FindOrCreateEntity(entity_user_name(module),strdup(STEP_MAX_NB_REQUEST_NAME));
      entity_type(step_max_nb_request)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), NIL);
      step_max_nb_request=MakeParameter(step_max_nb_request,
					make_op_exp("*",int_expr(2),
						    make_op_exp("*",int_expr(gen_length(comm)),
								entity_to_expr(step_max_nb_loopslices))));
      // creation du tableau
      step_requests = FindOrCreateEntity(entity_user_name(module),strdup(STEP_REQUEST_NAME));
      dimlist = CONS(DIMENSION,make_dimension(make_expression_1(),entity_to_expr(step_max_nb_request)),NIL);
      
      entity_type(step_requests)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), dimlist);
      area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);
      offset = add_variable_to_area(area,step_requests);
      
      entity_storage(step_requests)=make_storage(is_storage_ram,make_ram(module,area,offset,NIL));
      
      // ajout des nouvelles entites en fin de déclaration dans le nouveau module...
      //code_declarations(EntityCode(module)) =gen_append(code_declarations(EntityCode(module)), CONS(ENTITY,step_max_nb_request,CONS(ENTITY,step_requests,NIL)));
    }
  return step_requests;
}

statement build_call_STEP_AlltoAllRegion(boolean merge,entity array, entity nb_region,int tag)
{
  /*
    subroutine STEP_AlltoAllRegion_I(dim,
     &     nb_regions,regions,
     &     size,array,
   ( &     initial,buffer )
     &     tag,
     &     max_nb_request,requests,nb_request,STEP_NONBLOCKING)
  */ 
  expression expr_origine, expr_dim, expr_nb_region;
  expression expr_region, expr_size;
  expression expr_array,expr_tag;
  expression expr_max_nb_request, expr_requests, expr_nb_request;
  expression expr_algorithm;
  list arglist;
  expression expr_array_initial, expr_array_buffer;
  entity array_region = load_send_region_entities(array);

  expr_origine = make_expression(make_syntax_reference(make_reference(array_region,
								      CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
									   CONS(EXPRESSION,int_to_expression(1),
										CONS(EXPRESSION,int_to_expression(0),NIL))))),
				 normalized_undefined);
  
  expr_dim = copy_expression(dimension_upper(DIMENSION(gen_nth(1,variable_dimensions(type_variable(entity_type(array_region)))))));
  expr_nb_region = entity_to_expression(nb_region);
  
  expr_region = entity_to_expression(array_region);
  expr_size = make_call_expression(step_sizeRegion,CONS(EXPRESSION,copy_expression(expr_dim),
							 CONS(EXPRESSION,expr_origine,NIL)));
  
  expr_array = entity_to_expression(array);
  expr_tag = int_to_expression(tag);  
  expr_max_nb_request = entity_to_expression(step_max_nb_request);
  expr_requests = entity_to_expression(step_requests);
  expr_nb_request = entity_to_expression(step_nb_request);
  expr_algorithm = entity_to_expression(step_nonblocking);
  arglist = CONS(EXPRESSION,expr_tag,
		 CONS(EXPRESSION,expr_max_nb_request,
		      CONS(EXPRESSION,expr_requests,
			   CONS(EXPRESSION,expr_nb_request,
				CONS(EXPRESSION,expr_algorithm,NIL)))));
  if(merge)
    {
      expr_array_initial = entity_to_expression(load_initial_copy_entities(load_new_entities(array)));
      expr_array_buffer = entity_to_expression(load_buffer_copy_entities(load_new_entities(array)));
      arglist = CONS(EXPRESSION,expr_array_initial,
		     CONS(EXPRESSION,expr_array_buffer,arglist));
    }

  arglist = CONS(EXPRESSION,expr_dim,
		 CONS(EXPRESSION,expr_nb_region,
		      CONS(EXPRESSION,expr_region,
			   CONS(EXPRESSION,expr_size,
				CONS(EXPRESSION,expr_array,arglist)))));
  if (merge)
    return call_STEP_subroutine(strdup(concatenate(RT_STEP_AlltoAllRegion_Merge, step_type_suffix(array),NULL)),arglist);
  else
    return call_STEP_subroutine(strdup(concatenate(RT_STEP_AlltoAllRegion, step_type_suffix(array),NULL)),arglist);
}

static statement build_call_STEP_WaitALL(entity nb_request,entity requests)
{      
  list arglist;
  statement stmt;
  arglist = CONS(EXPRESSION,entity_to_expression(nb_request),
		 CONS(EXPRESSION,entity_to_expression(requests),NIL));
  
  stmt=call_STEP_subroutine(RT_STEP_WaitAll, arglist);
  statement_comments(stmt)=strdup("C     If STEP_Nb_Request equals 0, STEP_WAITALL does nothing\n");
  return stmt;
}

list step_handle_comm_requests(entity requests_array,list comm_stmt)
{
  list comm_request_stmt=NIL;

  if(!ENDP(comm_stmt))
    {
      statement stmt=make_assign_statement(entity_to_expression(step_nb_request),int_expr(0));
      statement_comments(stmt)=strdup("C     3 communication shemes for all-to-all personalized broadcast :\nC     STEP_NONBLOCKING, STEP_BLOCKING1 and STEP_BLOCKING2.\nC     A nonblocking algo increment STEP_Nb_Request.\n");
      //nb_request=0
      comm_request_stmt = CONS(STATEMENT, stmt,comm_request_stmt);
      //call comm_stmt
      comm_request_stmt=gen_append(comm_stmt,comm_request_stmt);
      //call waitALL
      comm_request_stmt=CONS(STATEMENT, build_call_STEP_WaitALL(step_nb_request,requests_array), comm_request_stmt);
    }
  step_requests=entity_undefined;
  step_nb_request=entity_undefined;
  return comm_request_stmt;
}


/*############################################################################################*/
string step_find_new_module_name(entity original, string suffix)
{
  int id=0;
  entity e;
  string newname;
  string original_name=entity_user_name(original);
  pips_debug(1,"original_name = %s\n", original_name);
   
  newname=strdup(concatenate(original_name,suffix,NULL));
  
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

string step_type_suffix(entity e)
{
  variable v;
  string s;
  pips_assert("step : check type_suffix",type_variable_p(entity_type(e)));
  v=type_variable(entity_type(e));

  switch (basic_tag(variable_basic(v)))
    {
    case is_basic_int: s=strdup(concatenate("_I",i2a(basic_int(variable_basic(v))),NULL));
      break;
    case is_basic_float: s=strdup(concatenate("_R",i2a(basic_int(variable_basic(v))),NULL));
      break;
    case is_basic_complex: s=strdup(concatenate("_C",i2a(basic_int(variable_basic(v))),NULL));
      break;
    default:
      pips_user_error("unexpected basic type : %i",basic_tag(variable_basic(v)));
      s=strdup("");
      break;
    }
  pips_debug(1,"%s\n",s);
  return s;
}

statement call_STEP_subroutine(string name, list args)
{
  statement statmt;

  pips_debug(1, "name = %s, arg = %p\n", name, args);

  statmt = make_call_statement(name, args, entity_undefined, string_undefined);

  STEP_DEBUG_STATEMENT(3, "call STEP_subroutine", statmt);

  pips_debug(1, "statmt = %p\n", statmt);
  return statmt;
}

/*############################################################################################*/
static bool step_mpi_module_filter(call c)
{
  pips_debug(1, "c = %p\n", c);

  if (bound_step_mpi_module_map_p(call_function(c)))
    {
      pips_debug(LOCAL_DEBUG, "substitution %s -> %s\n", entity_name(call_function(c)), entity_name(load_step_mpi_module_map(call_function(c))));

      call_function(c) = load_step_mpi_module_map(call_function(c));
    }
  return FALSE;
}


static string step_head_hook(entity __attribute__ ((unused)) e) 
{
  pips_debug(1, "step_head_hook\n");
  
  return strdup(concatenate
		("      implicit none\n",
		 "      include \"STEP.h\"\n", NULL));
}

bool add_mpi_module(entity mpi_module, statement mpi_body)
{
  bool result;
  set_prettyprinter_head_hook(step_head_hook);
  result=add_new_module(entity_local_name(mpi_module), mpi_module, mpi_body, TRUE);
  reset_prettyprinter_head_hook();
  return result;
}

static void compile_mpi(entity module)
{
  pips_assert("current module defined",!entity_undefined_p(module));
  pips_assert("global_directives loaded",!global_directives_undefined_p());
  pips_assert("undefined_seqlist",list_undefined_p(step_seqlist));
  string module_name=entity_local_name(module);

  if(bound_global_directives_p(module)) // directive module case
    {
      entity mpi_module=entity_undefined;
      statement directive_body;
      directive d=load_global_directives(module);

      pips_debug(1,"Directive module %s : %s\n", module_name,directive_txt(d));
      set_outline((outline_map)db_get_memory_resource(DBR_OUTLINED, "", TRUE));
      directive_body = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

      step_seqlist=NIL;
      switch(type_directive_tag(directive_type(d)))
	{
	case is_type_directive_omp_parallel:
	  add_mpi_module(module,copy_statement (directive_body));
	  break;
	case is_type_directive_omp_parallel_do:
	case is_type_directive_omp_do:
	  {
	    mpi_module = step_create_mpi_loop_module(module);
	    break;
	  }
	case is_type_directive_omp_barrier:
	  {
	    mpi_module = step_create_mpi_barrier(module);
	    break;
	  }
	case is_type_directive_omp_master:
	  {
	    mpi_module = step_create_mpi_master(module);
	    break;
	  }
	default:
	  pips_user_warning("Directive %s : MPI generation not yet implemented\n", directive_txt(d));
	}

      if(!entity_undefined_p(mpi_module))
	{
	  add_mpi_module(mpi_module,make_block_statement(gen_nreverse(gen_copy_seq(step_seqlist))));
	  store_step_mpi_module_map(module, mpi_module);
	}
      gen_free_list(step_seqlist);
      step_seqlist=list_undefined;
      reset_outline();
    }
  return;
}

bool step_mpi(string module_name)
{ 
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_MPI_DEBUG_LEVEL");

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  step_load_status();
  global_directives_load();

  // la ou ce fait la genearation du nouveau module
  compile_mpi(get_current_module_entity());
 
  global_directives_save();
  step_save_status();
  reset_current_module_entity();

  pips_debug(1, "End\n");
  debug_off(); 
  debug_off();
  return TRUE;
}


static void compile_omp(entity module)
{  
  pips_assert("current module defined",!entity_undefined_p(module));
  pips_assert("global_directives loaded",!global_directives_undefined_p());

  string module_name=entity_local_name(module);

  if(bound_global_directives_p(module))
    {
      statement omp_body;
      string new_name,name;
      entity omp_module;
      omp_body = copy_statement((statement)db_get_memory_resource(DBR_CODE, module_name, TRUE));
      new_name = step_find_new_module_name(module,STEP_OMP_SUFFIX);
      omp_module = make_empty_subroutine(new_name); // necessaire pour add_new_module

      /* Ajout des directives OpenMP
       */
      add_pragma_entity_to_statement(omp_body,module);

      /* Ce qui suit n'est pas propre...
       */
      name=entity_name(omp_module);
      omp_module = copy_entity(module);
      entity_name(omp_module)=name;
      free(code_decls_text(value_code(entity_initial(omp_module))));
      code_decls_text(value_code(entity_initial(omp_module)))=strdup("");
      /* ... On peut surement faire mieux pour avoir les déclarations et les arguments de la nouvelle fonction
       */      

      add_new_module(new_name, omp_module, omp_body, TRUE);
      store_step_omp_module_map(module, omp_module);

      free_statement(omp_body);
      free(new_name);
    }
  return;
}


bool step_omp(string module_name)
{
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_OMP_DEBUG_LEVEL");

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  step_load_status();
  global_directives_load();
  
  // la ou ce fait la genearation du nouveau module
  compile_omp(get_current_module_entity());
 
  global_directives_save();
  step_save_status();
  reset_current_module_entity();

  pips_debug(1, "End\n");
  debug_off(); 
  debug_off();
  return TRUE;
}

bool step_compile(string module_name)
{ 
  entity module;
  statement body;
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_COMPILE_DEBUG_LEVEL");

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  module = local_name_to_top_level_entity(module_name);  
  step_load_status();
  global_directives_load();

  body = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  if (entity_main_module_p(module)) // main module case
    {
      pips_debug(LOCAL_DEBUG, "Main module : %s\n",module_name);
      
      // STEP_Init and STEP_Finalize call insertion
      sequence_statements(instruction_sequence(statement_instruction(body)))=
	gen_insert_before(call_STEP_subroutine(RT_STEP_Finalize,NIL), // insert STEP_Finalize call befor  ...
			  find_last_statement(body),// ... return statement and ...
			  CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Init,NIL), // ... insert STEP_Init call at ... 
			       statement_block(body)) // ... body begining
			  );
    }

  /* Generation des versions _MPI et _OMP
     (TODO ne genere les versions que selon les besoin lors du la substitution dans le callgraph )
   */
  compile_mpi(module);
  compile_omp(module);

  /* substitution des modules outlines par les modules MPI deja compiles
     ceci impose de traiter les modules dans l'ordre inverse de l'outlining */
  gen_recurse(body, call_domain, step_mpi_module_filter, gen_null);

  module_reorder(body);
  if(ordering_to_statement_initialized_p())
    reset_ordering_to_statement();
  
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(body));
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, body);
  
  global_directives_save();
  step_save_status();
  reset_current_module_entity();
  pips_debug(1, "End\n");
  debug_off(); 
  debug_off();
  return TRUE;
}
