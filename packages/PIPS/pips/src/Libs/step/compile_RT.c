#include "defines-local.h"

/*
  Taille des entiers utilises en parametre pour l'API de la runtime

!!!!! Leur coherence avec  STEP.h, steprt_f.h et la runtime doit etre assure manuellement !!!!!

*/

#define STEP_RT_INTEGER4 DEFAULT_INTEGER_TYPE_SIZE
#define STEP_RT_INTEGER8 DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE

#define STEP_RT_SYMBOLIC_INTEGER STEP_RT_INTEGER4
#define STEP_RT_NB_WORKCHUNK_INTEGER STEP_RT_INTEGER4
#define STEP_RT_LOOP_INDEX_INTEGER STEP_RT_INTEGER4
#define STEP_RT_ARRAY_INDEX_INTEGER STEP_RT_INTEGER4

//######################################################
/*
  Declaration des parameter
*/
typedef struct {
  int declaration_file;
  int size_of_integer;
  int value;
  string name;
} SymbolicDescriptor;

#define DECLARE_NO 0
#define DECLARE_LOCAL 1
#define DECLARE_STEP_H 2
#define DECLARE_STEPRT_F_H 3

static list local_declaration=NIL;

void step_RT_set_local_declarations(entity module, statement body)
{

  FOREACH(ENTITY, e, local_declaration)
    {
      AddLocalEntityToDeclarations(e, module, body);
    }

  local_declaration=NIL;
}


static SymbolicDescriptor SymbolicDescriptorTable[] = {
  {DECLARE_LOCAL,      STEP_RT_NB_WORKCHUNK_INTEGER, -1, STEP_MAX_NB_REQUEST_NAME},
  {DECLARE_STEP_H,     STEP_RT_SYMBOLIC_INTEGER,      1, STEP_INDEX_SLICE_LOW_NAME},
  {DECLARE_STEP_H,     STEP_RT_SYMBOLIC_INTEGER,      2, STEP_INDEX_SLICE_UP_NAME},
  {DECLARE_STEP_H,     STEP_RT_NB_WORKCHUNK_INTEGER, 16, STEP_MAX_NB_LOOPSLICES_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      0, STEP_NONBLOCKING_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      0, STEP_TAG_DEFAULT_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      0, STEP_PARALLEL_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      1, STEP_DO_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      2, STEP_PARALLELDO_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      3, STEP_MASTER_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      3, STEP_CRITICAL_NAME},
  /*must be the last*/
  {DECLARE_NO, 0, 0, ""}
};

static SymbolicDescriptor find_SymbolicDescriptor_by_name(string name)
{
  int id = 0;
  size_t len=strlen(name);
  
  while ((SymbolicDescriptorTable[id].declaration_file != DECLARE_NO) &&
	 (strncmp(name, SymbolicDescriptorTable[id].name, len) != 0))
    id++;
  pips_assert("know symbol", SymbolicDescriptorTable[id].declaration_file != DECLARE_NO);
 
  return SymbolicDescriptorTable[id];
}

entity step_parameter(string name_, entity module, expression expr)
{
  entity e;
  string name = strdup(name_);
  SymbolicDescriptor sym_desc = find_SymbolicDescriptor_by_name(name);

  if (fortran_module_p(get_current_module_entity()))
    { 
      /* Fortran
	 PARAMETER (TEST = expr)
       */
      e = FindOrCreateEntity(entity_user_name(module), name);
      pips_assert("not null", e != NULL);
      pips_assert("entity defined", !entity_undefined_p(e));
      
      if (type_undefined_p(entity_type(e)))
	{
	  entity_type(e)=MakeTypeVariable(make_basic_int(sym_desc.size_of_integer), NIL);
	  if(expression_undefined_p(expr))
	    e = MakeParameter(e, int_to_expression(sym_desc.value));
	  else
	    e = MakeParameter(e, expr);
	  
	  if (sym_desc.declaration_file == DECLARE_LOCAL)
	    local_declaration = gen_nconc(local_declaration, CONS(ENTITY, e, NIL));
	}
      pips_assert("symbolic", entity_symbolic_p(e));
    }
  else 
    {
      /* C 
     const int TEST=1 
  
     DT74 "TOP-LEVEL:TEST"T77 2 T80 T31 0 4 )()(T63 0 U))))T70 1 T64 R74 "TOP-LEVEL:TOP-LEVEL" R74 "TOP-LEVEL:*STATIC*" 0 ()))T79 5 T44 T73 2 T33 R74 "TOP-LEVEL:1"

     T77: type T80: variable 
                    T31: basic 0: int
                    T63: qualifier 0: const
     T70: storage 1: ram T64: ram 
     T79: value 5: expression T44:Expression T73: syntax 2: call T33:call R74: entity"TOP-LEVEL:1"
*/
      e = FindOrCreateEntity(entity_user_name(module), name);
      pips_assert("not null", e != NULL);
      pips_assert("entity defined", !entity_undefined_p(e));

      local_declaration = gen_nconc(local_declaration, CONS(ENTITY, e, NIL));
      if (type_undefined_p(entity_type(e)))
	{
	  entity_type(e)=make_type(is_type_variable, make_variable(make_basic_int(sym_desc.size_of_integer), NIL, NIL));
	  entity_storage(e) = make_storage_rom(); // ROM ou RAM ???
	  if(expression_undefined_p(expr))
	    entity_initial(e) = make_value_expression(int_to_expression(sym_desc.value));
	  else
	    entity_initial(e) = make_value_expression(expr);
	}
    }
  free(name);
  return e;
}

expression step_parameter_max_nb_request(entity module, expression nb_communication)
{
  expression expr = expression_undefined;

  if(!expression_undefined_p(nb_communication))
    expr = make_op_exp("*", int_to_expression(2),
		       make_op_exp("*", nb_communication, 
				   step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, module)));

  return entity_to_expression(step_parameter(STEP_MAX_NB_REQUEST_NAME, module, expr));
}

expression step_symbolic(string name_, entity module)
{
  return entity_to_expression(step_parameter(name_ , module, expression_undefined));
}

//######################################################
expression step_function(string name,list args)
{
  entity e=gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,name,NULL), entity_domain);
  pips_assert("boostrap defined",!entity_undefined_p(e));
  pips_assert("functional",type_functional_p(entity_type(e)));
  return make_call_expression(e,args);
}

//######################################################
/*
  Pour declarations de tableaux/variables d'entier
*/
static entity step_local_RT_Integer(int size_of_integer, entity mpi_module, string name_, list dims, bool header)
{
  string name = strdup(name_);
  entity e = FindOrCreateEntity(entity_user_name(mpi_module),name);
  pips_assert("not null",e!=NULL);
  pips_assert("entity defined",!entity_undefined_p(e));

  if (type_undefined_p(entity_type(e)))
    {
      entity area = FindOrCreateEntity(entity_user_name(mpi_module), DYNAMIC_AREA_LOCAL_NAME);
      entity_type(e) = MakeTypeVariable(make_basic_int(size_of_integer), dims);
      entity_storage(e) = make_storage_ram(make_ram(mpi_module,area,add_variable_to_area(area,e),NIL));
      if(!header && mpi_module != get_current_module_entity())
	local_declaration=gen_nconc(local_declaration,CONS(ENTITY,e,NIL));
    }
  pips_assert("variable",entity_variable_p(e));
  free(name);
  return e;
}

/*
  A partir d'un module, d'un tableau, creation d'un tableau des regions SEND (ou RECV) pour chaque workchunk
*/
entity step_local_arrayRegions(string array_regions_name ,entity module, entity array, expression expr_nb_region)
{
  bool is_fortran=fortran_module_p(get_current_module_entity());

  string name = strdup(array_regions_name);
  list dims = NIL;
  if(is_fortran)
    {
      dimension dim_array = make_dimension(int_to_expression(1), int_to_expression(NumberOfDimension(array)));
      if(!expression_undefined_p(expr_nb_region))
	dims = CONS(DIMENSION, make_dimension(int_to_expression(1), expr_nb_region), dims);
      dimension bounds = make_dimension(step_symbolic(STEP_INDEX_SLICE_LOW_NAME, module),
					step_symbolic(STEP_INDEX_SLICE_UP_NAME, module));
      dims = CONS(DIMENSION, bounds, CONS(DIMENSION, dim_array, dims));
    }
  else
    {
      dimension bounds = make_dimension(int_to_expression(0), MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
									     step_symbolic(STEP_INDEX_SLICE_UP_NAME, module),
									     int_to_expression(1)));
      dims = CONS(DIMENSION, bounds, dims);
      
      dimension dim_array = make_dimension(int_to_expression(0), MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
										int_to_expression(NumberOfDimension(array)),
										int_to_expression(1)));
      dims = CONS(DIMENSION, dim_array, dims);

      if(!expression_undefined_p(expr_nb_region))
	dims = CONS(DIMENSION, make_dimension(int_to_expression(0), MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
										   expr_nb_region,
										   int_to_expression(1))), dims);
    }



  entity e = step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, name, dims, false);
  free(name);
  pips_assert("variable", entity_variable_p(e));
  return e;
}


entity step_local_slice_index(entity module)
{
  return step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_SLICE_INDEX_NAME, NIL, false);
}

/* 
   Ajout dans la nouvelle fonction MPI de la declaration du tableau
   contenant les tranches d'indices a traiter par chaque noeud

   Ce tableau prend comme nom le nom de l'indice de la boucle prefixe par STEP_, suffixe par LOOPSLICES:

   ex: STEP_I_LOOPSLICES
*/
entity step_local_loopSlices(entity module, entity i)
{
  list dims = CONS(DIMENSION, make_dimension(step_symbolic(STEP_INDEX_SLICE_LOW_NAME, module),
					     step_symbolic(STEP_INDEX_SLICE_UP_NAME, module)),
		   CONS(DIMENSION, make_dimension(int_to_expression(1), 
						  step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, module)), NIL));
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, STEP_LOOPSLICES_NAME(i), dims, false);
}


expression step_local_requests_array(entity module, expression nb_communication)
{
  expression step_max_nb_request = step_parameter_max_nb_request(module,nb_communication);
  list dims = CONS(DIMENSION, make_dimension(int_to_expression(1), step_max_nb_request), NIL);
  return entity_to_expression(step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_REQUEST_NAME, dims, false));
}

expression step_local_nb_request(entity module)
{
  entity e = step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_NBREQUEST_NAME, NIL, true);
  return entity_to_expression(e);
}

expression step_local_rank(entity module)
{
  bool in_header=fortran_module_p(get_current_module_entity());
  entity e = step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_COMM_RANK_NAME, NIL, in_header);
  return entity_to_expression(e);
}

expression step_local_size(entity module)
{
  bool in_header=fortran_module_p(get_current_module_entity());
  entity e = step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_COMM_SIZE_NAME, NIL, in_header);
  return entity_to_expression(e);
}

entity step_local_loop_index(entity module, string name)
{
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, name, NIL, false);
}


/*
  Generation de statement 
*/
static entity step_type(type t)
{
  basic b;
  pips_assert("check step_type",type_variable_p(t));
  b=variable_basic(type_variable(t));

  switch (basic_tag(b))
    {
    case is_basic_int:
      switch (basic_int(b))
	{
        case 1: return MakeConstant(STEP_INT_1_NAME,is_basic_string);
	case 2: return MakeConstant(STEP_INT_2_NAME,is_basic_string);
	case 4: return MakeConstant(STEP_INT_4_NAME,is_basic_string);
	case 8: return MakeConstant(STEP_INT_8_NAME,is_basic_string);
	default: pips_user_error("unexpected basic int : %i",basic_int(b));
	}
      break;
    case is_basic_float:
      switch (basic_float(b))
	{
	case 4: return MakeConstant(STEP_REAL4_NAME,is_basic_string);
	case 8: return MakeConstant(STEP_REAL8_NAME,is_basic_string);
	default: pips_user_error("unexpected basic float : %i",basic_float(b));
	}
      break;
    case is_basic_complex:
      switch (basic_complex(b))
	{
	case 8: return MakeConstant(STEP_COMPLEX8_NAME,is_basic_string);
	case 16: return MakeConstant(STEP_COMPLEX16_NAME,is_basic_string);
	default: pips_user_error("unexpected basic complex : %i",basic_complex(b));
	}
      break;
    default:
      pips_user_error("unexpected basic type : %i",basic_tag(b));
      break;
    }
  return entity_undefined;
}

static string step_type_suffix(string name, type t)
{
  string s,size, type_sufix;
  basic b;
  pips_assert("check step_type",type_variable_p(t));
  b=variable_basic(type_variable(t));
  switch (basic_tag(b))
    {
    case is_basic_int: type_sufix=strdup("_I");
      break;
    case is_basic_float: type_sufix=strdup("_R");
      break;
    case is_basic_complex: type_sufix=strdup("_C");
      break;
    default:
      pips_internal_error("unexpected basic type : %i",basic_tag(b));
      break;
    }
  size=i2a(basic_int(b));
  s=strdup(concatenate(name,type_sufix,size,NULL));
  free(type_sufix);
  free(size);
  pips_debug(1,"%s\n",s);
  return s;
}

statement call_STEP_subroutine(string name, list args, type t)
{
  statement statmt;

  pips_debug(1, "name = %s, arg = %p\n", name, args);
  if(!type_undefined_p(t))
    {
      string runtime=get_string_property("STEP_RUNTIME");
      if(strncmp("c",runtime,strlen(runtime))==0)
	{
	  entity step_symbolique_type=step_type(t);
	  pips_assert("defined symbolique type",!entity_undefined_p(step_symbolique_type));
	  args=gen_nconc(args,CONS(EXPRESSION,entity_to_expression(step_symbolique_type),NIL));
	}
      else
	name=step_type_suffix(name, t);
    }

  statmt = make_call_statement(name, args, entity_undefined, string_undefined);

 
  STEP_DEBUG_STATEMENT(3, "call STEP_subroutine", statmt);

  pips_debug(1, "statmt = %p\n", statmt);
  return statmt;
}



/*
  void step_init_arrayregions(void *array, int *type, int *dims, index_t L1, index_t U1,...)
*/
statement build_call_STEP_init_arrayregions(entity array)
{
  type t = entity_type(array);
  list dims = variable_dimensions(type_variable(t));

  expression expr_array = entity_to_expression(array);
  expression expr_type = entity_to_expression(step_type(t));
  
  expression expr_dims = int_to_expression(gen_length(dims));

  list args = CONS(EXPRESSION, expr_dims,
		   CONS(EXPRESSION, expr_type,
			CONS(EXPRESSION, expr_array, NIL)));

  FOREACH(DIMENSION, bounds_d, dims)
    {
      expression bounds_lower = copy_expression(dimension_lower(bounds_d));
      expression bounds_upper = copy_expression(dimension_upper(bounds_d));
      args = CONS(EXPRESSION, bounds_upper,
		  CONS(EXPRESSION, bounds_lower, args));
    }

  statement statmt = make_call_statement(RT_STEP_InitArrayRegions, gen_nreverse(args), entity_undefined, string_undefined);
  return statmt;
}


/*
  void step_alltoall_full(void *array, int *algorithm, int *tag)
  void step_alltoall_full_interlaced(void *array, int *algorithm, int *tag)
  void step_alltoall_partial(void *array, int *algorithm, int *tag)
  void step_alltoall_partial_interlaced(void *array, int *algorithm, int *tag)

*/
statement build_call_STEP_AllToAll(entity module, entity array, bool is_optimized, bool is_interlaced)
{
  statement statmt;
  string subroutine;

  if (is_optimized)
    {
      if(is_interlaced)
	subroutine=strdup(RT_STEP_AllToAll_PartialInterlaced);
      else
	subroutine=strdup(RT_STEP_AllToAll_Partial);
    }
  else
    {
      if(is_interlaced)
	subroutine=strdup(RT_STEP_AllToAll_FullInterlaced);
      else
	subroutine=strdup(RT_STEP_AllToAll_Full);
    }

  expression expr_array = entity_to_expression(array);
  expression expr_algorithm = step_symbolic(STEP_NONBLOCKING_NAME, module);
  expression expr_tag = step_symbolic(STEP_TAG_DEFAULT_NAME, module);
  list args = CONS(EXPRESSION, expr_array,
		   CONS(EXPRESSION, expr_algorithm,
			CONS(EXPRESSION, expr_tag, NIL)));

  statmt = make_call_statement(subroutine, args, entity_undefined, string_undefined);
  free(subroutine);

  return statmt;
}

statement build_call_STEP_WaitAll(list comm_stmt)
{
  if(ENDP(comm_stmt))
    return make_continue_statement(entity_undefined);
  else
    {
      statement block = make_block_statement(gen_nconc(comm_stmt,
						       CONS(STATEMENT, call_STEP_subroutine(RT_STEP_WaitAll, NIL, type_undefined),NIL)));
      string comment = strdup(concatenate("\nC     Communicating data to other nodes",
					  "\nC     3 communication shemes for all-to-all personalized broadcast :",
					  "\nC     STEP_NONBLOCKING, STEP_BLOCKING1 and STEP_BLOCKING2.\n",NULL));
      put_a_comment_on_a_statement(block, comment);
      return block;
    }
}
statement critical_build_call_STEP_WaitAll(list comm_stmt)
{
  if(ENDP(comm_stmt))
    return make_continue_statement(entity_undefined);
  else
    {
      statement block = make_block_statement(gen_nconc(comm_stmt,
						       CONS(STATEMENT, call_STEP_subroutine(RT_STEP_WaitAll, NIL, type_undefined),NIL)));
      return block;
    }
}

/*
  void step_set_sendregions(void *array, int *nb_workchunks, index_t *regions)
  void step_set_interlaced_sendregions(void *array, int *nb_workchunks, index_t *regions)
*/
statement build_call_STEP_set_send_region(entity array, expression expr_nb_workchunk, entity regions, bool is_interlaced, bool is_reduction)
{
  statement statmt;
  string subroutine;
  expression expr_array = entity_to_expression(array);
  expression expr_regions = entity_to_expression(regions);
  list args = CONS(EXPRESSION, expr_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions,NIL)));
  if (is_reduction)
    subroutine=strdup(RT_STEP_Set_ReductionSendRegions);
  else  if(is_interlaced)
    subroutine=strdup(RT_STEP_Set_InterlacedSendRegions);
  else
    subroutine=strdup(RT_STEP_Set_SendRegions);
  statmt = make_call_statement(subroutine, args, entity_undefined, string_undefined);
  free(subroutine);

  return statmt;
}

/*
  void step_set_recvregions(void *array, int *nb_workchunks, index_t *regions)
*/
statement build_call_STEP_set_recv_region(entity array, expression expr_nb_workchunk, entity regions)
{
  statement statmt;
  expression expr_array = entity_to_expression(array);
  expression expr_regions = entity_to_expression(regions);
  list args = CONS(EXPRESSION, expr_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions,NIL)));
  statmt = make_call_statement(RT_STEP_Set_RecvRegions, args, entity_undefined, string_undefined);
  return statmt;
}
