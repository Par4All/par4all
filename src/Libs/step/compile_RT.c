/*
  Taille des entiers utilises en parametre pour l'API de la runtime

!!!!! Leur coherence avec  STEP.h, steprt_f.h et la runtime doit etre assure manuellement !!!!!

*/

#include "defines-local.h"
#include "syntax.h" // for MakeParameter
#include "pipsmake.h" // for compilation_unit_of_module
#include "c_syntax.h" // for put_new_typedef
#include "preprocessor.h" // for AddEntityToModuleCompilationUnit

#define STEP_RT_INTEGER4 DEFAULT_INTEGER_TYPE_SIZE
#define STEP_RT_INTEGER8 DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE

#define STEP_RT_SYMBOLIC_INTEGER STEP_RT_INTEGER4
#define STEP_RT_ARRAY_INDEX_INTEGER STEP_RT_INTEGER4
#define STEP_RT_LOOP_INDEX_INTEGER STEP_RT_INTEGER4

//######################################################

static expression cast_STEP_ARG(expression expr, bool ptr)
{
  if (fortran_module_p(get_current_module_entity()))
    return expr;

  string cu = compilation_unit_of_module(get_current_module_name());
  entity step_arg_e = FindOrCreateEntity(cu, TYPEDEF_PREFIX "STEP_ARG");

  if(storage_undefined_p(entity_storage(step_arg_e)))
    {
      entity_storage(step_arg_e) = make_storage_rom();
      put_new_typedef("STEP_ARG");
    }

  type step_arg_t =MakeTypeVariable(make_basic_typedef(step_arg_e), NIL);
  if(ptr)
    step_arg_t = MakeTypeVariable(make_basic_pointer(step_arg_t), NIL);
  return MakeCastExpression(step_arg_t, expr);
}

/*
  Declaration des parameter
*/
static list local_declaration = NIL;
static set step_created_entity = set_undefined;

void step_RT_set_local_declarations(entity module, statement body)
{
  FOREACH(ENTITY, e, local_declaration)
    {
      AddLocalEntityToDeclarations(e, module, body);
    }

  local_declaration = NIL;
}

void set_RT_add_local(entity e)
{
  if(set_undefined_p(step_created_entity))
    step_created_entity = set_make(set_pointer);
  set_add_element(step_created_entity, step_created_entity, e);
}

void step_RT_clean_local()
{
  if(set_undefined_p(step_created_entity))
    return;

  /* reinitialisation du type sinon, la recompilation echoue */
  SET_FOREACH(entity, e, step_created_entity)
    {
      if(!type_undefined_p(entity_type(e)))
	{
	  free_type(entity_type(e));
	  entity_type(e) = type_undefined;
	  free_storage(entity_storage(e));
	  entity_storage(e) = storage_undefined;
	  free_value(entity_initial(e));
	  entity_initial(e) = value_undefined;
	}
    }
  set_clear(step_created_entity);
}

entity step_parameter(string name_, entity module, expression expr)
{
  entity e;
  string name = strdup(name_);

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
	  entity_type(e)=MakeTypeVariable(make_basic_int(STEP_RT_SYMBOLIC_INTEGER), NIL);
	  if(expression_undefined_p(expr))
	    e = MakeParameter(e, int_to_expression(0)); // a default value
	  else
	    e = MakeParameter(e, expr);
	}
      pips_assert("symbolic", entity_symbolic_p(e));
      set_RT_add_local(e);
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

      string cu = compilation_unit_of_module(get_current_module_name());
      e = FindOrCreateEntity(cu, name);
      pips_assert("not null", e != NULL);
      pips_assert("entity defined", !entity_undefined_p(e));

      if (type_undefined_p(entity_type(e)))
	{
	  entity area = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STATIC_AREA_LOCAL_NAME);
	  entity_type(e) = make_type(is_type_variable, make_variable(make_basic_int(STEP_RT_SYMBOLIC_INTEGER),
								     NIL, CONS(QUALIFIER, make_qualifier_const(), NIL)));
	  entity_storage(e) = make_storage_ram(make_ram(get_top_level_entity(), area, add_C_variable_to_area(area, e), NIL));
	  if(expression_undefined_p(expr))
	    entity_initial(e) = make_value_expression(int_to_expression(0)); // a default value
	  else
	    entity_initial(e) = make_value_expression(expr);
	  AddEntityToModuleCompilationUnit(e, get_current_module_entity());
	}
    }
  free(name);
  return e;
}

expression step_symbolic(string name, entity module)
{
  return entity_to_expression(step_parameter(name , module, expression_undefined));
}

//######################################################
expression step_function(string name, list args)
{
  entity e = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, name, NULL), entity_domain);
  pips_assert("boostrap defined", !entity_undefined_p(e));
  pips_assert("functional", type_functional_p(entity_type(e)));
  return make_call_expression(e, args);
}

//######################################################
/*
  Pour declarations de tableaux/variables d'entier
*/
static entity step_local_RT_Integer(int size_of_integer, entity mpi_module, string name_, list dims)
{
  string name = strdup(name_);
  entity e = FindOrCreateEntity(entity_user_name(mpi_module), name);
  pips_assert("not null", e != NULL);
  pips_assert("entity defined", !entity_undefined_p(e));

  if (type_undefined_p(entity_type(e)))
    {
      entity area = FindOrCreateEntity(entity_user_name(mpi_module), DYNAMIC_AREA_LOCAL_NAME);
      entity_type(e) = MakeTypeVariable(make_basic_int(size_of_integer), dims);
      entity_storage(e) = make_storage_ram(make_ram(mpi_module, area, add_variable_to_area(area, e), NIL));
      if(mpi_module != get_current_module_entity())
	local_declaration = gen_nconc(local_declaration, CONS(ENTITY, e, NIL));
    }
  pips_assert("variable", entity_variable_p(e));
  free(name);

  set_RT_add_local(e);
  return e;
}

/*
  A partir d'un module, d'un tableau, creation d'un tableau des regions SEND (ou RECV) pour chaque workchunk
*/
entity step_local_arrayRegions(string array_regions_name ,entity module, entity array, expression expr_nb_region)
{
  bool is_fortran = fortran_module_p(get_current_module_entity());

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

  entity e = step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, name, dims);
  free(name);
  pips_assert("variable", entity_variable_p(e));
  return e;
}


entity step_local_slice_index(entity module)
{
  return step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, STEP_SLICE_INDEX_NAME, NIL);
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
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, STEP_LOOPSLICES_NAME(i), dims);
}

expression step_local_rank(entity module)
{
  entity e = step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, STEP_COMM_RANK_NAME, NIL);
  return entity_to_expression(e);
}

expression step_local_size(entity module)
{
  entity e = step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, STEP_COMM_SIZE_NAME, NIL);
  return entity_to_expression(e);
}

entity step_local_loop_index(entity module, string name)
{
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, name, NIL);
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
        case 1: return MakeConstant(STEP_INTEGER1_NAME, is_basic_string);
	case 2: return MakeConstant(STEP_INTEGER2_NAME, is_basic_string);
	case 4: return MakeConstant(STEP_INTEGER4_NAME, is_basic_string);
	case 8: return MakeConstant(STEP_INTEGER8_NAME, is_basic_string);
	default: pips_user_error("unexpected basic int : %i", basic_int(b));
	}
      break;
    case is_basic_float:
      switch (basic_float(b))
	{
	case 4: return MakeConstant(STEP_REAL4_NAME, is_basic_string);
	case 8: return MakeConstant(STEP_REAL8_NAME, is_basic_string);
	default: pips_user_error("unexpected basic float : %i", basic_float(b));
	}
      break;
    case is_basic_complex:
      switch (basic_complex(b))
	{
	case 8: return MakeConstant(STEP_COMPLEX8_NAME, is_basic_string);
	case 16: return MakeConstant(STEP_COMPLEX16_NAME, is_basic_string);
	default: pips_user_error("unexpected basic complex : %i", basic_complex(b));
	}
      break;
    default:
      pips_user_error("unexpected basic type : %i", basic_tag(b));
      break;
    }
  return entity_undefined;
}

statement call_STEP_subroutine(string name, list args, type t)
{
  statement statmt;

  pips_debug(1, "name = %s, arg = %p\n", name, args);
  if(!type_undefined_p(t))
    {
      entity step_symbolique_type = step_type(t);
      pips_assert("defined symbolique type", !entity_undefined_p(step_symbolique_type));
      args=gen_nconc(args, CONS(EXPRESSION,entity_to_expression(step_symbolique_type), NIL));
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

  statement statmt = make_call_statement(RT_STEP_init_arrayregions, gen_nreverse(args), entity_undefined, string_undefined);
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
	subroutine=strdup(RT_STEP_alltoall_partial_interlaced);
      else
	subroutine=strdup(RT_STEP_register_alltoall_partial);
    }
  else
    {
      if(is_interlaced)
	subroutine=strdup(RT_STEP_alltoall_full_interlaced);
      else
	subroutine=strdup(RT_STEP_alltoall_full);
    }

  expression expr_array = entity_to_expression(array);
  expression expr_algorithm = step_symbolic(STEP_NBLOCKING_ALG_NAME, module);
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
						       CONS(STATEMENT, call_STEP_subroutine(RT_STEP_flush, NIL, type_undefined), NIL)));
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
						       CONS(STATEMENT, call_STEP_subroutine(RT_STEP_waitall, NIL, type_undefined), NIL)));
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
  expression expr_regions = cast_STEP_ARG(entity_to_expression(regions), true);
  list args = CONS(EXPRESSION, expr_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions,NIL)));
  if (is_reduction)
    subroutine = strdup(RT_STEP_set_reduction_sendregions);
  else  if(is_interlaced)
    subroutine = strdup(RT_STEP_set_interlaced_sendregions);
  else
    subroutine = strdup(RT_STEP_set_sendregions);
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
  expression expr_regions = cast_STEP_ARG(entity_to_expression(regions), true);
  list args = CONS(EXPRESSION, expr_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions, NIL)));
  statmt = make_call_statement(RT_STEP_set_recvregions, args, entity_undefined, string_undefined);
  return statmt;
}
