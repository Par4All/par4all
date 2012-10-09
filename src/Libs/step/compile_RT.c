/*
  Taille des entiers utilises en parametre pour l'API de la runtime

!!!!! Leur coherence avec  STEP.h, steprt_f.h et la runtime doit etre assuree manuellement !!!!!

*/
#include <stdarg.h>
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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

expression get_expression_addr(expression expr)
{
  expression expr2;

  /* Used to add & in front of variable name for C modules: var ---> &var */

  expr2 = expr;
  if (!fortran_module_p(get_current_module_entity()))
    expr2 = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr);

  return expr2;
}

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

  type step_arg_t = MakeTypeVariable(make_basic_typedef(step_arg_e), NIL);
  if(ptr)
    step_arg_t = MakeTypeVariable(make_basic_pointer(step_arg_t), NIL);
  return MakeCastExpression(step_arg_t, expr);
}

void generate_call_construct_begin_construct_end(entity new_module, step_directive drt, statement mpi_begin_stmt, statement mpi_end_stmt)
{
/*
    Generation of:
    STEP_CONSTRUCT_BEGIN(STEP_DO);
    ...
    STEP_CONSTRUCT_END(STEP_DO);
   */

  string directive_txt;
  get_step_directive_name(drt, &directive_txt);
  if(!string_undefined_p(directive_txt))
    {
      statement construct_begin_stmt;
      statement construct_end_stmt;

      construct_begin_stmt = call_STEP_subroutine2(RT_STEP_construct_begin, step_symbolic_expression(directive_txt, new_module), NULL);
      insert_statement(mpi_begin_stmt, construct_begin_stmt, true);

      construct_end_stmt = call_STEP_subroutine2(RT_STEP_construct_end, step_symbolic_expression(directive_txt, new_module), NULL);
      insert_statement(mpi_end_stmt, construct_end_stmt, false);
      free(directive_txt);
    }
}

void generate_call_flush(statement *stepalltoall_stmt)
{
  pips_debug(1, "begin\n");
  /*
    Generation of
    STEP_FLUSH();
  */
  if(!ENDP(statement_block(*stepalltoall_stmt)))
    {
      statement flush_stmt = build_call_STEP_flush();
      insert_statement(*stepalltoall_stmt, flush_stmt, false);
    }
  pips_debug(1, "end\n");
}


void generate_loop_workchunk(entity mpi_module, statement *compute_regions_stmt)
{
  statement loop_stmt;
  /* ATTENTION recrée alors qu'avant réutilisé */
  entity workchunk_id = step_local_slice_index(mpi_module);
  /*
    Generation of
    for(IDX = 1; IDX <= STEP_COMM_SIZE; IDX += 1) {}
  */
  range rng = make_range(int_to_expression(1), entity_to_expression(get_entity_step_commsize(mpi_module)), int_to_expression(1));

  loop_stmt = instruction_to_statement(make_instruction_loop(make_loop(workchunk_id, rng, *compute_regions_stmt, entity_empty_label(), make_execution_sequential(), NIL)));

  *compute_regions_stmt = make_empty_block_statement();
  insert_statement(*compute_regions_stmt, loop_stmt, true);

}

void generate_call_get_workchunk_loopbounds(entity mpi_module, loop loop_stmt, statement *compute_regions_stmt)
{
  pips_debug(1, "begin\n");
  /*
    Generation of:
    STEP_GET_LOOPBOUNDS(IDX-1, &STEP_i_LOW, &STEP_i_UP);
  */
  entity index = loop_index(loop_stmt);
  entity workchunk_id = step_local_slice_index(mpi_module);
  expression expr_id_workchunk = make_op_exp(PLUS_OPERATOR_NAME, int_to_expression(-1), entity_to_expression(workchunk_id));
  expression expr_index_low = entity_to_expression(step_local_loop_index(mpi_module, STEP_BOUNDS_LOW(index)));
  expression expr_index_up = entity_to_expression(step_local_loop_index(mpi_module, STEP_BOUNDS_UP(index)));

  statement get_bounds_stmt = call_STEP_subroutine2(RT_STEP_get_loopbounds, expr_id_workchunk, get_expression_addr(expr_index_low), get_expression_addr(expr_index_up), NULL);
  insert_statement(*compute_regions_stmt, get_bounds_stmt, true);

  pips_debug(1, "end\n");
}

statement generate_call_get_rank_loopbounds(entity new_module, loop loop_stmt)
{
  pips_debug(1, "begin\n");
  
  /*
    Generation of:
    CALL STEP_GETLOOPBOUNDS(STEP_Rank, &I_SLICE_LOW, &I_SLICE_UP)
  */
  entity index = loop_index(loop_stmt);
  entity index_low = step_local_loop_index(new_module, STEP_BOUNDS_LOW(index));
  expression expr_index_low = entity_to_expression(index_low);
  entity index_up = step_local_loop_index(new_module, STEP_BOUNDS_UP(index));
  expression expr_index_up = entity_to_expression(index_up);
  entity id_workchunk = get_entity_step_rank(new_module);
  expression expr_id_workchunk = entity_to_expression(id_workchunk);

  statement get_bounds_stmt = call_STEP_subroutine2(RT_STEP_get_loopbounds, expr_id_workchunk, get_expression_addr(expr_index_low), get_expression_addr(expr_index_up), NULL); 

  pips_debug(1, "end\n");
  
  return get_bounds_stmt;
}

statement generate_call_get_rank(entity new_module)
{
  statement stmt;
  /*
    Generation of:
    CALL STEP_GET_RANK(&STEP_COMM_RANK)
  */
  entity rank = get_entity_step_rank(new_module);
  expression expr_rank = entity_to_expression(rank);


  stmt = call_STEP_subroutine2(RT_STEP_get_rank, get_expression_addr(expr_rank), NULL);
  return stmt;
}

statement generate_call_compute_loopslices(entity new_module, loop loop_stmt)
{
  pips_debug(1, "begin new_module = %p, loop_stmt = %p\n", new_module, loop_stmt);
  statement stmt;
  /*
    Generation of:
    STEP_COMPUTE_LOOPSLICES(0, 99999, 1, STEP_COMM_SIZE);
  */
  entity commsize = get_entity_step_commsize(new_module);
  expression commsize_expr = entity_to_expression(commsize);
  range r = loop_range(loop_stmt);

  stmt = call_STEP_subroutine2(RT_STEP_compute_loopslices, range_lower(r), range_upper(r), range_increment(r), commsize_expr, NULL);
  pips_debug(1, "end\n");
  return stmt;
}

statement generate_call_get_commsize(entity new_module)
{
  pips_debug(1, "begin\n");
  statement stmt;
  /*
    Generation of:
    STEP_GET_COMMSIZE(&STEP_COMM_SIZE);
  */

  entity commsize = get_entity_step_commsize(new_module);
  expression commsize_expr = entity_to_expression(commsize);

  stmt = call_STEP_subroutine2(RT_STEP_get_commsize, get_expression_addr(commsize_expr), NULL);
  pips_debug(1, "end\n");
  return stmt;
}

void generate_call_init_regionArray(list referenced_entities, statement before, statement __attribute__ ((unused)) after)
{
  list init_block = NIL;

  /*
    Generation of

    STEP_INIT_ARRAYREGIONS(a, STEP_INTEGER4, 1, 0, 100000-1);
  */

  pips_debug(1, "begin\n");

  FOREACH(ENTITY, e, referenced_entities)
    {
      if (type_variable_p(entity_type(e)) && !entity_scalar_p(e))
	{
	  pips_debug(2,"build_call_STEP_init_regionArray entity %s\n", entity_name(e));
	  init_block = CONS(STATEMENT, build_call_STEP_init_regionArray(e), init_block);
	}
    }
  insert_statement(before, make_block_statement(gen_nreverse(init_block)), false);

  pips_debug(1, "end\n");
}

/*
  Parameters declaration.
*/

/*
  Macros for constants specific to STEP runtime API are defined in step_common.h.
  step_common.h is included in generated files.
  Macros must not be locally declared.
*/

set step_created_entity = set_undefined;

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

set step_created_symbolic = set_undefined;
static void step_add_created_symbolic(entity e)
{
  if(set_undefined_p(step_created_symbolic))
    step_created_symbolic = set_make(set_pointer);

  set_add_element(step_created_symbolic, step_created_symbolic, e);
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
	  entity_kind(e)=ABSTRACT_LOCATION;
	  AddEntityToModuleCompilationUnit(e, get_current_module_entity());
	}
      step_add_created_symbolic(e);
    }
  free(name);
  return e;
}

entity step_symbolic(string name, entity module)
{
  return step_parameter(name, module, expression_undefined);
}

expression step_symbolic_expression(string name, entity module)
{
  return entity_to_expression(step_symbolic(name, module));
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
/*
  liste contenant l'ensemble des tableaux/variables nécessaires à l'utilisation de l'API de STEP et devant être déclarer en temps que variables locales
*/
static list local_declaration = NIL;

void step_RT_set_local_declarations(entity module, statement body)
{
  FOREACH(ENTITY, e, local_declaration)
    {
      AddLocalEntityToDeclarations(e, module, body);
    }

  local_declaration = NIL;
}

static entity step_local_RT_Integer(int size_of_integer, entity mpi_module, string name_, list dims)
{
  pips_debug(1, "begin mpi_module = %s, name_ = %s\n", entity_name(mpi_module), name_);
  string name = strdup(name_);
  entity e = FindOrCreateEntity(entity_user_name(mpi_module), name);
  pips_assert("not null", e != NULL);
  pips_assert("entity defined", !entity_undefined_p(e));

  if (type_undefined_p(entity_type(e)))
    {
      pips_debug(2, "create entity VERIFIER CE MESSAGE\n");
 
      entity area = FindOrCreateEntity(entity_user_name(mpi_module), DYNAMIC_AREA_LOCAL_NAME);
      entity_type(e) = MakeTypeVariable(make_basic_int(size_of_integer), dims);
      entity_storage(e) = make_storage_ram(make_ram(mpi_module, area, add_variable_to_area(area, e), NIL));
      entity_kind(e)=ABSTRACT_LOCATION;
      if(mpi_module != get_current_module_entity())
	local_declaration = gen_nconc(local_declaration, CONS(ENTITY, e, NIL));
    }
  else
    pips_debug(2, "entity already exists\n");

  pips_assert("variable", entity_variable_p(e));
  free(name);

  set_RT_add_local(e);

  pips_debug(1, "end\n");
  return e;
}

/*
  For each module and array, creation of an array entity for SEND (and RECV) regions

  STEP_SR|RR_Array[STEP_MAX_NB_LOOPSLICES][NBDIMS][2]

  STEP_SR_d[1][STEP_INDEX_SLICE_UP]
  STEP_RR_a[STEP_MAX_NB_LOOPSLICES][1][STEP_INDEX_SLICE_UP]
  STEP_SR_d[STEP_MAX_NB_LOOPSLICES][1][STEP_INDEX_SLICE_UP]
  STEP_SR_c[STEP_MAX_NB_LOOPSLICES][2][STEP_INDEX_SLICE_UP]

*/
entity step_local_regionArray(entity module, entity array, string region_array_name, expression expr_nb_region)
{
  pips_debug(1, "begin module = %s, array = %s, region_array_name = %s, expr_nb_region = %p\n", entity_name(module), entity_name(array), region_array_name, expr_nb_region);

  bool is_fortran = fortran_module_p(get_current_module_entity());
  string name = strdup(region_array_name);
  list dims = NIL;
  if(is_fortran)
    {
      dimension dim_array = make_dimension(int_to_expression(1), int_to_expression(NumberOfDimension(array)));
      if(!expression_undefined_p(expr_nb_region))
	dims = CONS(DIMENSION, make_dimension(int_to_expression(1), expr_nb_region), dims);
      dimension bounds = make_dimension(step_symbolic_expression(STEP_INDEX_SLICE_LOW_NAME, module),
					step_symbolic_expression(STEP_INDEX_SLICE_UP_NAME, module));
      dims = CONS(DIMENSION, bounds, CONS(DIMENSION, dim_array, dims));
    }
  else
    {
      dimension bounds = make_dimension(int_to_expression(0), MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
									     step_symbolic_expression(STEP_INDEX_SLICE_UP_NAME, module),
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

  pips_debug(1, "end e = %p\n", e);
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
  list dims = CONS(DIMENSION, make_dimension(step_symbolic_expression(STEP_INDEX_SLICE_LOW_NAME, module),
					     step_symbolic_expression(STEP_INDEX_SLICE_UP_NAME, module)),
		   CONS(DIMENSION, make_dimension(int_to_expression(1),
						  step_symbolic_expression(STEP_MAX_NB_LOOPSLICES_NAME, module)), NIL));
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, STEP_LOOPSLICES_NAME(i), dims);
}

entity get_entity_step_rank(entity module)
{
  return step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, STEP_COMM_RANK_NAME, NIL);
}

/* Que signifie local? */
entity get_entity_step_commsize(entity module)
{
  entity e;

  pips_debug(3, "begin\n");

  e = step_local_RT_Integer(STEP_RT_ARRAY_INDEX_INTEGER, module, STEP_COMM_SIZE_NAME, NIL);

  pips_assert("type_variable_p", type_variable_p(entity_type(e)));
  pips_assert("entity_scalar_p", entity_scalar_p(e));

  pips_debug(3, "end entity name = %s (%p)\n", entity_name(e), e);
  return e;
}

entity step_local_loop_index(entity module, string name)
{
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, name, NIL);
}

/*
  Generation de statement
*/
entity step_type(entity data)
{
  type t;
  basic b;

  pips_assert("data", !entity_undefined_p(data));
  t = entity_type(data);
  pips_assert("check step_type", type_variable_p(t));
  b = variable_basic(type_variable(t));

  switch (basic_tag(b))
    {
    case is_basic_int:
      switch (basic_int(b))
	{
        case 1: return MakeConstant(STEP_INTEGER1_NAME, is_basic_string);
	case 2: return MakeConstant(STEP_INTEGER2_NAME, is_basic_string);
	case 4: return MakeConstant(STEP_INTEGER4_NAME, is_basic_string);
	case 8: return MakeConstant(STEP_INTEGER8_NAME, is_basic_string);
	default:
	  pips_debug(0, "unexpected basic int for entity %s\n", entity_name(data));
	  pips_user_error("unexpected basic int : %i\n", basic_int(b));
	}
      break;
    case is_basic_float:
      switch (basic_float(b))
	{
	case 4: return MakeConstant(STEP_REAL4_NAME, is_basic_string);
	case 8: return MakeConstant(STEP_REAL8_NAME, is_basic_string);
	default:
	  pips_debug(0, "unexpected basic float for entity %s\n", entity_name(data));
	  pips_user_error("unexpected basic float : %i\n", basic_float(b));
	}
      break;
    case is_basic_complex:
      switch (basic_complex(b))
	{
	case 8: return MakeConstant(STEP_COMPLEX8_NAME, is_basic_string);
	case 16: return MakeConstant(STEP_COMPLEX16_NAME, is_basic_string);
	default:
	  pips_debug(0, "unexpected basic complex for entity %s\n", entity_name(data));
	  pips_user_error("unexpected basic complex : %i\n", basic_complex(b));
	}
      break;
    default:
      pips_debug(0, "unexpected type for entity %s\n", entity_name(data));
      pips_user_error("unexpected basic type : %i\n", basic_tag(b));
      break;
    }
  return entity_undefined;
}

/*
  void step_init_regionArray(void *array, int *type, int *dims, index_t L1, index_t U1,...)

  Generation of:
  STEP_INIT_ARRAYREGIONS(a, STEP_INTEGER4, 1, 0, 100000-1);
*/
statement build_call_STEP_init_regionArray(entity array)
{
  pips_debug(1, "begin\n");

  type t = entity_type(array);
  list dims = variable_dimensions(type_variable(t));

  expression expr_array = entity_to_expression(array);
  expression expr_type = entity_to_expression(step_type(array));

  expression expr_dims = int_to_expression(gen_length(dims));

  list args = CONS(EXPRESSION, expr_dims,
		   CONS(EXPRESSION, expr_type,
			CONS(EXPRESSION, expr_array, NIL)));

  FOREACH(DIMENSION, bounds_d, dims)
    {
      if(unbounded_dimension_p(bounds_d))
	{
	  pips_debug(0, "Unbounded dimension for array : %s\n", entity_name(array));
	  pips_assert("bounded dimension", 0);
	}
      expression bounds_lower = copy_expression(dimension_lower(bounds_d));
      expression bounds_upper = copy_expression(dimension_upper(bounds_d));
      args = CONS(EXPRESSION, bounds_upper,
		  CONS(EXPRESSION, bounds_lower, args));
    }

  statement statmt = make_call_statement(RT_STEP_init_arrayregions, gen_nreverse(args), entity_undefined, string_undefined);
  pips_debug(1, "end\n");
  return statmt;
}

/*
  Generation of
      STEP_ALLTOALL_FULL(A, STEP_NBLOCKING_ALG, STEP_TAG_DEFAULT);
      STEP_REGISTER_ALLTOALL_PARTIAL(A, STEP_NBLOCKING_ALG, STEP_TAG_DEFAULT);

*/
statement build_call_STEP_AllToAll(entity module, entity array, bool is_partial, bool is_interlaced)
{
  statement statmt;
  string subroutine;

  if (is_partial)
    {
      /* Pourquoi register ? */

      if(is_interlaced)
	subroutine=strdup(RT_STEP_alltoall_partial_interlaced);
      else
	subroutine=strdup(RT_STEP_register_alltoall_partial);
    }
  else
    {
      /* Full communications */
      if(is_interlaced)
	subroutine=strdup(RT_STEP_alltoall_full_interlaced);
      else
	subroutine=strdup(RT_STEP_alltoall_full);
    }

  expression expr_array = entity_to_expression(array);
  expression expr_algorithm = step_symbolic_expression(STEP_NBLOCKING_ALG_NAME, module);
  expression expr_tag = step_symbolic_expression(STEP_TAG_DEFAULT_NAME, module);


  /*
    list args = CONS(EXPRESSION, expr_array,
    CONS(EXPRESSION, expr_algorithm,
    CONS(EXPRESSION, expr_tag, NIL)));
    statmt = make_call_statement(subroutine, args, entity_undefined, string_undefined);
  */
  statmt = call_STEP_subroutine2(subroutine, expr_array, expr_algorithm, expr_tag, NULL);
  
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
						       CONS(STATEMENT, call_STEP_subroutine2(RT_STEP_flush, NULL), NIL)));
      string comment = strdup(concatenate("\nC     Communicating data to other nodes",
					  "\nC     3 communication shemes for all-to-all personalized broadcast :",
					  "\nC     STEP_NONBLOCKING, STEP_BLOCKING1 and STEP_BLOCKING2.\n",NULL));
      put_a_comment_on_a_statement(block, comment);
      return block;
    }
}

/*
  Generation of
      STEP_SET_SENDREGIONS(a, 1, (STEP_ARG *) STEP_SR_a);
*/
statement build_call_STEP_set_sendregions(entity user_array, expression expr_nb_workchunk, entity regions_array, bool is_interlaced, bool is_reduction)
{
  statement statmt;
  string subroutine;

  pips_debug(1, "begin\n");
  expression expr_user_array = entity_to_expression(user_array);
  expression expr_regions_array = cast_STEP_ARG(entity_to_expression(regions_array), true);

  if (is_reduction)
    subroutine = strdup(RT_STEP_set_reduction_sendregions);
  else  if(is_interlaced)
    subroutine = strdup(RT_STEP_set_interlaced_sendregions);
  else
    subroutine = strdup(RT_STEP_set_sendregions);

  /*
  list args = CONS(EXPRESSION, expr_user_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions,NIL)));

  statmt = make_call_statement(subroutine, args, entity_undefined, string_undefined);
  */

  statmt = call_STEP_subroutine2(subroutine, expr_user_array, expr_nb_workchunk, expr_regions_array, NULL);

  free(subroutine);
  pips_debug(1, "end\n");
  return statmt;
}

/*
  Generation of
        STEP_SET_RECVREGIONS(a, STEP_COMM_SIZE, (STEP_ARG *) STEP_RR_a);
*/
statement build_call_STEP_set_recvregions(entity user_array, expression expr_nb_workchunk, entity regions_array)
{
  statement statmt;

  pips_debug(1, "begin\n");

  expression expr_user_array = entity_to_expression(user_array);
  expression expr_regions_array = cast_STEP_ARG(entity_to_expression(regions_array), true);

  statmt = call_STEP_subroutine2(RT_STEP_set_recvregions, expr_user_array, expr_nb_workchunk, expr_regions_array, NULL);

  /*
  list args = CONS(EXPRESSION, expr_user_array,
		   CONS(EXPRESSION, expr_nb_workchunk,
			CONS(EXPRESSION, expr_regions_array, NIL)));
  statmt = make_call_statement(RT_STEP_set_recvregions, args, entity_undefined, string_undefined);
  */

  pips_debug(1, "end\n");
  return statmt;
}

statement build_call_STEP_flush()
{
  return call_STEP_subroutine2(RT_STEP_flush, NULL);
}


/*

  Tentative de fonction se basant sur une liste variable d'arguments
  exprimés sous forme d'entity.

  Ne fonctionne pas à cause de range_lower et range_upper disponibles
  uniquement sous forme d'expressions.

  Tentative de passage par adresse automatique 

  Ne fonctionne pas parce qu'il n'y a pas de règle automatique

  Notamment ce n'est pas parce qu'une variable est de type scalaire
  qu'elle doit être passée par adresse.

 */

statement call_STEP_subroutine3(string name, ... )
{
  statement statmt;
  va_list va_args;
  list args_l = NIL;
  entity e;

  pips_debug(1, "name = %s\n", name);


  va_start (va_args, name);
  e = va_arg(va_args, entity);
  while (e != NULL) {
    expression expr;

    expr = entity_to_expression(e);
    
    if (type_variable_p(entity_type(e)) && entity_scalar_p(e))
      if (!fortran_module_p(get_current_module_entity()))
	expr = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr);

    args_l = CONS (EXPRESSION, expr, args_l);
    e = va_arg(va_args, entity);
  }
  va_end (va_args);

  statmt = make_call_statement(name, gen_nreverse(args_l), entity_undefined, string_undefined);

  STEP_DEBUG_STATEMENT(3, "call_STEP_subroutine2", statmt);

  pips_debug(1, "statmt = %p\n", statmt);
  return statmt;
}


/*
  call_STEP_subroutine2(string name, list of expressions representing arguments of the call )
 */
statement call_STEP_subroutine2(string name, ... )
{
  statement statmt;
  va_list va_args;
  list args_l = NIL;
  expression expr;

  pips_debug(1, "name = %s\n", name);


  va_start (va_args, name);
  expr = va_arg(va_args, expression);
  while (expr != NULL) {
    args_l = CONS (EXPRESSION, expr, args_l);
    expr = va_arg(va_args, expression);
  }
  va_end (va_args);

  statmt = make_call_statement(name, gen_nreverse(args_l), entity_undefined, string_undefined);

  STEP_DEBUG_STATEMENT(3, "call_STEP_subroutine2", statmt);

  pips_debug(1, "statmt = %p\n", statmt);
  return statmt;
}

void check_entity_step_type(entity data)
{
  entity step_symbolic_type;
  pips_debug(1, "begin\n");

  pips_assert("undefined entity", !entity_undefined_p(data));
  step_symbolic_type = step_type(data);
  pips_assert("defined symbolic type", !entity_undefined_p(step_symbolic_type));

  pips_debug(1, "end\n");
}

statement call_STEP_subroutine(string name, list args, entity data)
{
  statement statmt;

  pips_debug(1, "name = %s, arg = %p\n", name, args);
  if(!entity_undefined_p(data))
    {
      entity step_symbolic_type = step_type(data);
      pips_assert("defined symbolic type", !entity_undefined_p(step_symbolic_type));
      args = gen_nconc(args, CONS(EXPRESSION,entity_to_expression(step_symbolic_type), NIL));
    }

  statmt = make_call_statement(name, args, entity_undefined, string_undefined);


  STEP_DEBUG_STATEMENT(3, "call STEP_subroutine", statmt);

  pips_debug(1, "statmt = %p\n", statmt);
  return statmt;
}



