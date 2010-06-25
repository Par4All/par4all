#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
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

static SymbolicDescriptor SymbolicDescriptorTable[] = {
  {DECLARE_LOCAL,      STEP_RT_NB_WORKCHUNK_INTEGER, -1, STEP_MAX_NB_REQUEST_NAME},
  {DECLARE_STEP_H,     STEP_RT_SYMBOLIC_INTEGER,      1, STEP_INDEX_SLICE_LOW_NAME},
  {DECLARE_STEP_H,     STEP_RT_SYMBOLIC_INTEGER,      2, STEP_INDEX_SLICE_UP_NAME},
  {DECLARE_STEP_H,     STEP_RT_NB_WORKCHUNK_INTEGER, 16, STEP_MAX_NB_LOOPSLICES_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      0, STEP_NONBLOCKING_NAME},
  {DECLARE_STEPRT_F_H, STEP_RT_SYMBOLIC_INTEGER,      0, STEP_TAG_DEFAULT_NAME},
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
  string name = strdup(name_);
  entity e = FindOrCreateEntity(entity_user_name(module), name);
  pips_assert("not null", e != NULL);
  pips_assert("entity defined", !entity_undefined_p(e));

  if (type_undefined_p(entity_type(e)))
    {
      SymbolicDescriptor sym_desc = find_SymbolicDescriptor_by_name(name);

      entity_type(e)=MakeTypeVariable(make_basic_int(sym_desc.size_of_integer), NIL);
      if(expression_undefined_p(expr))
	e = MakeParameter(e, int_to_expression(sym_desc.value));
      else
	e = MakeParameter(e, expr);
      
      if (sym_desc.declaration_file == DECLARE_LOCAL)
	code_declarations(EntityCode(module)) = gen_nconc(code_declarations(EntityCode(module)), CONS(ENTITY, e, NIL));
    }
  free(name);
  pips_assert("symbolic", entity_symbolic_p(e));
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
      if(!header)
	code_declarations(EntityCode(mpi_module))=gen_nconc(code_declarations(EntityCode(mpi_module)),CONS(ENTITY,e,NIL));
    }
  pips_assert("variable",entity_variable_p(e));
  free(name);
  return e;
}


/*
  A partir d'un module, d'un tableau, creation d'un tableau SR stockant pour chaque noeud les bornes d'indices du tableau traites
*/
entity step_local_SR(entity module, entity array, expression expr_nb_region)
{
  string name = strdup(STEP_SR_NAME(array));
  dimension dim_array = make_dimension(int_to_expression(1), int_to_expression(NumberOfDimension(array)));
  dimension bounds = make_dimension(step_symbolic(STEP_INDEX_SLICE_LOW_NAME, module),
				    step_symbolic(STEP_INDEX_SLICE_UP_NAME, module));
  list dims = NIL;
  if(!expression_undefined_p(expr_nb_region))
    dims = CONS(DIMENSION, make_dimension(int_to_expression(0), expr_nb_region), dims);
  dims = CONS(DIMENSION, bounds, CONS(DIMENSION, dim_array, dims));

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
  entity e = step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_COMM_RANK_NAME, NIL, true);
  return entity_to_expression(e);
}

expression step_local_size(entity module)
{
  entity e = step_local_RT_Integer(STEP_RT_NB_WORKCHUNK_INTEGER, module, STEP_COMM_SIZE_NAME, NIL, true);
  return entity_to_expression(e);
}

entity step_local_loop_index(entity module, string name)
{
  return step_local_RT_Integer(STEP_RT_LOOP_INDEX_INTEGER, module, name, NIL, false);
}


/*
  Génération de statement 
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
      pips_error("step_type_suffix","unexpected basic type : %i",basic_tag(b));
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
