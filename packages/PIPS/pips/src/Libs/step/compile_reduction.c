/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"

GENERIC_LOCAL_FUNCTION(reduction_entities,map_entity_entity)

static string op_to_step_op(string op)
{
  if (!strcasecmp(op,"+")) return STEP_SUM_NAME;
  if (!strcasecmp(op,"*")) return STEP_PROD_NAME;
  pips_user_warning("STEP : reduction operator : \"%s\" not yet implemented\n\n",op);
  if (!strcasecmp(op,"-")) return STEP_MINUS_NAME;
  if (!strcasecmp(op,".and.")) return STEP_AND_NAME;
  if (!strcasecmp(op,".or.")) return STEP_OR_NAME;
  if (!strcasecmp(op,".eqv.")) return STEP_EQV_NAME;
  if (!strcasecmp(op,".neqv.")) return STEP_NEQV_NAME;
  if (!strcasecmp(op,"max")) return STEP_MAX_NAME;
  if (!strcasecmp(op,"min")) return STEP_MIN_NAME;
  if (!strcasecmp(op,"iand")) return STEP_IAND_NAME;
  if (!strcasecmp(op,"ior")) return STEP_IOR_NAME;
  if (!strcasecmp(op,"ieor")) return STEP_IEOR_NAME;
  return "";
}

static entity step_declare_reduction_variable(entity module,entity variable)
{
  entity new;
  entity area;
  int offset;

  pips_debug(1, "module = %p, reduction_entity = %p\n", module,  variable);

  new = FindOrCreateEntity(entity_user_name(module),strdup(STEP_REDUC_NAME(variable)));

    entity_type(new)=MakeTypeVariable(variable_basic(type_variable(entity_type(variable))),NIL);
  area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);
  offset = add_variable_to_area(area, new);
  entity_storage(new)=make_storage(is_storage_ram,make_ram(module,area,offset,NIL));
  store_reduction_entities(variable, new);
  AddEntityToDeclarations(new,module);
  return new;
}

static map_entity_string get_reductions_clause(entity module)
{
  map_entity_string reductions;
  directive d;

  pips_assert("directive module",bound_global_directives_p(module));
  d=load_global_directives(module);

  pips_assert("loop directive",type_directive_omp_parallel_do_p(directive_type(d))
	      ||  type_directive_omp_do_p(directive_type(d)));

  reductions=map_entity_string_undefined;
  // recherche de la clause reduction dans la listes des clauses de la directive
  FOREACH(CLAUSE,c,directive_clauses(d))
    {
      if (clause_reduction_p(c))
	{
	  pips_assert("only one clause reduction per directive",map_entity_string_undefined_p(reductions));
	  reductions=clause_reduction(c);
	}
    }

  return reductions;
}

list step_reduction_before(entity directive_module, entity mpi_module)
{
  list arglist;
  expression variable,variable_reduc,operator;
  list entity_reduction=NIL;
  map_entity_string reductions=get_reductions_clause(directive_module);
  list lstmt=NIL;

  init_reduction_entities();

  MAP_ENTITY_STRING_MAP(e, __attribute__ ((unused))op,{
      entity_reduction=CONS(ENTITY,e,entity_reduction);
    },reductions);
  sort_list_of_entities(entity_reduction);

  FOREACH(ENTITY,e,entity_reduction)
    {
      string op=apply_map_entity_string(reductions,e);
      step_declare_reduction_variable(mpi_module,e);
      variable=entity_to_expression(e);
      variable_reduc=entity_to_expression(load_reduction_entities(e));
      operator=entity_to_expression(MakeConstant(op_to_step_op(op),is_basic_string));
      arglist=CONS(EXPRESSION,variable,
		   CONS(EXPRESSION,variable_reduc,
      			CONS(EXPRESSION,operator,NIL)));
      lstmt=CONS(STATEMENT,call_STEP_subroutine(RT_STEP_InitReduction, arglist, entity_type(e)), lstmt);
      pips_debug(1,"reduction %s : %s\n",entity_name(e),op);
    }
  gen_free_list(entity_reduction);
  return lstmt;
}

list step_reduction_after(entity directive_module)
{
  list arglist;
  expression variable,variable_reduc,operator;
  list entity_reduction=NIL;
  map_entity_string reductions=get_reductions_clause(directive_module);
  list lstmt=NIL;

  MAP_ENTITY_STRING_MAP(e, __attribute__ ((unused))op,{
      entity_reduction=CONS(ENTITY,e,entity_reduction);
    },reductions);
  sort_list_of_entities(entity_reduction);

  FOREACH(ENTITY,e,entity_reduction)
    {
      string op=apply_map_entity_string(reductions,e); 
      variable=entity_to_expression(e);
      variable_reduc=entity_to_expression(load_reduction_entities(e));
      operator=entity_to_expression(MakeConstant(op_to_step_op(op),is_basic_string));
      arglist=CONS(EXPRESSION,variable,
		   CONS(EXPRESSION,variable_reduc,
      			CONS(EXPRESSION,operator,NIL)));
      lstmt=CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Reduction, arglist, entity_type(e)), lstmt);
      pips_debug(1,"reduction %s : %s\n",entity_name(e),op);
    }
  gen_free_list(entity_reduction);
  
  close_reduction_entities();
  return lstmt;
}

bool step_reduction_p(entity e)
{
  pips_assert("reduction initialized",!reduction_entities_undefined_p());
  return bound_reduction_entities_p(e);
}
