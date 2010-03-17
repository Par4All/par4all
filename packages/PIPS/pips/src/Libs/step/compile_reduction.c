/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

GENERIC_GLOBAL_FUNCTION(reduction_entities,step_entity_map)

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

static step_reduction get_reductions_clause(entity module)
{
  pips_assert("directive module",bound_global_directives_p(module));
  directive d=load_global_directives(module);
  step_reduction r=step_reduction_undefined;
  pips_assert("loop directive",type_directive_omp_parallel_do_p(directive_type(d))
	      ||  type_directive_omp_do_p(directive_type(d)));

  // recherche de la clause reduction dans la listes des clauses de la directive
  FOREACH(CLAUSE,c,directive_clauses(d))
    {
      if (clause_step_reduction_p(c))
	{
	  pips_assert("only one clause reduction per directive",step_reduction_undefined_p(r));
	  r=clause_step_reduction(c);
	}
    }

  return r;
}

void step_reduction_before(entity directive_module, entity mpi_module)
{
  list arglist;
  expression variable,variable_reduc,operator;
  STEP_REDUCTION_MAP(e,op,{
      step_declare_reduction_variable(mpi_module,e);
      variable=entity_to_expression(e);
      variable_reduc=entity_to_expression(load_reduction_entities(e));
      operator=entity_to_expression(MakeConstant(op_to_step_op(op),is_basic_string));
      arglist=CONS(EXPRESSION,variable,
		   CONS(EXPRESSION,variable_reduc,
      			CONS(EXPRESSION,operator,NIL)));
      step_seqlist=CONS(STATEMENT,call_STEP_subroutine(strdup(concatenate(RT_STEP_InitReduction,step_type_suffix(e),NULL)),arglist),step_seqlist);
      pips_debug(1,"reduction %s : %s\n",entity_name(e),op);
    },get_reductions_clause(directive_module));
  
  return ;
}

void step_reduction_after(entity directive_module)
{
  list arglist;
  expression variable,variable_reduc,operator;
  STEP_REDUCTION_MAP(e,op,{
      variable=entity_to_expression(e);
      variable_reduc=entity_to_expression(load_reduction_entities(e));
      operator=entity_to_expression(MakeConstant(op_to_step_op(op),is_basic_string));
      arglist=CONS(EXPRESSION,variable,
		   CONS(EXPRESSION,variable_reduc,
      			CONS(EXPRESSION,operator,NIL)));
      step_seqlist=CONS(STATEMENT,call_STEP_subroutine(strdup(concatenate(RT_STEP_Reduction,step_type_suffix(e),NULL)),arglist),step_seqlist);
      pips_debug(1,"reduction %s : %s\n",entity_name(e),op);}
    ,get_reductions_clause(directive_module));

  return;
}
