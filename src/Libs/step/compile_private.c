/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"

static list current_private_entity=list_undefined;

static step_private get_private_clause(entity module)
{
  pips_assert("directive module",bound_global_directives_p(module));
  directive d=load_global_directives(module);
  step_private privates=step_private_undefined;
  //  pips_assert("loop directive",type_directive_omp_parallel_do_p(directive_type(d))
  //      ||  type_directive_omp_do_p(directive_type(d)));
 
  // recherche de la clause private dans la listes des clauses de la directive
  FOREACH(CLAUSE,c,directive_clauses(d))
    {
      if (clause_step_private_p(c))
	{
	  pips_assert("only one clause private per directive",step_private_undefined_p(privates));
	  privates=clause_step_private(c);
	}
    }

  return privates;
}

void step_private_before(entity directive_module)
{
  pips_assert("current_private_entity not undefined",list_undefined_p(current_private_entity));
  current_private_entity=step_private_entity(get_private_clause(directive_module));
}

void step_private_after()
{
  pips_assert("current_private_entity undefined",!list_undefined_p(current_private_entity));
  current_private_entity=list_undefined;
}

bool bound_private_entities_p(entity e)
{
  pips_assert("current_private_entity undefined",!list_undefined_p(current_private_entity));
  return gen_in_list_p(e,current_private_entity);
}
