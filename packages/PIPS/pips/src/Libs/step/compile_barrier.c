/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"

statement step_compile_barrier(int step_transformation, entity new_module, statement work)
{
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is barrier directive",type_directive_omp_barrier_p(directive_type(d)));

  if (step_transformation == STEP_TRANSFORMATION_OMP)
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      statement call_stmt=call_STEP_subroutine(RT_STEP_Barrier,NIL,type_undefined);

      if (step_transformation == STEP_TRANSFORMATION_HYBRID)
	add_pragma_entity_to_statement(call_stmt,directive_module);

      return make_block_statement(CONS(STATEMENT, call_stmt,NIL));
    }
}
