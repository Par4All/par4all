/* Copyright 2009 Alain Muller

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"


statement step_compile_parallel(int step_transformation, entity new_module, statement work)
{
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is parallel directive",type_directive_omp_parallel_p(directive_type(d)));

  if (step_transformation == STEP_TRANSFORMATION_OMP ||
      step_transformation == STEP_TRANSFORMATION_HYBRID )
    add_pragma_entity_to_statement(work, directive_module);

  return work;
}

