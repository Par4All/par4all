/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"

statement step_share_before(entity directive_module, __attribute__((unused)) entity mpi_module)
{
  list l_block = NIL;

  FOREACH(ENTITY, e, code_declarations(entity_code(directive_module)))
    {
      if (type_variable_p(entity_type(e)) &&
	  !entity_scalar_p(e) && 
	  !step_private_p(e))
	{
	  l_block = CONS(STATEMENT, build_call_STEP_init_arrayregions(e), l_block);
	}
    }
  if (!ENDP(l_block))
    return make_block_statement(gen_nreverse(l_block));
  else
    return make_continue_statement(entity_undefined);
}

statement step_share_after(entity directive_module, entity mpi_module)
{
  list flush_list = NIL;
  FOREACH(ENTITY, e, code_declarations(entity_code(directive_module)))
    {
      if (type_variable_p(entity_type(e)) &&
	  !entity_scalar_p(e) && 
	  !step_private_p(e))
	{
	  bool is_optimized = false;
	  bool is_interlaced = false;
	  flush_list = CONS(STATEMENT, build_call_STEP_AllToAll(mpi_module, e, is_optimized, is_interlaced), flush_list);
	}
    }
  return build_call_STEP_WaitAll(flush_list);
}

