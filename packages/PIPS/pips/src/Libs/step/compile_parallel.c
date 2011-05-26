/* Copyright 2009 Alain Muller

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


#include "defines-local.h"


statement step_compile_parallel(int step_transformation, entity new_module, statement work,statement work_critical)
{
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is parallel directive",type_directive_omp_parallel_p(directive_type(d)));

  if (step_transformation == STEP_TRANSFORMATION_OMP)
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      if (step_transformation == STEP_TRANSFORMATION_HYBRID )
	add_pragma_entity_to_statement(work, directive_module);

      step_private_before(directive_module);
      statement begin = call_STEP_subroutine(RT_STEP_Begin, CONS(EXPRESSION,step_symbolic(STEP_PARALLEL_NAME, new_module),NIL), type_undefined);
      statement end = call_STEP_subroutine(RT_STEP_End, CONS(EXPRESSION,step_symbolic(STEP_PARALLEL_NAME, new_module),NIL), type_undefined);
      statement before_work = make_block_statement(CONS(STATEMENT, begin,
						       CONS(STATEMENT, step_share_before(directive_module, new_module), NIL)));

      statement after_work = make_block_statement(CONS(STATEMENT, step_share_after(directive_module, new_module),
						       CONS(STATEMENT, end, NIL)));
      
      if (step_transformation == STEP_TRANSFORMATION_HYBRID)
	{
	  before_work = step_guard_hybride(before_work);
	  after_work = step_guard_hybride(after_work);
	}
      step_private_after();
      //return make_block_statement(CONS(STATEMENT, before_work,
	//			       CONS(STATEMENT, work,
	//				    CONS(STATEMENT, after_work, NIL))));

      if(work_critical)  	
      		return make_block_statement(CONS(STATEMENT, before_work,
				       CONS(STATEMENT,work,
					CONS(STATEMENT, work_critical,
					    CONS(STATEMENT, after_work,
						 NIL)))));
      else
		return make_block_statement(CONS(STATEMENT, before_work,
				       CONS(STATEMENT,  work,
					    CONS(STATEMENT, after_work,
						 NIL))));



    }
}

