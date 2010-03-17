/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

entity step_create_mpi_barrier(entity directive_module)
{
  string new_name = step_find_new_module_name(directive_module,STEP_MPI_SUFFIX);
  entity mpi_module = make_empty_subroutine(new_name);

  step_seqlist = CONS(STATEMENT,make_return_statement(mpi_module),
		      CONS(STATEMENT,call_STEP_subroutine(RT_STEP_Barrier,NIL),
			   NIL));
  return mpi_module;
}
