/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"


directive make_directive_omp_parallel(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_PARALLEL,""),make_type_directive_omp_parallel(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_begin_directive_omp_parallel(directive __attribute__ ((unused)) current,directive next)
{
  bool b;

  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_parallel_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_end_parallel(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_END_PARALLEL,""),make_type_directive_omp_end_parallel(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;

}

bool is_end_directive_omp_end_parallel(directive current,directive next)
{
  bool b1, b2;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);

  b1 = type_directive_omp_end_parallel_p(directive_type(next));
  b2 = type_directive_omp_parallel_p(directive_type(current));

  pips_debug(1,"b1 = %d, b2 = %d\n", b1, b2);
  return b1 && b2;
}

instruction handle_omp_parallel(directive begin, directive end)
{
  statement call;
  instruction instr;
  entity directive_module;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  directive_module = outlining_start(directive_module_name(begin));
  outlining_scan_block(directive_body(begin));
  call = outlining_close();

  if(statement_comments(call) != empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call) = empty_comments;
    }
  statement_label(call) = entity_empty_label();
  call = step_keep_directive_txt(begin, call, end);

  instr = statement_instruction(call);
  statement_instruction(call) = instruction_undefined;
  free_statement(call);

  store_global_directives(directive_module, begin);
  free_directive(end);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

string directive_omp_parallel_to_string(directive d,bool close)
{
  pips_debug(1, "d=%p, close=%u\n",d,close);
  if (close)
    return strdup(END_PARALLEL_TEXT);
  else
    return strdup(PARALLEL_TEXT);
}
