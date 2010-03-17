/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "defines-local.h"


directive make_directive_omp_barrier(statement stmt)
{
  return make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_BARRIER,""),make_type_directive_omp_barrier(),NIL,NIL);
} 

bool begin_directive_omp_barrier(directive __attribute__ ((unused)) current,directive next)
{
  bool rep=type_directive_omp_barrier_p(directive_type(next));
  pips_debug(2,"%d\n",rep);
  return rep;
}

bool end_directive_omp_barrier(directive __attribute__ ((unused)) current,directive next)
{
  bool rep= type_directive_omp_barrier_p(directive_type(next));
  pips_debug(2,"%d\n",rep);
  return rep;
}

instruction handle_omp_barrier(directive begin,directive __attribute__ ((unused)) end)
{
  statement call;
  instruction i;
  entity directive_module=outlining_start(directive_module_name(begin));
  outlining_scan_block(gen_full_copy_list(directive_body(begin)));
  call=outlining_close();
  if(statement_comments(call)!=empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call)=empty_comments;
    }
  statement_label(call)=entity_empty_label();
  
  call=step_keep_directive_txt(begin,call,directive_undefined);
  
  i=statement_instruction(call);
  statement_instruction(call)=instruction_undefined;
  free_statement(call);

  store_global_directives(directive_module,begin);

  return i;
}


string directive_omp_barrier_to_string(directive d,bool close)
{
  pips_debug(1, "d=%p, close=%u\n",d,close);
  if (close)
    return string_undefined;
  else
    return strdup(BARRIER_TEXT);
}
