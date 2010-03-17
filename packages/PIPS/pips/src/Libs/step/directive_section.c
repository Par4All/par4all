/* Copyright 2007-2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"


directive make_directive_omp_parallel_sections(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_PARALLEL_SECTIONS,""),make_type_directive_omp_parallel_sections(NIL),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_begin_directive_omp_parallel_sections(directive __attribute__ ((unused)) current,directive next)
{
  bool b;

  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_parallel_sections_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_end_parallel_sections(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_SECTION,""),make_type_directive_omp_end_parallel_sections(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_end_directive_omp_end_parallel_sections(directive current,directive next)
{
  bool b1, b2;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);

  b1 = type_directive_omp_end_parallel_sections_p(directive_type(next));
  b2 = type_directive_omp_parallel_sections_p(directive_type(current));

  pips_debug(1,"b1 = %d, b2 = %d\n", b1, b2);
  return b1 && b2;
}

instruction handle_omp_parallel_sections(directive begin,directive end)
{
  instruction instr;
  statement call;
  entity directive_module;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  pips_assert("parallel sections",type_directive_omp_parallel_sections_p(directive_type(begin)));
  pips_assert("end parallel sections",type_directive_omp_end_parallel_sections_p(directive_type(end)));
  
  if (ENDP(type_directive_omp_parallel_sections(directive_type(begin))))
    pips_error("handle_omp_parallel_sections", "no section in parallel_sections directive");

  // handle the parallel sections directive
  directive_module = outlining_start(directive_module_name(begin));
  outlining_scan_block(type_directive_omp_sections(directive_type(begin)));
  call = outlining_close();
  if(statement_comments(call)!=empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call)=empty_comments;
    }
  statement_label(call) = entity_empty_label();
  
  call = step_keep_directive_txt(begin,call,end);

  instr = statement_instruction(call);
  statement_instruction(call) = instruction_undefined;
  free_statement(call);
  
  // reduction handling
  directive_clauses(begin) = gen_nconc(directive_clauses(begin),
				       CONS(CLAUSE,clause_reductions(call_function(instruction_call(instr)),directive_txt(begin)),NIL));
  
  store_global_directives(directive_module,begin);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

directive make_directive_omp_sections(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_SECTIONS,""),make_type_directive_omp_sections(NIL),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_begin_directive_omp_sections(directive __attribute__ ((unused)) current,directive next)
{
  bool b;
  
  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_sections_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_section(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_SECTION,""),make_type_directive_omp_section(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_end_directive_omp_section(directive current, directive next)
{
  bool b1, b2, b3;
  bool b;

  pips_debug(1,"current = %p, next = %p\n", current, next);

  STEP_DEBUG_DIRECTIVE(2, "current", current);
  STEP_DEBUG_DIRECTIVE(2, "next", next);

  b1 = type_directive_omp_sections_p(directive_type(current));
  b2 = type_directive_omp_parallel_sections_p(directive_type(current));

  b3 = type_directive_omp_section_p(directive_type(next));

  pips_debug(2,"sections_p(current) = %d, parallel_sections_p(current) = %d, section_p(next) = %d\n", b1, b2, b3);

  b = (b1 || b2) && b3;
  pips_debug(1, "b = %d\n", b);
  return b;
}

directive make_directive_omp_end_sections(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_SECTION,""),make_type_directive_omp_end_sections(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_end_directive_omp_end_sections(directive current, directive next)
{
  bool b1, b2;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);

  b1 = type_directive_omp_end_sections_p(directive_type(next));
  b2 = type_directive_omp_sections_p(directive_type(current));

  pips_debug(1,"b1 = %d, b2 = %d\n", b1, b2);
  return b1 && b2;
}

instruction handle_omp_sections(directive begin, directive end)
{
  instruction instr;
  statement call;
  entity directive_module;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  pips_assert("sections",type_directive_omp_sections_p(directive_type(begin)));
  pips_assert("end sections",type_directive_omp_end_sections_p(directive_type(end)));
  
  if (ENDP(type_directive_omp_sections(directive_type(begin))))
    pips_error("handle_omp_sections","no section in sections directive");

  // handle the sections directive
  directive_module = outlining_start(directive_module_name(begin));
  outlining_scan_block(type_directive_omp_sections(directive_type(begin)));
  call = outlining_close();
  if(statement_comments(call)!=empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call)=empty_comments;
    }
  statement_label(call) = entity_empty_label();
  
  call = step_keep_directive_txt(begin,call,end);

  instr = statement_instruction(call);
  statement_instruction(call)=instruction_undefined;
  free_statement(call);
  
  // reduction handling
  directive_clauses(begin)=gen_nconc(directive_clauses(begin),
				     CONS(CLAUSE,clause_reductions(call_function(instruction_call(instr)),directive_txt(begin)),NIL));

  store_global_directives(directive_module,begin);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

instruction handle_omp_section(directive begin, directive end)
{
  instruction instr;

  pips_debug(1, "begin = %p, end = %p\n", begin, end);
    
  pips_assert("sections",type_directive_omp_sections_p(directive_type(begin))
	      || type_directive_omp_parallel_sections_p(directive_type(begin)));
  pips_assert("end section",type_directive_omp_section_p(directive_type(end))
	      || type_directive_omp_end_sections_p(directive_type(end))
	      || type_directive_omp_end_parallel_sections_p(directive_type(end)));

  STEP_DEBUG_DIRECTIVE(2, "begin", begin);
  STEP_DEBUG_DIRECTIVE(2, "end", end);

  if (ENDP(directive_body(begin)))
    {
      pips_debug(2, "no outlining: empty body\n");

      /* il faudrait detruire l'entite contenant le nom sect1 par
	 exemple (genere avec le end) */

    }
  else 
    {
      statement call;
      list body;
      entity directive_module;

      pips_assert("empty body", !ENDP(directive_body(begin)));

      pips_debug(2, "current section outlining\n");

      body = directive_body(end)=directive_body(begin);
      directive_module = outlining_start(directive_module_name(end));
      outlining_scan_block(body);
      call = outlining_close();
      
      directive_body(begin)=NIL;
      
      if(statement_comments(call)!=empty_comments)
	{
	  free(statement_comments(call));
	  statement_comments(call)=empty_comments;
	}
      statement_label(call)=entity_empty_label();

      call = step_keep_directive_txt(make_directive("'section'","",make_type_directive_omp_section(),NIL,NIL),call,directive_undefined);

      //add the new outlined section call to the call section list
      if (type_directive_omp_sections_p(directive_type(begin)))
	type_directive_omp_sections(directive_type(begin))=
	  gen_nconc(type_directive_omp_sections(directive_type(begin)),CONS(STATEMENT,call,NIL));
      else
	type_directive_omp_parallel_sections(directive_type(begin))=
	  gen_nconc(type_directive_omp_parallel_sections(directive_type(begin)),CONS(STATEMENT,call,NIL));

      store_global_directives(directive_module,end);
    }

  if (type_directive_omp_end_sections_p(directive_type(end)))
    instr = handle_omp_sections(begin,end);
  else if (type_directive_omp_end_parallel_sections_p(directive_type(end)))
    instr = handle_omp_parallel_sections(begin,end);
  else
    {
      STEP_DEBUG_DIRECTIVE(2, "push current_directives", begin);
      current_directives_push(begin);
      instr = make_continue_instruction();
    }

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}
