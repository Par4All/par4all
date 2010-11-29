/* Copyright 2007-2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"

static directive _make_directive_omp_section(string directive_txt)
{
  string new_name;
  directive drt;

  pips_debug(1,"stmt = %s\n", directive_txt);

  new_name = step_make_new_directive_module_name(SUFFIX_OMP_SECTION,"");
  
  drt = make_directive(strdup(directive_txt), new_name, make_type_directive_omp_section(), NIL, NIL);

  pips_debug(1,"d = %p\n", drt);
  return drt;
}


void handle_directive_sections_push(list remaining, directive drt)
{
  pips_debug(1, "remaining = %p, drt = %p\n", remaining, drt);

  if (type_directive_omp_sections_p(directive_type(drt))
      || type_directive_omp_parallel_sections_p(directive_type(drt)))
    {
      directive new_drt;
      
      /*
	After any 'sections' or 'parallel sections'
	Automatically add a section
      */ 
      
      new_drt = _make_directive_omp_section("'section'");
      
      STEP_DEBUG_DIRECTIVE(2,"Automatical PUSH current_directives", new_drt);
      current_directives_push(new_drt);
    }

  pips_debug(1, "fin\n");
}

directive make_directive_omp_parallel_sections(statement stmt)
{
  string new_name;
  directive drt;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name(SUFFIX_OMP_PARALLEL_SECTIONS,"");

  drt = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_parallel_sections(NIL), NIL, NIL);

  pips_debug(1,"d = %p\n", drt);
  return drt;
}

bool is_begin_directive_omp_parallel_sections(directive __attribute__ ((unused)) current, directive next)
{
  bool b;

  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_parallel_sections_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_end_parallel_sections(statement stmt)
{
  string new_name;
  directive drt;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name("end","");

  drt = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_end_parallel_sections(), NIL, NIL);

  pips_debug(1,"drt = %p\n", drt);
  return drt;
}

bool is_end_directive_omp_end_parallel_sections(directive  __attribute__ ((unused)) current, directive  __attribute__ ((unused)) next)
{
  bool b;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);

  b = TRUE;

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_sections(statement stmt)
{
  string new_name;
  directive drt;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name(SUFFIX_OMP_SECTIONS,"");

  drt = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_sections(NIL),NIL,NIL);
  
  pips_debug(1,"d = %p\n", drt);
  return drt;
}

bool is_begin_directive_omp_sections(directive __attribute__ ((unused)) current, directive next)
{
  bool b;
  
  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_sections_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

bool is_begin_directive_omp_section(directive current, directive next)
{
  bool b;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);
  
  /* the first omp_section is automatically added
     when omp_sections is handled (push)
     thus already in the stack

     the next omp_section are added after 
     handling previous omp_section

     thus omp_section is considered as an end_directive
  */
  b = FALSE;

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_section(statement stmt)
{
  string directive_txt;
  directive drt;

  pips_debug(1,"stmt = %p\n", stmt);

  directive_txt = strdup(statement_to_directive_txt(stmt));

  drt = _make_directive_omp_section(directive_txt);


  pips_debug(1,"d = %p\n", drt);
  return drt;
}

bool is_end_directive_omp_section(directive current, directive next)
{
  bool b;

  pips_debug(1,"current = %p, next = %p\n", current, next);

  STEP_DEBUG_DIRECTIVE(2, "current", current);
  STEP_DEBUG_DIRECTIVE(2, "next", next);

  pips_assert("current is omp_section", type_directive_omp_section_p(directive_type(current)));
  pips_assert("next is omp_section", type_directive_omp_section_p(directive_type(next)));
  
  /* if body of current is not empty then it is end */
  b = !ENDP(directive_body(current));

  pips_debug(1, "b = %d\n", b);
  return b;
}

directive make_directive_omp_end_sections(statement stmt)
{
  directive drt;
  string new_name;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name(NULL,"");

  drt = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_end_sections(), NIL, NIL);

  pips_debug(1,"drt = %p\n", drt);
  return drt;
}

bool is_end_directive_omp_end_sections(directive current, directive next)
{
  bool b;
  
  pips_debug(1,"current = %p, next = %p\n", current, next);

  pips_assert("current is omp_section", type_directive_omp_section_p(directive_type(current)));

  b = type_directive_omp_end_sections_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

instruction handle_omp_sections(directive __attribute__ ((unused)) d1, directive __attribute__ ((unused)) d2)
{
  instruction instr = instruction_undefined;

  pips_internal_error("compatibily function: should never be called");

  return instr;
}

static instruction outline_omp_sections_body(directive begin, directive end)
{
  entity directive_module;
  statement call;
  instruction instr;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  STEP_DEBUG_DIRECTIVE(3, "begin", begin);
  STEP_DEBUG_DIRECTIVE(3, "end", end);

  directive_module = outlining_start(directive_module_name(begin));
  outlining_scan_block(gen_full_copy_list(directive_body(begin)));
  call = outlining_close(step_directives_USER_FILE_name());

  if(statement_comments(call) != empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call) = empty_comments;
    }
  statement_label(call) = entity_empty_label();
  
  /* add comments in the generated code to keep track of the OpenMP
     directives */

  call = step_keep_directive_txt(begin, call, directive_undefined);

  instr = statement_instruction(call);
  statement_instruction(call)=instruction_undefined;
  free_statement(call);
  

  store_global_directives(directive_module, begin); 

  pips_debug(1,"end instr = %p\n", instr);
  return instr;
}

/*
  will be also called with
   - omp_end_sections
   - omp_end_parallel_sections
*/
static void outline_omp_section_body(directive begin)
{
  entity directive_module;
  statement call;

  directive end;
  pips_assert("not empty body", !ENDP(directive_body(begin)));

  end = current_directives_head();
  STEP_DEBUG_DIRECTIVE(3, "begin", begin);
  STEP_DEBUG_DIRECTIVE(3, "end", end);

  pips_assert("end is omp_sections or omp_parallel_sections", type_directive_omp_sections_p(directive_type(end)) || type_directive_omp_parallel_sections_p(directive_type(end)));
  
  pips_debug(2, "begin section outlining\n");
  /* new call */
  directive_module = outlining_start(directive_module_name(begin));
  outlining_scan_block(gen_full_copy_list(directive_body(begin)));
  call = outlining_close(step_directives_USER_FILE_name());

  if(statement_comments(call) != empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call)=empty_comments;
    }
  statement_label(call) = entity_empty_label();

  /* add comments in the generated code to keep track of the OpenMP
     directives */

  call = step_keep_directive_txt(begin, call, directive_undefined);

  store_global_directives(directive_module, begin);  

  pips_debug(2, "ADD the new call to the body of the head of current_directives\n");
  directive_body(end) = gen_nconc(directive_body(end), CONS(STATEMENT, copy_statement(call), NIL));

  STEP_DEBUG_DIRECTIVE(2, "end", end);

  pips_debug(1, "end\n");
}


static instruction handle_omp_end_sections(directive begin, directive end)
{
  instruction instr;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  STEP_DEBUG_DIRECTIVE(2, "begin", begin);
  STEP_DEBUG_DIRECTIVE(2, "end", end);

  pips_assert("begin is omp_sections or omp_parallel_sections", type_directive_omp_sections_p(directive_type(begin)) || type_directive_omp_parallel_sections_p(directive_type(begin)));


  /* outline omp_sections body */
  instr = outline_omp_sections_body(begin, end);

  /* reduction handling */
  directive_clauses(begin)=gen_nconc(directive_clauses(begin),
				     CONS(CLAUSE,step_check_reduction(call_function(instruction_call(instr)),directive_txt(begin)),NIL));
  
  pips_debug(1,"instr = %p\n", instr);
  return instr;
}


instruction handle_omp_section(directive begin, directive end)
{
  instruction instr;

  pips_debug(1, "begin = %p, end = %p\n", begin, end);
  
  STEP_DEBUG_DIRECTIVE(2, "begin", begin);
  STEP_DEBUG_DIRECTIVE(2, "end", end);

  pips_assert("begin is omp_section", type_directive_omp_section_p(directive_type(begin)));
  pips_assert("end is omp_section or omp_end_sections or omp_end_parallel_sections", type_directive_omp_section_p(directive_type(end)) || type_directive_omp_end_sections_p(directive_type(end)) || type_directive_omp_end_parallel_sections_p(directive_type(end)));

  /* end is like sect1_sect1 or end... thus incorrect*/ 
  /* A VOIR supprimer end de la liste des entites */

  if (ENDP(directive_body(begin)))
    {
      /* case where the first optional section is present */
      /* begin has been automatically added when sections is handled */
      /* so do nothing */

      pips_debug(2, "section has already been automatically added so do nothing\n");
      instr = make_continue_instruction();
    }
  else 
    {
      outline_omp_section_body(begin);
      
      if (type_directive_omp_section_p(directive_type(end)))
	{
	  /* add the new section (like sect2) */
	  directive_module_name(end) = step_make_new_directive_module_name(SUFFIX_OMP_SECTION,"");
	  current_directives_push(end);
	  instr = make_continue_instruction();
	}
      else
	{
	  /* case end_sections or end_parallel_sections */
	  begin = current_directives_pop();
	  STEP_DEBUG_DIRECTIVE(2,"POP current_directives", begin);

	  instr = handle_omp_end_sections(begin, end);
	}
	
    }
  pips_debug(1,"END instr = %p\n", instr);
  return instr;
}
