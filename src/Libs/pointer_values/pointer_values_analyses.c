/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "misc.h"

#include "pointer_values.h"

/******************** MAPPINGS */

/* I don't know how to deal with these mappings if we have to analyse several modules 
   at the same time when performing an interprocedural analysis.
   We may need a stack of mappings, or a global mapping for the whole program, 
   and a temporary mapping to store the resources one module at a time
*/
GENERIC_GLOBAL_FUNCTION(pv, statement_cell_relations)
GENERIC_GLOBAL_FUNCTION(gen_pv, statement_cell_relations)
GENERIC_GLOBAL_FUNCTION(kill_pv, statement_effects)



/******************** PIPSDBM INTERFACES */

static statement_cell_relations db_get_simple_pv(char * module_name)
{
  return (statement_cell_relations) db_get_memory_resource(DBR_SIMPLE_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_pv(char * module_name, statement_cell_relations scr)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_POINTER_VALUES, module_name, (char*) scr);
}

static statement_cell_relations db_get_simple_gen_pv(char * module_name)
{
  return (statement_cell_relations) db_get_memory_resource(DBR_SIMPLE_GEN_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_gen_pv(char * module_name, statement_cell_relations scr)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_GEN_POINTER_VALUES, module_name, (char*) scr);
}

static statement_effects db_get_simple_kill_pv(char * module_name)
{
  return (statement_effects) db_get_memory_resource(DBR_SIMPLE_KILL_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_kill_pv(char * module_name, statement_effects se)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_KILL_POINTER_VALUES, module_name, (char*) se);
}


/******************** ANALYSIS CONTEXT */

typedef struct {
  /* PIPSDEBM INTERFACES */
  statement_cell_relations (*db_get_pv_func)(char *);
  void (*db_put_pv_func)(char * , statement_cell_relations);
  statement_cell_relations (*db_get_gen_pv_func)(char *);
  void (*db_put_gen_pv_func)(char * , statement_cell_relations);
  statement_effects (*db_get_kill_pv_func)(char *);
  void (*db_put_kill_pv_func)(char * , statement_effects);
} pv_context;

static pv_context make_simple_pv_context()
{
  pv_context ctxt;
  
  ctxt.db_get_pv_func = db_get_simple_pv;
  ctxt.db_put_pv_func = db_put_simple_pv;
  ctxt.db_get_gen_pv_func = db_get_simple_gen_pv;
  ctxt.db_put_gen_pv_func = db_put_simple_gen_pv;
  ctxt.db_get_kill_pv_func = db_get_simple_kill_pv;
  ctxt.db_put_kill_pv_func = db_put_simple_kill_pv;
  
  return ctxt;
}

#define UNDEF abort

typedef void (*void_function)();
typedef gen_chunk* (*chunks_function)();
typedef list (*list_function)();
typedef bool (*bool_function)();
typedef descriptor (*descriptor_function)();
typedef statement_cell_relations (*statement_cell_relations_function)();
typedef statement_effects (*statement_effects_function)();

static void reset_pv_context(pv_context *p_ctxt)
{
  p_ctxt->db_get_pv_func = (statement_cell_relations_function) UNDEF;
  p_ctxt->db_put_pv_func = (void_function) UNDEF;
  p_ctxt->db_get_gen_pv_func =(statement_cell_relations_function) UNDEF ;
  p_ctxt->db_put_gen_pv_func = (void_function) UNDEF;
  p_ctxt->db_get_kill_pv_func = (statement_effects_function) UNDEF;
  p_ctxt->db_put_kill_pv_func = (void_function) UNDEF;
}

/******************** LOCAL FUNCTIONS DECLARATIONS */

static 
list sequence_to_post_pv(sequence seq, list l_in, pv_context *ctxt);

static 
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt);

static
list declarations_to_post_pv(list l_decl, list l_in, pv_context *ctxt);

static
list declaration_to_post_pv(entity e, list l_in, pv_context *ctxt);

static
list instruction_to_post_pv(instruction inst, list l_in, pv_context *ctxt);

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt);

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt);

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt);

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt);

static
list unstructured_to_post_pv(unstructured u, list l_in, pv_context *ctxt);

static
list expression_to_post_pv(expression exp, list l_in, pv_context *ctxt);

static
list call_to_post_pv(call c, list l_in, pv_context *ctxt);


/**************** MODULE ANALYSIS *************/

static 
list sequence_to_post_pv(sequence seq, list l_in, pv_context *ctxt)
{  
  list l_out = l_in;
  list l_locals = NIL;
  FOREACH(STATEMENT, stmt, sequence_statements(seq))
    {
      /* keep local variables in declaration reverse order */
      if (declaration_statement_p(stmt))
	{
	  FOREACH(ENTITY, e, statement_declarations(stmt))
	    {
	      /* beware don't push static variables */
	      l_locals = CONS(ENTITY, e, l_locals);
	      l_out = declaration_to_post_pv(e, l_in, ctxt);
	    }
	  
	} 
      else
	l_out = instruction_to_post_pv(statement_instruction(stmt), l_in, ctxt);
    }
  
  /* don't forget to eliminate local declarations on exit */
  /* ... */
  return (l_out);
}

static 
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt)
{
  list l_out = NIL;

  if (declaration_statement_p(stmt))
    {
      list l_decl = statement_declarations(stmt);
      l_out = declarations_to_post_pv(l_decl, l_in, ctxt);
    }
  else
    {
      l_out = instruction_to_post_pv(statement_instruction(stmt), l_in, ctxt);
    }

  return (l_out);
}

static
list declarations_to_post_pv(list l_decl, list l_in, pv_context *ctxt)
{
  list l_out = l_in;

  FOREACH(ENTITY, e, l_decl)
    {
      l_out = declaration_to_post_pv(e, l_out, ctxt);
    }
  return (l_out);
}

static
list declaration_to_post_pv(entity e, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list instruction_to_post_pv(instruction inst, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  
  switch(instruction_tag(inst))
    {
    case is_instruction_sequence:
      l_out = sequence_to_post_pv(instruction_sequence(inst), l_in, ctxt);
      break;
    case is_instruction_test:
      l_out = test_to_post_pv(instruction_test(inst), l_in, ctxt);
      break;
    case is_instruction_loop:
      l_out = loop_to_post_pv(instruction_loop(inst), l_in, ctxt);
      break;
    case is_instruction_whileloop:
      l_out = whileloop_to_post_pv(instruction_whileloop(inst), l_in, ctxt);
      break;
    case is_instruction_forloop:
      l_out = forloop_to_post_pv(instruction_forloop(inst), l_in, ctxt);
      break;
    case is_instruction_unstructured:
      l_out = unstructured_to_post_pv(instruction_unstructured(inst), l_in, ctxt);
      break;
    case is_instruction_expression:
      l_out = expression_to_post_pv(instruction_expression(inst), l_in, ctxt);
      break;
    case is_instruction_call:
      l_out = call_to_post_pv(instruction_call(inst), l_in, ctxt);
      break;
    case is_instruction_goto:
      pips_internal_error("unexpected goto in pointer values analyses\n");
      break;
    case is_instruction_multitest:
      pips_internal_error("unexpected multitest in pointer values analyses\n");
      break;
    default:
      pips_internal_error("unknown instruction tag\n");
    }
  
  return (l_out);
}

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}


static
list unstructured_to_post_pv(unstructured u, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list expression_to_post_pv(expression exp, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}

static
list call_to_post_pv(call c, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
  return (l_out);
}



static void generic_module_pointer_values(char * module_name, pv_context *ctxt)
{
  list l_out;

  /* temporary settings : in an interprocedural context we need to keep track 
     of visited modules */
  /* Get the code of the module. */
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, TRUE));
  
  set_current_module_entity(module_name_to_entity(module_name));
  init_pv();
  init_gen_pv();
  init_kill_pv();

  debug_on("POINTER_VALUES_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  
  l_out = statement_to_post_pv(get_current_module_statement(), NIL, ctxt);

  (*ctxt->db_put_pv_func)(module_name, get_pv());
  (*ctxt->db_put_gen_pv_func)(module_name, get_gen_pv());
  (*ctxt->db_put_kill_pv_func)(module_name, get_kill_pv());

  pips_debug(1, "end\n");
  debug_off();
  reset_current_module_entity();
  reset_current_module_statement();
  return;
}

/**************** INTERFACE *************/

bool simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  generic_module_pointer_values(module_name, &ctxt);
  reset_pv_context(&ctxt);
  return(TRUE);
}
