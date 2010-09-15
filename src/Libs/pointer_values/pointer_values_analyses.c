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
#include "text-util.h"
#include "effects-simple.h"
#include "effects-generic.h"
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


pv_context make_simple_pv_context()
{
  pv_context ctxt;
  
  ctxt.db_get_pv_func = db_get_simple_pv;
  ctxt.db_put_pv_func = db_put_simple_pv;
  ctxt.db_get_gen_pv_func = db_get_simple_gen_pv;
  ctxt.db_put_gen_pv_func = db_put_simple_gen_pv;
  ctxt.db_get_kill_pv_func = db_get_simple_kill_pv;
  ctxt.db_put_kill_pv_func = db_put_simple_kill_pv;
  ctxt.make_pv_from_effects_func = make_simple_pv_from_simple_effects;
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
typedef cell_relation (*cell_relation_function)();

void reset_pv_context(pv_context *p_ctxt)
{
  p_ctxt->db_get_pv_func = (statement_cell_relations_function) UNDEF;
  p_ctxt->db_put_pv_func = (void_function) UNDEF;
  p_ctxt->db_get_gen_pv_func =(statement_cell_relations_function) UNDEF ;
  p_ctxt->db_put_gen_pv_func = (void_function) UNDEF;
  p_ctxt->db_get_kill_pv_func = (statement_effects_function) UNDEF;
  p_ctxt->db_put_kill_pv_func = (void_function) UNDEF;
  p_ctxt->make_pv_from_effects_func = (cell_relation_function) UNDEF;
}

/***************** ABSTRACT VALUES */

entity undefined_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     UNDEFINED_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
  }
  return u;
}

entity null_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     NULL_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
  }
  return u;
}


cell make_undefined_pointer_value_cell()
{
  entity u = undefined_pointer_value_entity();
  return make_cell(is_cell_reference, make_reference(u, NIL));
}

bool undefined_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), UNDEFINED_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;  
}

bool undefined_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else 
    r = preference_reference(cell_preference(c));
  return(undefined_pointer_value_entity_p(reference_variable(r)));
}

cell make_null_pointer_value_cell()
{
  entity u = null_pointer_value_entity();
  return make_cell(is_cell_reference, make_reference(u, NIL));
}

bool null_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), NULL_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;  
}

bool null_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else 
    r = preference_reference(cell_preference(c));
  return(null_pointer_value_entity_p(reference_variable(r)));
}



/******************** CONVERSION FROM EXPRESSIONS TO PATHS AND CELL_INTERPRETATION */

static effect intrinsic_to_interpreted_path(entity func, list args, cell_interpretation * ci, pv_context * ctxt);
static effect call_to_interpreted_path(call c, cell_interpretation * ci, pv_context * ctxt);
static 
effect expression_to_interpreted_path(expression exp, cell_interpretation * ci, pv_context * ctxt);


static effect intrinsic_to_interpreted_path(entity func, list args, cell_interpretation * ci, pv_context * ctxt)
{
  effect eff = effect_undefined;
  
  if(ENTITY_ASSIGN_P(func))
    /* if the call is an assignement, return the lhs path because the relation
       between the lhs and the rhs has been treated elsewhere
    */
    return(expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt));

  if(ENTITY_FIELD_P(func) || ENTITY_POINT_TO_P(func) || ENTITY_DEREFERENCING_P(func))
    {
      list l_tmp = generic_proper_effects_of_complex_address_expression(EXPRESSION(CAR(args)), &eff, true);
      *ci = make_cell_interpretation_value_of();
      gen_full_free_list(l_tmp);	
      return eff;
    }

  if(ENTITY_ADDRESS_OF_P(func))
    {
      list l_tmp = generic_proper_effects_of_complex_address_expression(EXPRESSION(CAR(args)), &eff, true);
      *ci = make_cell_interpretation_address_of();
      gen_full_free_list(l_tmp);	
      return eff;      
    }
  
  return eff;
}

static effect call_to_interpreted_path(call c, cell_interpretation * ci, pv_context * ctxt)
{
  effect eff = effect_undefined;

  entity func = call_function(c);
  value func_init = entity_initial(func);
  tag t = value_tag(func_init);
  string n = module_local_name(func);
  list func_args = call_arguments(c);
  type uet = ultimate_type(entity_type(func));
  
  if(type_functional_p(uet)) 
    {
      switch (t) 
	{
	case is_value_code:
	  pips_debug(5, "external function %s\n", n);
	  /* If it's an external call, we should try to retrieve the returned value 
	     if it's a pointer. 
	     do nothing for the moment, or we could return an anywhere effect....
	  */
	  pips_internal_error("not yet implemented\n", entity_name(func));	  
	  break;
      
	case is_value_intrinsic:
	  pips_debug(5, "intrinsic function %s\n", n);
	  eff = intrinsic_to_interpreted_path(func, func_args, ci, ctxt);
	  break;
      
	case is_value_symbolic:
	  pips_debug(5, "symbolic\n");
	  pips_internal_error("symbolic case, not yet implemented\n", entity_name(func));
	  break;
	    
	case is_value_constant:
	  pips_debug(5, "constant\n");
	  constant func_const = value_constant(func_init);
	  /* We should be here only in case of a pointer value rhs, and the value should be 0 */
	  if (constant_int_p(func_const))
	    {
	      if (constant_int(func_const) == 0)
		{
		  *ci = make_cell_interpretation_value_of();
		  cell path = make_null_pointer_value_cell();
		  /* use approximation_must to be consistent with effects, should be approximation_exact */
		  eff = make_effect(path, make_action_read_memory(), make_approximation_must(), make_descriptor_none());
		}
	      else
		pips_internal_error("unexpected integer constant value\n", entity_name(func));
	    }
	  else
	    pips_internal_error("unexpected constant case\n", entity_name(func));
	  break;
	    
	case is_value_unknown:
	  pips_internal_error("unknown function %s\n", entity_name(func));
	    
	default:
	  pips_internal_error("unknown tag %d\n", t);
	  break;
	}
    }
  return eff;
}

static 
effect expression_to_interpreted_path(expression exp, cell_interpretation * ci, pv_context * ctxt)
{
  effect eff = effect_undefined;
  syntax exp_syntax = expression_syntax(exp);
  *ci = cell_interpretation_undefined;

  pips_debug(1, "begin with_expression : %s\n", 
	     words_to_string(words_expression(exp,NIL)));

  switch(syntax_tag(exp_syntax))
    {
    case is_syntax_reference:
      pips_debug(5, "reference case\n");
      /* this function name should be stored in ctxt*/
      eff = (*reference_to_effect_func)(syntax_reference(exp_syntax), make_action_write_memory(), false);
      *ci = make_cell_interpretation_value_of();
      break;

    case is_syntax_range:
      pips_debug(5, "range case\n");
      pips_internal_error("not yet implemented\n");
      break;

    case is_syntax_call:
      pips_debug(5, "call case\n");
      eff = call_to_interpreted_path(syntax_call(exp_syntax), ci, ctxt);
      break;

    case is_syntax_cast:
      pips_debug(5, "cast case\n");
      eff = expression_to_interpreted_path(cast_expression(syntax_cast(exp_syntax)), ci, ctxt);
      break;

    case is_syntax_sizeofexpression:
      pips_debug(5, "sizeof case\n");
      pips_internal_error("sizeof not expected\n");
      break;
    
    case is_syntax_subscript:
      pips_debug(5, "subscript case\n");	
      list l_tmp = generic_proper_effects_of_complex_address_expression(exp, &eff, true);
      *ci = make_cell_interpretation_value_of();
      gen_full_free_list(l_tmp);	
      break;

    case is_syntax_application:
      pips_debug(5, "application case\n");		
      pips_internal_error("not yet implemented\n");
      break;

    case is_syntax_va_arg:
      pips_debug(5, "va_arg case\n");		
      pips_internal_error("va_arg not expected\n");
      break;
      
    default:
      pips_internal_error("unexpected tag %d\n", syntax_tag(exp_syntax));
    }

  pips_debug_effect(2, "returning path :",eff);
  pips_debug(2, "with %s interpretation\n", cell_interpretation_value_of_p(*ci)? "value_of": "address_of");
  pips_debug(1,"end\n");
  return eff;
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

static
list intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_context *ctxt);

static 
list assignment_to_post_pv(expression lhs, expression rhs, list l_in, pv_context *ctxt);



/**************** MODULE ANALYSIS *************/

static 
list sequence_to_post_pv(sequence seq, list l_in, pv_context *ctxt)
{  
  list l_cur = l_in;
  list l_locals = NIL;
  pips_debug(1, "begin\n");
  FOREACH(STATEMENT, stmt, sequence_statements(seq))
    {
      store_pv(stmt, make_cell_relations(gen_full_copy_list(l_cur)));
      /* keep local variables in declaration reverse order */
      if (declaration_statement_p(stmt))
	{
	  FOREACH(ENTITY, e, statement_declarations(stmt))
	    {
	      /* beware don't push static variables */
	      l_locals = CONS(ENTITY, e, l_locals);
	      l_cur = gen_nconc(l_cur, declaration_to_post_pv(e, l_cur, ctxt));
	    }
	  
	} 
      else
	l_cur = instruction_to_post_pv(statement_instruction(stmt), l_cur, ctxt);
      store_gen_pv(stmt, make_cell_relations(NIL));
      store_kill_pv(stmt, make_effects(NIL));

    }
  
  /* don't forget to eliminate local declarations on exit */
  /* ... */
  pips_debug_pvs(2, "returning: ", l_cur);
  pips_debug(1, "end\n");
  return (l_cur);
}

static 
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_debug_pvs(2, "input pvs: ", l_in);

  store_pv(stmt, make_cell_relations(gen_full_copy_list(l_in)));

  if (declaration_statement_p(stmt))
    {
      list l_decl = statement_declarations(stmt);
      l_out = declarations_to_post_pv(l_decl, l_in, ctxt);
    }
  else
    {
      l_out = instruction_to_post_pv(statement_instruction(stmt), l_in, ctxt);
    }

  store_gen_pv(stmt, make_cell_relations(NIL));
  store_kill_pv(stmt, make_effects(NIL));
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");

  return (l_out);
}

static
list declarations_to_post_pv(list l_decl, list l_in, pv_context *ctxt)
{
  list l_out = l_in;
  pips_debug(1, "begin\n");

  FOREACH(ENTITY, e, l_decl)
    {
      /* well, not exactly, we must take kills into account */
      l_out = gen_nconc(l_out, declaration_to_post_pv(e, l_out, ctxt));
    }
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}


static
list declaration_to_post_pv(entity e, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");

  expression lhs_exp = entity_to_expression(e);
  expression rhs_exp = expression_undefined;
  bool free_rhs_exp = false;

  value v_init = entity_initial(e);
  if (v_init == value_undefined)
    {
      pips_debug(2, "undefined inital value\n");
      rhs_exp = entity_to_expression(undefined_pointer_value_entity());
      free_rhs_exp = true;
    }
  else
    {
      switch (value_tag(v_init))
	{
	case is_value_expression:
	  rhs_exp = value_expression(v_init); 
	  break;
	case is_value_unknown:
	  pips_debug(2, "unknown inital value\n");
	  rhs_exp = expression_undefined;
	  break;
	case is_value_code:
	case is_value_symbolic:
	case is_value_constant:
	case is_value_intrinsic:
	default:
	  pips_internal_error("unexpected tag\n");
	}
    }

  l_out = assignment_to_post_pv(lhs_exp, rhs_exp, l_in, ctxt);
  
  free_expression(lhs_exp);
  if (free_rhs_exp) free_expression(rhs_exp);
  


  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}
  
static
list instruction_to_post_pv(instruction inst, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  
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
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  
  return (l_out);
}

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug(1, "end\n");
  return (l_out);
}

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}


static
list unstructured_to_post_pv(unstructured u, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
   pips_debug_pvs(2, "returning: ", l_out);
 pips_debug(1, "end\n");
  return (l_out);
}

static
list expression_to_post_pv(expression exp, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  syntax s = expression_syntax(exp);

  switch(syntax_tag(s))
    {
    case is_syntax_reference:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_range:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_call:
      {
	l_out = call_to_post_pv(syntax_call(s), l_in, ctxt);
	break;
      }
    case is_syntax_cast:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_sizeofexpression:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_subscript:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_application:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_va_arg:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    default:
      pips_internal_error("unexpected tag %d\n", syntax_tag(s));
    }

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list call_to_post_pv(call c, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  entity func = call_function(c);
  tag t = value_tag(entity_initial(func));
  list func_args = call_arguments(c);
  type func_type = ultimate_type(entity_type(func));

  pips_debug(1, "begin for %s\n", entity_local_name(func));

  if(type_functional_p(func_type)) 
    {
      switch (t) 
	{
	case is_value_code:
	  pips_debug(5, "external function \n");
	  pips_internal_error("not yet implemented\n");
	  break;
	  
	case is_value_intrinsic:
	  pips_debug(5, "intrinsic function\n");
	  l_out = intrinsic_to_post_pv(func, func_args, l_in, ctxt);
	  break;
	  
	case is_value_symbolic:
	  pips_debug(5, "symbolic\n");
	  l_out = l_in;
	  break;
	  
	case is_value_constant:
	  pips_debug(5, "constant\n");
	  l_out = l_in;
	  break;
	  
	case is_value_unknown:
	  pips_internal_error("unknown function \n");
	  break;
	  
	default:
	  pips_internal_error("unknown tag %d\n", t);
	  break;
	}
    }
  else if(type_variable_p(func_type)) 
    {
      pips_internal_error("not yet implemented\n");
    }
  else if(type_statement_p(func_type)) 
    {
      pips_internal_error("not yet implemented\n");
    }
  else 
    {
      pips_internal_error("Unexpected case\n");
    }

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin for %s\n", entity_local_name(func));

  /* only few intrinsics are currently handled : we should have a way to 
     describe the effects on aliasing of all intrinsics.
  */

  if (ENTITY_ASSIGN_P(func)) 
    {
      expression lhs = EXPRESSION(CAR(func_args));
      expression rhs = EXPRESSION(CAR(CDR(func_args)));
      l_out = assignment_to_post_pv(lhs, rhs, l_in, ctxt);
    }
  else if((ENTITY_STOP_P(func) || ENTITY_ABORT_SYSTEM_P(func)
	   || ENTITY_EXIT_SYSTEM_P(func))) 
    {
      /* The call is never returned from. No information is available
	 for the dead code that follows.
      */
      l_out = NIL;
    }
  else if (ENTITY_C_RETURN_P(func))
    {
      /* No information is available for the dead code that follows 
	 but we have to evaluate the impact
	 of the argument evaluation on pointer values
      */
      l_out = expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, ctxt);
    }
  else
    pips_internal_error("not yet implemented\n");

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}




static 
list assignment_to_post_pv(expression lhs, expression rhs, list l_in, pv_context *ctxt) 
{
  list l_out = NIL;
  list l_kill = NIL;
  list l_gen = NIL;
  effect lhs_eff = effect_undefined;
  cell_interpretation lhs_kind;
  effect rhs_eff = effect_undefined;
  cell_interpretation rhs_kind;
  pips_debug(1, "begin\n");

  type lhs_type = expression_to_type(lhs);

  if(type_variable_p(lhs_type))
    {
      /* first convert the lhs into a memory path */
      lhs_eff = expression_to_interpreted_path(lhs, &lhs_kind, ctxt);
  
      /* find all pointers that are or may be defined through this assignment */

      if (pointer_type_p(lhs_type))
	{
	  /* simple case first: lhs is a pointer*/
	  l_kill = CONS(EFFECT, lhs_eff, l_kill);
	  if (!expression_undefined_p(rhs))
	    rhs_eff = expression_to_interpreted_path(rhs, &rhs_kind, ctxt);
	  else 
	    {
	      rhs_eff = make_effect(make_undefined_pointer_value_cell(), 
				    make_action_read_memory(), make_approximation_must(), 
				    make_descriptor_none()); 
	      rhs_kind = make_cell_interpretation_value_of();
	    }
	  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)(lhs_eff, rhs_eff, rhs_kind);
	  l_gen = CONS(CELL_RELATION, gen_pv, NIL);
	}
      else
	{
	  /* lhs is not a pointer, but it may be an array of pointers, an aggregate type with pointers.... */
	  /* In this case, it cannot be an address_of case */
	  l_kill =  generic_effect_generate_all_accessible_paths_effects_with_level(lhs_eff, lhs_type, is_action_write, false, 0, true);
	  pips_debug_effects(2, "l_kill = ", l_kill);
      
	  if (!ENDP(l_kill))
	    {
	      
	      if (!expression_undefined_p(rhs))
		rhs_eff = expression_to_interpreted_path(rhs, &rhs_kind, ctxt);
	       else 
		 {
		   rhs_eff = make_effect(make_undefined_pointer_value_cell(), 
					 make_action_read_memory(), make_approximation_must(), 
					 make_descriptor_none());  
		   rhs_kind = make_cell_interpretation_value_of();
		 }
	      pips_assert("multiple pointer assignement requires a value_of rhs kind\n", cell_interpretation_value_of_p(rhs_kind));
	      cell rhs_cell = effect_cell(rhs_eff);

	      if(null_pointer_value_cell_p(rhs_cell))
		{
		  pips_internal_error("assigning NULL to several pointers at the same time!\n");
		}
	      else if (undefined_pointer_value_cell_p(rhs_cell) || (anywhere_effect_p(rhs_eff)))
		{
		  FOREACH(EFFECT, eff, l_kill)
		    {
		      cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)(eff, rhs_eff, rhs_kind); 
		      l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
		    }
		}
	      else
		{
		  reference rhs_ref = effect_any_reference(rhs_eff);
		  type rhs_type = cell_reference_to_type(rhs_ref);
	      
		  if (type_equal_p(lhs_type, rhs_type))
		    {
		      reference lhs_ref = effect_any_reference(lhs_eff);
		      size_t lhs_nb_dim = gen_length(reference_indices(lhs_ref));
		    
		      FOREACH(EFFECT, kill_eff, l_kill)
			{
			  reference kill_ref = effect_any_reference(kill_eff);
			  list kill_dims = reference_indices(kill_ref);
			  effect new_rhs_eff = copy_effect(rhs_eff);
		      
			  /* first skip dimensions of kill_ref similar to lhs_ref */
		      
			  for(size_t i = 0; i < lhs_nb_dim; i++, POP(kill_dims));
		      
			  /* add the remaining dimensions to the copy of rhs_eff */
		      
			  for(; !ENDP(kill_dims); POP(kill_dims))
			    {
			      expression dim = EXPRESSION(CAR(kill_dims));
			      (*effect_add_expression_dimension_func)(new_rhs_eff, dim);
			  
			    }

			  /* Then build the pointer_value relation */
			  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)(kill_eff, new_rhs_eff, rhs_kind);
			  l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
			}
		    }
		  else
		    pips_internal_error("not same lhs and rhs types, not yet supported. Please report.\n");
		}
	  
	    }
	  else
	    {
	      pips_debug(2, "no pointer assignment\n");
	    }	
	}

      l_out = l_gen; /* temporary shortcut for testing */
    }
  else if(type_functional_p(lhs_type))
    {
      pips_internal_error("not yet implemented\n");
    }
  else
    pips_internal_error("unexpected_type\n");

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
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

  reset_pv();
  reset_gen_pv();
  reset_kill_pv();

  return;
}

/**************** INTERFACE *************/

bool simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  set_methods_for_simple_pointer_effects();
  generic_module_pointer_values(module_name, &ctxt);
  reset_pv_context(&ctxt);
  generic_effects_reset_all_methods();
  return(TRUE);
}


cell_relation make_simple_pv_from_simple_effects(effect lhs_eff, effect rhs_eff, cell_interpretation ci)
{
  cell_relation pv;  

  pips_debug(1,"begin for %s cell_interpretation and effects :\n", cell_interpretation_value_of_p(ci)? "value_of" : "address_of");
  pips_debug_effect(2, "lhs_eff: ", lhs_eff);
  pips_debug_effect(2, "rhs_eff: ", rhs_eff);

  tag lhs_t = effect_approximation_tag(lhs_eff);
  tag rhs_t = effect_approximation_tag(rhs_eff);
  tag t = approximation_and(lhs_t, rhs_t);

  if (t == is_approximation_must) t = is_approximation_exact;
  
  cell lhs_c = effect_cell(lhs_eff);
  if (cell_preference_p(lhs_c))
    /* no need to copy the reference, it won't be freed when the effect is freed */
    lhs_c = make_cell(is_cell_reference, effect_any_reference(lhs_eff));
  else
    lhs_c = copy_cell(lhs_c);

  cell rhs_c = effect_cell(rhs_eff);
  if (cell_preference_p(rhs_c))
    /* no need to copy the reference, it won't be freed when the effect is freed */
    rhs_c = make_cell(is_cell_reference, effect_any_reference(rhs_eff));
  else
    rhs_c = copy_cell(rhs_c);



  if (cell_interpretation_value_of_p(ci))
    pv = make_value_of_pointer_value(lhs_c,
				     rhs_c,
				     t,
				     make_descriptor_none());	
  else
    pv = make_address_of_pointer_value(lhs_c,
				       rhs_c,
				       t,
				       make_descriptor_none());	
  
  pips_debug_pv(2, "generating: ", pv); 
  pips_debug(1,"end\n");
  return pv;
}
