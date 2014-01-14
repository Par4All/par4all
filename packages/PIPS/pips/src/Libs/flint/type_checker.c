/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/*
 * Typecheck Fortran code.
 * by Son PhamDinh 03-05/2000
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

#include "bootstrap.h" /* type of intrinsics stuff... */

/* Working with hash_table of basic
 */
#define GET_TYPE(h, e) ((basic)hash_get(h, (char*)(e)))
#define PUT_TYPE(h, e, b) hash_put(h, (char*)(e), (char*)(b))

/* ENTITY
 * It should be added in file "ri-util.h"
 */
#define CONCAT_OPERATOR_NAME "//"
#define ENTITY_CONCAT_P(e) (entity_an_operator_p(e, CONCAT))
#define ENTITY_EXTERNAL_P(e) (value_code_p(entity_initial(e)))
#define ENTITY_INTRINSIC_P(e) (value_intrinsic_p(entity_initial(e)))

#define ENTITY_CONVERSION_P(e,name) \
  (strcmp(entity_local_name(e), name##_GENERIC_CONVERSION_NAME)==0)
#define ENTITY_CONVERSION_CMPLX_P(e) ENTITY_CONVERSION_P(e, CMPLX)
#define ENTITY_CONVERSION_DCMPLX_P(e) ENTITY_CONVERSION_P(e, DCMPLX)

static void type_this_entity_if_needed(entity, type_context_p);

/**************************************************************************
 * Typing all the arguments of user-defined function C to its parameters
 * correspondent.
 *
 * Note: The call C must be an user-defined function
 */
static basic
typing_arguments_of_user_function(call c, type_context_p context)
{
  list            args = call_arguments(c);
  type            the_tp = entity_type(call_function(c));
  functional      ft = type_functional(the_tp);
  list            params = functional_parameters(ft);
  type            result = functional_result(ft);
  int             na = gen_length(args);
  int             nt = gen_length(params);
  parameter       param;
  basic b, b1;
  int argnumber = 0;

  if (na == nt ||
      (nt<=na &&
       type_varargs_p(parameter_type(PARAMETER(CAR(gen_last(params)))))))
  {
    while (args != NIL)
    {
      argnumber++;
      /* Here, parameter is always a variable */
      param = PARAMETER(CAR(params));
      b = variable_basic(type_variable(parameter_type(param)));
      b1 = GET_TYPE(context->types, EXPRESSION(CAR(args)));
      if (!basic_equal_p(b, b1))
      {
	add_one_line_of_comment((statement) stack_head(context->stats),
				"invalid arg #%d to '%s', %s instead of %s!",
				argnumber,
				entity_local_name(call_function(c)),
				basic_to_string(b1),
				basic_to_string(b));
	context->number_of_error++;
      }
      args = CDR(args);
      params = CDR(params);
    }
  }
  else if (na < nt)
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Too few argument(s) to '%s' (%d<%d)!",
			    entity_local_name(call_function(c)), na, nt);
    context->number_of_error++;
  }
  else
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Too many argument(s) to '%s' (%d>%d)!",
			    entity_local_name(call_function(c)), na, nt);
    context->number_of_error++;
  }

  /* Subroutine */
  if (type_void_p(result))
  {
    pips_debug(7, "type of %s is overloaded\n", entity_name(call_function(c)));
    b = make_basic_overloaded();
  }
  /* Function */
  else
  {
    pips_debug(7, "type of %s is a function\n", entity_name(call_function(c)));
    b = copy_basic(variable_basic(type_variable(result)));
  }
  return b;
}

/*****************************************************************************
 * Make typing an expression of type CALL
 * WARNING: The interpretation of COMPLEX !!!
 */
static basic
type_this_call(expression exp, type_context_p context)
{
  typing_function_t dotype;
  switch_name_function simplifier;
  call c = syntax_call(expression_syntax(exp));
  entity function_called = call_function(c);
  basic b;
  b = basic_undefined;

  pips_debug(2, "Call to %s; Its type is %s \n", entity_name(function_called),
	     type_to_string(entity_type(function_called)));

  /* Labels */
  if (entity_label_p(function_called))
  {
    b = make_basic_overloaded();
  }

  /* Constants */
  else if (entity_constant_p(function_called))
  {
    b = basic_of_call(c, true, true);
  }

  /* User-defined functions */
  else if (ENTITY_EXTERNAL_P(function_called))
  {
    b = typing_arguments_of_user_function(c, context);
  }

  /* All intrinsics */
  else if (ENTITY_INTRINSIC_P(function_called))
  {
    /* Typing intrinsics */
    dotype = get_typing_function_for_intrinsic(
				   entity_local_name(function_called));
    if (dotype != 0)
    {
      b = dotype(c, context);
    }

    /* Simplification */
    simplifier = get_switch_name_function_for_intrinsic(
				   entity_local_name(function_called));
    if (simplifier != 0)
    {
      simplifier(exp, context);
    }
  }
  else if (value_symbolic_p(entity_initial(function_called)))
  {
    /* lazy type entity contents... */
    type_this_entity_if_needed(function_called, context);
    b = GET_TYPE(context->types,
       symbolic_expression(value_symbolic(entity_initial(function_called))));
    b = copy_basic(b);
  }

  pips_debug(7, "Call to %s typed as %s\n", entity_name(function_called),
	     basic_to_string(b));

  return b;
}

/*****************************************************************************
 * Make typing an instruction
 * (Assignment statement (=) is the only instruction that is typed here)
 */
static void
type_this_instruction(instruction i, type_context_p context)
{
  basic b1;
  call c;
  typing_function_t dotype;

  if (instruction_call_p(i))
  {
    c = instruction_call(i);
    pips_debug(1, "Call to %s; Its type is %s \n",
	       entity_name(call_function(c)),
	       type_to_string(entity_type(call_function(c))));

    /* type check a SUBROUTINE call. */
    if (ENTITY_EXTERNAL_P(call_function(c)))
    {
      b1 = typing_arguments_of_user_function(c, context);

      if (!basic_overloaded_p(b1))
      {
	add_one_line_of_comment((statement) stack_head(context->stats),
				"Ignored %s value returned by '%s'",
				basic_to_string(b1),
				entity_local_name(call_function(c)));
	/* Count the number of errors */
	context->number_of_error++;
      }
      free_basic(b1);

      return;
    }

    /* Typing intrinsics:
     * Assignment, control statement, IO statement
     */
    dotype = get_typing_function_for_intrinsic(
				   entity_local_name(call_function(c)));
    if (dotype != 0)
    {
      b1 = dotype(c, context);
    }
  }
}

static void
check_this_test(test t, type_context_p context)
{
  basic b = GET_TYPE(context->types, test_condition(t));
  if (!basic_logical_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Test condition must be a logical expression!");
    context->number_of_error++;
  }
}

static void
check_this_whileloop(whileloop w, type_context_p context)
{
  basic b = GET_TYPE(context->types, whileloop_condition(w));
  if (!basic_logical_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "While condition must be a logical expression!");
    context->number_of_error++;
  }
}

/*****************************************************************************
 * Range of loop (lower, upper, increment), all must be Integer, Real or Double
 * (According to ANSI X3.9-1978, FORTRAN 77; Page 11-5)
 *
 * Return true if type of range is correct, otherwise FALSE
 */
bool
check_loop_range(range r, hash_table types)
{
  basic lower, upper, incr;
  lower = GET_TYPE(types, range_lower(r));
  upper = GET_TYPE(types, range_upper(r));
  incr = GET_TYPE(types, range_increment(r));
  if( (basic_int_p(lower) || basic_float_p(lower)) &&
      (basic_int_p(upper) || basic_float_p(upper)) &&
      (basic_int_p(incr) || basic_float_p(incr)))
  {
    return true;
  }
  return false;
}
/*****************************************************************************
 * Typing the loop if necessary
 */
static void
check_this_loop(loop l, type_context_p context)
{
  basic ind = entity_basic(loop_index(l));

  /* ok for F77, but not in F90? */
  if (!basic_int_p(ind))
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Obsolescent non integer loop index '%s'"
			    " (R822 ISO/IEC 1539:1991 (E))",
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }

  if( !(basic_int_p(ind) || basic_float_p(ind)) )
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Index '%s' must be Integer, Real or Double!",
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }
  else if (!check_loop_range(loop_range(l), context->types))
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
		    "Range of index '%s' must be Integer, Real or Double!",
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }
  else
  {
    type_loop_range(ind, loop_range(l), context);
  }
}

/*****************************************************************************
 * This function will be called in the function
 * gen_context_recurse(...) of Newgen as its parameter
 */
static void
type_this_expression(expression e, type_context_p context)
{
  syntax s = expression_syntax(e);
  basic b = basic_undefined;

  /* Specify the basic of the expression e  */
  switch (syntax_tag(s))
  {
  case is_syntax_call:
    b = type_this_call(e, context);
    break;

  case is_syntax_reference:
    b = copy_basic(entity_basic(reference_variable(syntax_reference(s))));
    pips_debug(2,"Reference: %s; Type: %s\n",
	       entity_name(reference_variable(syntax_reference(s))),
	       basic_to_string(b));
    break;

  case is_syntax_range:
    /* PDSon: For the range alone (not in loop),
     * I only check lower, upper and step, they must be all INT, REAL or DBLE
     */
    if (!check_loop_range(syntax_range(s), context->types))
    {
      add_one_line_of_comment((statement)stack_head(context->stats),
			      "Range must be INT, REAL or DBLE!");
      context->number_of_error++;
    }
    break;

  default:
    pips_internal_error("unexpected syntax tag (%d)", syntax_tag(s));
  }

  /* Push the basic in hash table "types" */
  if (!basic_undefined_p(b))
  {
    PUT_TYPE(context->types, e, b);
  }
}

static void check_this_reference(reference r, type_context_p context)
{
  MAP(EXPRESSION, ind,
  {
    /* cast expressions to INT if not already an int... ? */
    /* ??? maybe should update context->types ??? */

    basic b = GET_TYPE(context->types, ind);
    if (!basic_int_p(b))
    {
      basic bint = make_basic_int(4);
      insert_cast(bint, b, ind, context); /* and simplifies! */
      free_basic(bint);
    }
  },
      reference_indices(r));
}

static bool
stmt_flt(statement s, type_context_p context)
{
  stack_push(s, context->stats);
  return true;
}

static void
stmt_rwt(statement s, type_context_p context)
{
  pips_assert("pop same push", stack_head(context->stats)==s);
  stack_pop(context->stats);
}

static void type_this_chunk(void * c, type_context_p context)
{
  gen_context_multi_recurse
    (c, context,
     statement_domain, stmt_flt, stmt_rwt,
     instruction_domain, gen_true, type_this_instruction,
     test_domain, gen_true, check_this_test,
     whileloop_domain, gen_true, check_this_whileloop,
     loop_domain, gen_true, check_this_loop,
     expression_domain, gen_true, type_this_expression,
     reference_domain, gen_true, check_this_reference,
     NULL);
}

static void type_this_entity_if_needed(entity e, type_context_p context)
{
  value v = entity_initial(e);

  /* ??? TODO: type->variable->dimensions->dimension->expression */

  if (value_symbolic_p(v))
  {
    symbolic sy = value_symbolic(v);
    expression s = symbolic_expression(sy);
    basic b1, b2;

    if (hash_defined_p(context->types, s))
      return;

    type_this_chunk((void *) s, context);

    /* type as "e = s" */
    b1 = entity_basic(e);
    b2 = GET_TYPE(context->types, s);

    if (!basic_compatible_p(b1, b2))
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
		   "%s parameter '%s' definition from incompatible type %s",
			      basic_to_string(b1),
			      entity_local_name(e),
			      basic_to_string(b2));
      context->number_of_error++;
      return;
    }

    if (!basic_equal_p(b1, b2))
    {
      symbolic_expression(sy) = insert_cast(b1, b2, s, context);
      PUT_TYPE(context->types, symbolic_expression(sy), copy_basic(b1));
    }
}
}

static void put_summary(string name, type_context_p context)
{
  user_log("Type Checker Summary\n"
	   "\t%d errors found\n"
	   "\t%d conversions inserted\n"
	   "\t%d simplifications performed\n",
	   context->number_of_error,
	   context->number_of_conversion,
	   context->number_of_simplication);

  pips_user_warning("summary of '%s': "
		    "%d errors, %d convertions., %d simplifications\n",
		    name,
		    context->number_of_error,
		    context->number_of_conversion,
		    context->number_of_simplication);

  if (name && get_bool_property("TYPE_CHECKER_ADD_SUMMARY"))
  {
    entity module = local_name_to_top_level_entity(name);
    code c;
    char *buf;

    pips_assert("entity is a module", entity_module_p(module));

    c = value_code(entity_initial(module));

    asprintf( &buf,
	    "!PIPS TYPER: %d errors, %d conversions, %d simplifications\n",
	    context->number_of_error,
	    context->number_of_conversion,
	    context->number_of_simplication);

    if (!code_decls_text(c) || string_undefined_p(code_decls_text(c)))
      code_decls_text(c) = buf;
    else
    {
      string tmp = code_decls_text(c);
      code_decls_text(c) = strdup(concatenate(buf, tmp, NULL));
      free(buf);
      free(tmp);
    }
  }
}

/**************************************************************************
 * Type check all expressions in statements.
 * Returns false if type errors are detected.
 */
void typing_of_expressions(string name, statement s)
{
  type_context_t context;

  context.types = hash_table_make(hash_pointer, 0);
  context.stats = stack_make(statement_domain, 0, 0);
  context.number_of_error = 0;
  context.number_of_conversion = 0;
  context.number_of_simplication = 0;

  /* Bottom-up typing */
  type_this_chunk((void *) s, &context);

  /* Summary */
  put_summary(name, &context);

  /* Type checking ... */
  HASH_MAP(st, ba, free_basic(ba), context.types);
  hash_table_free(context.types);
  stack_free(&context.stats);
}

bool type_checker(string name)
{
  statement stat;
  debug_on("TYPE_CHECKER_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  /* Used to check the language */
  set_current_module_entity(module_name_to_entity(name));
  stat = (statement) db_get_memory_resource(DBR_CODE, name, true);
  set_current_module_statement(stat);

  typing_of_expressions(name, stat);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, name, stat);
  reset_current_module_statement();
  reset_current_module_entity();

  pips_debug(1, "done");
  debug_off();
  return true;
}
