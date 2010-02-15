/*

  $Id$

  Copyright 1989-2010 HPC Project

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
/*
  This file define methods to deal with objects extensions and pragma
  used as extensions to statements in the PIPS internal representation.

  A middle term, extensions method could go in another file.

  It is a trivial inplementation based on strings for a proof of concept.

  Pierre.Villalon@hpc-project.com
  Ronan.Keryell@hpc-project.com
*/

#include "linear.h"
#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "outlining_private.h"
#include "step_private.h"
#include "step.h"
#include "properties.h"

//***********************************************************Local constant
static const string C_PRAGMA_HEADER = "#pragma";
static const string FORTRAN_PRAGMA_HEADER = "!$";
static const string FORTRAN_OMP_CONTINUATION = "\n!$omp& ";



/***************************************************PRAGMA AS EXPRESSION PART
 */

/** @return "if (cond)" as an expression
 *  @param arg, the condition to be evaluted by the if clause
 */
expression pragma_if_as_expr (expression arg) {
  entity omp = CreateIntrinsic(OMP_IF_FUNCTION_NAME);
  list args_expr = gen_expression_cons (arg, NIL);
  call c = make_call (omp, args_expr);
  //  syntax s = make_syntax_call (c);
  expression expr_if = call_to_expression (c);// make_expression (s, normalized_undefined);
  return expr_if;
}

/** @return "private (x,y)" as an expression
 *  @param arg, the private variables as a list of entities
 */
expression pragma_private_as_expr (list args_ent) {
  // build the privates variable as a list of expression
  list args_expr = NIL;
  FOREACH (ENTITY, e, args_ent) {
    reference ref = make_reference (e, NULL);
    syntax s = make_syntax_reference (ref);
    expression expr = make_expression (s, normalized_undefined);
    // append the new expr to the list
    args_expr = gen_expression_cons (expr, args_expr);
  }
  entity omp = CreateIntrinsic(OMP_PRIVATE_FUNCTION_NAME);
  call c = make_call (omp, args_expr);
  syntax s = make_syntax_call (c);
  expression expr_omp = make_expression (s, normalized_undefined);
  return expr_omp;
}

/** @return "omp parallel" as a list of expression
 */
list pragma_omp_parallel_as_exprs (void) {
  // first prepare "omp" as an expression
  entity omp = CreateIntrinsic(OMP_OMP_FUNCTION_NAME);
  call c = make_call (omp, NULL);
  syntax s = make_syntax_call (c);
  expression expr_omp = make_expression (s, normalized_undefined);

  //secondly prepare "parallel" as an expression
  entity parallel = CreateIntrinsic(OMP_PARALLEL_FUNCTION_NAME);
  c = make_call (parallel, NULL);
  s = make_syntax_call (c);
  expression expr_parallel = make_expression (s, normalized_undefined);

  // build the list of expression
  list result = CONS(EXPRESSION, expr_omp, NIL);
  result = gen_expression_cons (expr_parallel, result);
  return result;
}

/** @return "omp parallel for" as an expression
 */
list pragma_omp_parallel_for_as_exprs (void) {
  // first prepare "for" as an expression
  entity e = CreateIntrinsic(OMP_FOR_FUNCTION_NAME);
  call c = make_call (e, NULL);
  syntax s = make_syntax_call (c);
  expression expr_for = make_expression (s, normalized_undefined);

  //secondly get "omp parallel as an expr and concat
  list result = pragma_omp_parallel_as_exprs ();
  result = gen_expression_cons (expr_for, result);

  return result;
}

/***************************************************** PRETTYPRINT PART
 */
/** @return a new allocated string to close the pragma.
 *  @param p, the pragma to be closed
 *
 */
string close_pragma (pragma p) {
  string result = string_undefined;
  if (get_prettyprint_is_fortran () == TRUE)
    {
      if (pragma_entity_p (p))
	result=directive_to_string(load_global_directives(pragma_entity(p)),true);
      else
	result = strdup(concatenate (FORTRAN_PRAGMA_HEADER, "omp end parallel do",
				     NULL));
    }
  return result;
}

/** @return a new allocated string with the pragma textual representation.
 */
string
pragma_to_string (pragma p) {
  bool flg   = FALSE;
  list l_str = NULL; //list of string
  list l_expr = NULL; // list of expression
  size_t line_sz = 0; // the pragma line size
  string s = string_undefined;
  string_buffer sb = string_buffer_make(FALSE);

  switch (pragma_tag (p)) {
  case is_pragma_string:
    s = pragma_string(p);
    break;
  case is_pragma_expression:
    l_expr = pragma_expression (p);
    FOREACH (EXPRESSION, e, l_expr) {
      if (flg == TRUE) {
	string_buffer_append (sb, strdup (" "));
	line_sz +=1;
      }
      flg = TRUE;
      l_str = words_expression(e, NIL);
      l_str = gen_nreverse (l_str);
      if (get_prettyprint_is_fortran() == TRUE) {
	// In fortran line size can not be more than 72
	FOREACH (STRING, str, l_str) {
	  pips_assert ("algo bug", line_sz < MAX_LINE_LENGTH - 7);
	  size_t size = strlen (str);
	  pips_assert ("not handled case need to split the str between two lines",
		       size < (MAX_LINE_LENGTH - 7));
	  line_sz += size;
	  if (line_sz >= MAX_LINE_LENGTH - 8) {
	    gen_insert_before (strdup (FORTRAN_OMP_CONTINUATION), str, l_str);
	    line_sz = size;
	  }
	}
      }
      string_buffer_append_list (sb, l_str);
      gen_free_list (l_str);
    }
    s = string_buffer_to_string_reverse (sb);
    // Free the buffer with its strings
    string_buffer_free_all(&sb);
    break;
  case is_pragma_entity:
    return directive_to_string(load_global_directives(pragma_entity(p)),false);
    break;
  default:
    pips_internal_error("Unknown pragama type\n");
    break;
  }
  if (s != string_undefined) {
    if (get_prettyprint_is_fortran() == TRUE) {
      s = strdup(concatenate (FORTRAN_PRAGMA_HEADER, s, NULL));
    }
    else
      s = strdup(concatenate (C_PRAGMA_HEADER, " ", s, NULL));
  }
  return s;
}

/********************************************************** PRAGMA MANAGEMENT
 */

/** @brief  Add a string as a pragma to a statement.
 *  @return void
 *  @param  st, the statement on which we want to add a pragma
 *  @param  s, the pragma string.
 *  @param  copy_flag, to be set to true to duplicate the string
 */
void
add_pragma_str_to_statement(statement st, string s, bool copy_flag) {
  extensions es = statement_extensions(st);
  /* Make a new pragma: */
  pragma p = pragma_undefined;
  if (copy_flag == TRUE) p = make_pragma(is_pragma_string, strdup(s));
  else p = make_pragma_string(s);
  extension e = make_extension(p);
  /* Add the new pragma to the extension list: */
  list el = extensions_extension(es);
  el = gen_extension_cons(e, el);
  extensions_extension(es) = el;
}


/** Add a list of strings as as many pragmas to a statement

    @param  st, the statement on which we want to add a pragma

    @param  l, a list of pragma string(s)

    @param  copy_flag, to be set to true to duplicate the string
 */
void
add_pragma_strings_to_statement(statement st, list l, bool copy_flag) {
  FOREACH(STRING, p, l)
    add_pragma_str_to_statement(st, p, copy_flag);
}


/** @brief  Add a pragma as a list of expression to a statement.
 *  @return void
 *  @param  st, the statement on which we want to add a pragma
 *  @param  l, the list of expression.
 */
void
add_pragma_expr_to_statement(statement st, list l) {
  extensions es = statement_extensions(st);
  /* Make a new pragma: */
  pragma p = pragma_undefined;
  p = make_pragma_expression(l);
  extension e = make_extension(p);
  /* Add the new pragma to the extension list: */
  list el = extensions_extension(es);
  el = gen_extension_cons(e, el);
  extensions_extension(es) = el;
}

/** @brief  Add an expression to the pragma current expression list.
 *  @return void
 *  @param  pr, the pragma to process.
 *  @param  ex, the expression to add.
 */
void
add_expr_to_pragma_expr_list (pragma pr, expression ex) {
  pips_assert ("the pragma need to be an expression", pragma_expression_p (pr) == TRUE);
  /* Add the new pragma to the extension list: */
  list exprs = pragma_expression (pr);
  exprs = gen_expression_cons (ex, exprs);
  pragma_expression (pr) = exprs;
  return;
}

void add_pragma_entity_to_statement(statement st, entity en)
{
  extensions es = statement_extensions(st);
  /* Make a new pragma: */
  pragma p = pragma_undefined;
  p = make_pragma_entity(en);
  extension e = make_extension(p);
  /* Add the new pragma to the extension list: */
  list el = extensions_extension(es);
  el = gen_extension_cons(e, el);
  extensions_extension(es) = el;
}


