/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
static const string FORT_PRAGMA_HEADER = "!$";
static const string FORT_OMP_CONTINUATION = "\n!$omp& ";


/*****************************************************A CONSTRUCTOR LIKE PART
 */

/** @return an empty extensions
 */
extensions empty_extensions (void) {
  return make_extensions (NIL);
}

/** @return TRUE if the extensions field is empty
    @param es the extensions to test
*/
bool empty_extensions_p (extensions es) {
  return (extensions_extension (es) == NIL);
}

/***************************************************PRAGMA AS EXPRESSION PART
 */

/** @return "private (x,y)" as an expression
 *  @param arg, the private variables as a list of entities
 */
expression pragma_private_as_expr (list args_ent) {
  // build the privates variavle as a list of expression
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


/** @return a new allocated string to close the extension.
 *  @param es, the extension to be closed
 *
 *  Today we only generate omp parallel do pragma so the close is pretty
 *  easy. Later we will have to analyze the extension to generate the
 *  close string accordingly.
 */
string close_extension (extension e) {
  string result = string_undefined;
  if (get_prettyprint_is_fortran () == TRUE)
    {
      if (pragma_entity_p(extension_pragma(e)))
	result=directive_to_string(load_global_directives(pragma_entity(extension_pragma(e))),true);
      else
	result = strdup(concatenate (FORT_PRAGMA_HEADER, "omp end parallel do",
				     NULL));
    }
  return result;
}

/** @return a new allocated string to close the extensions.
 *  @param es, the extensions to be closed
 *  @param nl, set to TRUE to get the string with a final new line character
 */
string
close_extensions (extensions es, bool nl) {
  string s = string_undefined;

  if (empty_extensions_p (es) == FALSE) {
    /* Use a string_buffer for efficient string concatenation: */
    string_buffer sb = string_buffer_make(FALSE);

    list el = extensions_extension(es);
    FOREACH(EXTENSION, e, el) {
      s = close_extension(e);
      if (s != string_undefined) {
	string_buffer_append(sb, s);
	if (nl == TRUE) string_buffer_append(sb, strdup ("\n"));
	nl = TRUE;
      }
    }
    s = string_buffer_to_string(sb);
    /* Free the buffer with its strings: */
    string_buffer_free_all(&sb);
  }

  return s;
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
      l_str = words_expression(e);
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
	    gen_insert_before (strdup (FORT_OMP_CONTINUATION), str, l_str);
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
      s = strdup(concatenate (FORT_PRAGMA_HEADER, s, NULL));
    }
    else
      s = strdup(concatenate (C_PRAGMA_HEADER, " ", s, NULL));
  }
  return s;
}

/** @return a new allocated string with the extension textual representation.
 */
string
extension_to_string(extension e) {
  string s;
  /* For later other extensions:
  switch (extension_tag(e)) {
  case is_extension_pragma:
  */
  s = pragma_to_string(extension_pragma(e));
  /*
  default:
    pips_internal_error("Unknown extension type\n");
  }
  */
  return s;
}

/** @brief return a new allocated string with the string representation of the
 *  extensions. Basically you'll get one extension per line
 *
 *  Assume that all the extension from the extensions (note the presence
 *  or not of the "s"...) are defined below.
 *
 *  @return string_undefined if es is extension_undefined, an malloc()ed
 *  textual string either.
 *  @param es, the extensions to translate to strings
 *  @param nl, set to TRUE to get the string with a final new line character
 */
string
extensions_to_string(extensions es, bool nl) {
  string s = string_undefined;

  /* Prettyprint in the correct language: */
  set_prettyprint_is_fortran_p(!get_bool_property("PRETTYPRINT_C_CODE"));

  if (empty_extensions_p (es) == FALSE) {
    /* Use a string_buffer for efficient string concatenation: */
    string_buffer sb = string_buffer_make(FALSE);

    list el = extensions_extension(es);
    FOREACH(EXTENSION, e, el) {
      s = extension_to_string(e);
      string_buffer_append(sb, s);
      if (nl == TRUE) string_buffer_append(sb, strdup ("\n"));
      nl = TRUE;
    }
    s = string_buffer_to_string(sb);
    /* Free the buffer with its strings: */
    string_buffer_free_all(&sb);
  }

  return s;
}

/********************************************************** PRAGMA MANAGEMENT
 */

/** @brief  Add a pragma as a string to a statement.
 *  @return void
 *  @param  stat, the statement on which we want to add a pragma
 *  @param  s, the string pragma.
 *  @param  copy_flag, to be set to true to duplicate the string
 */
void
add_pragma_str_to_statement(statement st, string s, bool copy_flg) {
  extensions es = statement_extensions(st);
  /* Make a new pragma: */
  pragma p = pragma_undefined;
  if (copy_flg == TRUE) p = make_pragma(is_pragma_string, strdup(s));
  else p = make_pragma_string(s);
  extension e = make_extension(p);
  /* Add the new pragma to the extension list: */
  list el = extensions_extension(es);
  el = gen_extension_cons(e, el);
  extensions_extension(es) = el;
}

/** @brief  Add a pragma as a list of expression to a statement.
 *  @return void
 *  @param  stat, the statement on which we want to add a pragma
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
