/*
  Copyright 1989-2010 MINES ParisTech

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

/**
 * @file manage_pragma.c
 * @brief This file holds transformations on OpenMP pragmas
 * store in the RI as extension expression.
 * Here is the list of transformations:
 *   1- add an OpenMP if clause
 */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

// The list of outter loop as a list of statements
static list l_outer = NIL;

// The list of pragma to be merged
static list l_pragma = NIL;

// The list of number of iteration (as expression) to be used in the if clause
static list l_iters = NIL;

// build a list of pragma to be merged. Also remove them from the
// list they currently belongs to.
static void build_pragma_list (extensions exts) {
  list tmp = NIL;
  list l_exts = extensions_extension (exts);
  FOREACH (EXTENSION, ext, l_exts) {
    // today extension is only pragma but can be something else in the future
    // a test will have to be done
    // if (extension_is_pragma_p ())
    tmp = gen_extension_cons (ext, tmp);
    l_pragma = gen_pragma_cons (extension_pragma (ext), l_pragma);
  }
  // remove the extensions that will be merger at outer level
  gen_list_and_not (&l_exts, tmp);
  // update the extensions field
  extensions_extension (exts) = l_exts;
}

// keep track of outer loop with pragma and return false
static bool build_outer (loop l) {
  statement stmt = (statement) gen_get_ancestor(statement_domain, l);
  list l_exts = extensions_extension (statement_extensions (stmt));
  FOREACH (EXTENSION, ext, l_exts) {
    // today extension is only pragma but can be something else in the future
    // a test will have to be done
    // if (extension_is_pragma_p ())
    pragma pr = extension_pragma (ext);
    pips_debug (5, "processing pragma : %s\n", pragma_to_string (pr));
    // only the pragma as expressions are managed
    if (pragma_expression_p (pr) == TRUE) {
      l_outer = gen_statement_cons (stmt, l_outer);
      return FALSE;
    }
  }
  return TRUE;
}

/// @brief merge the omp pragma on the most outer parallel loop
/// @return void
/// @param pr, the pragma to process
static void merge_on_outer () {
  FOREACH (STATEMENT, stmt, l_outer) {
    // collect the pragma
    gen_recurse  (stmt, extensions_domain, gen_true, build_pragma_list);
    list l_expr = pragma_omp_merge_expr (l_pragma);
    add_pragma_expr_to_statement (stmt, l_expr);
    gen_free_list (l_pragma);
    l_pragma = NIL;
  }
  return;
}

static void build_iteration_list (range r) {
  expression iter = range_to_expression(r , range_to_nbiter);
  l_iters = gen_expression_cons (iter, l_iters);
}

/// @brief add a if condition to the omp pragma
/// @return void
/// @param pr, the pragma to process
static void add_loop_parallel_threshold (pragma pr) {
  // only the pragma as expressions are managed
  if (pragma_expression_p (pr) == TRUE) {
    // we need to get the loop index
    statement stmt = (statement) gen_get_ancestor(statement_domain, pr);
    instruction inst = statement_instruction (stmt);
    if (instruction_tag (inst) == is_instruction_loop) {
      loop l = instruction_loop (inst);
      // evaluate the number of iteration according to the property value
      if (get_bool_property ("OMP_IF_CLAUSE_RECURSIVE") == TRUE) {
	// collect the number of iteration of current loop and inner loops
	gen_recurse  (stmt, range_domain, gen_true, build_iteration_list);
      }
      else {
	// get the number of iteration of the current loop only
	expression iter = range_to_expression(loop_range (l), range_to_nbiter);
	l_iters = gen_expression_cons (iter, l_iters);
      }
      // now we have a list of number of iteration we need to multiply them
      entity mul = CreateIntrinsic (MULTIPLY_OPERATOR_NAME);
      expression cond = expressions_to_operation (l_iters, mul);
      // compare the nb iteration to the threshold
      cond = pragma_build_if_condition (cond);
      // encapsulate the condition into the if clause
      expression expr_if = pragma_if_as_expr (cond);
      // bind the clause to the pragma
      add_expr_to_pragma_expr_list (pr, expr_if);
      // free list
      gen_free_list (l_iters);
      l_iters = NIL;
    }
  }
  return;
}

//////////////////////////////////////////////////////////////
// the phase function name

/**
   merge the pragma on the outer loop
**/
bool omp_merge_pragma (const string module_name) {
  debug_on("OPMIFY_CODE_DEBUG_LEVEL");
  statement mod_stmt = statement_undefined;
  // Get the code and tell PIPS_DBM we do want to modify it
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  // Set the current module entity and the current module statement that are
  // required to have many things working in PIPS
  set_current_module_statement(mod_stmt);
  set_current_module_entity(module_name_to_entity(module_name));

  // generate pragma string or expression using the correct language:
  value mv = entity_initial(module_name_to_entity(module_name));
  if(value_code_p(mv)) {
    code c = value_code(mv);
    set_prettyprint_language_from_property(language_tag(code_language(c)));
  } else {
    /* Should never arise */
    set_prettyprint_language_from_property(is_language_fortran);
  }

  // build the list of outer loop with pragma
  gen_recurse(mod_stmt, loop_domain, build_outer, gen_identity);

  // merge the pragma on the outer loop
  merge_on_outer ();
  gen_free_list (l_outer);
  l_outer = NIL;

  // Put the new CODE ressource into PIPS
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, mod_stmt);
  // There is no longer a current module:
  reset_current_module_statement();
  reset_current_module_entity();

  pips_debug(2, "done for %s\n", module_name);
  debug_off();

  return TRUE;
}

bool omp_loop_parallel_threshold_set (const string module_name) {
  debug_on("OPMIFY_CODE_DEBUG_LEVEL");
  statement mod_stmt = statement_undefined;
  // Get the code and tell PIPS_DBM we do want to modify it
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  // Set the current module entity and the current module statement that are
  // required to have many things working in PIPS
  set_current_module_statement(mod_stmt);
  set_current_module_entity(module_name_to_entity(module_name));

  // generate pragma string or expression using the correct language:
  value mv = entity_initial(module_name_to_entity(module_name));
  if(value_code_p(mv)) {
    code c = value_code(mv);
    set_prettyprint_language_from_property(language_tag(code_language(c)));
  } else {
    /* Should never arise */
    set_prettyprint_language_from_property(is_language_fortran);
  }

  // Add the parallel threshold to all the omp for pragmas
  gen_recurse(mod_stmt, pragma_domain, gen_true,
	      add_loop_parallel_threshold);

  /* Put the new CODE ressource into PIPS: */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, mod_stmt);
  // There is no longer a current module:
  reset_current_module_statement();
  reset_current_module_entity();

  pips_debug(2, "done for %s\n", module_name);
  debug_off();

  return TRUE;
}
