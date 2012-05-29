/*
  Copyright 1989-2012 MINES ParisTech - HPC Project

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
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "complexity_ri.h"
#include "complexity.h"


static list current = NIL;
static list next = NIL;
static list privates = NIL;

/// @brief the fonction aims at identifing the parallel loops and queues them
/// in the next list.
/// @return false when a parallel loop is found
/// @param l, the loop to process
static bool identify_outer_loops (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    next = gen_loop_cons (l, next);
    return false;
  }
  return true;
}

/// @brief collect the privates variables of inner loops
/// @return TRUE
/// @param l, the loop to process
static bool collect_privates (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    list var = loop_private_variables_as_entites (l, true, true);
    privates = gen_nconc (privates, var);
  }
  return true;
}

/// @brief make the inner loops sequential
/// @param l, the loop to process
static void  process_loop (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    // this loop is an innner loop -> make it sequential
    execution_tag (loop_execution (l)) = is_execution_sequential;
  }
}

/**
**/
bool limit_nested_parallelism (const const char* module_name) {

  // Use this module name and this environment variable to set
  statement mod_stmt = PIPS_PHASE_PRELUDE(module_name,
					  "MANAGE_PARALLEL_LOOPS_DEBUG_LEVEL");

  int threshold = get_int_property ("NESTED_PARALLELISM_THRESHOLD");
  if (threshold > 0) {
    // initialize the next list with all outer parallel loops
    gen_recurse(mod_stmt, loop_domain, identify_outer_loops, gen_identity);
    current = next;
    next = NIL;
    for (int i = 2; i <= threshold; i++) {
      // mark the nested loop at level i
      FOREACH (LOOP, l, current) {
	gen_recurse(loop_body (l), loop_domain, identify_outer_loops, gen_identity);
      }
      gen_free_list (current);
      current = next;
      next = NIL;
    }
  }

  // Targeted outer loops have been identified. They need to be processed
  // inne loops are marked sequential and local variables are moved
  // at the outer loop level
  FOREACH (LOOP, l, current) {
    gen_recurse(loop_body (l), loop_domain, collect_privates, process_loop);
    // need to merge entity one by one otherwise a newgen assertion
    // (about "no sharing of cons") raises
      list locals = loop_locals (l);
      FOREACH (ENTITY, e, privates) {
	if (gen_in_list_p (e, locals) == false) {
	  locals = gen_entity_cons (e, locals);
	}
      }
      loop_locals (l) = locals;
      gen_free_list (privates);
      privates = NIL;
  }

  gen_free_list (current);
  current = NIL;

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(mod_stmt);

  return true;
}

/*****************************************************************/

typedef struct  limit_uninteresting_parallelism_context{
  bool (*loop_cost_testing_function)(statement,  struct limit_uninteresting_parallelism_context *);
  int startup_overhead;
  int bandwidth;
  int frequency;
  list parallel_loops;
} limit_uninteresting_parallelism_context;

static void init_limit_uninteresting_parallelism_context(limit_uninteresting_parallelism_context *p_ctxt,
							 bool (*loop_cost_testing_function)(statement, limit_uninteresting_parallelism_context *))
{
  p_ctxt->loop_cost_testing_function = loop_cost_testing_function;
  p_ctxt->startup_overhead = get_int_property("COMPUTATION_INTENSITY_STARTUP_OVERHEAD");
  p_ctxt->bandwidth = get_int_property("COMPUTATION_INTENSITY_BANDWIDTH");
  p_ctxt->frequency = get_int_property("COMPUTATION_INTENSITY_FREQUENCY");
  p_ctxt->parallel_loops = NIL;
}


/** Cost function to test whether a loop is worth parallelizing

    Currently tests whether the highest coefficient in the whole loop
    complexity polynome divided by COMPUTATION_INTENSITY_FREQUENCY
    is higher than COMPUTATION_INTENSITY_STARTUP_OVERHEAD + 10.
 */
static bool complexity_cost_effective_loop_p(statement s,
					     limit_uninteresting_parallelism_context * p_ctxt)
{
  pips_assert("input statement must be a loop", statement_loop_p(s));
  bool result = true;
  complexity comp = load_statement_complexity(s);

  Ppolynome instruction_time = polynome_dup(complexity_polynome(comp));
  polynome_scalar_mult(&instruction_time, 1.f/p_ctxt->frequency);
  polynome_scalar_add(&instruction_time, p_ctxt->startup_overhead);

  int max_degree = polynome_max_degree(instruction_time);
  pips_debug(1, "max_degree is: %d\n", max_degree);
  float coeff=-1.f;
  for(Ppolynome p = instruction_time; !POLYNOME_NUL_P(p); p = polynome_succ(p))
    {
      int curr_degree =  (int)vect_sum(monome_term(polynome_monome(p)));
      if(curr_degree == max_degree) {
	coeff = monome_coeff(polynome_monome(p));
	break;
      }
    }
  polynome_rm(&instruction_time);

  pips_debug(1, "coeff is: %f\n", coeff);
  result = (coeff > (float) (p_ctxt->startup_overhead + 10));
  return result;
}

static bool limit_uninteresting_parallelism_statement_in(statement s,
							 limit_uninteresting_parallelism_context * p_ctxt)
{
    if (statement_loop_p(s))
    {
      pips_debug(1, "Entering loop statement with ordering: %03zd and number: %03zd\n",
		 statement_ordering(s), statement_number(s));
      ifdebug(1) {
	print_statement(s);
      }
      loop l = statement_loop(s);
      if (loop_parallel_p(l))
	p_ctxt->parallel_loops = CONS(LOOP, l, p_ctxt->parallel_loops);

    }
  return true;
}

static void limit_uninteresting_parallelism_statement_out(statement s,
							 limit_uninteresting_parallelism_context * p_ctxt)
{

  if (statement_loop_p(s))
    {
      pips_debug(1, "Dealing with loop statement with ordering: %03zd and number: %03zd\n",
		 statement_ordering(s), statement_number(s));
      ifdebug(1) {
	print_statement(s);
      }

      loop l = statement_loop(s);
      if (loop_parallel_p(l) && ! p_ctxt->loop_cost_testing_function(s, p_ctxt))
	{
	  POP(p_ctxt->parallel_loops);
	  execution_tag(loop_execution(l)) = is_execution_sequential;
	  /* now deal with loop locals: they must be propagated back to outer parallel loops */
	  list l_locals = loop_locals(l);
	  entity index = loop_index(l);
	  if (!ENDP(p_ctxt->parallel_loops) && !ENDP(l_locals))
	    {
	      loop previous_parallel_loop = LOOP(CAR(p_ctxt->parallel_loops));
	      list previous_parallel_loop_locals = loop_locals(previous_parallel_loop);
	      list to_add = NIL;
	      FOREACH(ENTITY, local, l_locals)
		{
		  if (local != index
		      && gen_find_eq( local, previous_parallel_loop_locals ) == entity_undefined )
		    to_add = CONS(ENTITY, local, to_add);
		}
	      loop_locals(previous_parallel_loop) = gen_append(previous_parallel_loop_locals, to_add);
	    }
	}
      pips_debug(1, "leaving loop\n");

    }
}

/**
**/
bool limit_parallelism_using_complexity(const const char* module_name)
{

  statement mod_stmt = PIPS_PHASE_PRELUDE(module_name,
					  "MANAGE_PARALLEL_LOOPS_DEBUG_LEVEL");
  set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

  limit_uninteresting_parallelism_context ctxt;
  init_limit_uninteresting_parallelism_context(&ctxt, complexity_cost_effective_loop_p);
  gen_context_recurse(get_current_module_statement(), &ctxt,
		      statement_domain, limit_uninteresting_parallelism_statement_in, limit_uninteresting_parallelism_statement_out);

  reset_complexity_map();
  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(mod_stmt);

  return true;
}

