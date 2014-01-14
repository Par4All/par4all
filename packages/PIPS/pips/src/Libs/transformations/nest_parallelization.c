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
 /* loop nest parallelization */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "text-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "resources.h"

#include "control.h"
#include "conversion.h"
/* #include "generation.h" */

#include "arithmetique.h"
#include "vecteur.h"

#include "transformations.h"

/* dependence graph */
static graph dg;

/* only one level of parallelism is allowed; this is checked
 * intra-procedurally only for the time being; pipsmake mechanism would
 * make an interprocedural propagation easy but it would be hard to defer
 * the parallel loop choice up to a whole program analysis; * This static
 * variable is a screwed up mechanism; the information should be
 * recursively maintainted by look_for_nested_loops() and propagated
 * downwards only (not forwards as with the static variable...).
 *
 * The same result could be achived by restarting a different occurence of 
 * look_for_nested_loops() with a simple vectorization trasnformation and
 * another predicate checking for internal loops or for anything I want.
 * No state memorization is needed...
 */
static bool parallel_loop_has_been_selected;

/* No lambda closure in C */
static entity current_loop_index = entity_undefined;

/* Transformation strategy for an isolated loop */

/* the parallelization and vectorization strategy is based on iteration numbers */
#define UNKNOWN_LOOP_COUNT 0
#define SMALL_LOOP_COUNT 1
#define MEDIUM_LOOP_COUNT 2
#define LARGE_LOOP_COUNT 3

#define SEQUENTIAL_DIRECTION 0
#define VECTOR_DIRECTION 1
#define PARALLEL_DIRECTION 2

/* the transformation strategy is chosen according to the loop iteration count
 */
typedef struct transformation_strategy {
    int maximum_iteration_count;
    statement (*loop_transformation)();
} transformation_strategy;



static statement loop_preserve(statement s, __attribute__((unused)) int c)
{
    debug(9, "loop_preserve", "begin\n");

    debug(9, "loop_preserve", "end\n");

    return s;
}
    
static statement loop_vectorize(statement s, __attribute__((unused)) int c)
{
    loop l = statement_loop(s);

    debug(9, "loop_vectorize", "begin\n");

    execution_tag(loop_execution(l)) = is_execution_parallel;

    debug(9, "loop_vectorize", "end\n");

    return s;
}

static statement tuned_loop_parallelize(statement s, __attribute__((unused)) int c)
{
    loop l = statement_loop(s);

    debug(9, "tuned_loop_parallelize", "begin\n");

    /* FI: the recursive computation of parallel_loop_has_been_selected is not implemented */
    if(parallel_loop_has_been_selected && !parallel_loop_has_been_selected)
	;
    else {
	/* the body complexity should be checked and, if it is a constant,
	   a strip-mining factor should be derived to get the best
	   possible load balancing */

	/* the easy way to go is to use the processor number to make chunks 
	 */

	loop_strip_mine(s, -1, get_processor_number());

	/* the outer loop is tagged as parallel, the inner loop is sequential */
	execution_tag(loop_execution(l)) = is_execution_parallel;
	parallel_loop_has_been_selected = true;
    }

    debug(9, "tuned_loop_parallelize", "end\n");

    return s;
}

static statement tuned_loop_unroll(statement s, int c)
{
    loop il = instruction_loop(statement_instruction(s));
    range lr = loop_range(il);
    expression lb = range_lower(lr),
               ub = range_upper(lr),
               inc = range_increment(lr);
    intptr_t lbval, ubval, incval;

    debug(9, "tuned_loop_unroll", "begin\n");

    if (expression_integer_value(lb, &lbval) 
	&& expression_integer_value(ub, &ubval) 
	&& expression_integer_value(inc, &incval)) {
	full_loop_unroll(s);
    }
    else if(c > 1) {
	loop_unroll(s, c);
    }

    debug(9, "tuned_loop_unroll", "end\n");

    return s;
}

static bool current_loop_index_p(reference r)
{
    return reference_variable(r) == current_loop_index;
}

static statement tuned_loop_strip_mine(statement s)
{

    debug(9, "tuned_loop_strip_mine", "begin\n");

    /* strip it for the vector registers and the parallel processors */
    loop_strip_mine(s, get_vector_register_length(), -1);

    /* outer parallel loop */
    /* FI: should be kept sequential if another outer parallel loop has been defined... */
    if(get_processor_number() != 1) {
	execution_tag(loop_execution(statement_loop(s))) = is_execution_parallel;
    }
    else {
	execution_tag(loop_execution(statement_loop(s))) = is_execution_sequential;
    }

    /* inner vector loop */
    pips_assert("tuned_loop_strip_mine", instruction_loop_p(statement_instruction(s)));
    if(get_vector_register_number() > 0) {
	execution_tag(loop_execution(statement_loop(s))) = is_execution_parallel;
    }
    else {
	execution_tag(loop_execution(statement_loop(s))) = is_execution_sequential;
    }

    debug(9, "tuned_loop_strip_mine", "end\n");

    return s;
}


static bool always_select_p(__attribute__((unused)) statement s)
{
    return true;
}
static bool carried_dependence_p(__attribute__((unused))statement s)
{
    return false;
}

static Pvecteur estimate_range_count(range r)
{
    normalized nlb = NORMALIZE_EXPRESSION(range_lower(r));
    normalized nub = NORMALIZE_EXPRESSION(range_upper(r));
    normalized ninc = NORMALIZE_EXPRESSION(range_increment(r));
    Pvecteur count = VECTEUR_UNDEFINED;

    debug(9, "estimate_range_count", "begin\n");

    if(normalized_linear_p(nlb) && normalized_linear_p(nub) && normalized_linear_p(ninc)) {
	Pvecteur vinc = (Pvecteur) normalized_linear(ninc);

	if(vect_constant_p(vinc)) {
	    Pvecteur vlb = (Pvecteur) normalized_linear(nlb);
	    Pvecteur vub = (Pvecteur) normalized_linear(nub);

	    count = vect_substract(vub, vlb);
	    vect_add_elem(&count, TCST, 1);
	    count = vect_div(count, vect_coeff(TCST, vinc));
	}
	else {
	    count = VECTEUR_UNDEFINED;
	}
    }
    else {
	count = VECTEUR_UNDEFINED;
    }

    ifdebug(9) {
	debug(9, "estimate_range_count", "output\n");
	vect_dump(count);
    }
    debug(9, "estimate_range_count", "end\n");

    return count;
}

static Pvecteur estimate_loop_iteration_count(loop l)
{
    return estimate_range_count(loop_range(l));
}
static int numerical_loop_iteration_count(loop l)
{
    Pvecteur count = estimate_loop_iteration_count(l);
    int c;

    if(VECTEUR_UNDEFINED_P(count))
	c = -1;
    else {
	if( vect_constant_p(count)) {
	    Value v = vect_coeff(TCST, count);
	    c = VALUE_TO_INT(v);
	}
	else
	    c = -1;
	vect_rm(count);
    }

    return c;
}

static transformation_strategy 
    one_loop_transformation_strategies
            [PARALLEL_DIRECTION+1][LARGE_LOOP_COUNT+1] = 
               {
	       {{-1, loop_preserve}, 
		{4, tuned_loop_unroll }, 
		{80, loop_preserve }, 
		{INT_MAX, loop_preserve}},
	       {{-1, loop_vectorize}, 
		{4, tuned_loop_unroll }, 
		{80, loop_vectorize }, 
		{INT_MAX, tuned_loop_strip_mine}},
	       {{-1, tuned_loop_parallelize}, 
		{4, tuned_loop_unroll }, 
		{80, tuned_loop_parallelize }, 
		{INT_MAX, tuned_loop_parallelize}}
	       };
static
statement one_loop_parallelization(statement s)
{
    statement new_s = s;
    int c;
    int kind;
    int size;

    debug(9,"one_loop_parallelization", "begin - input loop\n");
    if(get_debug_level()>=9) {
      print_text(stderr,text_statement(entity_undefined,0,s,NIL));
	pips_assert("one_loop_parallelization", statement_consistent_p(s));
    }

    /* find out the loop iteration count c */
    c = numerical_loop_iteration_count(statement_loop(s));

    /* find out the loop kind */
    if(carried_dependence_p(s))
	kind = SEQUENTIAL_DIRECTION;
    else if (assignment_block_or_statement_p(loop_body(statement_loop(s))))
	kind = VECTOR_DIRECTION;
    else
	kind = PARALLEL_DIRECTION;

    /* select the proper transformation */
    for( size = UNKNOWN_LOOP_COUNT; size <= LARGE_LOOP_COUNT; size++)
	if( c <= one_loop_transformation_strategies[kind][size].maximum_iteration_count)
	    break;

    if(size>LARGE_LOOP_COUNT) {
	pips_internal_error("cannot find a transformation strategy"
			    " for kind %d and count %d\n",
			    kind, c);
    }
    else {
	    debug(9, "one_loop_parallelization",
		  "kind = %d, size = %d, c = %d\n",
			   kind, size, c);
	(* one_loop_transformation_strategies[kind][size].loop_transformation)
	    (s, c);
    }


    ifdebug(9) {
	pips_debug(9, "output loop\n");
	print_text(stderr,text_statement(entity_undefined,0,s,NIL));
	pips_assert("one_loop_parallelization", statement_consistent_p(s));
	pips_debug(9, "end\n");
    }

    return new_s;
}
static int look_for_references_in_expression(expression , reference (*) (reference), bool (*) (reference));

static int look_for_references_in_range(range r, reference (*reference_transformation) (reference), bool (*reference_predicate) (reference))
{
    int count = 0;
    expression rl = range_lower(r);
    expression ru = range_upper(r);
    expression ri = range_increment(r);

    debug(5, "look_for_references_in_range", "begin\n");

    count += look_for_references_in_expression(rl, reference_transformation, reference_predicate);
    count += look_for_references_in_expression(ru, reference_transformation, reference_predicate);
    count += look_for_references_in_expression(ri, reference_transformation, reference_predicate);

    debug(5, "look_for_references_in_range", "end %d\n", count);

    return count;
}

static int look_for_references_in_call(call c, reference (*reference_transformation) (reference), bool (*reference_predicate) (reference))
{
    value vin;
    entity f;
    int count = 0;

    debug(5, "look_for_references_in_call", "begin\n");

    f = call_function(c);
    vin = entity_initial(f);

    switch (value_tag(vin)) {
        case is_value_constant:
            /* nothing to replace */
            break;
        case is_value_symbolic:
            pips_internal_error("case is_value_symbolic: not implemented");
            break;
        case is_value_intrinsic:
        case is_value_unknown:
        case is_value_code:
            {
                /* We assume that it is legal to replace arguments (because it should
                   have been verified with the effects that the index is not WRITTEN).
                   */
                FOREACH(EXPRESSION,e,call_arguments(c))
                {
                    count += look_for_references_in_expression(e, 
                            reference_transformation,
                            reference_predicate);
                }
            } break;
        default:
            pips_internal_error("unknown tag: %d", 
                    (int) value_tag(vin));

    }

    debug(5, __func__ , "end %d\n", count);

    return count;
}
static int look_for_references_in_expression(expression e, reference (*reference_transformation) (reference), bool (*reference_predicate) (reference))
{
    syntax s = expression_syntax(e);
    int count = 0;

    debug(5, "look_for_references_in_expression", "begin\n");

    switch(syntax_tag(s)) {
    case is_syntax_reference: {
	reference r = syntax_reference(s);
	if ( (*reference_predicate)(r)) {
	    reference new_r = (*reference_transformation)(syntax_reference(s));
	    /* FI: if a free must be performed, onlye reference_transformation()
	     * knows about it */
	    /* reference_free(syntax_reference(s)); */
	    syntax_reference(s) = new_r;
	    count = 1;
	}

	MAPL(lexpr, {
	    expression indice = EXPRESSION(CAR(lexpr));
	    count += look_for_references_in_expression(indice, reference_transformation,
						       reference_predicate);
	}, reference_indices(r));
    }
	break;
    case is_syntax_range:
	count = look_for_references_in_range(syntax_range(s), reference_transformation,
					     reference_predicate);
	break;
    case is_syntax_call:
	count = look_for_references_in_call(syntax_call(s), reference_transformation,
					   reference_predicate);
	break;
    default: 
	pips_internal_error("unknown tag: %d", 
		   (int) syntax_tag(expression_syntax(e)));
    }

    debug(5, "look_for_references_in_expression", "end %d\n", count);

    return count;
}
static int look_for_references_in_statement(statement s, reference (*reference_transformation) (reference), bool (*reference_predicate) (reference))
{
    instruction inst = statement_instruction(s);
    int count = 0;

    debug(5, "look_for_references_in_statement", "begin\n");

    switch(instruction_tag(inst)) {
    case is_instruction_block :
	MAPL( sts, {
	    count += look_for_references_in_statement(STATEMENT(CAR(sts)),
						      reference_transformation,
						      reference_predicate);
	}, instruction_block(inst));
	break;
    case is_instruction_test : {
	/* legal if no statement redefines ref */
	test t = instruction_test(inst);
	count = look_for_references_in_expression(test_condition(t), reference_transformation,
						  reference_predicate);
	count += look_for_references_in_statement(test_true(t), reference_transformation,
						  reference_predicate);
	count += look_for_references_in_statement(test_false(t), reference_transformation,
						  reference_predicate);
	break;
    }
    case is_instruction_loop : {
	loop l = instruction_loop(inst);
	count = look_for_references_in_range(loop_range(l), reference_transformation,
					     reference_predicate);
	count += look_for_references_in_statement(loop_body(l), reference_transformation,
						  reference_predicate);
	break;
    }
    case is_instruction_whileloop : {
	whileloop l = instruction_whileloop(inst);
	count = look_for_references_in_expression(whileloop_condition(l), reference_transformation,
						  reference_predicate);
	count += look_for_references_in_statement(whileloop_body(l), reference_transformation,
						  reference_predicate);
	break;
    }
    case is_instruction_call :
	count = look_for_references_in_call(instruction_call(inst), reference_transformation,
					    reference_predicate);
	break;
    case is_instruction_goto :
	pips_internal_error("case is_instruction_goto");
	break;
    case is_instruction_unstructured :
	/* FI: there is no reason not to do something here! */
	pips_internal_error("case is_instruction_unstructured");
	break;
    default:
      pips_internal_error("Bad instruction tag");
    }

    pips_debug(5, "end %d\n", count);

    return count;
}
static reference reference_identity(reference r)
{
    return r;
}

static bool constant_array_reference_p(reference r)
{
    /* Uses a static global variable, current_loop_index */
    list li = reference_indices(r);

    ifdebug(9) {
	pips_debug(9, "begin: index=%s reference=",
		   entity_local_name(current_loop_index));
	print_reference(r);
	putc('\n', stderr);
    }

    // The scalar references are constant with respect to any loop
    // FI: this should be upgraded to cope with C structures...
    if(!array_reference_p(r)) {
      pips_debug(9, "end: FALSE\n");
	return false;
    }

    /* FI: this is a very approximatw evaluation that assumes no
       induction variables, affine or not */
    FOREACH(EXPRESSION, i, li) {
      int count = look_for_references_in_expression(i, reference_identity,
						    current_loop_index_p);

      if(count!=0) {
	pips_debug(9, "end: count=%d FALSE\n", count);
	return false;
      }
    }

    pips_debug(9, "end: TRUE\n");
    return true;
}

static bool contiguous_array_reference_p(reference r)
{
  /* Uses a static global variable, current_loop_index */
  list li = reference_indices(r);
  expression se = expression_undefined; // subscript expression
  normalized nse = normalized_undefined;
  bool contiguous_p = false;

  /* The test could be improved by checking that the offset with
     respect to the loop index is constant within the loop nest:
     e.g. i+n**2 is a contiguous access */
  if(!ENDP(li)) {
    if(c_language_module_p(get_current_module_entity())) {
      se = EXPRESSION(CAR(gen_last(li)));
    }
    else if(fortran_language_module_p(get_current_module_entity())) {
      se = EXPRESSION(CAR(li));
    }
    nse = NORMALIZE_EXPRESSION(se);
    if(normalized_linear_p(nse)) {
      Pvecteur vse = normalized_linear(nse);
      if(vect_dimension(vse)==1
	 && vect_coeff((Variable) current_loop_index, vse)==VALUE_ONE)
	 contiguous_p = true;
    }
    // This consider 2*i as a contiguous reference... The above check
    //on VALUE_ONE might have to be relaxed
    //return expression_reference_p(first_index) &&
    //(reference_variable(expression_reference(first_subscript))
    // == current_loop_index);
  }

  return contiguous_p;
}


#if 0
static statement mark_loop_as_parallel(list lls, __attribute__((unused)) bool (*unused)(statement))
{
    statement ls = STATEMENT(CAR(lls));
    execution_tag(loop_execution(statement_loop(ls))) = is_execution_parallel;

    return ls;
}

static int current_loop_depth = -1;
static bool nth_loop_p(__attribute__((unused))statement s)
{
    /* FI: this is *wrong* but should work for a demo :-( */
    static int count = 0;

    count++;
    return count == current_loop_depth;
}
#endif







/* FI: there are at least two problems:
 *
 * - transformations like loop coalescing and full loop unrolling are
 * not considered when the iteration counts are small (although they
 * are considered in one_loop_parallelization!)
 *
 * - the cost function should be non-linear (C has conditional
 * expressions:-)
 *
 * Besides, it's bugged
 */
static statement loop_nest_parallelization(list lls)
{
  /* FI: see Corinne and Yi-Qing; in which order is this loop list?!? */
  statement s = STATEMENT(CAR(lls=gen_nreverse(lls)));
  int loop_count = gen_length(lls);
#define DIRECTION_CONTIGUOUS_COUNT 0
#define DIRECTION_PARALLEL_P 1
#define DIRECTION_REUSE_COUNT 2
#define DIRECTION_ITERATION_COUNT 3
#define CHARACTERISTICS_NUMBER 4
  int *characteristics[CHARACTERISTICS_NUMBER];
  int ln;
  int i;
  int vector_loop_number;
  int optimal_performance;

  pips_debug(8, "begin\n");

  /* gather information about each loop direction */
  // Allocate arrays to store the parallelism and locality information
  for(i=0; i < CHARACTERISTICS_NUMBER; i++)
    characteristics[i] = (int *) malloc(loop_count*(sizeof(ln)));

  // Look for contiguous references
  ln = 0;
  FOREACH(STATEMENT, ls, lls) {
    current_loop_index = loop_index(statement_loop(ls));
    *(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln) =
      look_for_references_in_statement(ls, reference_identity, contiguous_array_reference_p);
    ln++;
  }

  // Assume the the parallel code has been internalized, i.e. the
  // parallelization information has been stored into the resource "code".
  ln = 0;
  FOREACH(STATEMENT, ls, lls) {
    /* FI: the dependence graph should be used !!! */
    /* Or use internalize parallel code... */
    loop l = statement_loop(ls);
    *(characteristics[DIRECTION_PARALLEL_P]+ln) =
      execution_parallel_p(loop_execution(l));
    ln++;
  }

  ln = 0;
  FOREACH(STATEMENT, ls, lls) {
    current_loop_index = loop_index(statement_loop(ls));
    *(characteristics[DIRECTION_REUSE_COUNT]+ln) =
      look_for_references_in_statement(ls,
				       reference_identity,
				       constant_array_reference_p);
    ln++;
  }

  ln = 0;
  FOREACH(STATEMENT, ls, lls) {
    *(characteristics[DIRECTION_ITERATION_COUNT]+ln) =
      numerical_loop_iteration_count(statement_loop(ls));
    ln++;
  }

  /* Display the information obtained about each loop */
  ifdebug(8) {
    ln = 0;
    FOREACH(STATEMENT, ls, lls) {
      (void) fprintf(stderr,"loop %d index %s\t#contiguous %d\t// %s\t#reuse %d\t#range %d\n",
		     ln,
		     entity_local_name(loop_index(statement_loop(ls))),
		     *(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln),
		     bool_to_string(*(characteristics[DIRECTION_PARALLEL_P]+ln)),
		     *(characteristics[DIRECTION_REUSE_COUNT]+ln),
		     *(characteristics[DIRECTION_ITERATION_COUNT]+ln));
      ln++;
    }
  }

  /* choose and apply a transformation if at least one parallel loop
     has been found */

  /* choose as vector loop a parallel loop optimizing a tradeoff
   * between contiguity and iteration count
   */
  optimal_performance = 0; // FI: could now  be -1
  vector_loop_number = -1;
  for(ln = 0; ln < loop_count; ln++) {
    /* FI: these two constants should be provided by the target description (see target.c) */
#define REUSE_WEIGHT 8
#define CONTIGUITY_WEIGHT 4
#define ITERATION_COUNT_WEIGHT 1
    int performance =
      REUSE_WEIGHT*(*(characteristics[DIRECTION_REUSE_COUNT]+ln))
      + CONTIGUITY_WEIGHT*(*(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln));
    int iteration_count = *(characteristics[DIRECTION_ITERATION_COUNT]+ln);

    /* If the iteration count is unknown, the iteration count is
       -1, which may lead to a negative performance when all
       other coefficients are 0, which may happen with complex
       subscript expressions */
    performance += ITERATION_COUNT_WEIGHT*(iteration_count);

    // You can have no contiguity and no reuse and no known loop bound
    // and end up with performance==-1
    // pips_assert("performance is strictly greater than 0", performance > 0);

    // FI: see case nested06, the matrix multiplication
    // If the best loop is not parallel, then some kind of tiling
    // and unrolling should be performed to keep the best loop
    // inside (unrolled) while having a parallel loop around
    // Rgeister pressure should also be taken into account to
    // decide the tiling factor, which becomes the unrolling
    // factor

    // This kind of stuff can be performed at PyPS level or by PoCC

    // Look for the best vector loop
    if(*(characteristics[DIRECTION_PARALLEL_P]+ln)
       && performance > optimal_performance) {
      optimal_performance = performance;
      vector_loop_number = ln;
    }
  }
  if(vector_loop_number != -1) {

    ifdebug(8) {
      pips_debug(8, "Vector loop is loop %d with performance %d\n",
		 vector_loop_number, optimal_performance);
    }

    if(vector_loop_number != loop_count-1) {
      /* the vector direction is not the innermost loop: exchange! */
      pips_debug(8, "Interchange innermost loop with vector loop\n");
      /* lls is expected in the other order :-( */
      /* interchange_two_loops does now preserve parallel
	 loops; if parallel loops there are: do not forget to
	 intrernalize the parallelism.
      */
      s = interchange_two_loops(gen_nreverse(lls),
				vector_loop_number+1,
				loop_count);
    }
    else {
      // No vector loop has been found
      ;
    }
  }
  else {
    pips_debug(8, "No loop interchange\n");
  }

  /* mark vector loop as parallel */
  /* FI: this is very very bad code; interchange_two_loops() should preserve
   * loop execution
   */
  //current_loop_depth = loop_count;
  //look_for_nested_loop_statements(s, mark_loop_as_parallel, nth_loop_p);

  pips_debug(8, "end\n");

  return s;
}

static statement parallelization(list lls, __attribute__((unused)) bool (*loop_predicate) (statement))
{
    statement s = statement_undefined;

    // The debug level is reset by the function looking for nested loops
    debug_on("NEST_PARALLELIZATION_DEBUG_LEVEL");
    pips_debug(8, "begin\n");

    pips_assert("the loop list is not empty", gen_length(lls)!= 0);

    if(gen_length(lls) == 1) {
	s = one_loop_parallelization(STATEMENT(CAR(lls)));
    }
    else {
	s = loop_nest_parallelization(lls);
    }

    pips_debug(8, "end\n");

    debug_off();

    return s;
}

bool nest_parallelization(const char* module_name)
{
    entity module;
    statement mod_stat = statement_undefined;
    statement mod_parallel_stat = statement_undefined;

    set_current_module_entity(module_name_to_entity(module_name));
    module = get_current_module_entity();

    pips_assert("\"module\" is a module", entity_module_p(module));

    /* DBR_CODE will be changed into DBR_PARALLELIZED_CODE */
    set_current_module_statement(
		(statement) db_get_memory_resource(DBR_CODE, module_name, false) );
    mod_stat = get_current_module_statement();

    mod_parallel_stat = copy_statement(mod_stat);

    dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    /* Make sure the dependence graph points towards the code copy */
    set_ordering_to_statement(mod_parallel_stat);

    debug_on("NEST_PARALLELIZATION_DEBUG_LEVEL");

    parallel_loop_has_been_selected = false;

    look_for_nested_loop_statements(mod_parallel_stat,
				    parallelization,
				    always_select_p);

    /* Regenerate statement_ordering for the parallel
       code. module_body_reorder() checks the unique mapping
       ordering_to_statement to make sure that no inconsistency is
       introduced. */
    reset_ordering_to_statement();
    module_body_reorder(mod_parallel_stat);

    ifdebug(7)
    {
	fprintf(stderr, "\nparallelized code %s:", module_name);
	if (statement_consistent_p((statement)mod_parallel_stat))
	    fprintf(stderr," gen consistent ");
    }

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE,
			   strdup(module_name), 
			   (char*) mod_parallel_stat);

    reset_current_module_statement();
    reset_current_module_entity();
    // Already performed befpre the reordering
    //reset_ordering_to_statement();

    return true;
}


















