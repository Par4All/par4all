 /* loop nest parallelization */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
/* #include <values.h> */
#include <limits.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
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
static int current_loop_depth = -1;

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


statement loop_preserve(statement s, int c)
{
    debug(9, "loop_preserve", "begin\n");

    debug(9, "loop_preserve", "end\n");

    return s;
}
    
statement loop_vectorize(statement s, int c)
{
    loop l = statement_loop(s);

    debug(9, "loop_vectorize", "begin\n");

    execution_tag(loop_execution(l)) = is_execution_parallel;

    debug(9, "loop_vectorize", "end\n");

    return s;
}

statement tuned_loop_parallelize(statement s, int c)
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
	parallel_loop_has_been_selected = TRUE;
    }

    debug(9, "tuned_loop_parallelize", "end\n");

    return s;
}

statement tuned_loop_unroll(statement s, int c)
{
    loop il = instruction_loop(statement_instruction(s));
    range lr = loop_range(il);
    expression lb = range_lower(lr),
               ub = range_upper(lr),
               inc = range_increment(lr);
    int lbval, ubval, incval;

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

bool current_loop_index_p(reference r)
{
    return reference_variable(r) == current_loop_index;
}

statement tuned_loop_strip_mine(statement s)
{
    statement inner_loop = statement_undefined;

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
    inner_loop = loop_body(statement_loop(s));
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


static bool always_select_p(loop l)
{
    return TRUE;
}

bool nest_parallelization(string module_name)
{
    entity module;
    statement mod_stat = statement_undefined;
    statement mod_parallel_stat = statement_undefined;

    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    pips_assert("loop_interchange", entity_module_p(module));

    /* DBR_CODE will be changed into DBR_PARALLELIZED_CODE */
    set_current_module_statement(
		(statement) db_get_memory_resource(DBR_CODE, module_name, FALSE) );
    mod_stat = get_current_module_statement();

    mod_parallel_stat = copy_statement(mod_stat);

    dg = (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);

    /* Make sure the dependence graph points towards the code copy */
    if(ordering_to_statement_initialized_p())
	reset_ordering_to_statement();
    initialize_ordering_to_statement(mod_parallel_stat);

    debug_on("NEST_PARALLELIZATION_DEBUG_LEVEL");

    parallel_loop_has_been_selected = FALSE;

    look_for_nested_loop_statements(mod_parallel_stat, parallelization,
				    always_select_p);

    /* Regenerate statement_ordering for the parallel code */
    reset_ordering_to_statement();
    module_body_reorder(mod_parallel_stat);

    ifdebug(7)
    {
	fprintf(stderr, "\nparallelized code %s:", module_name);
	if (gen_consistent_p((statement)mod_parallel_stat))
	    fprintf(stderr," gen consistent ");
    }

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE,
			   strdup(module_name), 
			   (char*) mod_parallel_stat);

    reset_current_module_statement();
    reset_current_module_entity();
    reset_current_module_statement();

    return TRUE;
}

statement parallelization(list lls, bool (*loop_predicate) (/* ??? */))
{
    statement s = statement_undefined;

    debug(9,"parallelization", "begin\n");

    pips_assert("paralellization", gen_length(lls)!= 0);

    if(gen_length(lls) == 1) {
	s = one_loop_parallelization(STATEMENT(CAR(lls)));
    }
    else {
	s = loop_nest_parallelization(lls);
    }

    debug(9,"parallelization", "end\n");

    return s;
}

statement one_loop_parallelization(statement s)
{
    statement new_s = s;
    int c;
    int kind;
    int size;

    debug(9,"one_loop_parallelization", "begin - input loop\n");
    if(get_debug_level()>=9) {
	print_text(stderr,text_statement(entity_undefined,0,s));
	pips_assert("one_loop_parallelization", gen_consistent_p(s));
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
	pips_error("one_loop_parallelization", 
		   "cannot find a transformation strategy"
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
	debug(9,"one_loop_parallelization", "output loop\n");
	print_text(stderr,text_statement(entity_undefined,0,s));
	pips_assert("loop_unroll", gen_consistent_p(s));
	debug(9,"one_loop_parallelization", "end\n");
    }

    return new_s;
}


reference reference_identity(reference r)
{
    return r;
}

bool constant_array_reference_p(reference r)
{
    /* Uses a static global variable, current_loop_index */
    list li = reference_indices(r);

    ifdebug(9) {
	debug(9, "constant_array_reference_p", "begin: index=%s reference=", 
	      entity_local_name(current_loop_index));
	print_reference(r);
	putc('\n', stderr);
    }

    if(!array_reference_p(r)) {
	debug(9, "constant_array_reference_p", "end: FALSE\n");
	return FALSE;
    }

    /* FI: this is a very approximate evaluation that assumes no induction variables */
    MAPL(ci, {
	expression i = EXPRESSION(CAR(ci));
	int count = look_for_references_in_expression(i, reference_identity, current_loop_index_p);

	if(count!=0) {
	    debug(9, "constant_array_reference_p", "end: count=%d FALSE\n", count);
	    return FALSE;
	}
    }, li);

    debug(9, "constant_array_reference_p", "end: TRUE\n");
    return TRUE;
}


/* FI: there are at least two problems:
 *
 * - transformations like loop coalescing and full loop unrolling are not considered when
 * the iteration counts are small (although they are considered in one_loop_parallelization!)
 *
 * - the cost function should be non-linear (C has conditional expressions:-)
 *
 * Besides, it's bugged
 */
statement loop_nest_parallelization(list lls)
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

    debug(9, "loop_nest_parallelization", "begin\n");

    /* gather information about each loop direction */
    for(i=0; i < CHARACTERISTICS_NUMBER; i++)
	characteristics[i] = (int *) malloc(loop_count*(sizeof(ln)));

    ln = 0;
    MAPL(cls, {
	statement ls = STATEMENT(CAR(cls));

	current_loop_index = loop_index(statement_loop(ls));
	*(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln) = 
	    look_for_references_in_statement(ls, reference_identity, contiguous_array_reference_p);
	ln++;
    }, lls);

    ln = 0;
    MAPL(cls, {
	/* FI: the dependence graph should be used !!! */
	*(characteristics[DIRECTION_PARALLEL_P]+ln) = TRUE;
	ln++;
    }, lls);

    ln = 0;
    MAPL(cls, {
	statement ls = STATEMENT(CAR(cls));

	current_loop_index = loop_index(statement_loop(ls));
	*(characteristics[DIRECTION_REUSE_COUNT]+ln) = 
	    look_for_references_in_statement(ls, reference_identity, constant_array_reference_p);
	ln++;
    }, lls);

    ln = 0;
    MAPL(cls, {
	statement ls = STATEMENT(CAR(cls));

	*(characteristics[DIRECTION_ITERATION_COUNT]+ln) = 
	    numerical_loop_iteration_count(statement_loop(ls));
	ln++;
    }, lls);

    ifdebug(9) {
	ln = 0;
	MAPL(cls, {
	    statement ls = STATEMENT(CAR(cls));

	    (void) fprintf(stderr,"index %s\t#contiguous %d\t// %s\t#reuse %d\t#range %d\n",
			   entity_local_name(loop_index(statement_loop(ls))),
			   *(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln),
			   bool_to_string(*(characteristics[DIRECTION_PARALLEL_P]+ln)),
			   *(characteristics[DIRECTION_REUSE_COUNT]+ln),
			   *(characteristics[DIRECTION_ITERATION_COUNT]+ln));
	    ln++;
	}, lls);
    }

    /* choose and apply a transformation if at least one parallel loop has been found */

    /* choose as vector loop a parallel loop optimizing a tradeoff between contiguity
     * and iteration count
     */
    optimal_performance = 0;
    vector_loop_number = -1;
    for(ln = 0; ln < loop_count; ln++) {
	/* FI: these two constants should be provided by the target description (see target.c) */
#define CONTIGUITY_WEIGHT 4
#define ITERATION_COUNT_WEIGHT 1
	int performance = CONTIGUITY_WEIGHT*(*(characteristics[DIRECTION_CONTIGUOUS_COUNT]+ln))
	    + ITERATION_COUNT_WEIGHT*(*(characteristics[DIRECTION_ITERATION_COUNT]+ln));

	pips_assert("loop_nest_parallelization", performance > 0);
	if(*(characteristics[DIRECTION_PARALLEL_P]+ln) == TRUE
	   && performance > optimal_performance) {
	    optimal_performance = performance;
	    vector_loop_number = ln;
	}
    }
    pips_assert("loop_nest_parallelization", vector_loop_number != -1);
    ifdebug(9) {
	debug(9, "loop_nest_parallelization", "Vector loop is loop %d\n",
	      vector_loop_number);
    }

    if(vector_loop_number != loop_count-1) {
	/* the vector direction is not the innermost loop: exchange! */
	debug(9, "loop_nest_parallelization", "Interchange innermost loop with vector loop\n");
	/* lls is expected in the other order :-( */
	/* interchange_two_loops does not preserve parallel loops; they are all generated
	 * as sequential loops (FI, 18 January 1993) 
	 */
	s = interchange_two_loops(gen_nreverse(lls), vector_loop_number+1, loop_count);
    }
    else {
	debug(9, "loop_nest_parallelization", "No loop interchange\n");
    }

    /* mark vector loop as parallel */
    /* FI: this is very very bad code; interchange_two_loops() should preserve
     * loop execution
     */
    current_loop_depth = loop_count;
    look_for_nested_loop_statements(s, mark_loop_as_parallel, nth_loop_p);

    debug(9, "loop_nest_parallelization", "end\n");

    return s;
}

statement mark_loop_as_parallel(list lls)
{
    statement ls = STATEMENT(CAR(lls));
    execution_tag(loop_execution(statement_loop(ls))) = is_execution_parallel;

    return ls;
}

bool nth_loop_p(statement ls)
{
    /* FI: this is *wrong* but should work for a demo :-( */
    static int count = 0;

    count++;
    return count == current_loop_depth;
}

int numerical_loop_iteration_count(loop l)
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

Pvecteur estimate_loop_iteration_count(loop l)
{
    return estimate_range_count(loop_range(l));
}

Pvecteur estimate_range_count(range r)
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

bool contiguous_array_reference_p(reference r)
{
    /* Uses a static global variable, current_loop_index */
    list li = reference_indices(r);
    expression first_index = expression_undefined;

    if(!ENDP(li)) {
	first_index = EXPRESSION(CAR(li));
	return expression_reference_p(first_index) &&
	    reference_variable(expression_reference(first_index)) == current_loop_index;
    }

    return FALSE;
}




bool carried_dependence_p(statement s)
{
    return FALSE;
}

int look_for_references_in_statement(statement s, statement (*reference_transformation) (/* ??? */), bool (*reference_predicate) (/* ??? */))
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
    case is_instruction_call :
	count = look_for_references_in_call(instruction_call(inst), reference_transformation,
					    reference_predicate);
	break;
    case is_instruction_goto :
	pips_error("look_for_references_in_statement", "case is_instruction_goto");
	break;
    case is_instruction_unstructured :
	/* FI: there is no reason not to do something here! */
	pips_error("look_for_references_in_statement", 
		   "case is_instruction_unstructured");
	break;
	default : 
	pips_error("look_for_references_in_statement", 
		   "Bad instruction tag");
    }

    debug(5, "look_for_references_in_statement", "end %d\n", count);

    return count;
}

int look_for_references_in_expression(expression e, statement (*reference_transformation) (/* ??? */), bool (*reference_predicate) (/* ??? */))
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
	pips_error("look_for_references_in_expression", "unknown tag: %d\n", 
		   (int) syntax_tag(expression_syntax(e)));
    }

    debug(5, "look_for_references_in_expression", "end %d\n", count);

    return count;
}

int look_for_references_in_range(range r, statement (*reference_transformation) (/* ??? */), bool (*reference_predicate) (/* ??? */))
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

int look_for_references_in_call(call c, statement (*reference_transformation) (/* ??? */), bool (*reference_predicate) (/* ??? */))
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
	pips_error("CallReplaceReference", 
		   "case is_value_symbolic: replacement not implemented\n");
	break;
      case is_value_intrinsic:
      case is_value_unknown:
	/* We assume that it is legal to replace arguments (because it should
	   have been verified with the effects that the index is not WRITTEN).
	   */
	MAPL(a, {
	    count += look_for_references_in_expression(EXPRESSION(CAR(a)), 
						       reference_transformation,
						       reference_predicate);
	}, call_arguments(c));
	break;
      case is_value_code:
	MAPL(a, {
	    count += look_for_references_in_expression(EXPRESSION(CAR(a)), 
						       reference_transformation,
						       reference_predicate);
	}, call_arguments(c));
	break;
      default:
	pips_error("look_for_references_in_call", "unknown tag: %d\n", 
		   (int) value_tag(vin));

    }

    debug(5, "look_for_references_in_call", "end %d\n", count);

    return count;
}














