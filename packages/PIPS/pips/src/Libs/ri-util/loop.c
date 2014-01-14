/* Some generic methods about loops and list of loops.

   There are many things elsewher that should be factored out into here
   (static controlize...).

*/

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
#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "properties.h"
#include "ri-util.h"

/* @defgroup loop Methods dealing with loops

   @{
*/
extern int Nbrdo;

DEFINE_CURRENT_MAPPING(enclosing_loops, list)

void clean_enclosing_loops(void)
{
    /* warning: there are shared lists...
     */
    hash_table seen = hash_table_make(hash_pointer, 0);

    STATEMENT_MAPPING_MAP(s, l,
    {
	if (l && !hash_defined_p(seen, l))
	{
	    gen_free_list((list)l);
	    hash_put(seen, l, (char*) 1);
	}
    },
	get_enclosing_loops_map());

    hash_table_free(seen);
    free_enclosing_loops_map();
}

static void rloops_mapping_of_statement();

static void rloops_mapping_of_unstructured(
    statement_mapping m,
    list loops,
    unstructured u)
{
    list blocs = NIL ;

    CONTROL_MAP(c, rloops_mapping_of_statement(m, loops, control_statement(c)),
		unstructured_control(u), blocs) ;

    gen_free_list(blocs) ;
}

static void
rloops_mapping_of_statement(statement_mapping m,
			    list loops,
			    statement s)
{
    instruction i = statement_instruction(s);

    SET_STATEMENT_MAPPING(m, s, gen_copy_seq(loops));

    switch(instruction_tag(i)) {

      case is_instruction_block:
	MAP(STATEMENT, s, rloops_mapping_of_statement(m, loops, s),
	    instruction_block(i));
	break;

      case is_instruction_loop:
      {
	  list nl = gen_nconc(gen_copy_seq(loops), CONS(STATEMENT, s, NIL));
	  Nbrdo++;
	  rloops_mapping_of_statement(m, nl, loop_body(instruction_loop(i)));
	  gen_free_list(nl);
	  break;
      }

      case is_instruction_test:
	rloops_mapping_of_statement(m, loops, test_true(instruction_test(i)));
	rloops_mapping_of_statement(m, loops, test_false(instruction_test(i)));
	break;

      case is_instruction_whileloop:
	rloops_mapping_of_statement(m, loops, whileloop_body(instruction_whileloop(i)));
	break;

      case is_instruction_call:
	break;
      case is_instruction_expression:
	break;
      case is_instruction_goto:
	pips_internal_error("Go to instruction in CODE internal representation");
	break;

      case is_instruction_unstructured: {
	  rloops_mapping_of_unstructured(m, loops,instruction_unstructured(i));
	  break ;
      }

      case is_instruction_forloop: {
	/*
	  pips_user_error("Use property FOR_TO_WHILE_LOOP_IN_CONTROLIZER or "
			  "FOR_TO_DO_LOOP_IN_CONTROLIZER to convert for loops into while loops\n");
	*/
	rloops_mapping_of_statement(m, loops, forloop_body(instruction_forloop(i)));
	break ;
      }
      default:
	pips_internal_error("unexpected tag %d", instruction_tag(i));
    }
}


statement_mapping
loops_mapping_of_statement(statement stat)
{
    statement_mapping loops_map;
    loops_map = MAKE_STATEMENT_MAPPING();
    Nbrdo = 0;
    rloops_mapping_of_statement(loops_map, NIL, stat);

    if (get_debug_level() >= 7) {
      STATEMENT_MAPPING_MAP(stat, loops, {
	  fprintf(stderr, "statement %td in loops ",
		  statement_number((statement) stat));
	  FOREACH (STATEMENT, s,loops)
	    fprintf(stderr, "%td ", statement_number(s));
	  fprintf(stderr, "\n");
	}, loops_map)
    }
    return(loops_map);
}


static bool
distributable_statement_p(statement stat, set region)
{
    instruction i = statement_instruction(stat);

    switch(instruction_tag(i))
    {
    case is_instruction_block:
	MAPL(ps, {
	    if (!distributable_statement_p(STATEMENT(CAR(ps)),
					  region)) {
		return(false);
	    }
	}, instruction_block(i));
	return(true);

    case is_instruction_loop:
	region = set_add_element(region, region, (char *) stat);
	return(distributable_statement_p(loop_body(instruction_loop(i)),
					 region));

    case is_instruction_call:
	region = set_add_element(region, region, (char *) stat);
	return(true);

    case is_instruction_whileloop:
    case is_instruction_goto:
    case is_instruction_unstructured:
    case is_instruction_test:
    case is_instruction_expression:
	return(false);
    default:
	pips_internal_error("unexpected tag %d", instruction_tag(i));
    }

    return((bool) 0); /* just to avoid a gcc warning */
}


/* this functions checks if Kennedy's algorithm can be applied on the
loop passed as argument. If yes, it returns a set containing all
statements belonging to this loop including the initial loop itself.
otherwise, it returns an undefined set.

Our version of Kennedy's algorithm can only be applied on loops
containing no test, goto or unstructured control structures. */
set distributable_loop(l)
statement l;
{
    set r;

    pips_assert("distributable_loop", statement_loop_p(l));

    r = set_make(set_pointer);

    if (distributable_statement_p(l, r)) {
	return(r);
    }

    set_free(r);
    return(set_undefined);
}


/* returns true if loop lo's index is private for this loop */
bool index_private_p(lo)
loop lo;
{
    if( lo == loop_undefined ) {
	pips_internal_error("Loop undefined");
    }

    return((entity) gen_find_eq(loop_index(lo), loop_locals(lo)) !=
	   entity_undefined);
}


/* this function returns the set of all statements belonging to the given loop
   even if the loop contains test, goto or unstructured control structures */
set region_of_loop(l)
statement l;
{
    set r;

    pips_assert("distributable_loop", statement_loop_p(l));

    r = set_make(set_pointer);
    region_of_statement(l,r);
    return(r);
}


/* Should be rewritten with a gen_recurse to deal with the recent RI...
 */
void region_of_statement(stat, region)
statement stat;
set region;
{
  instruction i = statement_instruction(stat);

  switch(instruction_tag(i)) {

  case is_instruction_block:
      MAPL(ps, {
	  region_of_statement(STATEMENT(CAR(ps)),region);
      }, instruction_block(i));
      break;

  case is_instruction_loop:{
      region = set_add_element(region, region, (char *) stat);
      region_of_statement(loop_body(instruction_loop(i)),region);
      break;
  }

  case is_instruction_call:
  case is_instruction_expression:
      region = set_add_element(region, region, (char *) stat);
      break;

  case is_instruction_goto:
      region = set_add_element(region, region, (char *) stat);
      break;

  case is_instruction_test:
      /* The next statement is added by Y.Q. 12/9.*/
      region = set_add_element(region, region, (char *) stat);
      region_of_statement(test_true(instruction_test(i)), region);
      region_of_statement(test_false(instruction_test(i)), region);
      break;

  case is_instruction_unstructured:{
      unstructured u = instruction_unstructured(i);
      cons *blocs = NIL;

      CONTROL_MAP(c, {
	  region_of_statement(control_statement(c), region);
      }, unstructured_control(u), blocs) ;

      gen_free_list(blocs) ;
      break;
  }

  default:
      pips_internal_error("unexpected tag %d", instruction_tag(i));
  }
}


/** Get the variables local or private to a loop

    The function can also remove from that list all the variables that are
    localy declared in the loop statement body and the loop index using
    the apropriate flags.

    @param obj, the loop to look at.

    @param local, set to true to remove the the variables that are localy
    declared.

    @param index, set to true to remove the loop index variable

    @return a list of entities that are private in the current * context.
*/
list loop_private_variables_as_entites (loop obj, bool local, bool index) {
  // List of entities that are private to the loop according to the previous
  // phases. For historical reasons private variables are stored in the
  // locals field of the loop.
  list result = gen_copy_seq (loop_locals(obj));

  ifdebug(9) {
    pips_debug (9, "private entites to the loop:\n");
    print_entities (result);
    fprintf (stderr, "\n");
  }

  if (local ) {
    // List of localy declared entities that are stored in loop body
    list decl_var = statement_declarations (loop_body (obj));
    ifdebug(9) {
      pips_debug (9, "localy declaed entites:\n");
      print_entities (decl_var);
      fprintf (stderr, "\n");
    }
    gen_list_and_not (&result, decl_var);
  }

  if (index ) {
    pips_debug (9, "loop_indexl to remove : %s\n", entity_name (loop_index(obj)));
    gen_remove (&result, loop_index(obj));
  }

  sort_list_of_entities(result);

  return result;
}



/************************************** SORT ALL LOCALS AFTER PRIVATIZATION */

static void loop_sort_locals(loop l)
{
    list /* of entity */ le = loop_locals(l);
    if (le) sort_list_of_entities(le);
}

void sort_all_loop_locals(statement s)
{
    gen_multi_recurse(s, loop_domain, gen_true, loop_sort_locals, NULL);
}


/* Test if a loop is parallel

   @param l is the loop to test

   @return true if the loop has a parallel execution mode
*/
bool loop_parallel_p(loop l) {
  return execution_parallel_p(loop_execution(l));
}


/* Test if a loop is sequential

   @param l is the loop to test

   @return true if the loop has a sequential execution mode
*/
bool loop_sequential_p(loop l) {
  return execution_sequential_p(loop_execution(l));
}


/* Test if a statement is a parallel loop.

   It tests the parallel status of the loop but should test extensions
   such as OpenMP pragma and so on. TODO...

   @param s is the statement that may own the loop. We need this statement
   to get the pragma for the loop.
   instruction with a loop in it.

   @return true only if the statement is a parallel loop.
*/
bool parallel_loop_statement_p(statement s) {
  if (statement_loop_p(s)) {
    instruction i = statement_instruction(s);
    loop l = instruction_loop(i);

    return loop_parallel_p(l);
  }
  return false;
}


/** Compute the depth of a parallel perfect loop-nest

    @return the depth of parallel perfect loop-nest found. If there is no
    loop here, return 0
 */
int depth_of_parallel_perfect_loop_nest(statement s) {
  // We can have blocks and declarations surrounding loops
  while(statement_block_p(s)) {
    statement prev = s;
    for(list iter=statement_block(s);!ENDP(iter);POP(iter)) {
      statement st = STATEMENT(CAR(iter));
      // We can ignore declarations... until there is an initialization !
      if(declaration_statement_p(st)) {
        FOREACH(entity,e,statement_declarations(st)) {
          if(!value_undefined_p(entity_initial(e))
              && !value_unknown_p(entity_initial(e))) {
            return 0;
          }
        }
        continue;
      } else if(gen_length(iter)!=1) {
        return 0;
      } else {
        s = st;
      }
    }
    if(s == prev) return 0;
  }

  if(parallel_loop_statement_p(s)) {
    // Get the loop
    loop l = statement_loop(s);
    // Count the current one and dig into the statement of the loop:
    return 1 + depth_of_parallel_perfect_loop_nest(loop_body(l));
  } else {
    /* No parallel loop found here */
    return 0;
  }
}

/** Compute the depth of a perfect loop-nest

    @return the depth of perfect loop-nest found. If there is no
    loop here, return 0
 */
int depth_of_perfect_loop_nest(statement s) {
  // We can have blocks and declarations surrounding loops
  if(statement_block_p(s)) {
    if(ENDP(statement_block(s))) return 0;
    for(list iter=statement_block(s);!ENDP(iter);POP(iter)) {
      statement st = STATEMENT(CAR(iter));
      if(declaration_statement_p(st))//ok, skip this
        continue;
      else if(gen_length(iter)!=1) return 0;
      else
        s = st;
    }
  }

  if(statement_loop_p(s)) {
    // Get the loop
    loop l = statement_loop(s);
    // Count the current one and dig into the statement of the loop:
    return 1 + depth_of_perfect_loop_nest(loop_body(l));
  } else {
    /* No loop found here */
    return 0;
  }
}


/** Return the inner loop in a perfect loop-nest

    @param stat is the statement to test

    @return the loop statement if we have a perfect loop nest, else statement_undefined
*/
statement get_first_inner_perfectly_nested_loop(statement stat) {
  instruction ins = statement_instruction(stat);
  tag t = instruction_tag(ins);

  switch(t) {
    case is_instruction_block: {
      list lb = instruction_block(ins);

      if(lb != NIL && (lb->cdr) != NIL && (lb->cdr)->cdr == NIL
          && (continue_statement_p(STATEMENT(CAR(lb))))) {
        return get_first_inner_perfectly_nested_loop(STATEMENT(CAR(lb->cdr)));
      } else if(lb != NIL && (lb->cdr) == NIL) {
        return get_first_inner_perfectly_nested_loop(STATEMENT(CAR(lb)));
      }
      break;
    }
    case is_instruction_loop: {
      return stat;
    }
    default:
      break;
  }

  return statement_undefined;

}


/** Test if a statement is a perfect loop-nest

    @param stat is the statement to test

    @return true if the statement is a perfect loop-nest
*/
bool
perfectly_nested_loop_p(statement stat) {
  instruction ins = statement_instruction(stat);
  tag t = instruction_tag(ins);

  switch( t ) {
  case is_instruction_block: {
    list lb = instruction_block(ins);

    if ( lb != NIL && (lb->cdr) != NIL && (lb->cdr)->cdr == NIL
	 && ( continue_statement_p(STATEMENT(CAR(lb->cdr))) ) ) {
      if ( assignment_statement_p(STATEMENT(CAR(lb))) )
	return true;
      else
	return(perfectly_nested_loop_p(STATEMENT(CAR(lb))));
    }
    else if ( lb != NIL && (lb->cdr) == NIL )
      return(perfectly_nested_loop_p(STATEMENT(CAR(lb))));
    else if ( lb != NIL ) {
      /* biased for WP65 */
      return assignment_block_p(ins);
    }
    else
      /* extreme case: empty loop nest */
      return true;
    break;
  }
  case is_instruction_loop: {
    loop lo = instruction_loop(ins);
    statement sbody = loop_body(lo);

    if ( assignment_statement_p(sbody) )
      return true;
    else
      return(perfectly_nested_loop_p(sbody));
    break;
  }
  default:
    break;
  }

  return false;
}


/* Extract the body of a perfectly nested loop body.
 */
statement
perfectly_nested_loop_to_body(statement loop_nest) {
  instruction ins = statement_instruction(loop_nest);

  switch(instruction_tag(ins)) {

  case is_instruction_call:
  case is_instruction_whileloop:
  case is_instruction_test:
    /* By hypothesis we are in a perfectly nested loop and since it is
       not a loop, we've reached the loop body: */
    return loop_nest;

  case is_instruction_block: {
    list lb = instruction_block(ins);
    if (lb == NIL)
      /* The loop body is an empty block, such as { } in C: */
      return loop_nest;
    statement first_s = STATEMENT(CAR(lb));
    instruction first_i = statement_instruction(first_s);

    if(instruction_call_p(first_i))
      return loop_nest;
    else {
      if(instruction_block_p(first_i))
	return perfectly_nested_loop_to_body(STATEMENT(CAR(instruction_block(first_i))));
      else {
	pips_assert("perfectly_nested_loop_to_body",
		    instruction_loop_p(first_i));
	return perfectly_nested_loop_to_body( first_s);
      }
    }
    break;
  }
  case is_instruction_loop: {
    /* It is another loop: dig into it to reach the loop body: */
    statement sbody = loop_body(instruction_loop(ins));
    return (perfectly_nested_loop_to_body(sbody));
    break;
  }
  default:
    pips_internal_error("illegal tag");
    break;
  }
  return(statement_undefined); /* just to avoid a warning */
}


/** Extract the loop-body of a perfect loop-nest at a given depth

    @param s is the loop-nest statement to dig into

    @param depth is the diving depth

    @return the loop-body found at the given depth
 */
statement
perfectly_nested_loop_to_body_at_depth(statement s, int depth) {
  // To have it working for depth = 0 too:
  statement body = s;
  ifdebug(2) {
    pips_debug(1, "Look at statement at depth 0:\n");
    print_statement(body);
  }
  for(int i = 0; i < depth; i++) {

    // We can have blocks and declarations surrounding loops
    while(statement_block_p(body)) {
      for(list iter=statement_block(body);!ENDP(iter);POP(iter)) {
        statement st = STATEMENT(CAR(iter));
        if(declaration_statement_p(st))//ok, skip this
          continue;
        else if(gen_length(iter)!=1) pips_internal_error("should be a perfectly nested loop\n");
        else
          body = st;
        }
    }

    pips_assert("The statement is a loop", statement_loop_p(body));
    // Dive into one loop:
    body = loop_body(statement_loop(body));

    ifdebug(2) {
      pips_debug(1, "Look at statement at depth %d:\n", i + 1);
      print_statement(body);
    }
  }
  // We can have blocks and declarations surrounding loops
  while(statement_block_p(body)&&gen_length(statement_block(body))==1) {
	body=STATEMENT(CAR(statement_block(body)));
  }

  return body;
}


/** Get the index of the loop at a given depth inside a loop-nest

    @param s is the loop-nest statement to dig into

    @param depth is the diving depth

    @return the loop-body found at the given depth
 */
entity
perfectly_nested_loop_index_at_depth(statement s, int depth) {
  statement preloop = perfectly_nested_loop_to_body_at_depth(s, depth);
  /* there may be some declarations before */
  while(statement_block_p(preloop)) {
    for(list iter= statement_block(preloop);!ENDP(iter);POP(iter))
      if(declaration_statement_p(STATEMENT(CAR(iter)))) continue;
      else { preloop = STATEMENT(CAR(iter)) ;  break ;}
  }

  return loop_index(statement_loop(preloop));
}


/*
  returns the numerical value of loop l increment expression.
  aborts if this expression is not an integral constant expression.
  modification : returns the zero value when it isn't constant
  Y.Q. 19/05/92
*/
int
loop_increment_value(loop l) {
  range r = loop_range(l);
  expression ic = range_increment(r);
  normalized ni;
  int inc;

  ni = NORMALIZE_EXPRESSION(ic);

  if (! EvalNormalized(ni, &inc)){
    /*user_error("loop_increment_value", "increment is not constant");*/
    debug(8,"loop_increment_value", "increment is not constant");
    return(0);
  }
  return(inc);
}


/** Test if a loop has a constant step loop
 */
bool constant_step_loop_p(loop l) {
  pips_debug(7, "doing\n");
  return(expression_constant_p(range_increment(loop_range(l))));
}


/** Test if a loop does have a 1-increment step
 */
bool normal_loop_p(loop l) {
  expression ri;
  entity ent;

  pips_debug(7, "doing\n");

  if (!constant_step_loop_p(l))
    // No way for a non-constant step to be a 1-constant :-)
    return(false);

  ri = range_increment(loop_range(l));
  ent = reference_variable(syntax_reference(expression_syntax(ri)));
  return strcmp(entity_local_name(ent), "1") == 0;
}


/*************************************************************** COUNT LOOPS */

/** To store the number of sequential and parallel loops */
static int nseq, npar;


static void loop_update_statistics(loop l)
{
  if (loop_parallel_p(l))
    npar++;
  else
    nseq++;
}


/**
   Compute the number of parallel and sequential loops found in a
   statement and update given variables

   @param stat is the statement to dig into

   @param pseq point to the variable to update with the number of
   sequential loops found

   @param ppr point to the variable to update with the number of parallel
   loops found
 */
void number_of_sequential_and_parallel_loops(statement stat,
					     int * pseq,
					     int * ppar) {
  nseq=0, npar=0;
  gen_recurse(stat, loop_domain, gen_true, loop_update_statistics);
  *pseq=nseq, *ppar=npar;
}


/**
   Compute the number of parallel and sequential loops found in a
   statement and output them on a stream with a message before

   @param out is the stream to send the information to

   @param msg is the message to prepend

   @param s is the statement to dig into
 */
void print_number_of_loop_statistics(FILE * out,
				     string msg,
				     statement s) {
  int seq, par;
  number_of_sequential_and_parallel_loops(s, &seq, &par);
  fprintf(out, "%s: %d seq loops, %d par loops\n", msg, seq, par);
}


/* Print out the number of sequential versus parallel loops.
 */
void print_parallelization_statistics(
    const char* module, /**< the module name */
    const char* msg,    /**< an additional message */
    statement s    /**< the module statement to consider */) {
  if (get_bool_property("PARALLELIZATION_STATISTICS"))
    {
      fprintf(stderr, "%s %s parallelization statistics", module, msg);
      print_number_of_loop_statistics(stderr, "", s);
    }
}

/* Duplicate a loop list. */
list copy_loops(list ll)
{
  list nll = gen_full_copy_list(ll);

  return nll;
}

/* This is an ad'hoc function designed for
   do_loop_unroll_with_epilogue(). Th expression and execution
   parameters are reused directly in the new loop, but the body must
   be cloned. Compared to make_loop(), this function adds the cloning
   and the wrapping into an instruction and a statement.
*/
statement make_new_loop_statement(entity i,
				  expression low,
				  expression up,
				  expression inc,
				  statement b,
				  execution e)
{
  /* Loop range is created */
  range rg = make_range(low, up, inc);

  ifdebug(9) {
    pips_assert("new range is consistent", range_consistent_p(rg));
  }

  /* Create body of the loop, with updated index */
  clone_context cc = make_clone_context(
					get_current_module_entity(),
					get_current_module_entity(),
					NIL,
					get_current_module_statement() );
  statement body = clone_statement(b, cc);
  free_clone_context(cc);

  ifdebug(9) {
    pips_assert("cloned body is consistent", statement_consistent_p(body));
    /* "gen_copy_tree returns bad statement\n"); */
  }

  entity label_entity = entity_empty_label();

  ifdebug(9) {
    pips_assert("the cloned is consistent",
		statement_consistent_p(body));
  }

  instruction inst = make_instruction(is_instruction_loop,
			  make_loop(i,
				    rg,
				    body,
				    label_entity,
				    e,
				    NIL));

  ifdebug(9) {
    pips_assert("inst is consistent",
		instruction_consistent_p(inst));
  }

  statement stmt = instruction_to_statement(inst);
  return stmt;
}

/* If statement s is a perfectly loop nest, return the corresponding
   loop list. If not, the list returned is empty. */
list statement_to_loop_statement_list(statement s)
{
  list l = NIL;
  statement cs = s;

  while(statement_loop_p(cs)) {
    l = gen_nconc(l, CONS(STATEMENT,cs, NIL));
    cs = loop_body(statement_loop(cs));
    if(statement_block_p(cs)) {
      list sl = statement_block(cs);
      if(gen_length(sl)==1)
	cs = STATEMENT(CAR(sl));
    }
  }

  return l;
}

bool range_contains_at_least_one_point_p( range r )
{
  bool return_val = false;
  expression low = range_lower(r); int i_low;
  expression up = range_upper(r); int i_up;
  expression inc = range_increment(r); int i_inc;
  if(extended_integer_constant_expression_p_to_int(low, &i_low)
     && extended_integer_constant_expression_p_to_int(up, &i_up)
     && extended_integer_constant_expression_p_to_int(inc, &i_inc) ) {
    if(i_inc >0 && i_up > i_low) {
      // Increasing case
      return_val = true;
    } else if(i_inc < 0 && i_up < i_low) {
      // Decreasing case
      return_val = true;
    }
  }
  return return_val;
}


/**
 * @brief Check if loop bound are constant and then if upper > lower
 * @return true if the loop is always executed at least once
 */
bool loop_executed_at_least_once_p( loop l )
{
  return range_contains_at_least_one_point_p(loop_range(l));
}


/* @} */
