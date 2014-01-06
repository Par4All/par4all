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
/* Compiler Utilities
 *
 * Fabien Coelho, May 1993
 */

#include "defines-local.h"

/***************************************************************************/

void update_control_lists(c, map)
control c;
control_mapping map;
{
    control cprime = (control) GET_CONTROL_MAPPING(map, c);

    pips_assert("empty lists for cprime",
		control_predecessors(cprime)==NIL &&
		control_successors(cprime)==NIL);

    control_predecessors(cprime) = 
	updated_control_list(control_predecessors(c), map);
    control_successors(cprime) = 
	updated_control_list(control_successors(c), map);
}

list updated_control_list(lc, map)
list lc;
control_mapping map;
{
    list lc_result = NIL;
    control new_c = control_undefined;

    MAP(CONTROL, current,
     {
	 new_c = (control) GET_CONTROL_MAPPING(map, current);

	 pips_assert("defined",
		     !control_undefined_p(current) ||
		     !control_undefined_p(new_c));

	 lc_result = CONS(CONTROL, new_c, lc_result);
     },
	 lc);

    return(gen_nreverse(lc_result));
}

/* FindRefToDistArrayInStatement
 *
 * everything may be quite false regarding to the real effects of
 * the functions called, but it seems to me that the pips effect analysis
 * doesn't match my needs... and I have not much time to think about that...
 *
 * ??? these stupid functions assume no indirections on distributed arrays
 * ??? also that they are not written by a function other than assign
 */
static list
   found_written = NIL,
   found_read = NIL;

#define call_assign_p(c) \
  (strcmp(entity_local_name(call_function(c)), ASSIGN_OPERATOR_NAME)==0)

static bool FindRefToDistArrayInStatement_call_filter(c)
call c;
{
    list l;
    syntax s;

    if (!call_assign_p(c)) return(true);

    /*   else ASSIGN case
     */

    l = call_arguments(c);
    s = expression_syntax(EXPRESSION(CAR(l)));
    
    if (array_distributed_p(reference_variable(syntax_reference(s))))
	found_written = CONS(SYNTAX, s, found_written);
    
    found_read = 
	gen_nconc(FindRefToDistArray(EXPRESSION(CAR(CDR(l)))),
		  found_read);
    
    return(false);
}

static bool FindRefToDistArrayInStatement_expression_filter(e)
expression e;
{
    found_read = gen_nconc(FindRefToDistArray(e), found_read);
    return(false);
}

void FindRefToDistArrayInStatement(obj, lwp, lrp)
statement obj;
list *lwp, *lrp;
{
    list
	saved_r = found_read,
	saved_w = found_written;

    found_read = NIL, found_written = NIL;

    gen_multi_recurse(obj,
		      call_domain,
		      FindRefToDistArrayInStatement_call_filter,
		      gen_null,
		      expression_domain,
		      FindRefToDistArrayInStatement_expression_filter,
		      gen_null,
		      NULL);

    *lwp = found_written, *lrp = found_read,
    found_read = saved_r, found_written = saved_w;
}


/* computes the list of indices of the list of ref that are variables...
 */
list lIndicesOfRef(lsyn)
list lsyn;
{
    return((ENDP(lsyn))?
	   (NULL):
	   (AddOnceToIndicesList(IndicesOfRef(SYNTAX(CAR(lsyn))),
				 lIndicesOfRef(CDR(lsyn)))));
}

list IndicesOfRef(syn)
syntax syn;
{
    list l = NIL;

    pips_assert("reference", syntax_reference_p(syn));

    MAP(EXPRESSION, e,
     {
	 syntax s = expression_syntax(e);

	 switch (syntax_tag(s))
	 {
	 case is_syntax_reference:
	     l = CONS(SYNTAX, s, l);
	     break;
	 case is_syntax_range:
	     pips_internal_error("don't konw what to do with a range");
	     break;
	 case is_syntax_call:
	     /*     ??? could check that the given call is a constant.
	      */
	     break;
	 default:
	     pips_internal_error("unexpected syntax tag");
	 }	 
     },
	 reference_indices(syntax_reference(syn)));

    return(l);	 
}

list AddOnceToIndicesList(l, lsyn)
list l, lsyn;
{
    MAP(SYNTAX, s,
	if (!is_in_syntax_list(reference_variable(syntax_reference(s)), lsyn))
	     lsyn = CONS(SYNTAX, s, lsyn),
	l);

    gen_free_list(l);
    return(lsyn);
}

bool is_in_syntax_list(e, l)
entity e;
list l;
{
    MAP(SYNTAX, s,
	if (e==reference_variable(syntax_reference(s))) return(true),
	l);

    return(false);
}

/* ??? False!
 * The definition looked for must be an assignment call...
 */
static list
  syntax_list=NIL,
  found_definitions=NIL;

static void FindDefinitionsOf_rewrite(s)
statement s;
{
    instruction i = statement_instruction(s);

    /* ??? False! nothing is checked about the statement movement...
     */
    if (instruction_assign_p(i))
	if (is_in_syntax_list
	    (reference_variable
	     (expression_reference
	      (EXPRESSION(CAR(call_arguments(instruction_call(i)))))),
		 syntax_list))
	{
	    found_definitions = 
		CONS(STATEMENT, instruction_to_statement(i), found_definitions);
	    statement_instruction(s) = 
		make_continue_instruction();
	}
}

list FindDefinitionsOf(stat, lsyn)
statement stat;
list lsyn;
{
    list result = NIL;

    pips_assert("empty lists", ENDP(syntax_list) && ENDP(found_definitions));

    syntax_list = lsyn;

    gen_recurse(stat,
		statement_domain,
		gen_true,
		FindDefinitionsOf_rewrite);

    result = found_definitions,
    syntax_list = NIL,
    found_definitions = NIL;

    return(result);
}

/* atomic_accesses_only_p
 * 
 * checks that only atomic accesses to distributed variables are made
 * inside a parallel loop nest, for every iterations.
 *
 * ??? partially implemented, and the conclusions may be false...
 */
bool atomic_accesses_only_p(stat)
statement stat;
{
    ifdebug(1)
	hpfc_warning("not  implemented, returning TRUE\n");

    return true;
}

/* indirections_inside_statement_p
 *
 * ??? this may be checked using the dependences graph, looking for 
 * edges linking two distributed variables inside the loop...
 */
bool indirections_inside_statement_p(stat)
statement stat;
{
    ifdebug(1)
	hpfc_warning("not implemented yet, returning FALSE\n");
    return false;
}

/* ------------------------------------------------------------------
 * statement parallel_loop_nest_to_body(loop_nest, pblocks, ploops)
 * statement loop_nest;
 * list *pblocks, *ploops;
 *
 * What I want is to extract the parallel loops from loop_nest,
 * while keeping track of the structure if the loop nest is not
 * perfectly nested. Only a very simple structure is recognized.
 *
 * it returns the inner statement of the loop nest,
 * a list of pointers to the loops, and a list of pointers
 * to the blocks containing these loops if any.
 *
 * we may discuss the implementation based on static global variables...
 * but I cannot see how to do it otherwise with a gen_recurse.
 */

static list 
    loops, /* lisp of loops */
    blocks;/* list of lists, may be NIL if none */
static int n_loops, n_levels;
static statement inner_body;

static bool loop_filter(loop l)
{
    return execution_parallel_p(loop_execution(l));
}

static void sequence_rewrite(sequence s)
{
    if (n_loops==0 && n_levels==0) /* there was no doall inside */
	return;

    if (n_loops-n_levels!=1)
	pips_internal_error("block within a block encountered");

    n_levels++, blocks = CONS(LIST, sequence_statements(s), blocks);
}

static void loop_rewrite(loop l)
{
    if (n_loops!=n_levels) /* a loop was found directly as a body */
	n_levels++, blocks = CONS(LIST, NIL, blocks);

    if (n_loops==0) inner_body=loop_body(l);
    loops = CONS(LOOP, l, loops);
    n_loops++;
}

statement parallel_loop_nest_to_body(loop_nest, pblocks, ploops)
statement loop_nest;
list *pblocks, *ploops;
{
    loops=NIL, n_loops=0;
    blocks=NIL, n_levels=0;
    inner_body=statement_undefined;

    pips_assert("loop", instruction_loop_p(statement_instruction(loop_nest)));

    gen_multi_recurse(loop_nest,
		      sequence_domain, gen_true, sequence_rewrite,
		      loop_domain, loop_filter, loop_rewrite,
		      NULL);

    pips_assert("loops found", n_loops!=0 && (n_loops-n_levels==1));

    *pblocks = CONS(LIST, NIL, blocks); /* nothing was done for the first ! */
    *ploops=loops;

    return(inner_body);
}

/************************************************************ CURRENT LOOPS */

/*   management of a list of current loops. very burk ???
 */

static list current_loop_list=NIL;

static void set_current_loops_rewrite(loop l)
{
    current_loop_list =	CONS(LOOP, l, current_loop_list);
}

void set_current_loops(statement obj)
{
    pips_assert("no current loop", current_loop_list==NIL);
    gen_recurse(obj, loop_domain, gen_true, set_current_loops_rewrite);
}

void reset_current_loops()
{
    gen_free_list(current_loop_list);
    current_loop_list=NIL;
}

bool entity_loop_index_p(e)
entity e;
{
    MAP(LOOP, l, if (e == loop_index(l)) return true, current_loop_list);
    return false;
}

range loop_index_to_range(index)
entity index;
{
    MAP(LOOP, l, if (loop_index(l)==index) return(loop_range(l)),
	current_loop_list);
    return range_undefined;
}

/* that is all
 */
