/*
 * Compiler Utilities
 *
 * Fabien Coelho, May 1993
 *
 * SCCS Stuff:
 * $RCSfile: compiler-util.c,v $ ($Date: 1995/03/27 16:25:58 $, )
 * got on %D%, %T%
 * $Id$
 *
 */

#include <stdio.h>
extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"

#include "misc.h"
#include "ri-util.h"
#include "control.h"
#include "hpfc.h"
#include "defines-local.h"
#include "properties.h"

bool hpfc_empty_statement_p(stat)
statement stat;
{
    return((stat==statement_undefined) ||
	   (stat==NULL) ||
	   (statement_continue_p(stat)) ||
	   ((statement_block_p(stat)) && 
	    (hpfc_empty_statement_list_p
	      (instruction_block(statement_instruction(stat))))) ||
	   (empty_statement_p(stat)));
}

bool hpfc_empty_statement_list_p(l)
list l;
{
    return(ENDP(l) ?
	   TRUE :
	   hpfc_empty_statement_p(STATEMENT(CAR(l))) && 
	   hpfc_empty_statement_list_p(CDR(l)));
}

void update_control_lists(c, map)
control c;
control_mapping map;
{
    control
	cprime = (control) GET_CONTROL_MAPPING(map, c);

    assert(control_predecessors(cprime)==NIL &&
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
    list
	lc_result = NIL;
    control
	current = control_undefined,
	new_c = control_undefined;

    MAPL(cc,
     {
	 current = CONTROL(CAR(cc));
	 new_c = (control) GET_CONTROL_MAPPING(map, current);

	 assert(!control_undefined_p(current) ||
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

    if (!call_assign_p(c)) return(TRUE);

    /*   else ASSIGN case
     */

    l = call_arguments(c);
    s = expression_syntax(EXPRESSION(CAR(l)));
    
    if (array_distributed_p(reference_variable(syntax_reference(s))))
	found_written = CONS(SYNTAX, s, found_written);
    
    found_read = 
	gen_nconc(FindRefToDistArray(EXPRESSION(CAR(CDR(l)))),
		  found_read);
    
    return(FALSE);
}

static bool FindRefToDistArrayInStatement_expression_filter(e)
expression e;
{
    found_read = gen_nconc(FindRefToDistArray(e), found_read);
    return(FALSE);
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
    list 
	l = NIL;

    assert(syntax_reference_p(syn));

    MAPL(ce,
     {
	 syntax
	     s = expression_syntax(EXPRESSION(CAR(ce)));

	 switch (syntax_tag(s))
	 {
	 case is_syntax_reference:
	     l = CONS(SYNTAX, s, l);
	     break;
	 case is_syntax_range:
	     pips_error("IndicesOfRef","don't konw what to do with a range\n");
	     break;
	 case is_syntax_call:
	     /*     ??? could check that the given call is a constant.
	      */
	     break;
	 default:
	     pips_error("IndicesOfRef","unexpected syntax tag\n");
	 }	 
     },
	 reference_indices(syntax_reference(syn)));

    return(l);	 
}

list AddOnceToIndicesList(l, lsyn)
list l, lsyn;
{
    MAPL(cs,
     {
	 syntax
	     s = SYNTAX(CAR(cs));

	 if (!is_in_syntax_list(reference_variable(syntax_reference(s)), lsyn))
	     lsyn = CONS(SYNTAX, s, lsyn);
     },
	 l);

    gen_free_list(l);
    return(lsyn);
}

bool is_in_syntax_list(e, l)
entity e;
list l;
{
    MAPL(cs,
     {
	 if (e==reference_variable(syntax_reference(SYNTAX(CAR(cs)))))
	     return(TRUE);
     },
	 l);

    return(FALSE);
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
    instruction
	i = statement_instruction(s);

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
		CONS(STATEMENT, make_stmt_of_instr(i), found_definitions);
	    statement_instruction(s) = 
		make_continue_instruction();
	}
}

list FindDefinitionsOf(stat, lsyn)
statement stat;
list lsyn;
{
    list result = NIL;

    assert(ENDP(syntax_list) && ENDP(found_definitions));

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
    hpfc_warning("atomic_accesses_only_p", 
		 "not  implemented, returning TRUE\n");

    return(TRUE);
}

/* indirections_inside_statement_p
 *
 * ??? this may be checked using the dependences graph, looking for 
 * edges linking two distributed variables inside the loop...
 */
bool indirections_inside_statement_p(stat)
statement stat;
{
    hpfc_warning("indirections_inside_statement_p", 
		 "not implemented yet, returning FALSE\n");
    return(FALSE);
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
static int 
    n_loops, n_levels;
static statement 
    inner_body;

static bool inst_filter(i)
instruction i;
{
    /* descend only thru blocks and parallel loops 
     */
    switch(instruction_tag(i))
    {
    case is_instruction_block:
	return(TRUE);
    case is_instruction_loop:
	return(execution_parallel_p(loop_execution(instruction_loop(i))));
    }

    return(FALSE);
}

static void inst_rewrite(i)
instruction i;
{
    switch(instruction_tag(i))
    {
    case is_instruction_block:
    {
	if (n_loops==0 && n_levels==0) /* there was no doall inside */
	    return;

	if (n_loops-n_levels!=1)
	    pips_error("inst_rewrite",
		       "block within a block encountered\n");

	n_levels++, blocks = CONS(CONSP, instruction_block(i), blocks);

	break;
    }
    case is_instruction_loop:
    {
	loop l = instruction_loop(i);

	if (n_loops!=n_levels) /* a loop was found directly as a body */
	    n_levels++, 
	    blocks = CONS(CONSP, NIL, blocks);
	
	if (n_loops==0) inner_body=loop_body(l);
	loops = CONS(LOOP, l, loops);
	n_loops++;

	break;
    }
    default: /* consistent with inst_filter */
	pips_error("inst_rewrite",
		   "unexpected instruction tag (%d)\n",
		   instruction_tag(i));
    }
}

statement parallel_loop_nest_to_body(loop_nest, pblocks, ploops)
statement loop_nest;
list *pblocks, *ploops;
{
    loops=NIL, n_loops=0;
    blocks=NIL, n_levels=0;
    inner_body=statement_undefined;

    assert(instruction_loop_p(statement_instruction(loop_nest)));

    gen_recurse(loop_nest,
		instruction_domain,
		inst_filter,
		inst_rewrite);
    
    assert(n_loops!=0 && (n_loops-n_levels==1));

    *pblocks=CONS(CONSP, NIL, blocks); /* nothing was done for the first ! */
    *ploops=loops;

    return(inner_body);
}

/*------------------------------------------------------------------------
 *
 *   CURRENT LOOPS
 *
 *   management of a list of current loops. very burk ???
 *
 */

static list 
  current_loop_list=NIL;

static void set_current_loops_rewrite(l)
loop l;
{
    current_loop_list =	CONS(ENTITY, l, current_loop_list);
}

void set_current_loops(obj)
statement obj;
{
    assert(current_loop_list==NIL);

    gen_recurse(obj,
		loop_domain,
		gen_true,
		set_current_loops_rewrite);
}

void reset_current_loops()
{
    gen_free_list(current_loop_list);
    current_loop_list=NIL;
}

bool entity_loop_index_p(e)
entity e;
{
    MAPL(cl,
     {
	 if (e == loop_index(LOOP(CAR(cl)))) return(TRUE);
     },
	 current_loop_list);

    return(FALSE);
}

range loop_index_to_range(index)
entity index;
{
    loop l;

    MAPL(cl,
     {
	 l = LOOP(CAR(cl));

	 if (loop_index(l)==index) return(loop_range(l));
     },
	 current_loop_list);
    
    return(range_undefined);
}

/* that is all
 */
