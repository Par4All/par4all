/*
 * Compiler Utilities
 *
 * Fabien Coelho, May 1993
 *
 * SCCS Stuff:
 * $RCSfile: compiler-util.c,v $ ($Date: 1994/11/17 14:19:11 $, )
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

/*
 * hpfc_empty_statement_p 
 */
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

/*
 * bool hpfc_empty_statement_list_p(l)
 */
bool hpfc_empty_statement_list_p(l)
list l;
{
    return(ENDP(l) ?
	   TRUE :
	   hpfc_empty_statement_p(STATEMENT(CAR(l))) && 
	   hpfc_empty_statement_list_p(CDR(l)));
}

/*
 * update_control_lists
 */
void update_control_lists(c, map)
control c;
control_mapping map;
{
    control
	cprime = (control) GET_CONTROL_MAPPING(map, c);

    pips_assert("update_control_lists",
		(control_predecessors(cprime)==NIL) &&
		(control_successors(cprime)==NIL));

    control_predecessors(cprime) = 
	updated_control_list(control_predecessors(c), map);
    control_successors(cprime) = 
	updated_control_list(control_successors(c), map);
}

/*
 * updated_control_list
 */
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

	 pips_assert("updated_control_list",
		     !control_undefined_p(current) ||
		     !control_undefined_p(new_c));

	 lc_result = CONS(CONTROL, new_c, lc_result);
     },
	 lc);

    return(gen_nreverse(lc_result));
}

/*
 * FindRefToDistArrayInStatement
 *
 * everything may be quite false regarding to the real effects of
 * the functions called, but it seems to me that the pips effect analysis
 * doesn't match my needs... and I have not much time to think about that...
 */
void FindRefToDistArrayInStatement(stat, lwp, lrp)
statement stat;
list *lwp, *lrp;
{
    instruction 
	inst = statement_instruction(stat);

    (*lwp) = NULL ;
    (*lrp) = NULL ;

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
	debug(9, "FindRefToDistArrayInStatement", "block\n");

	MAPL(cs,
	 {
	     statement 
		 s = STATEMENT(CAR(cs));
	     list 
		 lw=NULL;
	     list 
		 lr=NULL;

	     FindRefToDistArrayInStatement(s, &lw, &lr);

	     (*lwp) = gen_nconc((*lwp), lw);
	     (*lrp) = gen_nconc((*lrp), lr);
	 },
	     instruction_block(inst));

	     debug(9, "FindRefToDistArrayInStatement", "end block\n");

	break;
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);
	list
	    lw = NULL,
	    lr = NULL;
	
	debug(9, "FindRefToDistArrayInStatement", "test\n");

	FindRefToDistArrayInStatement(test_true(t), &lw, &lr);
	(*lwp) = gen_nconc(lw, (*lwp));
	(*lrp) = gen_nconc(lr, (*lrp));
	
	lw = NULL;
	lr = NULL;
	FindRefToDistArrayInStatement(test_false(t), &lw, &lr);
	(*lwp) = gen_nconc(lw, (*lwp));
	(*lrp) = gen_nconc(lr, (*lrp));

	/*
	 * ??? False!
	 */
	(*lrp) = gen_nconc(FindRefToDistArray(test_condition(t)), (*lrp));

	break;
    }
    case is_instruction_loop:
    {
	loop
	    l = instruction_loop(inst);
	range
	    r = loop_range(l);
	list
	    lw = NULL,
	    lr = NULL;

	debug(9, "FindRefToDistArrayInStatement", "loop\n");

	FindRefToDistArrayInStatement(loop_body(l), &lw, &lr);
	(*lwp) = gen_nconc((*lwp), lw);
	(*lrp) = gen_nconc((*lrp), lr);

	/*
	 * ??? False!
	 */
	(*lrp) = gen_nconc(FindRefToDistArray(range_lower(r)), (*lrp));
	(*lrp) = gen_nconc(FindRefToDistArray(range_upper(r)), (*lrp));
	(*lrp) = gen_nconc(FindRefToDistArray(range_increment(r)), (*lrp));

	break;
    }
    case is_instruction_goto:
	break;
    case is_instruction_call:
    {
	call
	    c = instruction_call(inst);
	list
	    la = call_arguments(c);

	debug(9, "FindRefToDistArrayInStatement", "call\n");

	if (instruction_assign_p(inst))
	{
	    (*lwp) = FindRefToDistArray(EXPRESSION(CAR(la)));
	    (*lrp) = FindRefToDistArray(EXPRESSION(CAR(CDR(la))));
	}
	else
	{
	    /*
	     * ??? False!
	     */
	    (*lrp) = FindRefToDistArrayFromList(la);
	}
	break;
    }
    case is_instruction_unstructured:
    {
	unstructured
	    u = instruction_unstructured(inst);
	control
	    ct = unstructured_control(u);
	list
	    blocks=NIL;

	debug(9, "FindRefToDistArrayInStatement", "unstructured\n");

	CONTROL_MAP(c,
		{
		    list
			lw = NULL;
		    list 
			lr = NULL;

		    FindRefToDistArrayInStatement(control_statement(c), &lw, &lr);
		    
		    (*lwp) = gen_nconc(lw, (*lwp));
		    (*lrp) = gen_nconc(lr, (*lrp));
		},
		    ct,
		    blocks);

	gen_free_list(blocks);

	break;
    }
    default:
	pips_error("FindRefToDistArrayInStatement","unexpected instruction tag\n");
	break;
    }
}


/*
 * lIndicesOfRef
 *
 * computes the list of indices of the list of ref that are variables...
 */
list lIndicesOfRef(lsyn)
list lsyn;
{
    return((ENDP(lsyn))?
	   (NULL):
	   (AddOnceToIndicesList(IndicesOfRef(SYNTAX(CAR(lsyn))),
				 lIndicesOfRef(CDR(lsyn)))));
}

/*
 * IndicesOfRef
 */
list IndicesOfRef(syn)
syntax syn;
{
    list 
	l = NULL;

    pips_assert("IndicesOfRef",(syntax_reference_p(syn)));

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
	     /*
	      * ???
	      *
	      * could check that the given call is a constant.
	      */
	     break;
	 default:
	     pips_error("IndicesOfRef","unexpected syntax tag\n");
	 }	 
     },
	 reference_indices(syntax_reference(syn)));

    return(l);	 
}

/*
 * AddOnceToIndicesList
 */
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

/*
 * is_in_syntax_list
 */
bool is_in_syntax_list(e, l)
entity e;
list l;
{
    return((ENDP(l))?
	   (FALSE):
	   ((e==reference_variable(syntax_reference(SYNTAX(CAR(l))))) ||
	    (is_in_syntax_list(e, CDR(l)))));
}

/*
 * FindDefinitionsOf
 *
 * ??? False!
 * The definition looked for must be an assignment call...
 */
list FindDefinitionsOf(stat, lsyn)
statement stat;
list lsyn;
{
    list
	l = NULL;
    instruction
	inst = statement_instruction(stat);

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
	MAPL(cs,
	 {
	     l = gen_nconc(FindDefinitionsOf(STATEMENT(CAR(cs)), lsyn), l);
	 },
	     instruction_block(inst));
	break;
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	l = gen_nconc(FindDefinitionsOf(test_true(t), lsyn),
		      FindDefinitionsOf(test_false(t), lsyn));

	break;
    }
    case is_instruction_loop:
	l = FindDefinitionsOf(loop_body(instruction_loop(inst)), lsyn);
	break;
    case is_instruction_goto:
	break;
    case is_instruction_call:
	/*
	 * ??? False!
	 * nothing is checked about the statement movement...
	 */
	if (instruction_assign_p(inst))
	{
	    if (is_in_syntax_list
		(reference_variable
		 (expression_reference
		  (EXPRESSION(CAR(call_arguments(instruction_call(inst)))))),
		 lsyn))
	    {
		l = CONS(STATEMENT, make_stmt_of_instr(inst), NIL);
		statement_instruction(stat) = 
		    make_continue_instruction();
	    }
	}
	break;
    case is_instruction_unstructured:
    {
	unstructured
	    u = instruction_unstructured(inst);
	control
	    ct = unstructured_control(u);
	list
	    blocks=NIL;

	CONTROL_MAP(c, 
		{
		    l = gen_nconc(FindDefinitionsOf(control_statement(c), 
						    lsyn), 
				  l);
		},
		    ct, 
		    blocks);

	gen_free_list(blocks);

	break;
    }
    default:
	pips_error("FindDefinitionsOf","unexpected instruction tag\n");
	break;
    }
    
    return(l);
}

/*
 * atomic_accesses_only_p
 * 
 * checks that only atomic accesses to distributed variables are made
 * inside a parallel loop nest, for every iterations.
 *
 * ??? partially implemented, and the conclusions may be false...
 */
bool atomic_accesses_only_p(stat)
statement stat;
{
    bool
	result = TRUE;
    list
	lloop = NIL;

    user_warning("atomic_accesses_only_p", 
		 "only partially implemented\n");

/*
    result = (!sequential_loop_in_statement_p
	      (perfectly_nested_parallel_loop_to_body(stat, &lloop)));
*/

    gen_free_list(lloop);

    return(result);
}

bool sequential_loop_in_statement_p(stat)
statement stat;
{
    instruction
	inst = statement_instruction(stat);

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
	MAPL(cs,
	 {
	     if (sequential_loop_in_statement_p(STATEMENT(CAR(cs))))
		 return(TRUE);
	 },
	     instruction_block(inst));
        break;
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	return(sequential_loop_in_statement_p(test_true(t)) ||
	       sequential_loop_in_statement_p(test_false(t)));
    }
    case is_instruction_loop:
    {
	loop
	    l = instruction_loop(inst);

	if (execution_parallel_p(loop_execution(l)))
	    return(TRUE);
	else
	    return(sequential_loop_in_statement_p(loop_body(l)));
    }
    case is_instruction_goto:
        break;
    case is_instruction_call:
        break;
    case is_instruction_unstructured:
     {
	list
	    blocks = NULL;
	control
	    ct = unstructured_control(instruction_unstructured(inst));
	bool
	    sequential_loop_inside = FALSE;
	
	CONTROL_MAP(c,
		{
		    if (sequential_loop_in_statement_p(control_statement(c)))
			sequential_loop_inside = TRUE;
		},
		    ct,
		    blocks);
	gen_free_list(blocks);		
	return(sequential_loop_inside);
    }
    default:
        pips_error("","unexpected instruction tag\n");
        break;
    }

    return(FALSE);
}

/*
 * stay_inside_statement_p
 *
 * checks that no "goto" goes outside a loop nest.
 * ??? should also checks that no goto may go inside the loop nest...
 * but I don't think that is allowed, even in Fortran:-)
 */
bool stay_inside_statement_p(stat)
statement stat;
{
    list
	llabels = list_of_labels(stat),
	lgotos  = list_of_gotos(stat);
    bool
	stay_in = TRUE;

    MAPL(ce,
     {
	 if ((entity) gen_find_eq(ENTITY(CAR(ce)), llabels) == entity_undefined)
	     stay_in = FALSE;
     },
	 lgotos);
    
    gen_free_list(llabels);
    gen_free_list(lgotos);

    debug(6, "stay_inside_statement_p", "returning %d\n", stay_in);
    return(stay_in);
}

/*
 * io_inside_statement_p
 *
 * ??? should be with interprocedural informations...
 */
bool io_inside_statement_p(stat)
statement stat;
{
    instruction
	inst = statement_instruction(stat);

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
	debug(7, "io_inside_statement_p", "block beginning\n");
	MAPL(cs,
	 {
	     if (io_inside_statement_p(STATEMENT(CAR(cs))))
		 return(TRUE);
	 },
	     instruction_block(inst));
	debug(7, "io_inside_statement_p", "block end\n");
        break;
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	debug(7, "io_inside_statement_p", "test\n");

	return(io_inside_statement_p(test_true(t)) ||
	       io_inside_statement_p(test_false(t)));
    }
    case is_instruction_loop:
	debug(7, "io_inside_statement_p", 
	      "loop %s\n", entity_name(loop_index(instruction_loop(inst))));
	return(io_inside_statement_p(loop_body(instruction_loop(inst))));
    case is_instruction_goto:
	break;
    case is_instruction_call:
	/*
	 * ??? but what about calling a function that has io...
	 * there is the same problem in the expressions...
	 */
	debug(7, "io_inside_statement_p",
	      "call to %s\n", entity_name(call_function(instruction_call(inst))));
	if (IO_CALL_P(instruction_call(inst)))
	    return(TRUE);
	break;
    case is_instruction_unstructured:
    {
	list
	    blocks = NULL;
	control
	    ct = unstructured_control(instruction_unstructured(inst));
	bool
	    io_inside = FALSE;
	
	debug(7, "io_inside_statement_p", "unstructured\n");
	
	CONTROL_MAP(c,
		{
		    if (io_inside_statement_p(control_statement(c)))
			io_inside = TRUE;
		},
		    ct,
		    blocks);
	gen_free_list(blocks);		
	return(io_inside);
    }
    default:
        pips_error("io_inside_statement_p","unexpected instruction tag\n");
        break;
    }

    return(FALSE); /* just to avoid a gcc warning */
}

/*
 * indirections_inside_statement_p
 *
 * ??? this may be checked using the dependences graph, looking for 
 * edges linking two distributed variables inside the loop...
 */
bool indirections_inside_statement_p(stat)
statement stat;
{
    user_warning("indirections_inside_statement_p", 
		 "not implemented yet, returning FALSE\n");
    return(FALSE);
}

/*
 * list_of_labels
 *
 * list of labels in a given statement
 */
list list_of_labels(stat)
statement stat;
{
    entity
	label = statement_label(stat);
    instruction
	inst = statement_instruction(stat);
    list 
	ll = NULL;

    if (!entity_empty_label_p(label))
	ll = CONS(ENTITY, label, ll);

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	MAPL(ce,
	 {
	     ll = gen_nconc(list_of_labels(STATEMENT(CAR(ce))), ll);
	 },
	     instruction_block(inst));
        break;
    }
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	ll = gen_nconc(list_of_labels(test_false(t)),
		       gen_nconc(list_of_labels(test_true(t)), ll));
        break;
    }
    case is_instruction_loop:
	ll = gen_nconc(list_of_labels(loop_body(instruction_loop(inst))), ll);
        break;
    case is_instruction_goto:
        break;
    case is_instruction_call:
        break;
    case is_instruction_unstructured:
    {
	list
	    blocks = NULL;
	control
	    ct = unstructured_control(instruction_unstructured(inst));
	
	CONTROL_MAP(c,
		{
		    ll = gen_nconc(list_of_labels(control_statement(c)), ll);
		},
		    ct,
		    blocks);
	gen_free_list(blocks);		    

        break;
    }
    default:
        pips_error("list_of_labels","unexpected instruction tag\n");
        break;
    }

    return(ll);
}

/*
 * list_of_gotos
 *
 * list og gotos in a given statement
 */
list list_of_gotos(stat)
statement stat;
{
    instruction
	inst = statement_instruction(stat);
    list 
	ll = NULL;

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	MAPL(ce,
	 {
	     ll = gen_nconc(list_of_gotos(STATEMENT(CAR(ce))), ll);
	 },
	     instruction_block(inst));
        break;
    }
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	ll = gen_nconc(list_of_gotos(test_false(t)),
		       gen_nconc(list_of_gotos(test_true(t)), ll));
        break;
    }
    case is_instruction_loop:
	ll = gen_nconc(list_of_gotos(loop_body(instruction_loop(inst))), ll);
        break;
    case is_instruction_goto:
	ll = CONS(ENTITY, statement_label(instruction_goto(inst)), ll);
        break;
    case is_instruction_call:
        break;
    case is_instruction_unstructured:
    {
	list
	    blocks = NULL;
	control
	    ct = unstructured_control(instruction_unstructured(inst));
	
	CONTROL_MAP(c,
		{
		    ll = gen_nconc(list_of_gotos(control_statement(c)), ll);
		},
		    ct,
		    blocks);
	gen_free_list(blocks);		    

        break;
    }
    default:
        pips_error("list_of_gotos","unexpected instruction tag\n");
        break;
    }

    return(ll);
}

/*
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

    pips_assert("parallel_loop_nest_to_body",
		instruction_loop_p(statement_instruction(loop_nest)));

    gen_recurse(loop_nest,
		instruction_domain,
		inst_filter,
		inst_rewrite);
    
    pips_assert("parallel_loop_nest_to_body",
		n_loops!=0 && (n_loops-n_levels==1));

    *pblocks=CONS(CONSP, NIL, blocks); /* nothing was done for the first ! */
    *ploops=loops;

    return(inner_body);
}

/*
 * that is all
 */
