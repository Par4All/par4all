/*
 * Compiler Utilities
 *
 * Fabien Coelho, May 1993
 */

#include <stdio.h>
extern int fprintf();

#include "genC.h"
#include "hash.h"

#include "ri.h"
#include "hpf.h"

#include "misc.h"
#include "ri-util.h"
#include "control.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * UpdateExpressionForModule
 *
 * this function creates a new expression using the mapping of
 * old to new variables map.
 *
 * some of the structures generated may be shared...
 */
expression UpdateExpressionForModule(map, expr)
entity_mapping map;
expression expr;
{
    syntax
	s = expression_syntax(expr);
    expression
	e = expression_undefined;

    debug(8, "UpdateExpressionForModule", "updating...\n");

    switch(syntax_tag(s))
    {
    case is_syntax_reference:
	/* 
	 * connect to the new reference given by the mapping 
	 */
    {
	reference 
	    ref=syntax_reference(s);
	entity 
	    var=reference_variable(ref),
	    newvar=(entity) GET_ENTITY_MAPPING(map,var);

	debug(8, "UpdateExpressionForModule", "reference case\n");

	if (newvar == (entity) HASH_UNDEFINED_VALUE)
	{
	    e = reference_to_expression
		(make_reference(var, lUpdateExpr(map,reference_indices(ref))));
	}
	else
	{
	    debug(9,"UpdateExpressionForModule",
		  "updating reference %s to %s\n",
		  entity_name(var),
		  entity_name(newvar));
	    
	    e = reference_to_expression
		(make_reference(newvar, lUpdateExpr(map,reference_indices(ref))));
	}
	break;
    }
    case is_syntax_range:
    {
	range r=syntax_range(s);
	
	debug(8, "UpdateExpressionForModule", "range case\n");
	
	e=make_expression
	    (make_syntax(is_syntax_range,
			 make_range(UpdateExpressionForModule(map,range_lower(r)),
				    UpdateExpressionForModule(map,range_upper(r)),
				    UpdateExpressionForModule(map,range_increment(r)))),
	    normalized_undefined); 
	break;
    }
    case is_syntax_call:
    {
	call c=syntax_call(s);
	
	debug(8, "UpdateExpressionForModule", "call to %s case\n",
	      entity_name(call_function(c)));

	e=make_expression
	    (make_syntax(is_syntax_call,
			 make_call(call_function(c),
				   lUpdateExpr(map,call_arguments(c)))),
	    normalized_undefined); 
	break;
    }
    default:
	pips_error("UpdateExpressionForModule","unexpected syntax tag\n");
	break;
    }
    debug(8, "UpdateExpressionForModule", "end of update.\n");
    return(e);
}

/*
 * lUpdateExpr
 */
list lUpdateExpr(map,lexpr)
entity_mapping map;
list lexpr;
{
    list 
	l=NIL;

    debug(8, "lUpdateExpr", "updating %d expressions\n", gen_length(lexpr));

    MAPL(ce,
     {
	 expression
	     etmp = UpdateExpressionForModule(map, EXPRESSION(CAR(ce)));

	 l = gen_nconc(l, CONS(EXPRESSION, etmp, NIL));
     },
	 lexpr);

    debug(8, "lUpdateExpr", "end of update\n");

    return(l);
}

/*
 * lNewVariableForModule
 */
list lNewVariableForModule(map,le)
entity_mapping map;
list le;
{
    return((ENDP(le) ?
	    (NIL) :
	    CONS(ENTITY,
		 NewVariableForModule(map,ENTITY(CAR(le))),
		 lNewVariableForModule(map,CDR(le)))));
}

/*
 * NewVariableForModule
 */
entity NewVariableForModule(map,e)
entity_mapping map;
entity e;
{
    return((entity) GET_ENTITY_MAPPING(map,e));
}

/*
 * my_empty_statement_p 
 */
bool my_empty_statement_p(stat)
statement stat;
{
    return((stat==statement_undefined) ||
	   (stat==NULL) ||
	   (statement_continue_p(stat)) ||
	   ((statement_block_p(stat)) && 
	    (my_empty_statement_list_p
	      (instruction_block(statement_instruction(stat))))) ||
	   (empty_statement_p(stat)));
}

/*
 * bool my_empty_statement_list_p(l)
 */
bool my_empty_statement_list_p(l)
list l;
{
    return(ENDP(l) ?
	   TRUE :
	   my_empty_statement_p(STATEMENT(CAR(l))) && 
	   my_empty_statement_list_p(CDR(l)));
}

/*
 * update_control_lists
 */
void update_control_lists(c,map)
control c;
control_mapping map;
{
    control
	cprime = (control) GET_CONTROL_MAPPING(map, c);

    control_predecessors(cprime) = updated_control_list(control_predecessors(c),map);
    control_successors(cprime) = updated_control_list(control_successors(c),map);
}

/*
 * updated_control_list
 */
list updated_control_list(lc,map)
list lc;
control_mapping map;
{
    return((ENDP(lc))?
	   (NULL):
	   (CONS(CONTROL,
		 (control) GET_CONTROL_MAPPING(map, CONTROL(CAR(lc))),
		 updated_control_list(CDR(lc),map))));
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
	    blocks;

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
		l = CONS(STATEMENT, mere_statement(inst), NULL);
		statement_instruction(stat) = 
		    make_instruction(is_instruction_call,
				     make_call(entity_intrinsic(CONTINUE_FUNCTION_NAME),
					       NULL));
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
	    blocks;

	CONTROL_MAP(c, 
		{
		    l = gen_nconc(FindDefinitionsOf(control_statement(c), lsyn), l);
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
 * UpdateStatementForModule
 */
statement UpdateStatementForModule(map, stat)
entity_mapping map;
statement stat;
{
    statement
	updatedstat;
    instruction 
	inst = statement_instruction(stat);

    debug(7, "UpdateStatementForModule", "updating...\n");

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	list
	    lstat = NIL;
	
	debug(8, "UpdateStatementForModule", "block\n");

	MAPL(cs,
	 {
	     statement
		 stmp = UpdateStatementForModule(map, STATEMENT(CAR(cs)));

	     lstat = 
		 gen_nconc(lstat, CONS(STATEMENT, stmp, NULL));
	 },
	     instruction_block(inst));

	updatedstat = MakeStatementLike(stat, is_instruction_block, nodegotos);
	instruction_block(statement_instruction(updatedstat)) = lstat;
	break;
    }
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	debug(8, "UpdateStatementForModule", "test\n");

	updatedstat = MakeStatementLike(stat, is_instruction_test, nodegotos);
	instruction_test(statement_instruction(updatedstat)) = 
	    make_test(UpdateExpressionForModule(map, test_condition(t)),
		      UpdateStatementForModule(map, test_true(t)),
		      UpdateStatementForModule(map, test_false(t)));
	break;
    }
    case is_instruction_loop:
    {
	loop
	    l = instruction_loop(inst);
	range
	    r = loop_range(l);
	entity
	    nindex = NewVariableForModule(oldtonewnodevar,loop_index(l));

	debug(8, "UpdateStatementForModule", "loop\n");

	updatedstat = MakeStatementLike(stat, is_instruction_loop, nodegotos);
	instruction_loop(statement_instruction(updatedstat)) = 
	    make_loop(nindex,
		      make_range(UpdateExpressionForModule(map, range_lower(r)),
				 UpdateExpressionForModule(map, range_upper(r)),
				 UpdateExpressionForModule(map, range_increment(r))),
		      UpdateStatementForModule(map, loop_body(l)),
		      loop_label(l),
		      make_execution(is_execution_sequential,UU),
		      NULL);
	break;
    }
    case is_instruction_goto:
    {
	debug(8, "UpdateStatementForModule", "goto\n");

	updatedstat = MakeStatementLike(stat, is_instruction_goto, nodegotos);
	instruction_goto(statement_instruction(updatedstat)) = 
	    instruction_goto(inst);

	break;
    }
    case is_instruction_call:
    {
	call
	    c = instruction_call(inst);

	debug(8, "UpdateStatementForModule", 
	      "call to %s\n", 
	      entity_name(call_function(c)));

	updatedstat = MakeStatementLike(stat, is_instruction_call, nodegotos);
	instruction_call(statement_instruction(updatedstat)) = 
	    make_call(call_function(c), lUpdateExpr(map, call_arguments(c)));

	break;
    }
    case is_instruction_unstructured:
    {
	control_mapping 
	    ctrmap = MAKE_CONTROL_MAPPING();
	unstructured 
	    u=instruction_unstructured(inst);
	control 
	    ct = unstructured_control(u),
	    ce = unstructured_exit(u);
	list 
	    blocks = NIL;

	debug(8, "UpdateStatementForModule", "unstructured\n");

	CONTROL_MAP(c,
		{
		    statement
			statc = control_statement(c);
		    control
			ctr;

		    ctr = make_control(UpdateStatementForModule(map, statc),
				       NULL,
				       NULL);
		    SET_CONTROL_MAPPING(ctrmap, c, ctr);
		},
		    ct,
		    blocks);

	MAPL(cc,
	 {
	     control
		 c = CONTROL(CAR(cc));

	     update_control_lists(c, ctrmap);
	 },
	     blocks);

	updatedstat = MakeStatementLike(stat,is_instruction_unstructured,nodegotos);
	statement_instruction(instruction_unstructured(updatedstat)) =
	    make_unstructured((control) GET_CONTROL_MAPPING(ctrmap, ct),
			      (control) GET_CONTROL_MAPPING(ctrmap, ce));

	gen_free_list(blocks);
	FREE_CONTROL_MAPPING(ctrmap);
	break;
    }
    default:
	pips_error("UpdateStatementForModule","unexpected instruction tag\n");
	break;
    }
    
    debug(7, "UpdateStatementForModule", "end of update\n");
    return(updatedstat);
}

/*
 * atomic_accesses_only_p
 * 
 * checks that only atomic accesses to distributed variables are made
 * inside a parallel loop nest, for every iterations.
 *
 * partially implemented, and the conclusions may be false...
 */
bool atomic_accesses_only_p(stat)
statement stat;
{
    bool
	result;
    list
	lloop = NULL;

    user_warning("atomic_accesses_only_p", 
		 "only partially implemented\n");

    result = (!sequential_loop_in_statement_p
	      (perfectly_nested_parallel_loop_to_body(stat, &lloop)));

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
 * some little functions needed.
 */
bool entity_template_p(e)
entity e;
{
    return(gen_find_eq((chunk *) e, templates)==e);
}

bool entity_processor_p(e)
entity e;
{
    return(gen_find_eq((chunk *) e, processors)==e);
}
