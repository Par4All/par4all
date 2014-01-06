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
/*
 * Fabien Coelho, May 1993
 */

#include "defines-local.h"

/* ??? this should work (but that is not the case yet),
 * with every call with no write to distributed arrays.
 *
 * these conditions are to be verifyed, by calculating
 * the proper effects of the statement.
 *
 * to be corrected later on.
 */
void generate_c1_beta(stat, lhp, lnp)
statement stat;
list *lhp, *lnp;
{
    statement staths, statns;
    expression w;
    call the_call;
    list lreftodistarray = NIL;

    (*lhp) = NIL;
    (*lnp) = NIL;

    pips_assert("call", statement_call_p(stat));

    the_call = instruction_call(statement_instruction(stat));

    /* this shouldn't be necessary
     */
    pips_assert("assignment", ENTITY_ASSIGN_P(call_function(the_call)));

    w = EXPRESSION(CAR(call_arguments(the_call)));

    pips_assert("reference",
		syntax_reference_p(expression_syntax(w)) &&
	    (!array_distributed_p
	     (reference_variable(syntax_reference(expression_syntax(w))))));

    /* references to distributed arrays:
     * w(A(I)) = B(I)
     * so the whole list is to be considered.
     */
    lreftodistarray = FindRefToDistArrayFromList(call_arguments(the_call));

    /* generation of the code
     */
    MAP(SYNTAX, s,
     {
	 list lhost;
	 list lnode;

	 pips_debug(8, "considering reference to %s\n",
	       entity_name(reference_variable(syntax_reference(s))));

	 generate_read_of_ref_for_all(s, &lhost, &lnode);

	 (*lhp) = gen_nconc((*lhp), lhost);
	 (*lnp) = gen_nconc((*lnp), lnode);
     },
	 lreftodistarray);

    /*
     * then updated statements are to be added to both host and nodes:
     */

    staths = instruction_to_statement(make_instruction
			  (is_instruction_call,
			   make_call(call_function(the_call),
				     lUpdateExpr(host_module,
						 call_arguments(the_call)))));

    statns = instruction_to_statement(make_instruction
			  (is_instruction_call,
			   make_call(call_function(the_call),
				     lUpdateExpr(node_module,
						 call_arguments(the_call)))));

    DEBUG_STAT(9, entity_name(host_module), staths);
    DEBUG_STAT(9, entity_name(node_module), statns);

    (*lhp) = gen_nconc((*lhp), CONS(STATEMENT, staths, NIL));
    (*lnp) = gen_nconc((*lnp), CONS(STATEMENT, statns, NIL));

    gen_free_list(lreftodistarray);
}


/* generate_c1_alpha
 *
 * a distributed array variable is defined
 */
void generate_c1_alpha(stat, lhp, lnp)
statement stat;
list *lhp, *lnp;
{
    statement statcomputation, statcomputecomputer, statifcomputer;
    expression writtenexpr, newreadexpr;
    call the_call;
    list
 	lupdatecomp = NIL,
 	lupdatenotcomp = NIL,
	lreadreftodistarray = NIL,
 	lstatcomp = NIL,
	lstatnotcomp = NIL,
	lstat = NIL,
 	linds = NIL;
    reference ref, newref;
    entity newarray;

    (*lhp) = NIL;
    (*lnp) = NIL;

    pips_assert("call", statement_call_p(stat));

    the_call = instruction_call(statement_instruction(stat));
    pips_assert("assignment", ENTITY_ASSIGN_P(call_function(the_call)));

    writtenexpr = EXPRESSION(CAR(call_arguments(the_call)));
    pips_assert("reference",
		syntax_reference_p(expression_syntax(writtenexpr)) &&
	   (array_distributed_p
	    (reference_variable
	     (syntax_reference(expression_syntax(writtenexpr))))));

    /* read references to distributed arrays.
     */
    lreadreftodistarray =
	FindRefToDistArray(EXPRESSION(CAR(CDR(call_arguments(the_call)))));

    /* generation of the code to get the necessary values...
     */
    MAP(SYNTAX, s,
    {
	list lnotcomp;
	list lcomp;

	pips_debug(8, "considering reference to %s\n",
	      entity_name(reference_variable(syntax_reference(s))));

	generate_read_of_ref_for_computer(s, &lcomp, &lnotcomp);

	lstatcomp = gen_nconc(lstatcomp, lcomp);
	lstatnotcomp = gen_nconc(lstatnotcomp, lnotcomp);
    },
	lreadreftodistarray);

    gen_free_list(lreadreftodistarray);

    /* then the updated statement is to be added to node:
     */
    ref = syntax_reference(expression_syntax(writtenexpr));
    newarray  =  load_new_node(reference_variable(ref));
    generate_compute_local_indices(ref, &lstat, &linds);
    newref = make_reference(newarray, linds);
    newreadexpr = UpdateExpressionForModule
	(node_module, EXPRESSION(CAR(CDR(call_arguments(the_call)))));

    statcomputation =
	make_assign_statement(reference_to_expression(newref),
			      newreadexpr);

    lstatcomp = gen_nconc(lstatcomp, lstat);
    lstatcomp = gen_nconc(lstatcomp, CONS(STATEMENT, statcomputation, NIL));

    /* Update the values of the defined distributed variable
     * if necessary...
     */
    generate_update_values_on_nodes(ref, newref, &lupdatecomp, &lupdatenotcomp);

    lstatcomp = gen_nconc(lstatcomp, lupdatecomp);
    lstatnotcomp = gen_nconc(lstatnotcomp, lupdatenotcomp);

    /* the overall statements are generated.
     */
    statcomputecomputer = st_compute_current_computer(ref);
    statifcomputer = st_make_nice_test(condition_computerp(),
				       lstatcomp,
				       lstatnotcomp);

    DEBUG_STAT(8, entity_name(node_module), statifcomputer);

    (*lnp) = CONS(STATEMENT, statcomputecomputer,
	     CONS(STATEMENT, statifcomputer,
		  NIL));
    (*lhp) = NIL;

    return;
 }

/* generate_update_values_on_nodes
 *
 * computer is doing the job
 */
void generate_update_values_on_nodes(ref, newref, lscompp, lsnotcompp)
reference ref, newref;
list *lscompp, *lsnotcompp;
{
    entity array = reference_variable(ref);

    if (replicated_p(array))
    {
	statement statif, statco, statco2, statsndtoon, statrcvfromcomp;
	list lstat, linds;

	/* remote values have to be updated
 	 */
 	statco = st_compute_current_owners(ref);

	/* the computer has to compute the owners in order to call
	 * the send to other owners function, because the owners
	 * are used to define the owners set, which was quite obvious:-)
	 *
	 * bug fixed by FC, 930629
	 */
 	statco2 = st_compute_current_owners(ref);
	statsndtoon = st_send_to_other_owners(newref);
	generate_compute_local_indices(ref, &lstat, &linds);
 	statrcvfromcomp =
	    st_receive_from_computer(make_reference(reference_variable(newref),
						    linds));
	statif = st_make_nice_test(condition_ownerp(),
				   gen_nconc(lstat,
					     CONS(STATEMENT,
						  statrcvfromcomp,
						  NIL)),
				   NIL);

	(*lscompp) = CONS(STATEMENT, statco2,
		     CONS(STATEMENT, statsndtoon,
			  NIL));
	(*lsnotcompp) = CONS(STATEMENT, statco,
			CONS(STATEMENT, statif,
			     NIL));
    }
    else
    {
	(*lscompp) = NIL;
	(*lsnotcompp) = NIL;
    }
}

/* generate_read_of_ref_for_computer
 *
 * en cours d'adaptation...
 */
void generate_read_of_ref_for_computer(s, lcompp, lnotcompp)
syntax s;
list *lcompp, *lnotcompp;
{
    statement statcompco, statcompgv, statnotcompco, statnotcompmaysend;
    reference ref = syntax_reference(s);
    entity tempn,
 	var = reference_variable(ref),
	temp = make_new_scalar_variable(get_current_module_entity(),
				    entity_basic(var));

    pips_assert("distributed array", array_distributed_p(var));

    AddEntityToHostAndNodeModules(temp);
    tempn = load_new_node(temp);

    statcompco = st_compute_current_owners(ref);
    statcompgv = st_get_value_for_computer(ref, make_reference(tempn, NIL));

    (*lcompp) = CONS(STATEMENT, statcompco,
		CONS(STATEMENT, statcompgv, NIL));

    statnotcompco = st_compute_current_owners(ref);
    statnotcompmaysend = st_send_to_computer_if_necessary(ref);


    (*lnotcompp) =
	CONS(STATEMENT, statnotcompco,
	CONS(STATEMENT, statnotcompmaysend,
	     NIL));

    /* the new variable is inserted in the expression...
     */
    syntax_reference(s) = make_reference(temp, NIL);
}

/* generate_read_of_ref_for_all
 *
 * this function organise the read of the given reference
 * for all the nodes, and for host.
 */
void generate_read_of_ref_for_all(s, lhp, lnp)
syntax s;
list *lhp, *lnp;
{
    statement stathco, stathrcv, statnco, statngv;
    reference ref = syntax_reference(s);
    entity temph, tempn,
	var = reference_variable(ref),
	temp = make_new_scalar_variable(get_current_module_entity(),
				    entity_basic(var));

    pips_assert("distributed array", array_distributed_p(var));

    AddEntityToHostAndNodeModules(temp);
    temph = load_new_host(temp);
    tempn = load_new_node(temp);

    /* the receive statement is built for host:
     *
     * COMPUTE_CURRENT_OWNERS(ref)
     * temp = RECEIVEFROMSENDER(...)
     */
    stathco = st_compute_current_owners(ref);

    /* a receive from sender is generated, however replicated the variable is
     * FC 930623 (before was a call to st_receive_from(ref, ...))
     */
    stathrcv = st_receive_from(ref, make_reference(temph, NIL));

    (*lhp) = CONS(STATEMENT, stathco,
	     CONS(STATEMENT, stathrcv,
		  NIL));

    DEBUG_STAT(9, entity_name(host_module), stathco);
    DEBUG_STAT(9, entity_name(host_module), stathrcv);

    /* the code for node is built, in order that temp has the
     * wanted value.
     *
     * COMPUTE_CURRENT_OWNERS(ref)
     * IF OWNERP(ME)
     * THEN
     *   local_ref = COMPUTE_LOCAL(ref)
     *   temp = (local_ref)
     *   IF SENDERP(ME) // this protection in case of replicated arrays.
     *   THEN
     *       SENDTOHOST(temp)
     *   SENDTONODE(ALL-OWNERS,temp)
     *   ENDIF
     * ELSE
     *   temp = RECEIVEFROM(SENDER(...))
     * ENDIF
     */
    statnco = st_compute_current_owners(ref);
    statngv = st_get_value_for_all(ref, make_reference(tempn, NIL));

    (*lnp) = CONS(STATEMENT, statnco, CONS(STATEMENT, statngv, NIL));

    DEBUG_STAT(9, entity_name(node_module), statnco);
    DEBUG_STAT(9, entity_name(node_module), statngv);

    /* the new variable is inserted in the expression...
     */
    syntax_reference(s) = make_reference(temp, NIL);
}

/* generate_compute_local_indices
 *
 * this function generate the list of statement necessary to compute
 * the local indices of the given reference. It gives back the new list
 * of indices for the reference.
 */
void generate_compute_local_indices(ref, lsp, lindsp)
reference ref;
list *lsp, *lindsp;
{
    int i;
    entity array = reference_variable(ref);
    list inds = reference_indices(ref);

    pips_assert("distributed array", array_distributed_p(array));

    (*lsp) = NIL, (*lindsp) = NIL;

    pips_debug(9, "number of dimensions of %s to compute: %d\n",
	  entity_name(array), NumberOfDimension(array));

    for(i=1; i<=NumberOfDimension(array); i++, inds = CDR(inds))
    {
	if (local_index_is_different_p(array, i))
	{
	    syntax s;
	    statement stat = st_compute_ith_local_index(array, i,
					      EXPRESSION(CAR(inds)), &s);

	    DEBUG_STAT(9, entity_name(node_module), stat);

	    (*lsp) = gen_nconc((*lsp), CONS(STATEMENT, stat, NIL));
	    (*lindsp) =
		gen_nconc((*lindsp), CONS(EXPRESSION,
			   make_expression(s, normalized_undefined), NIL));
	}
	else
	{
	    expression expr =
		UpdateExpressionForModule(node_module, EXPRESSION(CAR(inds)));

	    (*lindsp) =
		gen_nconc((*lindsp),
			  CONS(EXPRESSION, expr, NIL));
	}
    }

    pips_debug(8, "result:\n");
    ifdebug(8)
	MAP(STATEMENT, s, DEBUG_STAT(8, entity_name(node_module), s), (*lsp));
}

/* generate_get_value_locally
 *
 * put the local value of ref in the variable local.
 *
 * for every indexes, if necessary, compute
 * the local value of the indices, by calling
 * RTR support function LOCAL_INDEX(array_number, dimension)
 * then the assignment is performed.
 *
 * tempi = LOCAL_INDEX(array_number, dimension, indexi) ...
 * goal = ref_local(tempi...)
 */
void generate_get_value_locally(ref, goal, lstatp)
reference ref, goal;
list *lstatp;
{
    statement stat;
    expression expr;
    entity array = reference_variable(ref),
 	   newarray = load_new_node(array);
    list ls = NIL, newinds = NIL;

    pips_assert("distributed array", array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    expr = reference_to_expression(make_reference(newarray, newinds));
    stat = make_assign_statement(reference_to_expression(goal), expr);

    DEBUG_STAT(9, entity_name(node_module), stat);

    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, stat, NIL));
}

/* generate_send_to_computer
 *
 * sends the local value of ref to the current computer
 */
void generate_send_to_computer(ref, lstatp)
reference ref;
list *lstatp;
{
    statement statsnd;
    entity
	array = reference_variable(ref),
 	newarray = load_new_node(array);
    list ls = NIL, newinds = NIL;

    pips_assert("distributed array", array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    statsnd = st_send_to_computer(make_reference(newarray, newinds));

    DEBUG_STAT(9, entity_name(node_module), statsnd);

    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, statsnd, NIL));
}

void generate_receive_from_computer(ref, lstatp)
reference ref;
list *lstatp;
{
    statement statrcv;
    entity
	array = reference_variable(ref),
 	newarray = load_new_node(array);
    list ls = NIL, newinds = NIL;

    pips_assert("distributed array", array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    statrcv = st_receive_from_computer(make_reference(newarray, newinds));

    DEBUG_STAT(9, entity_name(node_module), statrcv);

    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, statrcv, NIL));
}

void generate_parallel_body(body, lstatp, lw, lr)
statement body;
list *lstatp, lw, lr;
{
    statement statcc, statbody;
    list lcomp = NIL, lcompr = NIL, lcompw = NIL,
        lnotcomp = NIL, lnotcompr = NIL, lnotcompw = NIL;
    syntax comp;

    pips_assert("at leat one written or read",
		(gen_length(lw)+gen_length(lr))>0);

    /* the first written or read distributed ref is chosen as the computer
     */
    comp = lw? SYNTAX(CAR(lw)): SYNTAX(CAR(lr));
    statcc = st_compute_current_computer(syntax_reference(comp));

    MAP(SYNTAX, s,
     {
	 list lco = NIL;
	 list lnotco = NIL;

	 generate_read_of_ref_for_computer(s, &lco, &lnotco);

	 lcompr = gen_nconc(lcompr, lco);
	 lnotcompr = gen_nconc(lnotcompr, lnotco);
     },
	 lr);

    MAP(SYNTAX, s,
     {
	 list lco = NIL;
	 list lnotco = NIL;
	 reference r = syntax_reference(s);
	 entity var = reference_variable(r);
	 entity temp = make_new_scalar_variable(get_current_module_entity(),
					 entity_basic(var));
	 entity tempn ;

	 AddEntityToHostAndNodeModules(temp);
	 tempn = load_new_node( temp);

	 if (comp == s)
	 {
	     /* we are sure that computer is one of the owners
	      */
	     list lstat = NIL;
	     list linds = NIL;
	     entity newarray = load_new_node(var);

	     generate_compute_local_indices(r, &lstat, &linds);
	     lstat =
		 gen_nconc
		     (lstat,
		      CONS(STATEMENT,
			   make_assign_statement
			     (reference_to_expression(make_reference(newarray,
								     linds)),
			      entity_to_expression(tempn)),
			   NIL));

	     generate_update_values_on_nodes
		 (r, make_reference(tempn, NIL), &lco, &lnotco);

	     lco = gen_nconc(lstat, lco);
	 }
	 else
	 {
	     generate_update_values_on_computer_and_nodes
		 (r, make_reference(tempn, NIL), &lco, &lnotco);
	 }

	 lcompw = gen_nconc(lcompw, lco);
	 lnotcompw = gen_nconc(lnotcompw, lnotco);

	 syntax_reference(s) = make_reference(temp, NIL);
     },
	 lw);


    pips_debug(6, "%zd statements for computer write:\n", gen_length(lcompw));

    ifdebug(8)
    {
	MAP(STATEMENT, s, DEBUG_STAT(8, entity_name(node_module), s), lcompw);
    }

    pips_debug(6, "%zd statements for not computer write:\n",
	       gen_length(lnotcompw));

    ifdebug(8)
    {
	MAP(STATEMENT, s, DEBUG_STAT(8, entity_name(node_module), s),
	    lnotcompw);
    }

    statbody = UpdateStatementForModule(node_module, body);
    DEBUG_STAT(7, entity_name(node_module), statbody);

    lcomp = gen_nconc(lcompr, CONS(STATEMENT, statbody, lcompw));
    lnotcomp = gen_nconc(lnotcompr, lnotcompw);

    (*lstatp) = CONS(STATEMENT, statcc,
		CONS(STATEMENT, st_make_nice_test(condition_computerp(),
						  lcomp,
						  lnotcomp),
		     NIL));

    ifdebug(6)
    {
	pips_debug(6, "final statement:\n");
	MAP(STATEMENT, s, DEBUG_STAT(6, entity_name(node_module),s), (*lstatp));
    }
}


/* generate_update_values_on_computer_and_nodes
 *
 * inside a loop, a variable is defined, and the values have to be updated
 * on the computer node itself, and on the other owners of the given variable.
 *
 * computer is doing the job
 */
void generate_update_values_on_computer_and_nodes(ref, val, lscompp, lsnotcompp)
reference ref, val;
list *lscompp, *lsnotcompp;
{
    entity
	array = reference_variable(ref),
	newarray = load_new_node(array);
    statement
	statif, statcompif, statco, statcompco, statcompassign,
	statsndtoO, statsndtoOO, statrcvfromcomp;
    list lstat, lstatcomp, linds, lindscomp;

    /* all values have to be updated...
     */
    statcompco = st_compute_current_owners(ref);
    generate_compute_local_indices(ref, &lstatcomp, &lindscomp);

    statcompassign =
	make_assign_statement
	    (reference_to_expression(make_reference(newarray, lindscomp)),
	     reference_to_expression(val));

    if (replicated_p(array))
    {
	statsndtoOO = st_send_to_other_owners(val);
	statsndtoO  = st_send_to_owners(val);
    }
    else
    {
	statsndtoOO = make_continue_statement(entity_undefined);
	statsndtoO  = st_send_to_owner(val);
    }
    statcompif = st_make_nice_test
	(condition_ownerp(),
	 gen_nconc(lstatcomp, CONS(STATEMENT, statcompassign,
			      CONS(STATEMENT, statsndtoOO,
				   NIL))),
	 CONS(STATEMENT, statsndtoO,
	      NIL));

    statco = st_compute_current_owners(ref);
    generate_compute_local_indices(ref, &lstat, &linds);
    statrcvfromcomp =
	st_receive_from_computer(make_reference(newarray,linds));
    statif = st_make_nice_test(condition_ownerp(),
			       gen_nconc(lstat, CONS(STATEMENT, statrcvfromcomp,
						     NIL)),
			       NIL);

    (*lscompp) = CONS(STATEMENT, statcompco,
		 CONS(STATEMENT, statcompif, NIL));
    (*lsnotcompp) = CONS(STATEMENT, statco,
		    CONS(STATEMENT, statif, NIL));
}

/* generate_update_distributed_value_from_host
 */
void generate_update_distributed_value_from_host(s, lhstatp, lnstatp)
syntax s;
list *lhstatp, *lnstatp;
{
    reference r = syntax_reference(s);
    entity array, newarray, temp, temph;
    statement statnco, stathco, stnrcv, stnif, sthsnd;
    list linds = NIL, lnstat = NIL;

    pips_assert("distrivuted reference",
		array_distributed_p(reference_variable(syntax_reference(s))));

    array    = reference_variable(r);
    newarray = load_new_node(array);

    temp     = make_new_scalar_variable(get_current_module_entity(),
				    entity_basic(array));

    AddEntityToHostAndNodeModules(temp);
    temph = load_new_host(temp);

    generate_compute_local_indices(r, &lnstat, &linds);
    stnrcv = st_receive_from_host(make_reference(newarray, linds));
    stnif = st_make_nice_test(condition_ownerp(),
			      gen_nconc(lnstat,
					CONS(STATEMENT, stnrcv, NIL)),
			      NIL);

    /* call the necessary communication function
     */

    sthsnd = (replicated_p(array) ?
	      st_send_to_owners(make_reference(temph, NIL)) :
	      st_send_to_owner(make_reference(temph, NIL))) ;

    syntax_reference(s) = make_reference(temp, NIL);

    statnco = st_compute_current_owners(r);
    stathco = st_compute_current_owners(r);

    (*lhstatp) = CONS(STATEMENT, stathco, CONS(STATEMENT, sthsnd, NIL));
    (*lnstatp) = CONS(STATEMENT, statnco, CONS(STATEMENT, stnif, NIL));

}

/* generate_update_private_value_from_host
 */
void generate_update_private_value_from_host
  (syntax s, list* lhstatp, list* lnstatp)
{
  entity
    var  = reference_variable(syntax_reference(s)),
    varn = load_new_node(var),
    varh = load_new_host(var);

  pips_assert("not distributed reference",
	      !array_distributed_p(reference_variable(syntax_reference(s))));

  (*lhstatp) = CONS(STATEMENT,
		    st_host_send_to_all_nodes(make_reference(varh, NIL)),
		    NIL);

  (*lnstatp) = CONS(STATEMENT,
		    st_receive_mcast_from_host(make_reference(varn, NIL)),
		    NIL);
}

/************************************************************** STATEMENTS */

statement st_get_value_locally_and_send(reference ref, reference goal)
{
    list ls=NIL;

    generate_get_value_locally(ref,goal,&ls);
    ls=gen_nconc(ls, CONS(STATEMENT, st_send_to_host_and_nodes(ref, goal),
			  NIL));

    return(make_block_statement(ls));
}


/* the returned expression is translated into variables
 * of the node module.
 */
statement st_compute_ith_local_index(array, i, expr, sp)
entity array;
int i;
expression expr;
syntax *sp;
{
    /* the necessity is not checked, but it could be done,
     * looking that the new declaration of the given array
     * is smaller or not on the given dimension...
     */

    entity temp = make_new_scalar_variable(node_module,
					   MakeBasic(is_basic_int));
    statement stat;
    AddEntityToCurrentModule(temp);

    (*sp)=make_syntax(is_syntax_reference, make_reference(temp, NIL));

    stat=instruction_to_statement
 	(make_assign_instruction
	 (reference_to_expression(make_reference(temp, NIL)),
	  expr_compute_local_index(array,
				   i,
				   UpdateExpressionForModule(node_module,
							     expr))));

    return(stat);
}

statement st_send_to_host_and_nodes(ref,val)
reference ref,val;
{
    if(replicated_p(reference_variable(ref)))
 	return(st_make_nice_test(condition_senderp(),
				 CONS(STATEMENT,
				      st_send_to_host_and_not_owners(val),
				      NIL),
				 NIL));
    else
	return st_send_to_host_and_all_nodes(val);
}

statement st_send_to_computer_if_necessary(reference ref)
{
    entity
 	array = reference_variable(ref),
	newarray = load_new_node(array);
    expression
	condition=((replicated_p(array))?
		   (MakeBinaryCall
		    (entity_intrinsic(AND_OPERATOR_NAME),
		     condition_senderp(),
		     condition_not_computer_in_owners())):
		   (condition_senderp()));
    list lstat = NIL, lnewinds = NIL;

    generate_compute_local_indices(ref, &lstat, &lnewinds);

    return
      st_make_nice_test(condition,
	gen_nconc(lstat,
	  CONS(STATEMENT,
	       st_send_to_computer(make_reference(newarray, lnewinds)),
	       NIL)),
			NIL);
}

statement st_get_value_for_all(reference ref, reference goal)
{
    return
	st_make_nice_test(condition_ownerp(),
	 CONS(STATEMENT, st_get_value_locally_and_send(ref,goal), NIL),
	 CONS(STATEMENT, st_receive_from(ref,goal), NIL));
}

statement st_get_value_for_computer(ref,goal)
reference ref, goal;
{
    list ltrue=NIL, lfalse=NIL;
    generate_get_value_locally(ref,goal,&ltrue);
    lfalse=CONS(STATEMENT,st_receive_from_sender(goal),NIL);
    return st_make_nice_test(condition_ownerp(),ltrue,lfalse);
}

/* if ref is replicated:
 * goal = Receive_From_Sender()
 *
 * if ref is not replicated
 * goal = Receive_Multi_Cast_From_Sender()
 */
statement st_receive_from(ref,goal)
reference ref,goal;
{
    if (replicated_p(reference_variable(ref)))
	return st_receive_from_sender(goal);
    else
	return st_receive_mcast_from_sender(goal);
}

/*   That is all
 */
