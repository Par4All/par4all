/* 
 * generate.c 
 * 
 * code generation 
 * 
 * Fabien Coelho, May 1993
 *
 */

/*
 * included files, from C libraries, newgen and pips libraries.
 */

#include <stdio.h> 
#include <string.h> 

extern fprintf();

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

#include "misc.h" 
#include "ri-util.h" 
#include "bootstrap.h" 
#include "hpfc.h"
#include "defines-local.h"

extern instruction MakeAssignInst(syntax l, expression r);
extern entity CreateIntrinsic(string name); /* in syntax.h */


/*
 * ??? this should work (but that is not the case yet),
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
    statement
 	staths,
 	statns;
    expression 
	w; 
    call 
	the_call; 
    list
	lreftodistarray = NIL;
    
    (*lhp) = NIL;
    (*lnp) = NIL;
    
    /*
     * assertions...  
     */
    
    pips_assert("generate_c1_beta",
		(instruction_call_p(statement_instruction(stat))));

    the_call = instruction_call(statement_instruction(stat));

    /*
     * this shouldn't be necessary
     */
    pips_assert("generate_c1_beta",
		ENTITY_ASSIGN_P(call_function(the_call)));
    
    w = EXPRESSION(CAR(call_arguments(the_call)));
    
    pips_assert("generate_c1_beta",
		(syntax_reference_p(expression_syntax(w)) &&
		 (!array_distributed_p
		  (reference_variable(syntax_reference(expression_syntax(w)))))));

    /*
     * references to distributed arrays:
     * w(A(I)) = B(I)
     * so the whole list is to be considered. 
     */
    lreftodistarray = FindRefToDistArrayFromList(call_arguments(the_call));

    /*
     * generation of the code 
     */ 
    MAPL(cs,
     { 	 
	 list lhost; 	 
	 list lnode;
 	 syntax s = SYNTAX(CAR(cs));
 	 	
	 debug(8, "generate_c1_beta", "considering reference to %s\n", 	
	       entity_name(reference_variable(syntax_reference(s))));

	 generate_read_of_ref_for_all(s, &lhost, &lnode);

	 (*lhp) = gen_nconc((*lhp), lhost); 	
	 (*lnp) = gen_nconc((*lnp), lnode);
     }, 
	 lreftodistarray);

    /*
     * then updated statements are to be added to both host and nodes:
     */

    staths = mere_statement(make_instruction 			
			  (is_instruction_call, 			
			   make_call(call_function(the_call), 				
				     lUpdateExpr(oldtonewhostvar,
						 call_arguments(the_call)))));

    statns = mere_statement(make_instruction 			
			  (is_instruction_call, 			
			   make_call(call_function(the_call), 				
				     lUpdateExpr(oldtonewnodevar,
						 call_arguments(the_call)))));

    IFDBPRINT(9, "generate_c1_beta", hostmodule, staths);
    IFDBPRINT(9, "generate_c1_beta", nodemodule, statns);

    (*lhp) = gen_nconc((*lhp), CONS(STATEMENT, staths, NIL));
    (*lnp) = gen_nconc((*lnp), CONS(STATEMENT, statns, NIL));

    gen_free_list(lreftodistarray); 
}


/*
 * generate_c1_alpha
 *
 * a distributed array variable is defined
 */
void generate_c1_alpha(stat, lhp, lnp)
statement stat;
list *lhp, *lnp;
{
    statement
 	statcomputation = statement_undefined,
	statcomputecomputer = statement_undefined,
 	statifcomputer = statement_undefined;
    expression
 	writtenexpr = expression_undefined,
 	newreadexpr = expression_undefined;
    call
	the_call;
    list
 	lupdatecomp = NIL,
 	lupdatenotcomp = NIL,
	lreadreftodistarray = NIL,
 	lstatcomp = NIL,
	lstatnotcomp = NIL,
	lstat = NIL,
 	linds = NIL;
    reference
 	ref,
	newref;
    entity newarray;

    (*lhp) = NIL;
    (*lnp) = NIL;
    
    /*
     * assertions... 
     */
    pips_assert("generate_c1_alpha",
		(instruction_call_p(statement_instruction(stat))));

    the_call = instruction_call(statement_instruction(stat));
    pips_assert("generate_c1_alpha",
		ENTITY_ASSIGN_P(call_function(the_call)));

    writtenexpr = EXPRESSION(CAR(call_arguments(the_call)));
    pips_assert("generate_c1_alpha",
		(syntax_reference_p(expression_syntax(writtenexpr)) &&
		 (array_distributed_p 		
		  (reference_variable
		   (syntax_reference(expression_syntax(writtenexpr)))))));
    
    /*
     * read references to distributed arrays. 
     */
    lreadreftodistarray =
	FindRefToDistArray(EXPRESSION(CAR(CDR(call_arguments(the_call)))));

    /*
     * generation of the code to get the necessary values... 
     */
    MAPL(cs, 
     { 	 
	 list lnotcomp;
 	 list lcomp;
 	 syntax
	     s = SYNTAX(CAR(cs)); 	 	

	 debug(8, "generate_c1_alpha", "considering reference to %s\n",  	
	       entity_name(reference_variable(syntax_reference(s))));
	 
	 generate_read_of_ref_for_computer(s, &lcomp, &lnotcomp);

	 lstatcomp = gen_nconc(lstatcomp, lcomp); 	
	 lstatnotcomp = gen_nconc(lstatnotcomp, lnotcomp);
     },
	 lreadreftodistarray);

    gen_free_list(lreadreftodistarray);

    /*
     * then the updated statement is to be added to node:
     */
    ref = syntax_reference(expression_syntax(writtenexpr));
    newarray  =  (entity) GET_ENTITY_MAPPING(oldtonewnodevar, reference_variable(ref));
    generate_compute_local_indices(ref, &lstat, &linds);
    newref = make_reference(newarray, linds);
    newreadexpr = 
	UpdateExpressionForModule(oldtonewnodevar,
				  EXPRESSION(CAR(CDR(call_arguments(the_call)))));


    statcomputation = 
	mere_statement(MakeAssignInst(make_syntax(is_syntax_reference, newref),
				      newreadexpr));

    lstatcomp = gen_nconc(lstatcomp, lstat);
    lstatcomp = gen_nconc(lstatcomp, CONS(STATEMENT, statcomputation, NIL));

    /*
     * Update the values of the defined distributed variable
     * if necessary... 
     */
    generate_update_values_on_nodes(ref, newref, &lupdatecomp, &lupdatenotcomp);

    lstatcomp = gen_nconc(lstatcomp, lupdatecomp);
    lstatnotcomp = gen_nconc(lstatnotcomp, lupdatenotcomp);

    /*
     * the overall statements are generated.
     */
    statcomputecomputer = st_compute_current_computer(ref);
    statifcomputer = st_make_nice_test(condition_computerp(),
				       lstatcomp,
				       lstatnotcomp);

    IFDBPRINT(8,"generate_c1_alpha", nodemodule, statifcomputer);


    (*lnp) = CONS(STATEMENT, statcomputecomputer, CONS(STATEMENT, statifcomputer, NIL));
    (*lhp) = NIL;

    return;
 }

/*
 * generate_update_values_on_nodes
 *
 * computer is doing the job
 */
void generate_update_values_on_nodes(ref, newref, lscompp, lsnotcompp)
reference ref, newref;
list *lscompp, *lsnotcompp;
{ 
    entity
	array = reference_variable(ref);

    if (replicated_p(array))
    {
	statement
	    statif,
	    statco, 
	    statco2,
	    statsndtoon,
	    statrcvfromcomp;
	list
	    lstat,
	    linds;

	/* 
	 * remote values have to be updated
 	 */
 	statco = st_compute_current_owners(ref);
	/*
	 * the computer has to compute the owners in order to call
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
	
/*	(*lscompp) = CONS(STATEMENT, statsndtoon, NIL); */
	(*lscompp) = CONS(STATEMENT, 
			statco2,
			CONS(STATEMENT,
			     statsndtoon,
			     NIL));
	(*lsnotcompp) = CONS(STATEMENT, statco, CONS(STATEMENT, statif, NIL));
    } 
    else
    { 
	(*lscompp) = NIL;
	(*lsnotcompp) = NIL;
    }
}

/*
 * generate_read_of_ref_for_computer
 *
 * en cours d'adaptation... 
 */
void generate_read_of_ref_for_computer(s, lcompp, lnotcompp)
syntax s;
list *lcompp, *lnotcompp;
{
    statement
 	statcompco,
 	statcompgv,
	statnotcompco,
 	statnotcompmaysend;
    reference
 	ref = syntax_reference(s);
    entity
 	tempn,
 	var = reference_variable(ref),
	temp = NewTemporaryVariable(get_current_module_entity(), 
				    entity_basic(var));

    pips_assert("generate_read_of_ref_for_computer", 
		(array_distributed_p(var)));

    AddEntityToHostAndNodeModules(temp);
    tempn = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, temp);

    statcompco = st_compute_current_owners(ref);
    statcompgv = st_get_value_for_computer(ref, make_reference(tempn, NIL));

    (*lcompp) = CONS(STATEMENT, statcompco, 
		CONS(STATEMENT, statcompgv, NIL));

/*
    IFDBPRINT(9, "generate_read_of_ref_for_computer", hostmodule, statcompco);
    IFDBPRINT(9, "generate_read_of_ref_for_computer", hostmodule, statcompgv);
*/
    statnotcompco = st_compute_current_owners(ref);
    statnotcompmaysend = st_send_to_computer_if_necessary(ref);


    (*lnotcompp) = 
	CONS(STATEMENT, statnotcompco, 
	CONS(STATEMENT, statnotcompmaysend, NIL));

/*
    IFDBPRINT(9, "generate_read_of_ref_for_computer", 
    nodemodule, statnotcompco);
    IFDBPRINT(9, "generate_read_of_ref_for_computer", 
    nodemodule, statnotcompmaysend);
*/
    /*
     * the new variable is inserted in the expression...
     */
    syntax_reference(s) = make_reference(temp, NIL); 
}

/*
 * generate_read_of_ref_for_all
 *
 * this function organise the read of the given reference
 * for all the nodes, and for host. 
 */
void generate_read_of_ref_for_all(s, lhp, lnp)
syntax s;
list *lhp, *lnp;
{
    statement
 	stathco,
 	stathrcv,
 	statnco,
 	statngv;
    reference 
	ref = syntax_reference(s);
    entity
	temph,
 	tempn,
	var = reference_variable(ref),
	temp = NewTemporaryVariable(get_current_module_entity(), 
				    entity_basic(var));


    pips_assert("generate_read_of_ref_for_all", (array_distributed_p(var)));
    
    AddEntityToHostAndNodeModules(temp);
    temph = (entity) GET_ENTITY_MAPPING(oldtonewhostvar, temp); 
    tempn = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, temp);

    /*
     * the receive statement is built for host:
     *
     * COMPUTE_CURRENT_OWNERS(ref)
     * temp = RECEIVEFROMSENDER(...)
     */
    stathco = st_compute_current_owners(ref);
    /*
     * a receive from sender is generated, however replicated the variable is
     * FC 930623 (before was a call to st_receive_from(ref, ...))
     */
    stathrcv = st_receive_from_sender(make_reference(temph, NIL));

    (*lhp) = CONS(STATEMENT, stathco, CONS(STATEMENT, stathrcv, NIL));

    IFDBPRINT(9, "generate_read_of_ref_for_all", hostmodule, stathco);
    IFDBPRINT(9, "generate_read_of_ref_for_all", hostmodule, stathrcv);

    /*
     * the code for node is built, in order that temp has the
     * wanted value. 
     *
     * COMPUTE_CURRENT_OWNERS(ref)
     * IF OWNERP(ME)
     * THEN
     * local_ref = COMPUTE_LOCAL(ref)
     * temp = (local_ref)
     * IF SENDERP(ME) // this protection in case of replicated arrays. 
     * THEN
     * SENDTOHOST(temp)
     * SENDTONODE(ALL-OWNERS,temp)
     * ENDIF
     * ELSE
     * temp = RECEIVEFROM(SENDER(...)) 
     * ENDIF
     */
    statnco = st_compute_current_owners(ref);
    statngv = st_get_value_for_all(ref, make_reference(tempn, NIL));

    (*lnp) = CONS(STATEMENT, statnco, CONS(STATEMENT, statngv, NIL));

    IFDBPRINT(9, "generate_read_of_ref_for_all", nodemodule, statnco);
    IFDBPRINT(9, "generate_read_of_ref_for_all", nodemodule, statngv);

    /*
     * the new variable is inserted in the expression... 
     */
    syntax_reference(s) = make_reference(temp, NIL);
}

/*
 * generate_compute_local_indices
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
    entity
 	array = reference_variable(ref);
    list
 	inds = reference_indices(ref);


    pips_assert("generate_compute_local_indexes", array_distributed_p(array));

    (*lsp) = NIL;
    (*lindsp) = NIL;

    debug(9, "generate_compute_local_indices", 
	  "number of dimensions of %s to compute: %d\n",
	  entity_name(array),
	  NumberOfDimension(array));

    for(i=1;i<=NumberOfDimension(array);i++) 
    {
	if (local_index_is_different_p(array, i))
	{
	    statement stat;
	    syntax s;
	    
	    stat = st_compute_ith_local_index(array, i, EXPRESSION(CAR(inds)), &s);
	    IFDBPRINT(9, "generate_compute_local_indexes", nodemodule, stat);
	    
	    (*lsp) = gen_nconc((*lsp), CONS(STATEMENT, stat, NIL));
	    (*lindsp) =
		gen_nconc((*lindsp), 		
			  CONS(EXPRESSION, make_expression(s, normalized_undefined), NIL));
	}
	else
	{
	    expression expr = 
		UpdateExpressionForModule(oldtonewnodevar, EXPRESSION(CAR(inds)));

	    (*lindsp) =
		gen_nconc((*lindsp),  		
			  CONS(EXPRESSION, expr, NIL));
	} 
	inds = CDR(inds);
    }

    debug(8, "generate_compute_local_indices", "result:\n");
    MAPL(cs, {IFDBPRINT(8, "generate_compute_local_indices",
		       nodemodule, STATEMENT(CAR(cs)));}, (*lsp));
	      
}

/*
 * generate_common_hpfrtds(...)
 *
 * must add a common in teh declarations that define
 * the data structures needed for the run time resolution...
 */
void generate_common_hpfrtds()
{
    pips_error("generate_common_hpfrtds", "not yet implemented\n");
}


/*
 * generate_get_value_locally
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
    statement 
	stat;
    expression 
	expr;
    entity 
	array = reference_variable(ref),
 	newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
    list 
	ls = NIL,
	newinds = NIL;
    
    pips_assert("st_get_local",array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    expr = reference_to_expression(make_reference(newarray, newinds));
    stat = mere_statement(MakeAssignInst(make_syntax(is_syntax_reference, goal), expr));

    IFDBPRINT(9, "generate_get_value_locally", nodemodule, stat);

    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, stat, NIL));
}

/*
 * generate_send_to_computer
 *
 * sends the local value of ref to the current computer
 */
void generate_send_to_computer(ref, lstatp)
reference ref;
list *lstatp;
{ 
    statement 
	statsnd;
    entity 
	array = reference_variable(ref),
 	newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
    list 
	ls = NIL,
	newinds = NIL;
    
    pips_assert("st_get_local",array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    statsnd = st_send_to_computer(make_reference(newarray, newinds));

    IFDBPRINT(9, "generate_send_to_computer", nodemodule, statsnd);
    
    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, statsnd, NIL));
}

/*
 * generate_receive_from_computer
 *
 *
 */
void generate_receive_from_computer(ref, lstatp)
reference ref; 
list *lstatp;
{
    statement 
	statrcv;
    entity 
	array = reference_variable(ref), 
 	newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
    list 
	ls = NIL,
	newinds = NIL;
    
    pips_assert("st_receive_from_computer", array_distributed_p(array));

    generate_compute_local_indices(ref, &ls, &newinds);
    statrcv = st_receive_from_computer(make_reference(newarray, newinds));
    
    IFDBPRINT(9, "st_receive_val_from_computer", nodemodule, statrcv);

    (*lstatp) = gen_nconc(ls, CONS(STATEMENT, statrcv, NIL));
}

/*
 * generate_parallel_body
 */
void generate_parallel_body(body, lstatp, lw, lr)
statement body;
list *lstatp, lw, lr;
{
    statement
	statcc,
	statbody;
    list
	lcomp = NIL,
	lcompr = NIL,
	lcompw = NIL,
	lnotcomp = NIL,
	lnotcompr = NIL,
	lnotcompw = NIL;
    syntax
	comp = SYNTAX(CAR(lw));

    pips_assert("generate_parallel_body", (gen_length(lw)>0));

    statcc = st_compute_current_computer(syntax_reference(comp));

    MAPL(cs,
     {
	 syntax
	     s = SYNTAX(CAR(cs));
	 list
	     lco = NIL;
	 list
	     lnotco = NIL;

	 generate_read_of_ref_for_computer(s, &lco, &lnotco);

	 lcompr = gen_nconc(lcompr, lco);
	 lnotcompr = gen_nconc(lnotcompr, lnotco);	 
     },
	 lr);

/*
    debug(7, "generate_parallel_body", "read for computer:\n");
    MAPL(cs, {IFDBPRINT(7, "generate_parallel_body", 
		       nodemodule, STATEMENT(CAR(cs)));}, lcompr);
    debug(7, "generate_parallel_body", "read for not computer:\n");
    MAPL(cs, {IFDBPRINT(7, "generate_parallel_body", 
		       nodemodule, STATEMENT(CAR(cs)));}, lnotcompr);
*/

    MAPL(cs,
     {
	 list
	     lco = NIL;
	 list
	     lnotco = NIL;
	 syntax
	     s = SYNTAX(CAR(cs));
	 reference 
	     r = syntax_reference(s);
	 entity
	     var = reference_variable(r);
	 entity
	     temp = NewTemporaryVariable(get_current_module_entity(),
					 entity_basic(var));
	 entity
	     tempn ;

	 AddEntityToHostAndNodeModules(temp);
	 tempn = (entity) GET_ENTITY_MAPPING(oldtonewnodevar,  temp);

	 if (comp == s)
	 {
	     /*
	      * we are sure that computer is one of the owners
	      */
	     list
		 lstat = NIL;
	     list
		 linds = NIL;
	     entity
		 newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, var);
							

	     generate_compute_local_indices(r, &lstat, &linds);
	     lstat = 
		 gen_nconc
		     (lstat,
		      CONS(STATEMENT,
			   mere_statement
			   (MakeAssignInst
			    (make_syntax(is_syntax_reference,
					 make_reference(newarray, 
							linds)),
			     reference_to_expression(make_reference(tempn, 
								    NIL)))),
			   NIL));
									      
	     generate_update_values_on_nodes(r, 
					     make_reference(tempn, NIL), 
					     &lco, 
					     &lnotco);

	     lco = gen_nconc(lstat, lco);
	 }
	 else
	 {
	     generate_update_values_on_computer_and_nodes(r, 
							  make_reference(tempn, NIL), 
							  &lco, 
							  &lnotco);
	 }
	 
	 lcompw = gen_nconc(lcompw, lco);
	 lnotcompw = gen_nconc(lnotcompw, lnotco);

	 syntax_reference(s) = make_reference(temp, NIL);
     },
	 lw);


    debug(6, "generate_parallel_body", 
	  "%d statements for computer write:\n",
	  gen_length(lcompw));

    ifdebug(8)
    {
	MAPL(cs,
	 {
	     IFDBPRINT(8,"generate_parallel_body",
		       nodemodule,STATEMENT(CAR(cs)));
	 },
	     lcompw);
    }

    debug(6, "generate_parallel_body", 
	  "%d statements for not computer write:\n",
	  gen_length(lnotcompw));

    ifdebug(8)
    {
	MAPL(cs,
	 {
	     IFDBPRINT(8,"generate_parallel_body",
		       nodemodule,STATEMENT(CAR(cs)));
	 },
	     lnotcompw);
    }

    statbody = UpdateStatementForModule(oldtonewnodevar, body);
    IFDBPRINT(7, "generate_parallel_body", nodemodule, statbody);

    lcomp = gen_nconc(lcompr, CONS(STATEMENT, statbody, lcompw));
    lnotcomp = gen_nconc(lnotcompr, lnotcompw);

    (*lstatp) = CONS(STATEMENT,
		     statcc,
		     CONS(STATEMENT,
			  st_make_nice_test(condition_computerp(),
					    lcomp,
					    lnotcomp),
			  NIL));

    debug(6, "generate_parallel_body", "final statement:\n");
    MAPL(cs,{IFDBPRINT(6,"generate_parallel_body",
		       nodemodule,STATEMENT(CAR(cs)));},(*lstatp));
}


/*
 * generate_update_values_on_computer_and_nodes
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
	newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
    statement
	statif,
	statcompif,
	statco,
	statcompco,
	statcompassign,
	statsndtoO,
	statsndtoOO,
	statrcvfromcomp;
    list
	lstat,
	lstatcomp,
	linds,
	lindscomp;
    
    /* 
     * all values have to be updated...
     */
    statcompco = st_compute_current_owners(ref);
    generate_compute_local_indices(ref, &lstatcomp, &lindscomp);

/*
    debug(8, "generate_update_values_on_computer_and_nodes","lstatcomp:\n");
    MAPL(cs,{IFDBPRINT(8,"generate_update_values_on_computer_and_nodes",
		       nodemodule,STATEMENT(CAR(cs)));}, lstatcomp);
*/

    statcompassign = 
	mere_statement(MakeAssignInst(make_syntax(is_syntax_reference,
						  make_reference(newarray,lindscomp)),
				      reference_to_expression(val)));   
/*
    IFDBPRINT(8,"generate_update_values_on_computer_and_nodes",
	      nodemodule,statcompassign);
*/
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
    statcompif = st_make_nice_test(condition_ownerp(),
				   gen_nconc(lstatcomp, 
					    CONS(STATEMENT,
						 statcompassign,
						 CONS(STATEMENT,
						      statsndtoOO,
						      NIL))),
				   CONS(STATEMENT,
					statsndtoO,
					NIL));
/*
    IFDBPRINT(8, "generate_update_values_on_computer_and_nodes",
	      nodemodule, statcompif);
*/
    statco = st_compute_current_owners(ref);
    generate_compute_local_indices(ref, &lstat, &linds);
    statrcvfromcomp = 
	st_receive_from_computer(make_reference(newarray,linds));
    statif = st_make_nice_test(condition_ownerp(),
			       gen_nconc(lstat,
					 CONS(STATEMENT,
					      statrcvfromcomp,
					      NIL)),
			       NIL);
    
    (*lscompp)=CONS(STATEMENT, statcompco, 
		    CONS(STATEMENT, statcompif, NIL));
    (*lsnotcompp)=CONS(STATEMENT, statco,
		       CONS(STATEMENT,statif,NIL));
/*
    debug(8, "generate_update_values_on_computer_and_nodes","result for computer:\n");
    MAPL(cs,{IFDBPRINT(8,"generate_update_values_on_computer_and_nodes",
		       nodemodule,STATEMENT(CAR(cs)));},(*lscompp));

    debug(8, "generate_update_values_on_computer_and_nodes","result for not computer:\n");
    MAPL(cs,{IFDBPRINT(8,"generate_update_values_on_computer_and_nodes",
		       nodemodule,STATEMENT(CAR(cs)));},(*lsnotcompp));
*/
}

/*
 * generate_update_distributed_value_from_host
 */
void generate_update_distributed_value_from_host(s, lhstatp, lnstatp)
syntax s;
list *lhstatp, *lnstatp;
{
    reference
	r = syntax_reference(s);
    entity
	array,
	newarray,
	temp,
	temph;
    statement
	statnco,
	stathco,
	stnrcv,
	stnif,
	sthsnd;
    list
	linds = NIL,
	lnstat = NIL;

    pips_assert("generate_update_distributed_value_from_host", 
		(array_distributed_p(reference_variable(syntax_reference(s)))));

    array    = reference_variable(r);
    newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);

    temp     = NewTemporaryVariable(get_current_module_entity(), 
				    entity_basic(array));

    AddEntityToHostAndNodeModules(temp);
    temph = (entity) GET_ENTITY_MAPPING(oldtonewhostvar, temp);

    generate_compute_local_indices(r, &lnstat, &linds);
    stnrcv = st_receive_from_host(make_reference(newarray, linds));
    stnif = st_make_nice_test(condition_ownerp(),
			      gen_nconc(lnstat,
					CONS(STATEMENT, stnrcv, NIL)),
			      NIL);
    
    /*
     * call the necessary communication function
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


/*
 * generate_update_private_value_from_host
 */
void generate_update_private_value_from_host(s, lhstatp, lnstatp)
syntax s;
list *lhstatp, *lnstatp;
{
    entity
	var  = reference_variable(syntax_reference(s)),
	varn = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, var),
	varh = (entity) GET_ENTITY_MAPPING(oldtonewhostvar, var);
    
    pips_assert("generate_update_private_value_from_host", 
		(!array_distributed_p(reference_variable(syntax_reference(s)))));

    (*lhstatp) = CONS(STATEMENT,
		      st_host_send_to_all_nodes(make_reference(varh, NIL)),
		      NIL);

    (*lnstatp) = CONS(STATEMENT,
		      st_receive_mcast_from_host(make_reference(varn, NIL)),
		      NIL);
}
