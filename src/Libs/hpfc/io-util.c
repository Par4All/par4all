/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-util.c,v $ ($Date: 1994/05/05 09:48:56 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

entity CreateIntrinsic(string name); /* in syntax */

/*
 * ??? beurk
 */
static statement define_node_processor_id(proc)
entity proc;
{
    int 
	i=0,
	procn = get_hpf_number(proc);
    expression /* ??? gonna be shared! */
	proce = int_to_expression(procn);
    entity
	dummy = entity_undefined;
    list
	mypos_indices = NIL,
	ls = NIL;
    reference
	mypos = reference_undefined;

    for(i=NumberOfDimension(proc); i>=1; i--)
	dummy = get_ith_processor_dummy(i),
	mypos_indices = CONS(EXPRESSION, int_to_expression(i),
			CONS(EXPRESSION, proce, 
			     NIL)),
	mypos = make_reference(hpfc_name_to_entity(MYPOS), mypos_indices),
	ls = CONS(STATEMENT,
		  make_assign_statement(entity_to_expression(dummy),
					reference_to_expression(mypos)),
		  ls);
				       
    return(make_block_statement(ls));	
}

/* statement generate_deducables(list le)
 *
 * the le list of expression is used to generate the deducables.
 * The fields of interest are the variable which is referenced and 
 * the normalized filed which is the expression that is going to
 * be used to define the variable.
 */
statement generate_deducables(le)
list le;
{
    list
	ls = NIL;

    MAPL(ce,
     {
	 expression
	     e = EXPRESSION(CAR(ce));
	 entity
	     var = reference_variable(expression_reference(e));
	 Pvecteur
	     v = vect_dup(normalized_linear(expression_normalized(e)));

	 ls = CONS(STATEMENT,
		   make_assign_statement
		      (entity_to_expression(var),
		       make_vecteur_expression((vect_erase_var(&v, 
							       (Variable) var),
						v))),
		   ls);	

	 vect_rm(v);
     },
	 le);

    return(make_block_statement(ls));
}

list hpfc_gen_n_vars_expr(creation, number)
entity (*creation)();
int number;
{
    list
	result = NIL;
    int i;

    pips_assert("hpfc_gen_n_vars", (number>=0) && (number<=7));

    for(i=number; i>=1; i--)
	result=CONS(EXPRESSION,
		    entity_to_expression(creation(i)),
		    result);

    return(result);
}

/* the following functions generate the statements to appear in
 * the I/O loop nest.
 */

static statement hpfc_initsend()
{
    /*
     * 2 args to pvmfinitsend
     */
    return
	(my_make_call_statement
	     (hpfc_name_to_entity(PVM_INITSEND), 
	      CONS(EXPRESSION, MakeCharacterConstantExpression("PVMRAW"),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(BUFID)),
		   NIL))));
}

static statement hpfc_pack(array, local)
entity array;
bool local;
{
    /*
     * 5 args to pvmfpack
     */
    return
	(my_make_call_statement
	     (hpfc_name_to_entity(PVM_PACK),
	      CONS(EXPRESSION, pvm_what_option_expression(array),
	      CONS(EXPRESSION, reference_to_expression
		                  (make_reference
				       (array, 
					hpfc_gen_n_vars_expr
					    (local?
					     get_ith_local_dummy:
					     get_ith_array_dummy,
					     NumberOfDimension(array)))),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(INFO)),
		   NIL)))))));
}

static statement hpfc_unpack(array, local)
entity array;
bool local;
{
    /*
     * 5 args to pvmfunpack
     */

    return
	(my_make_call_statement
	     (hpfc_name_to_entity(PVM_UNPACK),
	      CONS(EXPRESSION, pvm_what_option_expression(array),
	      CONS(EXPRESSION, reference_to_expression
		                  (make_reference
				       (array, 
					hpfc_gen_n_vars_expr
					    (local?
					     get_ith_local_dummy:
					     get_ith_array_dummy,
					     NumberOfDimension(array)))),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(INFO)),
		   NIL)))))));
}

list make_list_of_constant(val, number)
int val, number;
{
    list
	l=NIL;
    int i;

    for(i=1; i<=number; i++)
	l = CONS(EXPRESSION, make_integer_constant_expression(val), l);

    return(l);
}

/* static statement st_compute_lid(proc)
 *
 *       T_LID=CMP_LID(pn, pi...)
 */
static statement hpfc_compute_lid(proc)
entity proc;
{
    int
	ndim = NumberOfDimension(proc);
    entity
	lid = hpfc_name_to_entity(T_LID),
	cmp_lid = hpfc_name_to_entity(CMP_LID);

    return(make_assign_statement
   	       (entity_to_expression(lid),
		make_call_expression
		(cmp_lid,
		 CONS(EXPRESSION, 
		      int_to_expression(get_hpf_number(proc)),
		      gen_nconc(hpfc_gen_n_vars_expr(get_ith_processor_dummy,
						     ndim),
				make_list_of_constant(0, 7-ndim))))));

}

/* static statement add_2(exp)
 *
 *       exp = exp + 2
 */
static statement hpfc_add_2(exp)
expression exp;
{
    entity
	plus = CreateIntrinsic(PLUS_OPERATOR_NAME);

    return(make_assign_statement(expression_dup(exp),
				 MakeBinaryCall(plus,
						exp,
						int_to_expression(2))));

}

/* static statement hpfc_hsend(proc)
 * 
 *       T_LID=CMP_LID(pn, pi...)
 *       PVM_SEND(NODETIDS(T_LID), NODE_CHANNELS(T_LID), INFO)
 *       NODE_CHANNELS(T_LID) = NODE_CHANNELS(T_LID) + 2
 *
 */
static statement hpfc_hsend(proc)
entity proc;
{
    entity
	lid = hpfc_name_to_entity(T_LID),
	channel = hpfc_name_to_entity(NODE_CHANNELS),
	nodetid = hpfc_name_to_entity(NODETIDS),
	pvm_send = hpfc_name_to_entity(PVM_SEND),
	info = hpfc_name_to_entity(INFO);
    statement
	cmp_lid = hpfc_compute_lid(proc),
	send =
	    my_make_call_statement
		(pvm_send,
		 CONS(EXPRESSION, 
		      reference_to_expression
		          (make_reference
			       (nodetid,
				CONS(EXPRESSION, entity_to_expression(lid), 
				     NIL))),
		 CONS(EXPRESSION,
		      reference_to_expression
		          (make_reference
			       (channel,
				CONS(EXPRESSION, entity_to_expression(lid), 
				     NIL))),
		 CONS(EXPRESSION, entity_to_expression(info),
		      NIL)))),
	incr2 = hpfc_add_2(reference_to_expression
		      (make_reference
		       (channel,
			CONS(EXPRESSION, entity_to_expression(lid), 
			     NIL))));

    return(make_block_statement(CONS(STATEMENT, cmp_lid,
				CONS(STATEMENT, send,
				CONS(STATEMENT, incr2,
				     NIL)))));
}

/*
 *
 *      PVM_SEND(HOST_TID, HOST_CHANNEL, INFO)
 *      HOST_CHANNEL = HOST_CHANNEL + 2
 */
static statement hpfc_nsend()
{
    entity
	pvm_send = hpfc_name_to_entity(PVM_SEND),
	info = hpfc_name_to_entity(INFO),
	channel = hpfc_name_to_entity(HOST_CHANNEL),
	htid = hpfc_name_to_entity(HOST_TID);

    statement
	send = 
	    my_make_call_statement
		(pvm_send,
		 CONS(EXPRESSION, entity_to_expression(htid),
		 CONS(EXPRESSION, entity_to_expression(channel),
		 CONS(EXPRESSION, entity_to_expression(info),
		      NIL)))),
	incr2 = hpfc_add_2(entity_to_expression(channel));

    return(make_block_statement(CONS(STATEMENT, send,
				CONS(STATEMENT, incr2,
				     NIL))));
}

/* static statement hpfc_hcast()
 *
 *       PVM_CAST(NBTASKS, NODETIDS, MCASTHOST, INFO)
 *       MCASTHOST = MCASTHOST + 2
 *
 */
static statement hpfc_hcast()
{
    entity
	pvm_cast = hpfc_name_to_entity(PVM_CAST),
	nbtasks = hpfc_name_to_entity(NBTASKS),
	nodetid = hpfc_name_to_entity(NODETIDS),
	mcasthost = hpfc_name_to_entity(MCASTHOST),
	info = hpfc_name_to_entity(INFO);

    statement
	st_cast = 
	    my_make_call_statement
		(pvm_cast,
		 CONS(EXPRESSION, entity_to_expression(nbtasks),
		 CONS(EXPRESSION, entity_to_expression(nodetid),
		 CONS(EXPRESSION, entity_to_expression(mcasthost),
		 CONS(EXPRESSION, entity_to_expression(info),
		      NIL))))),
	incr2 = hpfc_add_2(entity_to_expression(mcasthost));

    return(make_block_statement(CONS(STATEMENT, st_cast,
				CONS(STATEMENT, incr2,
				     NIL))));
}

/*
 *       T_LID=CMP_LID(pn, pi...)
 *       PVM_RECV(NODETID(T_LID), NODE_CHANNELS(T_LID), INFO)
 *       NODE_CHANNELS(T_LID) = NODE_CHANNELS(T_LID) + 2
 */
static statement hpfc_hrecv(proc)
entity proc;
{
    entity
	lid = hpfc_name_to_entity(T_LID),
	channel = hpfc_name_to_entity(NODE_CHANNELS),
	nodetid = hpfc_name_to_entity(NODETIDS),
	pvm_recv = hpfc_name_to_entity(PVM_RECV),
	info = hpfc_name_to_entity(INFO);

    statement
	cmp_lid = hpfc_compute_lid(proc),
	recv =
	    my_make_call_statement
		(pvm_recv,
		 CONS(EXPRESSION, 
		      reference_to_expression
		          (make_reference
			       (nodetid,
				CONS(EXPRESSION, entity_to_expression(lid), 
				     NIL))),
		 CONS(EXPRESSION,
		      reference_to_expression
		          (make_reference
			       (channel,
				CONS(EXPRESSION, entity_to_expression(lid), 
				     NIL))),
		 CONS(EXPRESSION, entity_to_expression(info),
		      NIL)))),
	incr2 = hpfc_add_2(reference_to_expression
		      (make_reference
		       (channel,
			CONS(EXPRESSION, entity_to_expression(lid), 
			     NIL))));

    return(make_block_statement(CONS(STATEMENT, cmp_lid,
				CONS(STATEMENT, recv,
				CONS(STATEMENT, incr2,
				     NIL)))));
}

/*
 *       PVM_RECV(HOST_TID, {HOST_CHANNEL, MCASTHOST}, INFO)
 *       {} += 2
 */
static statement hpfc_nrecv(cast) /* from host */
bool cast;
{
    entity
	channel = (cast?
		   hpfc_name_to_entity(MCASTHOST):
		   hpfc_name_to_entity(HOST_CHANNEL)),
	hosttid = hpfc_name_to_entity(HOST_TID),
	info = hpfc_name_to_entity(INFO),
	pvm_rcv = hpfc_name_to_entity(PVM_RECV);
    statement
	receive = 
	    my_make_call_statement(pvm_rcv,
				CONS(EXPRESSION, entity_to_expression(hosttid),
				CONS(EXPRESSION, entity_to_expression(channel),
				CONS(EXPRESSION, entity_to_expression(info),
				     NIL)))),
	incr2 = hpfc_add_2(entity_to_expression(channel));

    return(make_block_statement(CONS(STATEMENT, receive,
				CONS(STATEMENT, incr2,
				     NIL))));
}

static statement node_pre_io(array, move)
entity array;
tag move;
{
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = hpfc_initsend();
	break;
    case is_movement_update:
	stat = hpfc_nrecv(FALSE);
	break;
    default:
        pips_error("node_pre_io", "unexpected movement tag\n");
        break;
    }

    return(stat);
}

static statement node_in_io(array, move)
entity array;
tag move;
{
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = hpfc_pack(array, TRUE);
	break;
    case is_movement_update:
	stat = hpfc_unpack(array, TRUE);
	break;
    default:
        pips_error("node_in_io", "unexpected movement tag\n");
        break;
    }

    return(stat);
}

static statement node_post_io(array, move)
entity array;
tag move;
{
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = hpfc_nsend();
	break;
    case is_movement_update:
	stat = make_empty_statement();
	break;
    default:
        pips_error("node_post_io", "unexpected movement tag\n");
        break;
    }

    return(stat);
}

static statement host_pre_io(array, move)
entity array;
tag move;
{
    entity
	proc = template_to_processors(array_to_template(array));
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = hpfc_hrecv(proc);
	break;
    case is_movement_update:
	stat = hpfc_initsend();
	break;
    default:
        pips_error("host_pre_io", "unexpected movement tag\n");
        break;
    }

    return(stat);
}

static statement host_in_io(array, move)
entity array;
tag move;
{
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = hpfc_unpack(array, FALSE);
	break;
    case is_movement_update:
	stat = hpfc_pack(array, FALSE);
	break;
    default:
        pips_error("host_in_io", "unexpected movement tag\n");
        break;
    }

    return(stat);
}

static statement host_post_io(array, move)
entity array;
tag move;
{
    entity
	proc = template_to_processors(array_to_template(array));
    statement
	stat = statement_undefined;

    switch(move)
    {
    case is_movement_collect:
	stat = make_empty_statement();
	break;
    case is_movement_update:
	stat = hpfc_hsend(proc);
	break;
    default:
        pips_error("host_post_io","unexpected movement tag\n");
        break;
    }

    return(stat);
}

/*
 * generate_io_statements_for_distributed_arrays
 *
 * ??? the variables are not generated in the right module
 *
 *
 */
void generate_io_statements_for_distributed_arrays
  (array, move, 
   condition, proc_echelon, tile_echelon,
   parameters, processors, scanners, rebuild,
   psh, psn)
entity array;
tag move;
Psysteme condition, proc_echelon, tile_echelon;
list parameters, processors, scanners, rebuild;
statement *psh, *psn;
{
    entity
	proc = template_to_processors(array_to_template(array));
    Psysteme
	/* proc_cond computation:
	 * well, it may have been kept before the new loop bound computation?
	 */
	proc_decl = entity_to_declaration_constraints(proc),
	proc_cond_tmp = sc_dup(proc_echelon),
	proc_cond = (sc_nredund(&proc_cond_tmp),
		     non_redundent_subsystem(proc_cond_tmp, proc_decl));
    statement
	inner_host_proc = statement_undefined,
	inner_host_scan = statement_undefined,
	inner_node_scan = statement_undefined,
	node_tmp = statement_undefined,
	host_proc_loop = 
	    systeme_to_loop_nest(proc_echelon, processors, &inner_host_proc),
	host_scan_loop = 
	    systeme_to_loop_nest(tile_echelon, scanners, &inner_host_scan),
	node_scan_loop = 
	    systeme_to_loop_nest(tile_echelon, scanners, &inner_node_scan),
	node_defproc = define_node_processor_id(proc),
	node_deduce = generate_deducables(rebuild),
	host_deduce = generate_deducables(rebuild),
	h_pre = host_pre_io(array, move),
	h_in = host_in_io(array, move),
	h_post = host_post_io(array, move),
	n_pre = node_pre_io(array, move),
	n_in = node_in_io(array, move),
        n_post = node_post_io(array, move);

    ifdebug(8)
    {
	fprintf(stderr, 
		"[generate_io_statements_for_distributed_array] statements:\n");
	fprintf(stderr, "Host:\n");
	fprintf(stderr, "host_pre_io:\n");
	print_statement(h_pre);
	fprintf(stderr, "host_deduce:\n");
	print_statement(host_deduce);
	fprintf(stderr, "host_in_io:\n");
	print_statement(h_in);
	fprintf(stderr, "host_post_io:\n");
	print_statement(h_post);
	fprintf(stderr, "Node:\n");
	fprintf(stderr, "node_pre_io:\n");
	print_statement(n_pre);
	fprintf(stderr, "node_deduce:\n");
	print_statement(node_deduce);
	fprintf(stderr, "node_in_io:\n");
	print_statement(n_in);
	fprintf(stderr, "node_post_io:\n");
	print_statement(n_post);
    }

    /*
     * code to be generated:
     * 
     * Host:
     *
     * [ IF condition then ]
     *     DO host_proc_loop
     *         host_pre_io
     *         DO host_scan_loop
     *            host_deduce
     *            host_in_io
     *         ENDDO
     *         host_post_io
     *     ENDDO
     * [ ENDIF ]
     *
     * Node:
     *
     * [ IF condition then ]
     *     node_defproc
     *     [ IF proc_cond then ]
     *         node_pre_io
     *         DO node_scan_loop
     *            node_deduce
     *            node_in_io
     *         ENDDO
     *         node_post_io
     *     [ ENDIF ]
     * [ ENDIF ]
     */
     
    loop_body(instruction_loop(statement_instruction(inner_host_scan))) = 
	make_block_statement(CONS(STATEMENT, host_deduce,
			     CONS(STATEMENT, h_in,
				  NIL)));
    loop_body(instruction_loop(statement_instruction(inner_host_proc))) = 
	make_block_statement(CONS(STATEMENT, h_pre,
			     CONS(STATEMENT, host_scan_loop,
			     CONS(STATEMENT, h_post,
				  NIL))));
    *psh = generate_optional_if(condition, host_proc_loop);

    loop_body(instruction_loop(statement_instruction(inner_node_scan))) = 
	make_block_statement(CONS(STATEMENT, node_deduce,
			     CONS(STATEMENT, n_in,
				  NIL)));
    node_tmp = 
	make_block_statement(CONS(STATEMENT, n_pre,
			     CONS(STATEMENT, node_scan_loop,
			     CONS(STATEMENT, n_post,
				  NIL))));
    *psn = 
	generate_optional_if
	    (condition,
	     make_block_statement(CONS(STATEMENT, node_defproc,
				  CONS(STATEMENT, 
				       generate_optional_if(proc_cond,
							    node_tmp),
				       NIL))));

    sc_rm(proc_cond_tmp),
    sc_rm(proc_cond);

    ifdebug(5)
    {
	fprintf(stderr, 
		"[generate_io_statements_for_distributed_array] result:\n");
	fprintf(stderr, "Host:\n");
	print_statement(*psh);
	fprintf(stderr, "Node:\n");
	print_statement(*psn);
    }
}

/*
 * that's all
 */
