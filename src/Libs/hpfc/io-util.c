/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-util.c,v $ ($Date: 1994/12/06 14:42:50 $, ) version $Revision$,
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
#include "conversion.h"
#include "properties.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"
/* #include "compiler_parameters.h" */

entity CreateIntrinsic(string name); /* in syntax */

/*
 * ??? beurk
 */
static statement define_node_processor_id(proc)
entity proc;
{
    int 
	i=0,
	procn = load_entity_hpf_number(proc);
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
	rev = gen_nreverse(gen_copy_seq(le)),
	ls = NIL;

    MAPL(ce,
     {
	 expression
	     e = EXPRESSION(CAR(ce));
	 entity
	     var = reference_variable(expression_reference(e));
	 Pvecteur
	     v = vect_dup(normalized_linear(expression_normalized(e)));
	 int
	     coef = vect_coeff((Variable) var, v);

	 pips_assert("generate_deducables", (abs(coef)==1));

	 vect_erase_var(&v, (Variable) var);
	 if (coef==1) vect_chg_sgn(v);

	 ls = CONS(STATEMENT,
		   make_assign_statement(entity_to_expression(var),
					 make_vecteur_expression(v)),
		   ls);	

	 vect_rm(v);
     },
	 rev);

    gen_free_list(rev);
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
	(hpfc_make_call_statement
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
	(hpfc_make_call_statement
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
	(hpfc_make_call_statement
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

    pips_assert("make_list_of_constant", number>=0);

    for(i=1; i<=number; i++)
	l = CONS(EXPRESSION, make_integer_constant_expression(val), l);

    return(l);
}

#define psi(i) entity_to_expression(get_ith_processor_dummy(i))

/* expression linearized_processor(proc, creator) */

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
	lid = hpfc_name_to_entity(T_LID);

    if (!get_bool_property("HPFC_EXPAND_CMPLID"))
    {
	entity
	    cmp_lid = hpfc_name_to_entity(CMP_LID);
	
	return(make_assign_statement
   	       (entity_to_expression(lid),
		make_call_expression
		(cmp_lid,
		 CONS(EXPRESSION, 
		      int_to_expression(load_entity_hpf_number(proc)),
		      gen_nconc(hpfc_gen_n_vars_expr(get_ith_processor_dummy,
						     ndim),
				make_list_of_constant(0, 7-ndim))))));
    }
    else
    {
	int i = 0;
	entity
	    plus = CreateIntrinsic(PLUS_OPERATOR_NAME),
	    minus = CreateIntrinsic(MINUS_OPERATOR_NAME),
	    multiply = CreateIntrinsic(MULTIPLY_OPERATOR_NAME);
	expression
	    value = expression_undefined;
	
	/* if (NODIMP(pn).EQ.0) then
	 *   lid = 1
	 * else
	 *   t = indp(1) - RANGEP(pn, 1, 1)
	 *   do i=2, NODIMP(pn)
	 *     t = (t * RANGEP(pn, i, 3)) + (indp(i) - RANGEP(pn, i, 1))
	 *   enddo
	 *   lid = t+1
	 * endif
	 */
	
	if (ndim==0) 
	    return(make_assign_statement(entity_to_expression(lid),
					 int_to_expression(1)));
	
	value = make_call_expression(minus,
	    CONS(EXPRESSION, psi(1),
	    CONS(EXPRESSION, 
		 copy_expression(dimension_lower(FindIthDimension(proc, 1))),
		 NIL)));
	
	for(i=2;
	    i<=ndim;
	    i++)
	{
	    dimension
		dim = FindIthDimension(proc, i);
	    expression
		t1 = make_call_expression(minus,
		     CONS(EXPRESSION, copy_expression(dimension_upper(dim)),
		     CONS(EXPRESSION, copy_expression(dimension_lower(dim)),
			  NIL))),
		t2 = make_call_expression(plus,
		     CONS(EXPRESSION, t1,
		     CONS(EXPRESSION, int_to_expression(1),
			  NIL))),
		t3 = make_call_expression(multiply,
		     CONS(EXPRESSION, t2,
		     CONS(EXPRESSION, value,
			  NIL))),
		t4 = make_call_expression(minus,
		     CONS(EXPRESSION, psi(i),
		     CONS(EXPRESSION, copy_expression(dimension_lower(dim)),
			  NIL)));
	
	    value = make_call_expression(plus,
		    CONS(EXPRESSION, t3,
		    CONS(EXPRESSION, t4,
			 NIL)));
	}
    
	value = MakeBinaryCall(plus, value, int_to_expression(1));
	return(make_assign_statement(entity_to_expression(lid), value));
    }
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
	    hpfc_make_call_statement
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
	    hpfc_make_call_statement
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
	    hpfc_make_call_statement
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
	    hpfc_make_call_statement
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
	    hpfc_make_call_statement(pvm_rcv,
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
	proc = array_to_processors(array);
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
	proc = array_to_processors(array);
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
	proc = array_to_processors(array),
	divide = hpfc_name_to_entity(IDIVIDE);
    Psysteme
	/* proc_cond computation:
	 * well, it may have been kept before the new loop bound computation?
	 */
	proc_decl = entity_to_declaration_constraints(proc),
	proc_cond_tmp = sc_dup(proc_echelon),
	proc_cond = (sc_nredund(&proc_cond_tmp),
		     non_redundent_subsystem(proc_cond_tmp, proc_decl));
    statement
	node_tmp = statement_undefined,
	node_defproc = define_node_processor_id(proc),
	node_deduce = generate_deducables(rebuild),
	host_deduce = generate_deducables(rebuild),
	h_pre = host_pre_io(array, move),
	h_in = host_in_io(array, move),
	h_post = host_post_io(array, move),
	n_pre = node_pre_io(array, move),
	n_in = node_in_io(array, move),
        n_post = node_post_io(array, move),
	host_scan_loop = 
	    systeme_to_loop_nest
		(tile_echelon, 
		 scanners, 
		 make_block_statement(CONS(STATEMENT, host_deduce,
				      CONS(STATEMENT, h_in,
					   NIL))),
		 divide),
	host_proc_loop = 
	    systeme_to_loop_nest
		(proc_echelon, 
		 processors, 
		 make_block_statement(CONS(STATEMENT, h_pre,
				      CONS(STATEMENT, host_scan_loop,
				      CONS(STATEMENT, h_post,
					   NIL)))),
		 divide),
	node_scan_loop = 
	    systeme_to_loop_nest
		(tile_echelon, 
		 scanners, 	
		 make_block_statement(CONS(STATEMENT, node_deduce,
				      CONS(STATEMENT, n_in,
					   NIL))),
		 divide);

    *psh = generate_optional_if(condition, host_proc_loop);

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
 * generate_io_statements_for_shared_arrays
 *
 * code to be generated:
 *
 * Host:
 *
 * [ IF condition then ]
 *     init_send
 *     DO scanners
 *         rebuild
 *         pack
 *     ENDDO
 *     host_cast
 * [ ENDIF ]
 *
 * Node:
 *
 * [ IF condition then ]
 *     receive_hcast
 *     DO scanners
 *        rebuild
 *        unpack
 *     ENDDO
 * [ ENDIF ]
 *
 */
void generate_io_statements_for_shared_arrays
  (array, move,
   condition, echelon,
   parameters, scanners, rebuild,
   psh, psn)
entity array;
tag move;
Psysteme condition, echelon;
list parameters, scanners, rebuild;
statement *psh, *psn;
{
    entity
	divide = hpfc_name_to_entity(IDIVIDE);
    statement
	h_pre = hpfc_initsend(),
	h_rebuild = generate_deducables(rebuild),
	h_pack = hpfc_pack(array, FALSE),
	h_cast = hpfc_hcast(),
	n_rcv = hpfc_nrecv(TRUE),
	n_rebuild = generate_deducables(rebuild),
	n_unpack = hpfc_unpack(array, FALSE),
	h_scan = systeme_to_loop_nest
	             (echelon, 
		      scanners,
		      make_block_statement(CONS(STATEMENT, h_rebuild,
					   CONS(STATEMENT, h_pack,
						NIL))),
		      divide),
	n_scan = systeme_to_loop_nest
	             (echelon, 
		      scanners,
		      make_block_statement(CONS(STATEMENT, n_rebuild,
					   CONS(STATEMENT, n_unpack,
						NIL))),
		      divide);

    pips_assert("generate_io_statements_for_shared_arrays",
		movement_update_p(move));

    *psh = generate_optional_if(condition,
				make_block_statement(CONS(STATEMENT, h_pre,
						     CONS(STATEMENT, h_scan,
						     CONS(STATEMENT, h_cast,
							  NIL)))));

    *psn = generate_optional_if(condition,
				make_block_statement(CONS(STATEMENT, n_rcv,
						     CONS(STATEMENT, n_scan,
							  NIL))));
}

/*
 * that's all
 */
