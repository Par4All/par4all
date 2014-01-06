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
/* HPFC module by Fabien COELHO
 */

#include "defines-local.h"

#include "conversion.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/************************************************** ONLY_IO MAP DEFINITION */

/* this is enough for the distribution purpose, because more clever
 * analysis would be as checking for the distributablity of the enclosed
 * code. A distribution code dedicated to IO will be implemented later on.
 * only io encoding:
 * 0 - not an IO
 * 1 - is an IO
 * 3 - may be considered as an IO along real IO functions...
 */

GENERIC_CURRENT_MAPPING(only_io, bool, statement)

static statement_mapping 
  stat_bool_map = hash_table_undefined;

#define Load(stat) \
    ((bool)(intptr_t) (hash_get(stat_bool_map,  stat)))

#define Store(stat, val) \
    (hash_put(stat_bool_map,  (void*)stat, (void*) (val)))

/* true if the first statement of a block is a host section marker.
 * looks like a hack. should be managed in directives.c...
 */
static bool 
host_section_p(
    list /* of statement */ ls)
{
    if (gen_length(ls)>=1)
    {
	instruction one = statement_instruction(STATEMENT(CAR(ls)));
	if (instruction_call_p(one))
	    return same_string_p
		(entity_local_name(call_function(instruction_call(one))), 
		 HPF_PREFIX HOSTSECTION_SUFFIX);
    }

    return false;
}

/* ??? neglect expression side effects...
 */
DEFINE_LOCAL_STACK(current_statement, statement)

void hpfc_io_util_error_handler()
{
    error_reset_current_statement_stack();
}

static void only_io_sequence(sequence q)
{
    intptr_t is_io;
    if (host_section_p(sequence_statements(q)))
	is_io=1;
    else
    {
	is_io=3;
	MAP(STATEMENT, s, is_io = (is_io & Load(s)), sequence_statements(q));
    }

    pips_debug(5, "block %p: %"PRIdPTR"\n", q, is_io);
    Store(current_statement_head(), is_io);
}

static void only_io_test(test t)
{
    intptr_t is_io=3;
    is_io = (Load(test_true(t)) & Load(test_false(t)));
    pips_debug(5, "test %p: %"PRIdPTR"\n", t, is_io);
    Store(current_statement_head(), is_io);
}

static void only_io_loop(loop l)
{
    intptr_t is_io = Load(loop_body(l));
    pips_debug(5, "loop %p: %"PRIdPTR"\n", l, is_io);
    Store(current_statement_head(), is_io);
}

static void only_io_call(call c)
{
    entity f = call_function(c);
    intptr_t is_io = entity_continue_p(f)? 3:
                io_intrinsic_p(f) ||     /* Fortran IO intrinsics */
		    hpfc_special_io(f) ||    /* declared with FCD */
			hpfc_io_like_function(f);/* runtime managed */

    pips_debug(5, "call %p (%s): %"PRIdPTR"\n", c, entity_name(f), is_io);
    Store(current_statement_head(), is_io);
}

static void only_io_unstructured(unstructured u)
{
    intptr_t is_io = 3;
    control c = unstructured_control(u);
    list blocks = NIL;

    CONTROL_MAP(ct, is_io = is_io & Load(control_statement(ct)), c, blocks);
    gen_free_list(blocks);
    pips_debug(5, "unstructured %p: %"PRIdPTR"\n", u, is_io);
    Store(current_statement_head(), (void*)is_io);
}

static statement_mapping 
only_io_mapping(
    statement program,
    statement_mapping map)
{
    stat_bool_map = map;

    make_current_statement_stack();
    gen_multi_recurse(program, 
       statement_domain, current_statement_filter, current_statement_rewrite,
       loop_domain, gen_true, only_io_loop,
       test_domain, gen_true, only_io_test,
       sequence_domain, gen_true, only_io_sequence,
       call_domain, gen_true, only_io_call,
       unstructured_domain, gen_true, only_io_unstructured,
       expression_domain, gen_false, gen_null,
       NULL);
    free_current_statement_stack();
    return stat_bool_map;
}

void 
only_io_mapping_initialize(
    statement program)
{
    debug_on("HPFC_IO_DEBUG_LEVEL");
    set_only_io_map(only_io_mapping(program, MAKE_STATEMENT_MAPPING()));
    debug_off();
}



/********************************************* GENERATION OF IO STATEMENTS */

/*       T_LID=CMP_LID(pn, pi...)
 *     ? init buffer...
 *     /  CALL (type) PACK...
 *       CALL HPFC {SND TO,RCV FROM} NODE(T_LID)
 *     /  CALL (type) UPCK...
 */
static statement hpfc_hmessage(
    entity array,
    entity proc,
    bool send)
{
    entity ld;
    expression lid;
    statement cmp_lid, comm, pack;
    list /* statement */ lp, ls;

    ld = hpfc_name_to_entity(T_LID);
    lid = entity_to_expression(ld);
    cmp_lid = hpfc_compute_lid(ld, proc, (entity (*)())get_ith_processor_dummy, NULL);

    lp = CONS(STATEMENT, cmp_lid, NIL);

    if (!send)
	lp = CONS(STATEMENT, hpfc_buffer_initialization(false, false, false), 
		  lp);

    comm = hpfc_make_call_statement(hpfc_name_to_entity(
	send? HPFC_sH2N: HPFC_rN2H), CONS(EXPRESSION, lid, NIL));

    pack = hpfc_packing_of_current__buffer(array, send);

    if (send)
	ls = CONS(STATEMENT, pack, CONS(STATEMENT, comm, NIL));
    else
	ls = CONS(STATEMENT, comm, CONS(STATEMENT, pack, NIL));

    return make_block_statement(gen_nconc(lp, ls));
}

/*      ! init buffer 
 *        CALL HPFC {RCV FROM HOST, NCAST}
 *        CALL (type) BUFFER UNPACK 
 */
static statement hpfc_nrecv(
    entity array, 
    bool cast) /* from host */
{
    return make_block_statement(
	CONS(STATEMENT, hpfc_buffer_initialization(false, false, false),
	CONS(STATEMENT, hpfc_make_call_statement
	     (hpfc_name_to_entity(cast? HPFC_NCAST: HPFC_rH2N), NIL),
	CONS(STATEMENT, hpfc_packing_of_current__buffer(array, false),
	     NIL))));
}

/*      PVM_SEND(HOST_TID, HOST SND CHANNEL, INFO)
 *      HOST SND CHANNEL += 2
 */
static statement hpfc_nsend(entity array)
{
    return make_block_statement(
	CONS(STATEMENT, hpfc_packing_of_current__buffer(array, true),
	CONS(STATEMENT, hpfc_make_call_statement
	     (hpfc_name_to_entity(HPFC_sN2H), NIL),
	     NIL)));
}

/* static statement hpfc_hcast()
 *
 *       PVM_CAST(NBTASKS, NODETIDS, MCASTHOST, INFO)
 *       MCASTHOST = MCASTHOST + 2
 */
static statement hpfc_hcast(entity array)
{
    return make_block_statement(
	CONS(STATEMENT, hpfc_packing_of_current__buffer(array, true),
	CONS(STATEMENT, 
	     hpfc_make_call_statement(hpfc_name_to_entity(HPFC_HCAST), NIL),
	     NIL)));
}

#define GENERATION(NAME, COLLECT, UPDATE)\
static statement NAME(array, move) entity array; tag move;\
{pips_assert("valid move", \
  move==is_movement_collect || move==is_movement_update);\
 return((move==is_movement_collect) ? (COLLECT) : (UPDATE));}

GENERATION(node_pre_io,
	   hpfc_buffer_initialization(true, false, true),
	   hpfc_nrecv(array, false))
GENERATION(node_in_io,
	   hpfc_buffer_packing(array, (entity (*)())get_ith_local_dummy, true),
	   hpfc_buffer_packing(array, (entity (*)())get_ith_local_dummy, false))
GENERATION(node_post_io,
	   hpfc_nsend(array),
	   make_empty_statement())
GENERATION(host_pre_io,
	   hpfc_hmessage(array, array_to_processors(array), false),
	   hpfc_buffer_initialization(true, false, true))
GENERATION(host_in_io,
	   hpfc_buffer_packing(array, (entity (*)())get_ith_array_dummy, false),
	   hpfc_buffer_packing(array, (entity (*)())get_ith_array_dummy, true))
GENERATION(host_post_io,
	   make_empty_statement(),
	   hpfc_hmessage(array, array_to_processors(array), true))

/* generate_io_statements_for_distributed_arrays
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
    string comment;
    entity
	proc = array_to_processors(array),
	divide = hpfc_name_to_entity(IDIVIDE);
    Psysteme
	/* proc_cond computation:
	 * well, it may have been kept before the new loop bound computation?
	 */
	proc_decl = entity_to_declaration_constraints(proc, 2),
	proc_cond_tmp = sc_dup(proc_echelon),
	proc_cond = (sc_nredund(&proc_cond_tmp),
		     extract_nredund_subsystem(proc_cond_tmp, proc_decl));
    statement
	h_cont = make_empty_statement(),
	n_cont = make_empty_statement(),
	node_tmp = statement_undefined,
	node_defproc = define_node_processor_id(proc, get_ith_processor_dummy),
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

    *psh = make_block_statement(
	CONS(STATEMENT, generate_optional_if(condition, host_proc_loop),
	CONS(STATEMENT, h_cont, NIL)));

    node_tmp = 
	make_block_statement(CONS(STATEMENT, n_pre,
			     CONS(STATEMENT, node_scan_loop,
			     CONS(STATEMENT, n_post,
				  NIL))));
    *psn = make_block_statement(
	CONS(STATEMENT, generate_optional_if
	     (condition, make_block_statement(CONS(STATEMENT, node_defproc,
					      CONS(STATEMENT, 
			 generate_optional_if(proc_cond, node_tmp), NIL)))),
	CONS(STATEMENT, n_cont, NIL)));

    sc_rm(proc_cond_tmp), sc_rm(proc_cond);

    comment = strdup(concatenate("! ", 
				 movement_update_p(move) ? "updating" : "collecting",
				 " distributed variable ",
				 entity_local_name(array), "\n", NULL));

    insert_comments_to_statement(*psh, comment);
    insert_comments_to_statement(*psn, comment);
    free(comment);
    
    comment = strdup(concatenate("! end of ",
				 movement_update_p(move) ? "update" : "collect",
				 "\n", NULL));
    insert_comments_to_statement(h_cont, comment);
    /* Do not forget to move forbidden information associated with
       block: */
    insert_comments_to_statement(n_cont, comment);
    free(comment);
    
    DEBUG_STAT(5, "Host", *psh);
    DEBUG_STAT(5, "Node", *psn);
}

/* generate_io_statements_for_shared_arrays
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
void generate_io_statements_for_shared_arrays(
    entity array,
    tag move,
    Psysteme condition,
    Psysteme echelon,
    list parameters, 
    list scanners, 
    list rebuild,
    statement *psh, 
    statement *psn)
{
    entity divide = hpfc_name_to_entity(IDIVIDE);
    string comment;
    statement
	h_cont = make_empty_statement(),
	n_cont = make_empty_statement(),
	h_pre = hpfc_buffer_initialization(true, false, true),
	h_rebuild = generate_deducables(rebuild),
	h_pack = hpfc_buffer_packing(array, (entity (*)())get_ith_array_dummy, true),
	h_cast = hpfc_hcast(array),
	n_rcv = hpfc_nrecv(array, true),
	n_rebuild = generate_deducables(rebuild),
	n_unpack = hpfc_buffer_packing(array, (entity (*)())get_ith_local_dummy, false),
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

    pips_assert("update", movement_update_p(move));

    *psh = make_block_statement(
	CONS(STATEMENT, generate_optional_if
	     (condition, make_block_statement(CONS(STATEMENT, h_pre,
					      CONS(STATEMENT, h_scan,
					      CONS(STATEMENT, h_cast,
						   NIL))))),
	CONS(STATEMENT, h_cont, NIL)));

    *psn = make_block_statement(
	CONS(STATEMENT, generate_optional_if
	     (condition, make_block_statement(CONS(STATEMENT, n_rcv,
					      CONS(STATEMENT, n_scan,
						   NIL)))),
	CONS(STATEMENT, n_cont, NIL)));

    /*   some comments are generated to help understand the code
     */
    comment = strdup(concatenate("! updating shared variable ",
				 entity_local_name(array), "\n", NULL));
    insert_comments_to_statement(*psh, comment);
    insert_comments_to_statement(*psn, comment);
    free(comment);
    insert_comments_to_statement(h_cont, "! end of update\n");
    insert_comments_to_statement(n_cont, "! end of update\n");
}

/* that is all
 */
