/* HPFC module by Fabien COELHO
 *
 * $RCSfile: remapping.c,v $ version $Revision$
 * ($Date: 1995/04/24 16:34:48 $, ) 
 *
 * generates a remapping code. 
 * debug controlled with HPFC_REMAPPING_DEBUG_LEVEL.
 */

#include "defines-local.h"
#include "conversion.h"

entity CreateIntrinsic(string name); /* in syntax */

static Psysteme generate_remapping_system(src, trg)
entity src, trg;
{
    int ndim = variable_entity_dimension(src);
    Psysteme 
	result = sc_rn(NULL),
	s_src = generate_system_for_distributed_variable(src),
	s_trg = shift_system_to_prime_variables
	    (generate_system_for_distributed_variable(trg)),
	s_equ = generate_system_for_equal_variables
	    (ndim, get_ith_array_dummy, get_ith_array_prime);
    
    DEBUG_SYST(6, concatenate("source ", entity_name(src), NULL), s_src);
    DEBUG_SYST(6, concatenate("target ", entity_name(trg), NULL), s_trg);
    DEBUG_SYST(6, "link", s_equ);

    result = sc_append(result, s_src), sc_rm(s_src);
    result = sc_append(result, s_trg), sc_rm(s_trg);
    result = sc_append(result, s_equ), sc_rm(s_equ);

    return(result);
}

/*   ??? assumes that there are no parameters. what should be the case...
 */
static void remapping_variables(s, a1, a2, pl, plp, pll, plrm, plo)
Psysteme s;
entity a1, a2;
list *pl, 	/* P */
     *plp, 	/* P' */
     *pll, 	/* locals */
     *plrm, 	/* to remove */
     *plo;	/* others */
{
    entity
	t1 = align_template(load_entity_align(a1)),
	p1 = distribute_processors(load_entity_distribute(t1)),
	t2 = align_template(load_entity_align(a2)),
	p2 = distribute_processors(load_entity_distribute(t2));
    int
	a1dim = variable_entity_dimension(a1),
	a2dim = variable_entity_dimension(a2),
	t1dim = variable_entity_dimension(t1),
	t2dim = variable_entity_dimension(t2),
	p1dim = variable_entity_dimension(p1),
	p2dim = variable_entity_dimension(p2);

    /*   processors.
     */
    *pl  = NIL; add_to_list_of_vars(*pl, get_ith_processor_dummy, p1dim);
    *plp = NIL; add_to_list_of_vars(*plp, get_ith_processor_prime, p2dim);

    /*   to be removed.
     */
    *plrm = NIL;
    add_to_list_of_vars(*plrm, get_ith_template_dummy, t1dim);
    add_to_list_of_vars(*plrm, get_ith_template_prime, t2dim);
    add_to_list_of_vars(*plrm, get_ith_array_dummy, a1dim);
    add_to_list_of_vars(*plrm, get_ith_array_prime, a2dim);

    /*   locals.
     */
    *pll = NIL;
    add_to_list_of_vars(*pll, get_ith_local_dummy, a1dim);
    add_to_list_of_vars(*pll, get_ith_local_prime, a2dim);

    /*   others.
     */
    *plo = base_to_list(sc_base(s)),
    gen_remove(plo, (entity) TCST);
    MAPL(ce, gen_remove(plo, ENTITY(CAR(ce))), *pl);
    MAPL(ce, gen_remove(plo, ENTITY(CAR(ce))), *plp);
    MAPL(ce, gen_remove(plo, ENTITY(CAR(ce))), *plrm);
    MAPL(ce, gen_remove(plo, ENTITY(CAR(ce))), *pll);

    ifdebug(7)
    {
	fprintf(stderr, "[remapping_variables] list of variables:");
	fprintf(stderr, "\nP: "); fprint_entity_list(stderr, *pl);
	fprintf(stderr, "\nP': "); fprint_entity_list(stderr, *plp);
	fprintf(stderr, "\nRM: "); fprint_entity_list(stderr, *plrm);
	fprintf(stderr, "\nLOCALS: "); fprint_entity_list(stderr, *pll);
	fprintf(stderr, "\nOTHERS: "); fprint_entity_list(stderr, *plo);
	fprintf(stderr, "\n");
    }
}

/* to be generated:
 *
 *   PSI_i's definitions
 *   [ IF (I AM IN S(PSI_i)) THEN ]
 *     DO OTH_i's in S(OTH_i's)[PSI_i's]
 *       LID computation(OTH_i's)
 *       body
 *     ENDDO
 *   [ ENDIF ]
 */
static statement
processor_loop(s, l_psi, l_oth, psi, oth, lid, create_psi, create_oth, body, sh)
Psysteme s;
list /* of entities */ l_psi, l_oth;
entity psi, oth, lid;
entity  (*create_psi)(/* int */), (*create_oth)(/* int */);
statement body;
boolean sh; /* wether to shift the psi's */
{
    entity divide = hpfc_name_to_entity(IDIVIDE);
    Psysteme condition, enumeration, known, simpler;
    statement
	define_psis = define_node_processor_id(psi, create_psi), 
	compute_lid = hpfc_compute_lid(lid, oth, create_oth),
	oth_loop, if_guard;

    known = sc_dup(entity_to_declaration_constraints(psi));
    if (sh) known = shift_system_to_prime_variables(known);

    DEBUG_SYST(7, "initial system", s);
    DEBUG_ELST(7, "loop indexes", l_oth);

    hpfc_algorithm_row_echelon(s, l_oth, &condition, &enumeration);

    DEBUG_SYST(7, "P condition", condition);

    simpler = extract_nredund_subsystem(condition, known);
    sc_rm(condition), sc_rm(known);

    DEBUG_SYST(5, "P simpler", simpler);
    DEBUG_SYST(5, "P enumeration", enumeration);

    oth_loop = systeme_to_loop_nest(enumeration, l_oth, 
	       make_block_statement(CONS(STATEMENT, compute_lid,
				    CONS(STATEMENT, body,
					 NIL))), divide);
    if_guard = generate_optional_if(simpler, oth_loop);

    sc_rm(simpler); sc_rm(enumeration);

    return(make_block_statement(CONS(STATEMENT, define_psis,
				CONS(STATEMENT, if_guard,
				     NIL))));
}

/* to be generated:
 *
 *   DO ll's in S(ll)[...]
 *     DEDUCABLES(ld)
 *     body
 *   ENDDO
 */
static statement
elements_loop(s, ll, ld, body)
Psysteme s;
list /* of entities */ ll, /* of expressions */ ld;
statement body;
{
    return(systeme_to_loop_nest(s, ll,
	    make_block_statement(CONS(STATEMENT, generate_deducables(ld),
				 CONS(STATEMENT, body,
				      NIL))), 
	    hpfc_name_to_entity(IDIVIDE)));
}

/* to be generated:
 * 
 *   IF (MYLID.NE.LID)
 *   THEN true
 *   ELSE false
 *   ENDIF
 */
static statement
if_different_pe(lid, true, false)
entity lid;
statement true, false;
{
    return(test_to_statement
	   (make_test(MakeBinaryCall(CreateIntrinsic(NON_EQUAL_OPERATOR_NAME),
			     entity_to_expression(hpfc_name_to_entity(MYLID)),
			     entity_to_expression(lid)),
		      true, false)));
}

static statement msg(ld, send)
entity ld;
bool send;
{
    entity 
	nc = hpfc_name_to_entity(send ? SEND_CHANNELS : RECV_CHANNELS),
	nt = hpfc_name_to_entity(NODETIDS);
    expression
	lid = entity_to_expression(ld),
	tid = reference_to_expression
	    (make_reference(nt, CONS(EXPRESSION, lid, NIL))),
	chn = reference_to_expression
	    (make_reference(nc, CONS(EXPRESSION, copy_expression(lid), NIL)));
    return(hpfc_message(tid, chn, send));
}

/* 0 is copy
 * 1 is send
 * 2 is receive
 */
static statement pre(t, lid)
int t;
entity lid;
{
    switch(t)
    {
    case 0: return(make_empty_statement());
    case 1: return(hpfc_initsend());
    case 2: return(msg(lid, FALSE));
    default: return(pips_error("pre", "invalid tag"), statement_undefined);
    }
}

static statement in(t, src, trg, create_src, create_trg)
int t;
entity src, trg;
entity (*create_src)(), (*create_trg)();
{
    switch(t)
    {
    case 0: return(make_assign_statement
		   (make_reference_expression(trg, create_trg),
		    make_reference_expression(src, create_src)));
    case 1: return(hpfc_packing(src, create_src, TRUE));
    case 2: return(hpfc_packing(trg, create_trg, FALSE));
    default: return(pips_error("in", "invalid tag"), statement_undefined);
    }
}

static statement post(t, lid)
int t;
entity lid;
{
    switch(t)
    {
    case 0: return(make_empty_statement());
    case 1: return(msg(lid, TRUE));
    case 2: return(make_empty_statement());
    default: return(pips_error("post", "invalid tag"), statement_undefined);
    }
}

static statement 
remapping_stats(t, s, ll, ld, lid, src, trg)
int t;
Psysteme s;
list /* of entities */ ll, /* of expressions */ ld;
entity lid, src, trg;
{
    statement 
	inner =	in(t, src, trg, get_ith_local_dummy, get_ith_local_prime),
	result = make_block_statement
	    (CONS(STATEMENT, pre(t, lid),
	     CONS(STATEMENT, elements_loop(s, ll, ld, inner),
	     CONS(STATEMENT, post(t, lid),
		  NIL))));

    statement_comments(result) = 
	strdup(concatenate("c - %sing\n",
			   t==0 ? "copy" :
			   t==1 ? "send" :
			   t==2 ? "receiv" : "?", NULL));
}

static statement
generate_remapping_code(src, trg, procs, locals, l, lp, ll, ld)
entity src, trg;
Psysteme procs, locals;
list /* of entities */ l, lp, ll, /* of expressions */ ld;
{
    entity lid = hpfc_name_to_entity(T_LID),
           p_src = array_to_processors(src),
           p_trg = array_to_processors(trg);
    statement 
	remap_copy = remapping_stats(0, locals, ll, ld, lid, src, trg),
	remap_send = remapping_stats(1, locals, ll, ld, lid, src, trg),
	remap_recv = remapping_stats(2, locals, ll, ld, lid, src, trg),
	send = processor_loop
	    (procs, l, lp, p_src, p_trg, lid,
	     get_ith_processor_dummy, get_ith_processor_prime,
	     if_different_pe(lid, remap_send, make_empty_statement()), FALSE),
	recv = processor_loop
	    (procs, lp, l, p_trg, p_src, lid,
	     get_ith_processor_prime, get_ith_processor_dummy,
	     if_different_pe(lid, remap_recv, remap_copy), TRUE),
	result = make_block_statement(CONS(STATEMENT, send,
				      CONS(STATEMENT, recv,
					   NIL)));
    
    DEBUG_STAT(3, "result", result);
    
    return(result);
}

statement hpf_remapping(src, trg)
entity src, trg;
{
    Psysteme p, proc, enume;
    statement s;
    list /* of entities */ l, lp, ll, lrm, lo, left, scanners,
         /* of expressions */ lddc;

    debug(3, "hpf_remapping", "%s -> %s\n", entity_name(src), entity_name(trg));
    
    p = generate_remapping_system(src, trg);
    remapping_variables(p, src, trg, &l, &lp, &ll, &lrm, &lo);

    clean_the_system(&p, &lrm, &lo);

    lddc = simplify_deducable_variables(p, ll, &left);
    gen_free_list(ll);
    scanners = gen_nconc(lo, left);

    DEBUG_SYST(4, "cleaned system", p);
    DEBUG_ELST(4, "scanners", scanners);

    hpfc_algorithm_row_echelon(p, scanners, &proc, &enume);
    sc_rm(p);

    DEBUG_SYST(3, "proc", proc);
    DEBUG_SYST(3, "enum", enume);

    s = generate_remapping_code(src, trg, proc, enume, l, lp, scanners, lddc);
    statement_comments(s) =
	strdup(concatenate("c remapping ", entity_local_name(src), 
			   " -> ", entity_local_name(trg), "\n", NULL));
    
    return(s);
}

/* void remapping_compile(s, hsp, nsp)
 * statement s, *hsp, *nsp;
 *
 * what: generates the remapping code for s.
 * how: polyhedron technique.
 * input: s, the statement.
 * output: statements *hsp and *nsp, the host and SPMD node code.
 * side effects: (node?)
 * bugs or features:
 */
void remapping_compile(s, hsp, nsp)
statement s, *hsp /* Host Statement Pointer */, *nsp /* idem Node */;
{
    list /* of statements */ l = NIL;

    debug_on("HPFC_REMAPPING_DEBUG_LEVEL");
    debug(1, "remapping_compile",
	  "dealing with statement 0x%x\n", (unsigned int) s);

    *hsp = make_continue_statement(entity_empty_label()); /* nothing for host */

    MAPL(cr,
     {
	 renaming r = RENAMING(CAR(cr));

	 l = CONS(STATEMENT,
		  hpf_remapping(renaming_old(r), renaming_new(r)), l);
     },
	 load_renamings(s));

    *nsp = make_block_statement(l); /* block of remaps for the nodes */

    debug_off();
}

/* that is all
 */
