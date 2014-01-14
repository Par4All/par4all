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
 *
 * generates a remapping code. 
 * debug controlled with HPFC_REMAPPING_DEBUG_LEVEL.
 * ??? should drop the renaming domain?
 */

#include "defines-local.h"

#include "conversion.h"
#include "resources.h"
#include "pipsdbm.h" 

/* linearize the processor arrangement on which array is distributed
 * or replicated (driven by distributed). The variables are those created 
 * or returned by the create_var function. var is set as the linearization 
 * result, if not NULL. *psize the the extent of the returned array.
 * Special care is taken about the declaration shifts.
 */
static Pcontrainte
partial_linearization(
    entity array,              /* array variable */
    bool distributed,          /* distributed or replicated dimensions lin. */
    entity var,                /* assigned variable if desired */
    int *psize,                /* returned array extent */
    entity (*create_var)(int)) /* dimension variable builder */
{
    entity proc = array_to_processors(array);
    int dim = NumberOfDimension(proc), low, up, size;
    Pvecteur v = VECTEUR_NUL;

    for(*psize=1; dim>0; dim--)
	if (distributed ^ processors_dim_replicated_p(proc, array, dim))
	{
	    get_entity_dimensions(proc, dim, &low, &up);
	    size = up - low + 1;
	    *psize *= size;
	    v = vect_multiply(v, int_to_value(size));
	    vect_add_elem(&v, (Variable) create_var(dim), VALUE_ONE);
	    vect_add_elem(&v, TCST, int_to_value(-low));
	}

    if (var) vect_add_elem(&v, (Variable) var, VALUE_MONE);

    return contrainte_make(v);
}

/* builds a linarization equation of the dimensions of obj.
 * var is set to the result. *psize returns the size of obj.
 * Done the Fortran way, or the other way around...
 */ 
Pcontrainte
full_linearization(
    entity obj,                /* array being lin. */
    entity var,                /* assigned variable if desired */
    int *psize,                /* returned array extent */
    entity (*create_var)(int), /* dimension variable builder */
    bool fortran_way,          /* Fortran/C linearization way */
    int initial_offset)        /* initial offset if desired */
{
    int dim = NumberOfDimension(obj), low, up, size, i;
    Pvecteur v = VECTEUR_NUL;

    for(*psize=1, i=fortran_way ? dim : 1 ;
	fortran_way ? i>0 : i<=dim; 
	i+= fortran_way ? -1 : 1)
    {
	get_entity_dimensions(obj, i, &low, &up);
	size = up - low + 1;
	*psize *= size;
	v = vect_multiply(v, int_to_value(size));
	vect_add_elem(&v, (Variable) create_var(i), VALUE_ONE);
	vect_add_elem(&v, TCST, int_to_value(-low));
    }

    if (var) vect_add_elem(&v, (Variable) var, VALUE_MONE);
    if (initial_offset) 
	vect_add_elem(&v, TCST, 
		      int_to_value(initial_offset));

    return contrainte_make(v);
}

/* load-balancing equation as suggested in the A-274 report.
 * here some choices are quite arbitrary: 
 *  - the dimension order linearization of both source and target procs.
 *  - the initial offset
 * some clever choice could be suggested so as to minimize the required
 * communications by maximizing the local copies, knowing the actual
 * hpf processors to real processors affectation...
 *
 * should allow a stupid system with some property?
 */
static Psysteme 
generate_work_sharing_system(
    entity src, /* source array */
    entity trg) /* target array */
{
    entity psi_r = get_ith_temporary_dummy(1),
           psi_d = get_ith_temporary_dummy(2),
           delta = get_ith_temporary_dummy(3);
    int size_r, size_d;
    Psysteme sharing = sc_new();

    sc_add_egalite(sharing, partial_linearization(src, false, psi_r, &size_r, 
						  get_ith_processor_dummy));
    sc_add_egalite(sharing, partial_linearization(trg, true, psi_d, &size_d,
						  get_ith_processor_prime));
    
    /* psi_d = psi_r + |psi_r| delta
     */
    sc_add_egalite(sharing, contrainte_make(vect_make
	(VECTEUR_NUL, psi_d, VALUE_MONE, 
	              psi_r, VALUE_ONE, 
	              delta, int_to_value(size_r), 
	              TCST, VALUE_ZERO)));

    if (size_d > size_r)
    {
	/* 0 <= delta (there are cycles)
	 */
	sc_add_inegalite(sharing, contrainte_make
	   (vect_make(VECTEUR_NUL, delta, VALUE_MONE, TCST, VALUE_ZERO)));

	/* delta <= -((size_d-1)/size_r) // not really necessary, but may help.
	 */
	sc_add_inegalite(sharing, contrainte_make
	   (vect_make(VECTEUR_NUL, delta, VALUE_ONE, 
		      TCST, int_to_value(-((size_d-1)/size_r)))));
    }
    else
    {
	/* delta == 0
	 */
	sc_add_egalite(sharing, 
	   contrainte_make(vect_new((Variable) delta, VALUE_ONE)));
    }

    sc_creer_base(sharing);

    return sharing;
}

/* returns the full remapping system, including the source and target
 * mappings systems, the link and the work sharing system 
 */
static Psysteme 
generate_remapping_system(
    entity src, /* source array for the remapping */
    entity trg) /* target array */
{
    int ndim = variable_entity_dimension(src);
    Psysteme 
	result,
	s_src = generate_system_for_distributed_variable(src),
	s_trg = shift_system_to_prime_variables
	    (generate_system_for_distributed_variable(trg)),
	s_equ = generate_system_for_equal_variables
	    (ndim, get_ith_array_dummy, get_ith_array_prime),
	s_shr = generate_work_sharing_system(src, trg);
    
    DEBUG_SYST(6, concatenate("source ", entity_name(src), NULL), s_src);
    DEBUG_SYST(6, concatenate("target ", entity_name(trg), NULL), s_trg);
    DEBUG_SYST(6, "link", s_equ);
    DEBUG_SYST(6, "sharing", s_shr);

    result = s_src;
    result = sc_append(result, s_trg), sc_rm(s_trg);
    result = sc_append(result, s_equ), sc_rm(s_equ);
    result = sc_append(result, s_shr), sc_rm(s_shr);

    return result;
}

/* ??? assumes that there are no parameters. what should be the case...
 * generates the list of variables needed by the code generation.
 */
static void 
remapping_variables(
    Psysteme s, /* full remapping system */
    entity a1,  /* source array */
    entity a2,  /* target array */
    list *pl, 	/* P */
    list *plp, 	/* P' */
    list *pll, 	/* locals */
    list *plrm, /* to remove */
    list *pld,  /* diffusion processor variables */
    list *plo)	/* others */
{
    entity
	t1 = array_to_template(a1),
	p1 = template_to_processors(t1),
	t2 = array_to_template(a2),
	p2 = template_to_processors(t2);
    int
	a1dim = variable_entity_dimension(a1),
	a2dim = variable_entity_dimension(a2),
	t1dim = variable_entity_dimension(t1),
	t2dim = variable_entity_dimension(t2),
	p1dim = variable_entity_dimension(p1),
	p2dim = variable_entity_dimension(p2), i;

    /*   processors.
     */
    *pl  = NIL; add_to_list_of_vars(*pl, get_ith_processor_dummy, p1dim);
    *plp = NIL; add_to_list_of_vars(*plp, get_ith_processor_prime, p2dim);

    *pld = NIL; 
    for (i=p2dim; i>0; i--)
	if (processors_dim_replicated_p(p2, a2, i))
	    *pld = CONS(ENTITY, get_ith_processor_prime(i), *pld);

    /*   to be removed.
     */
    *plrm = NIL;
    add_to_list_of_vars(*plrm, get_ith_template_dummy, t1dim);
    add_to_list_of_vars(*plrm, get_ith_template_prime, t2dim);
    add_to_list_of_vars(*plrm, get_ith_array_dummy, a1dim);
    add_to_list_of_vars(*plrm, get_ith_array_prime, a2dim);

    /*    corresponding equations generated in the sharing system
     */
    add_to_list_of_vars(*plrm, get_ith_temporary_dummy, 2);
    

    /*    Replicated dimensions associated variables must be removed.
     *    A nicer approach would have been not to generate them, but
     * it's not so hard to remove them, and the equation generation is
     * kept simpler this way. At least in my own opinion:-)
     */
    for (i=p1dim; i>0; i--)
	if (processors_dim_replicated_p(p1, a1, i))
	    *plrm = CONS(ENTITY, get_ith_block_dummy(i), 
		    CONS(ENTITY, get_ith_cycle_dummy(i), *plrm));

    for (i=p2dim; i>0; i--)
	if (processors_dim_replicated_p(p2, a2, i))
	    *plrm = CONS(ENTITY, get_ith_block_prime(i), 
		    CONS(ENTITY, get_ith_cycle_prime(i), *plrm));

    /*   locals.
     */
    *pll = NIL;
    add_to_list_of_vars(*pll, get_ith_local_dummy, a1dim);
    add_to_list_of_vars(*pll, get_ith_local_prime, a2dim);

    /*   others.
     */
    *plo = base_to_list(sc_base(s)); 
    gen_remove(plo, (entity) TCST);

    MAP(ENTITY, e, gen_remove(plo, e), *pl);
    MAP(ENTITY, e, gen_remove(plo, e), *plp);
    MAP(ENTITY, e, gen_remove(plo, e), *plrm);
    MAP(ENTITY, e, gen_remove(plo, e), *pll);

    DEBUG_ELST(7, "P", *pl);
    DEBUG_ELST(7, "P'", *plp);
    DEBUG_ELST(7, "RM", *plrm);
    DEBUG_ELST(7, "LOCALS", *pll);
    DEBUG_ELST(7, "DIFFUSION", *pld);
    DEBUG_ELST(7, "OTHERS", *plo);
}

/* to be generated:
 * ??? the Proc cycle should be deduce directly in some case...
 *
 *   PSI_i's definitions
 *   [ IF (I AM IN S(PSI_i)) THEN ]
 *     DO OTH_i's in S(OTH_i's)[PSI_i's]
 *       LID computation(OTH_i's) // if LID is not NULL!
 *       body
 *     ENDDO
 *   [ ENDIF ]
 */
static statement
processor_loop(
    Psysteme s,                   /* system of comm. couples of processors */
    list /* of entities */ l_psi, /* processor local dimensions */
    list /* of entities */ l_oth, /* communicating proc. dimensions */
    entity psi,                   /* local processor arrangement */
    entity oth,                   /* communicating processor arrangement */
    entity lid,                   /* variable for the comm. proc local id */
    entity array,                 /* array being remapped */
    entity (*create_psi)(),    /* to create a local proc. dim. */
    entity (*create_oth)(),    /* to create a comm. proc. dim. */
    statement body,               /* loop body */
    bool sh)                   /* whether to shift the psi's */
{
    entity divide = hpfc_name_to_entity(IDIVIDE);
    Psysteme condition, enumeration, known, simpler;
    statement define_psis, compute_lid, oth_loop, if_guard;
    
    define_psis = define_node_processor_id(psi, create_psi);

    /* the lid computation is delayed in the body for broadcasts.
     */
    compute_lid = (lid) ?
	hpfc_compute_lid(lid, oth, create_oth, array) : 
	make_empty_statement(); 

    /* simplifies the processor arrangement for the condition
     */
    known = sc_dup(entity_to_declaration_constraints(psi, 2));
    if (sh) known = shift_system_to_prime_variables(known);

    DEBUG_SYST(7, "initial system", s);
    DEBUG_ELST(7, "loop indexes", l_oth);

    hpfc_algorithm_row_echelon(s, l_oth, &condition, &enumeration);

    DEBUG_SYST(7, "P condition", condition);

    simpler = extract_nredund_subsystem(condition, known);
    sc_rm(condition), sc_rm(known);

    /*  the processor must be in the psi processor arrangement
     */
    sc_add_inegalite(simpler, 
		     contrainte_make
		     (vect_make(VECTEUR_NUL,
				hpfc_name_to_entity(MYLID), VALUE_ONE, 
				TCST, int_to_value(- element_number
      (variable_basic(type_variable(entity_type(psi))),
       variable_dimensions(type_variable(entity_type(psi))))))));

    DEBUG_SYST(5, "P simpler", simpler);
    DEBUG_SYST(5, "P enumeration", enumeration);

    /* target processors enumeration loop
     */
    oth_loop = systeme_to_loop_nest(enumeration, l_oth, 
	       make_block_statement(CONS(STATEMENT, compute_lid,
				    CONS(STATEMENT, body,
					 NIL))), divide);

    if_guard = generate_optional_if(simpler, oth_loop);

    sc_rm(simpler); sc_rm(enumeration);

    return make_block_statement(CONS(STATEMENT, define_psis,
				CONS(STATEMENT, if_guard,
				     NIL)));
}

/* to be generated:
 *
 *   DO ll's in S(ll)[...]
 *     DEDUCABLES(ld)
 *     body
 *   ENDDO
 */
static statement
elements_loop(
    Psysteme s,                   /* triangular bounds on ll */
    list /* of entities */ ll,    /* loop indexes */
    list /* of expressions */ ld, /* deduced scalars */
    statement body)               /* loop body */
{
    return systeme_to_loop_nest(s, ll,
	   make_block_statement(CONS(STATEMENT, generate_deducables(ld),
				CONS(STATEMENT, body,
				     NIL))), 
				hpfc_name_to_entity(IDIVIDE));
}

static expression 
mylid_ne_lid(entity lid)
{
    return ne_expression(entity_to_expression(hpfc_name_to_entity(MYLID)),
			 entity_to_expression(lid));
}

/* to be generated:
 * IF (MYLID.NE.LID[.AND.NOT.HPFC_TWIN_P(an, LID)])
 * THEN true 
 * ELSE false
 * ENDIF
 */
static statement 
if_different_pe_and_not_twin(
    entity src,      /* source array processor */
    entity lid,      /* process local id variable */
    statement strue,  /* then statement */
    statement sfalse) /* else statement */
{
    expression cond = mylid_ne_lid(lid);

    if (get_bool_property("HPFC_GUARDED_TWINS") && replicated_p(src))
    {
	list /* of expression */ largs;
	expression not_twin;

	largs = CONS(EXPRESSION, int_to_expression(load_hpf_number(src)),
		CONS(EXPRESSION, entity_to_expression(lid), NIL));
	not_twin = not_expression(call_to_expression
	    (make_call(hpfc_name_to_entity(TWIN_P), largs)));

	cond = and_expression(cond, not_twin);
    }

    return test_to_statement(make_test(cond, strue, sfalse));
}

/* builds the diffusion loop.
 *
 * [ IF (LAZY_SEND) THEN ]
 *   DO ldiff in sr
 *     LID computation(...)
 *     IF (MYLID.NE.LID) send to LID
 *   ENDDO
 * [ ENDIF ]
 */
static statement 
broadcast(
    entity src,			/* source array */
    entity lid,                 /* variable to store the target local id */
    entity proc,                /* target processors for the broadcast */
    Psysteme sr,                /* broadcast polyhedron */
    list /* of entity */ ldiff, /* dimension variables for the broadcast */
    bool lazy)                  /* whether to send empty messages */
{
    statement cmp_lid, body, nest;

    cmp_lid = hpfc_compute_lid(lid, proc, (entity(*)())get_ith_processor_prime, NULL);
    body = make_block_statement
	(CONS(STATEMENT, cmp_lid,
	 CONS(STATEMENT, if_different_pe_and_not_twin
	      (src, lid, hpfc_generate_message(lid, true, false),
	       make_empty_statement()),
	      NIL)));

    nest = systeme_to_loop_nest(sr, ldiff, body, hpfc_name_to_entity(IDIVIDE));

    return lazy ? hpfc_lazy_guard(true, nest) : nest ;
}

/* in the following functions tag t controls the code generation, 
 * depending of what is to be generated (send, receive, copy, broadcast)...
 * tag t may have the following values:
 */

#define CPY	0
#define SND	1
#define RCV	2
#define BRD     3

#define PRE 	0
#define INL 	4
#define PST 	8

#define NLZ 	0
#define LZY 	16

#define NBF 	0
#define BUF 	32

/* I have to deal with a 4D space to generate some code:
 *
 * buffer/nobuf
 * lazy/nolazy
 *
 * pre/in/post
 * cpy/snd/rcv/brd
 */

#define ret(name) result = name; break

/* arguments: all that may be useful to generate some code */
static statement 
gen(int what,
    entity src, entity trg, entity lid, entity proc,
    entity (*create_src)(), entity (*create_trg)(),
    Psysteme sr, list /* of entity */ ldiff)
{
    statement result;
    bool is_lazy = what & LZY, 
         is_buff = what & BUF;

    switch (what)
    {
	/* cpy pre/post
	 * rcv post nbuf
	 */
    case CPY+PRE: 
    case CPY+PST:
    case CPY+PRE+LZY: 
    case CPY+PST+LZY:
    case CPY+PRE+BUF:
    case CPY+PST+BUF:
    case CPY+PRE+LZY+BUF: 
    case CPY+PST+LZY+BUF:
    case RCV+PST: 
    case RCV+PST+LZY:
    case RCV+PST+BUF:
    case RCV+PST+LZY+BUF:
	ret(make_empty_statement());

	/* snd.brd pre nbuf
	 */
    case SND+PRE: 
    case BRD+PRE:
    case SND+PRE+LZY: 
    case BRD+PRE+LZY: 
	ret(hpfc_initsend(is_lazy));

    case RCV+PRE+LZY:
	ret(set_logical(hpfc_name_to_entity(LAZY_RECV), true));

    case RCV+PRE:
	ret(hpfc_generate_message(lid, false, false));

	/* cpy inl 
	 */
    case CPY+INL: 
    case CPY+INL+LZY:
    case CPY+INL+BUF:
    case CPY+INL+LZY+BUF:
	ret(make_assign_statement(make_reference_expression(trg, create_trg),
				  make_reference_expression(src, create_src)));

	/* snd/brd inl nbf
	 */
    case SND+INL:
    case BRD+INL:
    case SND+INL+LZY:
    case BRD+INL+LZY:
	ret(hpfc_lazy_packing(src, lid, create_src, true, is_lazy));

    case SND+INL+BUF:
    case SND+INL+LZY+BUF:
    case BRD+INL+BUF:
    case BRD+INL+LZY+BUF:
	ret(hpfc_lazy_buffer_packing(src, trg, lid, proc, create_src,
				     true /* send! */, is_lazy));
    case RCV+INL+BUF:
    case RCV+INL+LZY+BUF:
	ret(hpfc_lazy_buffer_packing(src, trg, lid, proc, create_trg,
				     false /* receive! */, is_lazy));
    case RCV+INL:
    case RCV+INL+LZY:
	ret(hpfc_lazy_packing(trg, lid, create_trg, false, is_lazy));

    case SND+PST:
    case SND+PST+LZY:
	ret(hpfc_generate_message(lid, true, is_lazy));

    case BRD+PST:
    case BRD+PST+LZY:
	ret(broadcast(src, lid, proc, sr, ldiff, is_lazy));

    case SND+PST+BUF:
    case SND+PST+LZY+BUF:
    case BRD+PST+BUF:
    case BRD+PST+LZY+BUF:
	ret(hpfc_broadcast_if_necessary(src, trg, lid, proc, is_lazy));

	/* snd/rcv pre buf
	 */
    case SND+PRE+BUF:
    case SND+PRE+LZY+BUF:
    case BRD+PRE+BUF:
    case BRD+PRE+LZY+BUF:
	ret(hpfc_buffer_initialization(true /* send! */, is_lazy, true));
    case RCV+PRE+BUF:
    case RCV+PRE+LZY+BUF:
	ret(hpfc_buffer_initialization(false /* receive! */, is_lazy, true));
	
	/* default is a forgotten case, I guess
	 */
    default:
	pips_internal_error("invalid tag %d", what);
	ret(statement_undefined); /* to avoid a gcc warning */
    }

    pips_debug(7, "tag is %d (%s, %s)\n", what, 
	       is_lazy ? "lazy" : "not lazy",
	       is_buff ? "bufferized" : "not bufferized");
    DEBUG_STAT(7, "returning", result);

    return result;
}

/* generates a remapping loop nest (copy/send/receive/broadcast)
 * basically a loop on elements with pre/in/post statements...
 */
static statement 
remapping_stats(
    int t,                        /* tag: CPY/SND/RCV/BRD */
    Psysteme s,                   /* elements polyhedron */
    Psysteme sr,                  /* broadcast polyhedron */
    list /* of entity */ ll,      /* indexes for element loops */
    list /* of entity */ ldiff,   /* indexes for broadcasts */
    list /* of expressions */ ld, /* deducable elements */
    entity lid,                   /* lid variable to be used */
    entity src,                   /* source array */
    entity trg)                   /* target array */
{
    entity proc = array_to_processors(trg);
    statement inner, prel, postl, result;
    bool is_buffer, is_lazy;
    int what;

    is_lazy = get_bool_property(LAZY_MESSAGES);
    is_buffer = get_bool_property(USE_BUFFERS);
    what = (is_lazy ? LZY : NLZ) + (is_buffer ? BUF : NBF) + t;
    
    prel  = gen(what+PRE, src, trg, lid, proc, 
		get_ith_local_dummy, get_ith_local_prime, sr, ldiff);
    inner = gen(what+INL, src, trg, lid, proc, 
		get_ith_local_dummy, get_ith_local_prime, sr, ldiff);
    postl = gen(what+PST, src, trg, lid, proc, 
		get_ith_local_dummy, get_ith_local_prime, sr, ldiff);

    result = make_block_statement
	    (CONS(STATEMENT, prel,
	     CONS(STATEMENT, elements_loop(s, ll, ld, inner),
	     CONS(STATEMENT, postl,
		  NIL))));

    {
	/* comments are added to help understand the generated code
	 */
	string comment = strdup(concatenate("! - ", (what & LZY) ? "lazy " : "",
	       t==CPY ? "copy" : t==SND ? "send" : t==RCV ? "receiv" :
	       t==BRD ? "broadcast" : "?",  "ing\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    return result;
}


/* generates a full remapping code, given the systems and indexes
 * to be used in the different loops. that is complementary send/broadcast
 * and receive/copy. The distribution of target onto source processors
 * may require special care if lambda. 
 *
 * output: 
 * AS a source
 *   DO target
 *     DO element
 *       PACK 
 *     SEND/BROADCAST to target(s)
 * AS a target
 *   DO source
 *     DO element RECEIVE/UNPACK
 *     OR 
 *     DO element COPY
 */
static statement
generate_remapping_code(
    entity src,                  /* source array */
    entity trg,                  /* target array */
    Psysteme procs,              /* communicating processors polyhedron */
    Psysteme locals,             /* elements polyhedron */
    list /* of entity */ l,      /* source processor indexes */
    list lp,                     /* target processor indexes */
    list ll,                     /* element indexes */
    list ldiff,                  /* broadcast target processor indexes */
    list /* of expression */ ld, /* deducable elements */
    bool dist_p)                 /* true if must take care of lambda */
{
    entity lid = hpfc_name_to_entity(T_LID),
           p_src = array_to_processors(src),
           p_trg = array_to_processors(trg),
           lambda = get_ith_temporary_dummy(3),
           primary = load_primary_entity(src);
    statement copy, recv, send, receive, cont, result;
    bool is_buffer = get_bool_property(USE_BUFFERS);

    pips_debug(3, "%s taking care of processor cyclic distribution\n", 
	       dist_p ? "actually" : "not");

    copy = remapping_stats(CPY, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);
    recv = remapping_stats(RCV, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);

    if (dist_p) lp = CONS(ENTITY, lambda, lp);

    /* the send is different for diffusions
     */
    if (ENDP(ldiff))
    {
	statement rp_send;

	rp_send =
	    remapping_stats(SND, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);
    
	send = processor_loop
	    (procs, l, lp, p_src, p_trg, lid, NULL,
	     get_ith_processor_dummy, get_ith_processor_prime,
	     if_different_pe_and_not_twin(src, lid, rp_send, 
					  make_empty_statement()), false);
    }
    else
    {
	Pbase b = entity_list_to_base(ldiff);
	list lpproc = gen_copy_seq(lp);
	statement diff;
	Psysteme 
	    sd /* distributed */, 
	    sr /* replicated */;

	MAP(ENTITY, e, gen_remove(&lpproc, e), ldiff);

	/* polyhedron separation to extract the diffusions.
	 */
	sc_separate_on_vars(procs, b, &sr, &sd);
	base_rm(b);

	diff = remapping_stats(BRD, locals, sr, ll, ldiff, ld, lid, src, trg);

	send = processor_loop
	    (sd, l, lpproc, p_src, p_trg, lid, is_buffer ? trg : NULL,
	     get_ith_processor_dummy, get_ith_processor_prime,
	     diff, false);

	gen_free_list(lpproc); sc_rm(sd); sc_rm(sr);
    }

    if (dist_p) 
    {
	gen_remove(&lp, lambda);
	l = gen_nconc(l, CONS(ENTITY, lambda, NIL)); /* ??? to be deduced */
    }

    receive = processor_loop
	(procs, lp, l, p_trg, p_src, lid, NULL,
	 get_ith_processor_prime, get_ith_processor_dummy,
	 if_different_pe_and_not_twin(src, lid, recv, copy), true);

    if (dist_p) gen_remove(&l, lambda);

    cont = make_empty_statement();

    result = make_block_statement(CONS(STATEMENT, send,
				  CONS(STATEMENT, receive,
				  CONS(STATEMENT, cont,
				       NIL))));

    /*   some comments to help understand the generated code
     */
    {
	char *buffer;
	
	asprintf(&buffer, "! remapping %s[%"PRIdPTR"]: %s[%"PRIdPTR"] -> %s[%"PRIdPTR"]\n",
		entity_local_name(primary), load_hpf_number(primary),
		entity_local_name(src), load_hpf_number(src),
		entity_local_name(trg), load_hpf_number(trg));
	
	insert_comments_to_statement(result, buffer);
    free(buffer);
	insert_comments_to_statement(send, "! send part\n");
	insert_comments_to_statement(receive, "! receive part\n");
	insert_comments_to_statement(cont, "! end of remapping\n");
    }
    
    DEBUG_STAT(3, "result", result);
    
    return result;
}

/* returns LIVEMAPPING(index) 
 */
static expression 
live_mapping_expression(int index)
{
    entity live_status = hpfc_name_to_entity(LIVEMAPPING);
    return reference_to_expression(make_reference
        (live_status, CONS(EXPRESSION, int_to_expression(index), NIL)));
}

static statement 
set_array_status_to_target(entity trg)
{
    int trg_n = load_hpf_number(trg),
        prm_n = load_hpf_number(load_primary_entity(trg));
    entity m_status = hpfc_name_to_entity(MSTATUS); /* mapping status */
    expression m_stat_ref;

    /* MSTATUS(primary_n) */
    m_stat_ref =
	reference_to_expression
	    (make_reference(m_status, 
			    CONS(EXPRESSION, int_to_expression(prm_n), NIL)));

    return make_assign_statement(m_stat_ref, int_to_expression(trg_n));
}

static statement
set_live_status(
    entity trg,
    bool val)
{
    int trg_n = load_hpf_number(load_similar_mapping(trg));

    return make_assign_statement(live_mapping_expression(trg_n),
				 bool_to_expression(true));
}

static statement 
update_runtime_for_remapping(entity trg)
{
    statement s = set_live_status(trg, true);

    {
	string comment =
	    strdup(concatenate("! direct remapping for ", 
			       entity_local_name(load_primary_entity(trg)), "\n", NULL));
	insert_comments_to_statement(s, comment);
	free(comment);
    }

    return make_block_statement
	(CONS(STATEMENT, s,
         CONS(STATEMENT, set_array_status_to_target(trg), NIL)));
}

/* Runtime descriptors management around the remapping code.
 * performs the remapping if reaching mapping is ok, and update the 
 * mapping status.
 *
 * IF (MSTATUS(primary_number).eq.src_number) THEN
 *  [ IF (.not.LIVEMAPPING(target_number)]) THEN ]
 *     the_code
 *  [ ENDIF ]
 *  [ LIVEMAPPING(target_number) = .TRUE. ]
 *   MSTATUS(primary_number) = trg_number
 * ENDDIF
 */
static statement
generate_remapping_guard(
    entity src,         /* source array */
    entity trg,         /* target array */ 
    statement the_code) /* remapping code */
{
    int src_n = load_hpf_number(src), /* source, target and primary numbers */
        trg_n = load_hpf_number(trg),
        prm_n = load_hpf_number(load_primary_entity(src));
    entity m_status = hpfc_name_to_entity(MSTATUS); /* mapping status */
    expression m_stat_ref, cond;
    statement result;
    list /* of statement */ l = NIL;
    
    /* MSTATUS(primary_n) */
    m_stat_ref =
	reference_to_expression
	    (make_reference(m_status, 
			    CONS(EXPRESSION, int_to_expression(prm_n), NIL)));
    
    /* MSTATUS(primary_number) = trg_number */
    l = CONS(STATEMENT, set_array_status_to_target(trg), l);

    /* MSTATUS(primary_number).eq.src_number */
    cond = eq_expression(m_stat_ref, int_to_expression(src_n));

    /* checks whether alive or not */
    if (get_bool_property("HPFC_DYNAMIC_LIVENESS"))
    {
	expression live_cond =
	    not_expression(live_mapping_expression(trg_n));

	the_code =  test_to_statement(make_test
	    (live_cond, the_code, make_empty_statement()));

	l = CONS(STATEMENT,
		 make_assign_statement(live_mapping_expression(trg_n),
				   bool_to_expression(true)), l);
    }

    result = test_to_statement(make_test(cond, 
      make_block_statement(CONS(STATEMENT, the_code, l)),
      make_empty_statement()));

    return result;
}

static statement 
generate_all_liveness_but(
    entity primary,
    bool val, 
    entity butthisone)
{
    statement result;
    list /* of statement */ ls = NIL;

    MAP(ENTITY, array,
    {
	/* LIVEMAPPING(array) = val (.TRUE. or .FALSE.)
	 */
	if (array != butthisone)
	    ls = CONS(STATEMENT, 
		      make_assign_statement
		      (live_mapping_expression(load_hpf_number(array)),
		       bool_to_expression(val)), ls);
    },
	entities_list(load_dynamic_hpf(primary)));

    /* commented result
     */
    result = make_block_statement(ls);
    {
	string comment =
	    strdup(concatenate("! all livenesss for ", 
			       entity_local_name(primary), "\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    return result;
}

statement 
generate_all_liveness(
    entity primary,
    bool val)
{
    return generate_all_liveness_but(primary, val, entity_undefined);
}

static statement 
generate_dynamic_liveness_for_primary(
    entity primary, 
    list /* of entity */ tokeep)
{
    statement result;
    list /* of statement */ ls = NIL;

    /* clean not maybeuseful instances of the primary
     */
    MAP(ENTITY, array,
    {
	if (!gen_in_list_p(array, tokeep))
	{
	    /* LIVEMAPPING(array) = .FALSE.
	     */
	    ls = CONS(STATEMENT, 
	      make_assign_statement
                  (live_mapping_expression(load_hpf_number(array)),
		   bool_to_expression(false)), ls);
	}	    
    },
	entities_list(load_dynamic_hpf(primary)));

    /* commented result
     */
    result = make_block_statement(ls);
    if (ls) {
	string comment =
	    strdup(concatenate("! clean live set for ", 
			       entity_local_name(primary), "\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    return result;
}

static statement
generate_dynamic_liveness_management(statement s)
{
    statement result;
    list /* of entity */ already_seen = NIL,
                         tokeep = entities_list(load_maybeuseful_mappings(s)),
         /* of statement */ ls;
    
    result = make_empty_statement();
    {
	string comment =
	    strdup(concatenate("! end of liveness management\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    ls = CONS(STATEMENT, result, NIL);

    /* for each primary remapped at s, generate the management code.
     */
    MAP(ENTITY, array,
    {
	entity primary = load_primary_entity(array);

	if (!gen_in_list_p(primary, already_seen))
	{
	    ls = CONS(STATEMENT, 
		      generate_dynamic_liveness_for_primary(primary, tokeep),
		      ls);
	    already_seen = CONS(ENTITY, primary, already_seen);
	}
    },
	tokeep);

    /* commented result 
     */
    result = make_block_statement(ls);
    {
	string comment =
	    strdup(concatenate("! liveness management\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    return result;
}

/* remaps src to trg.
 * first builds the equation and needed lists of indexes,
 * then call the actual code generation phase.
 */
static statement 
hpf_remapping(
    entity src,
    entity trg)
{
    Psysteme p, proc, enume;
    statement s;
    entity lambda = get_ith_temporary_dummy(3) ; /* P cycle */
    bool proc_distribution_p;
    list /* of entities */ l, lp, ll, lrm, ld, lo, left, scanners,
         /* of expressions */ lddc;
    bool time_remapping = get_bool_property("HPFC_TIME_REMAPPINGS");

    pips_debug(1, "%s -> %s\n", entity_name(src), entity_name(trg));

    if (src==trg) /* (optimization:-) */
	return make_empty_statement();

    if (time_remapping) push_performance_spy();

    /*   builds and simplifies the systems.
     */
    p = generate_remapping_system(src, trg);
    set_information_for_code_optimizations(p);

    remapping_variables(p, src, trg, &l, &lp, &ll, &lrm, &ld, &lo);

    /* it is not obvious to decide where to place
     * the cleaning, the equalities detection and the deducables...
     */

    clean_the_system(&p, &lrm, &lo);
    DEBUG_SYST(4, "cleaned system", p);

    if (get_bool_property("HPFC_EXTRACT_EQUALITIES"))
    {
	sc_find_equalities(&p);
	DEBUG_SYST(4, "more equalities", p);
    }

    lddc = simplify_deducable_variables(p, ll, &left);
    gen_free_list(ll);
    DEBUG_SYST(4, "without deducables system", p);
    
    /* the P cycle ?
     */
    proc_distribution_p = gen_in_list_p(lambda, lo);
    if (proc_distribution_p) gen_remove(&lo, lambda);
    scanners = gen_nconc(lo, left);

    DEBUG_ELST(4, "scanners", scanners);

    /* processors/array elements separation.
     *
     * ??? why row echelon? why not... it is not too bad a projection!
     * actually, what I want is to separate processors and elements. 
     * Then I would like to perform some manipulations on the systems
     * to improve them, extracting deducables and lattices... 
     * but the conservation of the optimized order is not obvious... 
     * May/should I give it up in some cases? Well, I guess so.
     *
     * What's missing:
     *  - extraction of the lattice if equalities remain in a system.
     *    1 Hermite transformation + variable list update...
     *  - also some deducables could be extracted once the code is transformed.
     *  - Q: what about variables which were kept althought not desired?
     */
    hpfc_algorithm_row_echelon(p, scanners, &proc, &enume);
    sc_rm(p);

    sc_transform_ineg_in_eg(proc);
    sc_transform_ineg_in_eg(enume);

    if (sc_egalites(proc))
	hpfc_warning("lattice extraction not implemented (proc)\n");
    /* {
	list ns = NIL;
	extract_lattice(proc, lp, &ns, &lp.. */


    if (sc_egalites(enume))
    {
	list ns = NIL;
	extract_lattice(enume, scanners, &ns, &lddc);
	gen_free_list(scanners), scanners=ns;
    }

    DEBUG_SYST(3, "proc", proc);
    DEBUG_SYST(3, "enum", enume);

    /*   generates the code.
     */
    s = generate_remapping_guard(src, trg, generate_remapping_code
      (src, trg, proc, enume, l, lp, scanners, ld, lddc, proc_distribution_p));

    /*   clean.
     */
    sc_rm(proc), sc_rm(enume);
    gen_free_list(scanners);
    gen_free_list(l), gen_free_list(lp), gen_free_list(ld);
    gen_map((gen_iter_func_t)gen_free, lddc), gen_free_list(lddc);  /* ??? */

    reset_information_for_code_optimizations();
    
    if (time_remapping) 
	pop_performance_spy(stderr, 
	concatenate("remapping ", 
		    entity_name(src), " -> ",
		    entity_name(trg), NULL));

    DEBUG_STAT(6, "result", s);

    return s;
}

/* file name for storing the remapping code.
 * {module}_{array}_{src}_{trg}_node.h
 */
static string
remapping_file_name(
    renaming remap)
{
    char *buffer;
    entity src = renaming_old(remap), trg = renaming_new(remap);
    string module, array;

    module = strdup(entity_module_name(src));
    array  = strdup(entity_local_name(load_primary_entity(src)));

    asprintf(&buffer, "%s_%s_%"PRIdPTR"_%"PRIdPTR"_node.h", module, array,
	    load_hpf_number(src), load_hpf_number(trg));

    free(module);
    free(array);

    return buffer;
}

/* quick recursion to find the entities referenced in a statement.
 * the list is allocated and returned.
 */
static list /* of entity */ l_found;
static void loop_rwt(loop l)
{ l_found = gen_once(loop_index(l), l_found);}
static void reference_rwt(reference r)
{ l_found = gen_once(reference_variable(r), l_found);}

static list
list_of_referenced_entities(statement s)
{
    l_found = NIL;
    gen_multi_recurse(s, loop_domain, gen_true, loop_rwt,
                         reference_domain, gen_true, reference_rwt, NULL);
    return l_found;
}

static text
protected_text_statement(statement s)
{
    text t;
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    t = text_statement(entity_undefined, 0, s, NIL);
    debug_off();
    return t;
}

static void
generate_hpf_remapping_file(renaming r)
{
    string file_name, dir;
    FILE * f;
    statement remap;
    text t;
    list /* of entity */ l;
    entity src = renaming_old(r), trg = renaming_new(r);

    pips_debug(1, "%s -> %s\n",
	       entity_name(src), entity_name(trg));

    /* generates the remapping code and text
     * !!! generated between similar arrays...
     */
    remap = hpf_remapping(load_similar_mapping(src),
			  load_similar_mapping(trg));
    update_object_for_module(remap, node_module);
    t = protected_text_statement(remap);

    /* stores the remapping as computed
     */
    l = list_of_referenced_entities(remap);
    add_remapping_as_computed(r, l);

    /* put it in a file
     */
    dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    string rfn = remapping_file_name(r);
    asprintf(&file_name,"%s/%s",dir,rfn);
    free(rfn);
    free(dir);

    f = hpfc_fopen(file_name);
    print_text(f, t);
    hpfc_fclose(f, file_name);

    free_text(t);
    free(file_name);
    free_statement(remap);
    gen_free_list(l);
}

/* just a hack because pips does not have 'include'
 */
static statement
generate_remapping_include(renaming r)
{
    statement result;

    result = make_empty_statement();
    {
	string comment =
	    strdup(concatenate("      include '",
			       remapping_file_name(r), "'\n", NULL));
	insert_comments_to_statement(result, comment);
	free(comment);
    }
    return result;
}

/* returns the initialization statement:
 * must initialize the status and liveness of arrays
 */
statement
root_statement_remapping_inits(
    statement root)
{
    list /* of statement */ ls = NIL;
    list /* of entity */ le =
	list_of_distributed_arrays_for_module(get_current_module_entity());

    /* LIVENESS(...) = .TRUE.
     * STATUS(...) = ...
     */
    MAP(RENAMING, r,
	ls = CONS(STATEMENT, update_runtime_for_remapping(renaming_new(r)),ls),
	load_renamings(root));

    /* LIVENESS(...) = .FALSE.
     */
    MAP(ENTITY, array,
	if (bound_dynamic_hpf_p(array) && primary_entity_p(array))
	    ls = CONS(STATEMENT, generate_all_liveness(array, false), ls),
	le);

    gen_free_list(le), le = NIL;
    
    return make_block_statement(ls);
}

/* void remapping_compile(s, hsp, nsp)
 * statement s, *hsp, *nsp;
 *
 * what: generates the remapping code for s.
 * how: polyhedron technique.
 * input: s, the statement.
 * output: statements *hsp and *nsp, the host and SPMD node code.
 * side effects: (none?)
 * bugs or features:
 */
void 
remapping_compile(
    statement s,      /* initial statement in the source code */
    statement *hsp,   /* Host Statement Pointer */
    statement *nsp)   /* idem Node */
{
    statement tmp;
    list /* of statements */ l = NIL;
    
    debug_on("HPFC_REMAPPING_DEBUG_LEVEL");
    what_stat_debug(1, s);

    *hsp = make_empty_statement(); /* nothing for host */

    /* comment at the end
     */
    tmp = make_empty_statement();
    {
	string comment = strdup(concatenate("! end remappings\n", NULL));
	insert_comments_to_statement(tmp, comment);
	free(comment);
    }
    l = CONS(STATEMENT, tmp, l);

    /* dynamic liveness management if required
     */
    if (get_bool_property("HPFC_DYNAMIC_LIVENESS"))
    {
	l = CONS(STATEMENT, generate_dynamic_liveness_management(s), l);
    }

    /* remapping codes (indirect thru include)
     */
    MAP(RENAMING, r,
    {
	pips_debug(7, "remapping %s -> %s\n", entity_name(renaming_old(r)),
		   entity_name(renaming_new(r)));
	if (renaming_old(r)==renaming_new(r)) /* KILL => status update */
	    l = CONS(STATEMENT, set_array_status_to_target(renaming_new(r)),
		     l);
	else 
	{
	    if (!remapping_already_computed_p(r))
		generate_hpf_remapping_file(r);
	    
	    add_remapping_as_used(r);
	    l = CONS(STATEMENT, generate_remapping_include(r), l);
	}
    },
	load_renamings(s));

    /* comment at the beginning
     */
    tmp = make_empty_statement();
    {
	string comment = strdup(concatenate("! begin remappings\n", NULL));
	insert_comments_to_statement(tmp, comment);
	free(comment);
    }
    l = CONS(STATEMENT, tmp, l);

    *nsp = make_block_statement(l); /* block of remaps for the nodes */
    DEBUG_STAT(8, "result", *nsp);

    debug_off();
}

/* that is all
 */
