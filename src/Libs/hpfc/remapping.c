/* HPFC module by Fabien COELHO
 *
 * $RCSfile: remapping.c,v $ version $Revision$
 * ($Date: 1995/07/20 18:40:47 $, ) 
 *
 * generates a remapping code. 
 * debug controlled with HPFC_REMAPPING_DEBUG_LEVEL.
 */

#include "defines-local.h"
#include "conversion.h"

entity CreateIntrinsic(string name); /* in syntax */

/* linearize the processor arrangement on which array is distributed
 * or replicated (driven by distributed). The variables are those created 
 * or returned by the create_var function. var is set as the linearization 
 * result, if not NULL. *psize the the extent of the returned array.
 * Special care is taken about the declaration shifts.
 */
static Pcontrainte
partial_linearization(array, distributed, var, psize, create_var)
entity array;
bool distributed;
entity var;
int *psize;
entity (*create_var)(/* int */);
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
	    v = vect_multiply(v, size);
	    vect_add_elem(&v, (Variable) create_var(dim), 1);
	    vect_add_elem(&v, TCST, - low);
	}

    if (var) vect_add_elem(&v, (Variable) var, -1);

    return(contrainte_make(v));
}

/* builds a linarization equation of the dimensions of obj.
 * var is set to the result. *psize returns the size of obj.
 * Done the Fortran way.
 */ 
Pcontrainte
full_linearization(obj, var, psize, create_var)
entity obj, var;
int *psize;
entity (*create_var)(/* int */);
{
    int dim = NumberOfDimension(obj), low, up, size;
    Pvecteur v = VECTEUR_NUL;

    for(*psize=1; dim>0; dim--)
    {
	get_entity_dimensions(obj, dim, &low, &up);
	size = up - low + 1;
	*psize *= size;
	v = vect_multiply(v, size);
	vect_add_elem(&v, (Variable) create_var(dim), 1);
	vect_add_elem(&v, TCST, - low);
    }

    if (var) vect_add_elem(&v, (Variable) var, -1);

    return(contrainte_make(v));
}

/* lambda = Psi'_D / |Psi_R|
 */
/*
static statement
compute_lambda(src, trg)
entity src, trg;
{
    entity idiv = hpfc_name_to_entity(HPFC_DIVIDE),
           lambda = get_ith_temporary_dummy(3);
    Pcontrainte p1, p2;
    int s1, s2;

    p1 = partial_linearization(src, FALSE, NULL, &s1, get_ith_processor_dummy);
    p2 = partial_linearization(src, TRUE, NULL, &s2, get_ith_processor_prime);
    
    
}
*/

static Psysteme 
generate_work_sharing_system(src, trg)
entity src, trg;
{
    entity psi_r = get_ith_temporary_dummy(1),
           psi_d = get_ith_temporary_dummy(2),
           delta = get_ith_temporary_dummy(3);
    int size_r, size_d;
    Psysteme sharing = sc_new();

    sc_add_egalite(sharing, partial_linearization(src, FALSE, psi_r, &size_r, 
						  get_ith_processor_dummy));
    sc_add_egalite(sharing, partial_linearization(trg, TRUE, psi_d, &size_d,
						  get_ith_processor_prime));
    
    /* psi_d = psi_r + |psi_r| delta
     */
    sc_add_egalite(sharing, contrainte_make(vect_make(VECTEUR_NUL, 
						      psi_d, -1, 
						      psi_r, 1,
						      delta, size_r)));

    if (size_d >= size_r)
    {
	/* 0 <= delta (there are cycles)
	 */
	sc_add_inegalite(sharing, contrainte_make
			 (vect_make(VECTEUR_NUL, delta, -1, TCST, 0)));
    }
    else
    {
	/* delta == 0
	 */
	sc_add_egalite(sharing, contrainte_make(vect_new((Variable) delta, 1)));
    }

    sc_creer_base(sharing);

    return sharing;
}

static Psysteme 
generate_remapping_system(src, trg)
entity src, trg;
{
    int ndim = variable_entity_dimension(src);
    Psysteme 
	result = sc_rn(NULL),
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

    result = sc_append(result, s_src), sc_rm(s_src);
    result = sc_append(result, s_trg), sc_rm(s_trg);
    result = sc_append(result, s_equ), sc_rm(s_equ);
    result = sc_append(result, s_shr), sc_rm(s_shr);

    return(result);
}

/*   ??? assumes that there are no parameters. what should be the case...
 */
static void 
remapping_variables(s, a1, a2, pl, plp, pll, plrm, pld, plo)
Psysteme s;
entity a1, a2;
list *pl, 	/* P */
     *plp, 	/* P' */
     *pll, 	/* locals */
     *plrm, 	/* to remove */
     *pld,      /* diffusion processor variables */
     *plo;	/* others */
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
processor_loop(s, l_psi, l_oth, psi, oth, lid, create_psi, create_oth, body, sh)
Psysteme s;
list /* of entities */ l_psi, l_oth;
entity psi, oth, lid;
entity  (*create_psi)(/* int */), (*create_oth)(/* int */);
statement body;
boolean sh; /* whether to shift the psi's */
{
    entity divide = hpfc_name_to_entity(IDIVIDE);
    Psysteme condition, enumeration, known, simpler;
    statement define_psis, compute_lid, oth_loop, if_guard;
    
    define_psis = define_node_processor_id(psi, create_psi);

    /* the lid computation is delayed in the body for diffusions
     */
    compute_lid = (lid) ?
	hpfc_compute_lid(lid, oth, create_oth) : make_empty_statement(); 

    /* simplifies the processor arrangement for the condition
     */
    known = sc_dup(entity_to_declaration_constraints(psi));
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
				hpfc_name_to_entity(MYLID), 1, 
				TCST, - NumberOfElements
      (variable_dimensions(type_variable(entity_type(psi)))))));

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
    entity nc, nt;
    expression lid, tid, chn;

    nc = hpfc_name_to_entity(send ? SEND_CHANNELS : RECV_CHANNELS);
    nt = hpfc_name_to_entity(NODETIDS);
    lid = entity_to_expression(ld);
    tid = reference_to_expression
	(make_reference(nt, CONS(EXPRESSION, lid, NIL)));
    chn = reference_to_expression
	(make_reference(nc, CONS(EXPRESSION, copy_expression(lid), NIL)));

    return(hpfc_message(tid, chn, send));
}

/* builds the diffusion loop.
 *
 * DO ldiff in sr
 *   LID computation(...)
 *   IF (MYLID.NE.LID) send to LID
 * ENDDO
 */
static statement diffuse(lid, proc, sr, ldiff)
entity lid, proc;
Psysteme sr;
list /* of entity */ ldiff;
{
    statement body;

    body = make_block_statement
	(CONS(STATEMENT, hpfc_compute_lid(lid, proc, get_ith_processor_prime),
	 CONS(STATEMENT, if_different_pe(lid, msg(lid, TRUE),
					 make_empty_statement()),
	      NIL)));

    return(systeme_to_loop_nest(sr, ldiff, body, hpfc_name_to_entity(IDIVIDE)));
}

/* in the following functions tag t controls the code generation, 
 * depending of what is to be generated (send, receive, copy, broadcast)...
 * tag t may have the following values:
 */

#define COPY 0
#define SEND 1
#define RECV 2
#define DIFF 3

static statement 
pre(t, lid)
int t;
entity lid;
{
    return(t==COPY ? make_empty_statement() :
	   t==SEND || t==DIFF ? hpfc_initsend() :
           t==RECV ? msg(lid, FALSE) :
	   statement_undefined);
}

static statement 
in(t, src, trg, create_src, create_trg)
int t;
entity src, trg;
entity (*create_src)(), (*create_trg)();
{
    return(t==COPY ? make_assign_statement
	              (make_reference_expression(trg, create_trg),
		       make_reference_expression(src, create_src)) :
	   t==SEND || t==DIFF ? hpfc_packing(src, create_src, TRUE) :
	   t==RECV ? hpfc_packing(trg, create_trg, FALSE) :
	   statement_undefined);
}

static statement 
post(t, lid, proc, sr, ldiff)
int t;
entity lid, proc;
Psysteme sr;
list /* of entities */ ldiff;
{
    return(t==COPY || t==RECV ? make_empty_statement() :
           t==SEND ? msg(lid, TRUE) :
           t==DIFF ? diffuse(lid, proc, sr, ldiff) :
	   statement_undefined);
}

static statement 
remapping_stats(t, s, sr, ll, ldiff, ld, lid, src, trg)
int t;
Psysteme s, sr;
list /* of entities */ ll, ldiff, /* of expressions */ ld;
entity lid, src, trg;
{
    statement inner, result;

    inner  = in(t, src, trg, get_ith_local_dummy, get_ith_local_prime),
    result = make_block_statement
	    (CONS(STATEMENT, pre(t, lid),
	     CONS(STATEMENT, elements_loop(s, ll, ld, inner),
	     CONS(STATEMENT, post(t, lid, array_to_processors(trg), sr, ldiff),
		  NIL))));

    statement_comments(result) = 
	strdup(concatenate("c - ", 
	   t==COPY ? "copy" : t==SEND ? "send" : t==RECV ? "receiv" :
			   t==DIFF ? "diffus" : "?",  "ing\n", NULL));

    return(result);
}

void sc_separate_on_vars(s, b, pwith, pwithout)
Psysteme s;
Pbase b;
Psysteme *pwith, *pwithout;
{
    Pcontrainte i_with, e_with, i_without, e_without;

    Pcontrainte_separate_on_vars(sc_inegalites(s), b, &i_with, &i_without);
    Pcontrainte_separate_on_vars(sc_egalites(s), b, &e_with, &e_without);

    *pwith = sc_make(e_with, i_with),
    *pwithout = sc_make(e_without, i_without);
}

static statement
generate_remapping_code(src, trg, procs, locals, l, lp, ll, ldiff, ld, dist_p)
entity src, trg;
Psysteme procs, locals;
list /* of entities */ l, lp, ll, ldiff, /* of expressions */ ld;
bool dist_p; /* true if must take care of lambda */
{
    entity lid = hpfc_name_to_entity(T_LID),
           p_src = array_to_processors(src),
           p_trg = array_to_processors(trg),
           lambda = get_ith_temporary_dummy(3);
    statement rp_copy, rp_recv, send, recv, cont, result;

    pips_debug(3, "%s taking care of processor cyclic distribution\n", 
	       dist_p ? "actually" : "not");

    rp_copy = remapping_stats(0, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);
    rp_recv = remapping_stats(2, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);

    if (dist_p) lp = CONS(ENTITY, lambda, lp);

    /* the send is different for diffusions
     */
    if (ENDP(ldiff))
    {
	statement rp_send;

	rp_send =
	    remapping_stats(1, locals, SC_EMPTY, ll, NIL, ld, lid, src, trg);
    
	send = processor_loop
	    (procs, l, lp, p_src, p_trg, lid,
	     get_ith_processor_dummy, get_ith_processor_prime,
	     if_different_pe(lid, rp_send, make_empty_statement()), FALSE);
    }
    else
    {
	Pbase b = entity_list_to_base(ldiff);
	list lpproc = gen_copy_seq(lp);
	statement rp_diff;
	Psysteme 
	    sd /* distributed */, 
	    sr /* replicated */;

	MAP(ENTITY, e, gen_remove(&lpproc, e), ldiff);

	/* polyhedron separation to extract the diffusions.
	 */
	sc_separate_on_vars(procs, b, &sr, &sd);
	base_rm(b);

	rp_diff = 
	    remapping_stats(3, locals, sr, ll, ldiff, ld, lid, src, trg);

	send = processor_loop
	    (sd, l, lpproc, p_src, p_trg, NULL,
	     get_ith_processor_dummy, get_ith_processor_prime,
	     rp_diff, FALSE);

	gen_free_list(lpproc); sc_rm(sd); sc_rm(sr);
    }

    if (dist_p) 
    {
	gen_remove(&lp, lambda);
	l = gen_nconc(l, CONS(ENTITY, lambda, NIL)); /* ??? to be deduced */
    }

    recv = processor_loop
	(procs, lp, l, p_trg, p_src, lid,
	 get_ith_processor_prime, get_ith_processor_dummy,
	 if_different_pe(lid, rp_recv, rp_copy), TRUE);

    if (dist_p) gen_remove(&l, lambda);

    cont = make_empty_statement();

    result = make_block_statement(CONS(STATEMENT, send,
				  CONS(STATEMENT, recv,
				  CONS(STATEMENT, cont,
				       NIL))));

    /*   some comments to help understand the generated code
     */
    statement_comments(result) =
	strdup(concatenate("c remapping ", entity_local_name(src), 
			   " -> ", entity_local_name(trg), "\n", NULL));
    statement_comments(send) = strdup("c send part\n");
    statement_comments(recv) = strdup("c receive part\n");
    statement_comments(cont) = strdup("c end of remapping\n");
    
    DEBUG_STAT(3, "result", result);
    
    return(result);
}

/*  remaps src to trg.
 */
statement hpf_remapping(src, trg)
entity src, trg;
{
    Psysteme p, proc, enume;
    statement s;
    entity lambda = get_ith_temporary_dummy(3) ; /* P cycle */
    bool proc_distribution_p;
    list /* of entities */ l, lp, ll, lrm, ld, lo, left, scanners,
         /* of expressions */ lddc;

    pips_debug(3, "%s -> %s\n", entity_name(src), entity_name(trg));

    if (src==trg) return(make_empty_statement()); /* (optimization:-) */

    /*   builds and simplifies the systems.
     */
    p = generate_remapping_system(src, trg);
    remapping_variables(p, src, trg, &l, &lp, &ll, &lrm, &ld, &lo);
    clean_the_system(&p, &lrm, &lo);
    lddc = simplify_deducable_variables(p, ll, &left);
    gen_free_list(ll);
    
    /* the P cycle ?
     */
    proc_distribution_p = gen_in_list_p(lambda, lo);

    if (proc_distribution_p) gen_remove(&lo, lambda);

    scanners = gen_nconc(lo, left);

    DEBUG_SYST(4, "cleaned system", p);
    DEBUG_ELST(4, "scanners", scanners);

    /*   processors/array elements separation.
     */
    hpfc_algorithm_row_echelon(p, scanners, &proc, &enume);
    sc_rm(p);

    DEBUG_SYST(3, "proc", proc);
    DEBUG_SYST(3, "enum", enume);

    /*   generates the code.
     */
    s = generate_remapping_code
	(src, trg, proc, enume, l, lp, scanners, ld, lddc, proc_distribution_p);

    /*   clean.
     */
    sc_rm(proc), sc_rm(enume);
    gen_free_list(scanners);
    gen_free_list(l), gen_free_list(lp), gen_free_list(ld);
    gen_map(gen_free, lddc), gen_free_list(lddc);  /* ??? */
    
    DEBUG_STAT(6, "result", s);

    return(s);
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
void remapping_compile(s, hsp, nsp)
statement s, *hsp /* Host Statement Pointer */, *nsp /* idem Node */;
{
    list /* of statements */ l = NIL;

    debug_on("HPFC_REMAPPING_DEBUG_LEVEL");
    pips_debug(1, "dealing with statement 0x%x\n", (unsigned int) s);

    *hsp = make_empty_statement(); /* nothing for host */

    MAP(RENAMING, r,
     {
	 l = CONS(STATEMENT, 
		  hpf_remapping(renaming_old(r), renaming_new(r)), l);
     },
	 load_renamings(s));

    *nsp = make_block_statement(l); /* block of remaps for the nodes */

    DEBUG_STAT(8, "result", *nsp);

    debug_off();
}

/* that is all
 */
