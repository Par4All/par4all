/* Overlap Analysis Module for HPFC
 * 
 * Fabien Coelho, August 1993
 *
 * $RCSfile: o-analysis.c,v $ ($Date: 1995/04/21 14:32:54 $, )
 * version $Revision$
 */

#include "defines-local.h"
#include "access_description.h"

#include "effects.h"

entity CreateIntrinsic(string name); /* in syntax.h */

static list
    lblocks = NIL,
    lloop  = NIL;

GENERIC_LOCAL_MAPPING(variable_used, int, entity);

/* Overlap_Analysis
 *
 * check conditions and compile...
 */
bool Overlap_Analysis(stat, pstat)
statement stat, *pstat;
{
    list 
	lw = NIL,
	lr = NIL,
	Ra = NIL,
	Ro = NIL,
	Rrt = NIL,
	lWa = NIL,
	lRa = NIL,
	lRo = NIL, /* to keep the lkind and lvect computed */
	W  = NIL,
	Wa = NIL,
	Wrt = NIL,
	lvect = NIL,
	lkind = NIL;
    syntax
	the_written_syntax = syntax_undefined;
    reference
	the_written_reference = reference_undefined;
    statement
	innerbody,
	messages_stat, 
	newloopnest;

    DEBUG_STAT(9, "considering statement", stat);

    set_hpfc_current_statement(stat);

    reset_current_loops();   /* ??? should be on exit */
    set_current_loops(stat);

    lblocks = NIL,
    lloop = NIL;
    innerbody = parallel_loop_nest_to_body(stat, &lblocks, &lloop);

    FindRefToDistArrayInStatement(innerbody, &lw, &lr);

    /* keeps only written references of which dimensions are block distributed,
     * and indices simple enough (=> normalization of loops may be usefull).
     * ??? bug: should also search for A(i,i) things that are forbidden...
     */
    MAPL(cs,
     {
	 syntax
	     s = SYNTAX(CAR(cs));
	 reference 
	     r = syntax_reference(s);
	 entity
	     array = reference_variable(r);

	 if ((block_distributed_p(array)) && 
	     (simple_indices_p(r)) &&
	     (!replicated_p(array)))
	     W = CONS(SYNTAX, s, W);
	 else
	     Wrt = CONS(SYNTAX, s, Wrt);
     },
	 lw);

    gen_free_list(lw);
    if (W==NIL) /* no ok distributed variable written ! */
    {
	debug(7, "Overlap_Analysis", 
	      "returning FALSE because no ok distributed variable written\n");
	return(FALSE);
    }
	
    
    the_written_syntax = choose_one_syntax_in_references_list(&W);

    the_written_reference = syntax_reference(the_written_syntax);
    Wa = CONS(SYNTAX, the_written_syntax, NIL);
    if (!align_check(the_written_reference,
		     the_written_reference, &lvect, &lkind))
	pips_error("Overlap_Analysis","no self alignment!\n");
    
    lWa = CONS(CONSP, CONS(CONSP, lkind,
		      CONS(CONSP, lvect, NIL)), NIL);

    MAPL(cs,
     {
	 syntax
	     s = SYNTAX(CAR(cs));
	 reference 
	     r = syntax_reference(s);

	 if (align_check(the_written_reference, r, &lvect, &lkind))
	 {
	     if (aligned_p(the_written_reference, r, lvect, lkind))
	     {
		 Wa = gen_nconc(Wa, CONS(SYNTAX, s, NIL));
		 lWa = gen_nconc(lWa, CONS(CONSP,
					   CONS(CONSP, lkind, 
						CONS(CONSP, lvect, NIL)),
					   NIL));
	     }
	     else /* ??? what about loop splitting */
	     {
		 Wrt = gen_nconc(Wrt, CONS(SYNTAX, s, NIL));
		 gen_free_list(lvect);
		 gen_free_list(lkind); /* ??? memory leak */
	     }
	 }
	 else
	 {
	     Wrt = gen_nconc(Wrt, CONS(SYNTAX, s, NIL));
	     gen_free_list(lvect);
	     gen_free_list(lkind); /* ??? memory leak */
	 }
     },
	 CDR(W));

    gen_free_list(W);
    
    debug(5, "Overlap_Analysis",
	  "Wa length is %d (%d), Wrt lenght is %d\n",
	  gen_length(Wa), gen_length(lWa), gen_length(Wrt));

    if (gen_length(Wrt)!=0) 
    {
	gen_free_list(Wa);
	gen_free_list(lWa);
	gen_free_list(Wrt);
	gen_free_list(lblocks);
	gen_free_list(lloop);
	return(FALSE);
    }

    /* Now, we have the fellowing situation:
     *
     * Wa: set of aligned written references, the first of which is ``the'' ref.
     */
    MAPL(cs,
     {
	 syntax
	     s = SYNTAX(CAR(cs));
	 reference 
	     r = syntax_reference(s);
	 entity
	     array = reference_variable(r);
	 list lvect = NIL;
	 list lkind = NIL;

	 debug(6, "Overlap_Analysis",
	       "dealing with reference of array %s\n",
	       entity_name(array));

	 ifdebug(6)
	 {
	     fprintf(stderr, "[Overlap_Analysis]\nreference is:\n");
	     print_reference(r);
	     fprintf(stderr, "\n");
	 }

	 if (align_check(the_written_reference, r, &lvect, &lkind))
	 {
	     if (aligned_p(the_written_reference, r, lvect, lkind))
	     {
		 Ra = gen_nconc(Ra, CONS(SYNTAX, s, NIL));
		 lRa = gen_nconc(lRa, CONS(CONSP,
					   CONS(CONSP, lkind, 
						CONS(CONSP, lvect, NIL)),
					   NIL));    
	     }
	     else
	     if (message_manageable_p(array, lvect, lkind))
	     {
		 Ro = gen_nconc(Ro, CONS(SYNTAX, s, NIL));
		 lRo = gen_nconc(lRo, CONS(CONSP,
					   CONS(CONSP, lkind, 
						CONS(CONSP, lvect, NIL)),
					   NIL));
	     }
	     else
	     {
		 Rrt = gen_nconc(Rrt, CONS(SYNTAX, s, NIL));
		 gen_free_list(lvect);
		 gen_free_list(lkind);
	     }
	 }
	 else
	 {
	     Rrt = gen_nconc(Rrt, CONS(SYNTAX, s, NIL));
	     gen_free_list(lvect);
	     gen_free_list(lkind);
	 }
     },
	 lr);

    gen_free_list(lr);

    debug(5, "Overlap_Analysis",
	  "Ra length is %d, Ro length is %d, Rrt lenght is %d\n",
	  gen_length(Ra), gen_length(Ro), gen_length(Rrt));

    if (gen_length(Rrt)!=0) 
    {
	gen_free_list(Wa);
	gen_free_list(lWa);
	gen_free_list(Ra);
	gen_free_list(lRa);
	gen_free_list(Ro);
	gen_free_list(lRo);
	gen_free_list(Rrt);
	gen_free_list(lblocks);
	gen_free_list(lloop);
	return(FALSE);
    }

    /* here is the situation now:
     *
     * Wa set of aligned references written,
     * Ra set of aligned references read,
     * Ro set of nearly aligned references that suits the overlap analysis
     */

    /* messages handling
     */

    messages_stat = ((gen_length(Ro)>0)?
		     (messages_handling(Ro, lRo)):
		     (make_continue_statement(entity_empty_label())));

    /* generate the local loop for every processor, given the global loop
     * bounds. The former indexes have to be computed, and the loops are
     * based upon new indexes, of which names have to be propagated in the
     * body of the loop. This generation is to be based on the normalized
     * form computed for every references of Ro, but it is direct for
     * Ra and Wa, since new declarations implied that the alignment is
     * performed for distributed indices. Not distributed dimensions
     * indices have not to be touched, (at least if no new declarations are
     * the common case)
     */
    
    if (!generate_optimized_code_for_loop_nest(innerbody, 
					       &newloopnest,
					       Wa, Ra, Ro, 
					       lWa, lRa, lRo))
	return(FALSE);

    (*pstat) = 
	make_block_statement
	    (CONS(STATEMENT, messages_stat,
	     CONS(STATEMENT, loop_nest_guard(newloopnest,
					     the_written_reference,
					     CONSP(CAR(CONSP(CAR(lWa)))),
					     CONSP(CAR(CDR(CONSP(CAR(lWa)))))),
		  NIL)));    

    IFDBPRINT(8, "Overlap_Analysis", node_module, (*pstat));

    reset_hpfc_current_statement();

    return(TRUE);
}

/* bool block_distributed_p(array)
 *
 * true if there is no cyclic distribution for the array
 */
bool block_distributed_p(array)
entity array;
{
    int	dim = NumberOfDimension(array);
    tag n;

    for(; dim>0; dim--)
    {
	n = new_declaration(array, dim);
	if ((n==is_hpf_newdecl_gamma) || (n==is_hpf_newdecl_delta))
	    /* distributed && (nd==is_hpf_newdecl_none)) ?
	     * ??? the case is not handled later on 
	     */
	    return(FALSE);
    }

    return(TRUE);
}

/* bool simple_indices_p(r)
 *
 * true if indices are constants or index
 */
bool simple_indices_p(r)
reference r;
{
    entity
	array = reference_variable(r);
    int
	dim = 1;

    MAPL(ce,
     {
	 expression
	     e = EXPRESSION(CAR(ce));
	 normalized /* must be normalized somewhere! */
	     n = expression_normalized(e); 
	 int 
	     p;
	 bool
	     b1 = ith_dim_distributed_p(array, dim, &p);
	 bool
	     b2 = ((!b1) ? local_integer_constant_expression(e) : FALSE);

	 debug(7, "simple_indices_p",
	       "array %s, dim %d, distributed %d, locally constant %d\n",
	       entity_name(array), dim, b1, b2);

	 if (!b2)
	 {
	 if (normalized_complex_p(n)) 
	     /* cannot decide, so it is supposed to be FALSE */
	 {
	     debug(7, "simple_indices_p",
		   "returning FALSE (complex) for ref to %s, dim %d\n",
		   entity_name(reference_variable(r)), dim);
	     return(FALSE); 
	 }
	 else
	 {
	     Pvecteur
		 v = (Pvecteur) normalized_linear(n);
	     int
		 s = vect_size(v);
	     
	     if (s>1) return(FALSE);
	     
	     if ((s==1) && 
		 (!entity_loop_index_p((entity)v->var)) &&
		 ((int) vect_coeff(TCST, v)==0))
	     {
		 debug(7, "simple_indices_p",
		   "returning FALSE (not simple) for ref to %s, dim %d\n",
		   entity_name(reference_variable(r)), dim);
		 return(FALSE);
	     }
	     else
	     if (entity_loop_index_p((entity)v->var))
	     {
		 /* ??? checks that there is a shift alignment, 
		  * what shouldn't be necessary...
		  */
		 alignment
		     al = FindArrayDimAlignmentOfArray(array, dim);
		 int
		     rate = 
			 (al==alignment_undefined) ?
			 (0) : HpfcExpressionToInt(alignment_rate(al)) ;

		 if (rate!=0 && rate!=1)
		 {
		     debug(7, "simple_indices_p",
			   "returning FALSE (stride) for %s, dim %d\n",
			   entity_name(reference_variable(r)), dim);
		     return(FALSE);
		 }
	     }
	 }
         }
	 
	 dim++;
     },
	 reference_indices(r));
    
    debug(7, "simple_indices_p",
	  "returning TRUE for reference to %s\n",
	  entity_name(reference_variable(r)));
    return(TRUE);
}

/* bool aligned_p(r1, r2, lvref, lkref)
 *
 * true if references are aligned or, for constants, on the same processor...
 */
bool aligned_p(r1, r2, lvref, lkref)
reference r1, r2;
list lvref, lkref;
{
    entity
	e2 = reference_variable(r2),
	template = array_to_template(e2);
    list
	lv = lvref,
	lk = lkref;
    bool
	result = TRUE;
    int
	i = 1 ;

    debug(7, "aligned_p",
	  "arrays %s and %s\n",
	  entity_name(reference_variable(r1)), entity_name(e2));

    for ( ; (lk!=NIL) ; lv=CDR(lv), lk=CDR(lk))
    {
	tag
	    t = access_tag(INT(CAR(lk)));
	Pvecteur
	    v = PVECTOR(CAR(lv));
	int
	    p,
	    tpldim = template_dimension_of_array_dimension(e2, i),
/*	    s   = vect_size(v),
	    cst = vect_coeff(TCST, v), */
	    tpl = vect_coeff(TEMPLATEV, v),
	    dlt = vect_coeff(DELTAV, v),
	    tsh = vect_coeff(TSHIFTV, v);

	if ((t==not_aligned) ||
	    ((t==aligned_constant) && 
	     (processor_number(template, tpldim, tpl, &p)!=
	      processor_number(template, tpldim, tpl-dlt, &p))) ||
	    ((t==aligned_shift) && (tsh!=0)))
	    return(FALSE);

	i++;
    }
    
    return(result);
}

/* bool message_manageable_p(array, lpref, lkref) 
 *
 * conditions:
 * every thing should be manageable, i.e.
 * no star in the dimensions, and the width has to be accepted...
 */
bool message_manageable_p(array, lpref, lkref)
entity array;
list lpref, lkref;
{
    list
	lp = NIL,
	lk = NIL;
    int
	i;

    for(i=1, lk=lkref, lp=lpref ; lk!=NIL ; lk=CDR(lk), lp=CDR(lp))
    {
	tag
	    ta = access_tag(INT(CAR(lk)));
	int
	    p = 0,
	    shift = vect_coeff(TSHIFTV, PVECTOR(CAR(lp))),
	    dlt = vect_coeff(DELTAV, PVECTOR(CAR(lp))),
	    t2 = vect_coeff(TEMPLATEV, PVECTOR(CAR(lp)));

	if ((ta==not_aligned) ||
	    (ta==local_star) ||
	    (ta==aligned_star) ||
	    ((ta==aligned_constant) &&
	     (!on_same_proc_p(t2-dlt, t2,
			     array_to_template(array), 
			     template_dimension_of_array_dimension(array, i)))) ||
	    ((ta==aligned_shift) &&
	     (shift>DistributionParameterOfArrayDim(array, i, &p))))
	{
	    debug(5, "message_manageable_p",
		  "returning false for %s, dim %d, access %d\n",
		  entity_name(array), i, ta);

	    return(FALSE);
	}

	i++;
    }

    if (!block_distributed_p(array)) return(FALSE);

    /*
     * here the overlap is accepted, and stored
     *
     * ??? this should be done elsewhere, because too much overlap
     * may be too much memory, allocated... in generate_one_message()?
     */


    for(i=1, lk=lkref, lp=lpref ; lk!=NIL ; lk=CDR(lk), lp=CDR(lp))
    {
	tag
	    ta = access_tag(INT(CAR(lk)));
	int
	    shift = vect_coeff(TSHIFTV, PVECTOR(CAR(lp)));

	if ((ta==aligned_shift) && (shift!=0))
	    set_overlap(array, 
			i, 
			((shift<0)?(0):(1)), 
			abs(shift));

	i++;
    }

    return(TRUE); /* accepted! */
}

bool statically_decidable_loops(l)
list l;
{
    range
	r = ((ENDP(l))?(NULL):(loop_range(LOOP(CAR(l)))));

    return((ENDP(l))?
	   (TRUE):
	   (expression_integer_constant_p(range_lower(r)) &&
	    expression_integer_constant_p(range_upper(r)) &&
	    expression_integer_constant_p(range_increment(r)) &&
	    (HpfcExpressionToInt(range_increment(r))==1) &&
	    statically_decidable_loops(CDR(l))));
}

bool expression_integer_constant_p(e)
expression e;
{
    syntax
	s = expression_syntax(e);
    normalized
	n = expression_normalized(e);

    if ((n!=normalized_undefined) && (normalized_linear_p(n)))
    {
	Pvecteur
	    v = normalized_linear(n);
	int
	    s = vect_size(v);

	ifdebug(8)
	{
	    fprintf(stderr, 
		    "[expression_integer_constant_p] normalized linear, size %d, TCST %d\n",
		    s, (int) vect_coeff(TCST,v));
	    print_expression(e);
	}

	if (s==0) return(TRUE);
	if (s>1) return(FALSE);
	return((s==1) && ((int) vect_coeff(TCST,v)!=0));
    }
    else
    if (syntax_call_p(s))
    {
	call 
	    c = syntax_call(s);
	value
	    v = entity_initial(call_function(c));

	/* I hope a short evaluation is made by the compiler */
	return((value_constant_p(v)) && (constant_int_p(value_constant(v))));
    }
    
    return(FALSE);
}

/*   generate the call to the dynamic loop bounds computation
 */
static statement statement_compute_bounds(newlobnd, newupbnd, oldidxvl, 
				   lb, ub, an, dp)
entity newlobnd, newupbnd, oldidxvl;
expression lb, ub;
int an, dp;
{
    list
	l = CONS(EXPRESSION, entity_to_expression(newlobnd),
	    CONS(EXPRESSION, entity_to_expression(newupbnd),
	    CONS(EXPRESSION, entity_to_expression(oldidxvl),
	    CONS(EXPRESSION, lb,
	    CONS(EXPRESSION, ub,
	    CONS(EXPRESSION, int_to_expression(an),
	    CONS(EXPRESSION, int_to_expression(dp),
		 NIL)))))));

    return(hpfc_make_call_statement(hpfc_name_to_entity(LOOP_BOUNDS), l));
}

static statement make_loop_nest_for_overlap(lold, lnew, lbl,
					    new_indexes, old_indexes, 
					    innerbody)
list lold, lnew, lbl;
entity_mapping new_indexes, old_indexes;
statement innerbody;
{
    entity
	index, 
	oldindexvalue;
    loop
	oldloop,
	newloop;
    list
	l,
	lnew_loop = NIL,
	lnew_body = NIL;
    bool
	compute_index = FALSE;

    if (ENDP(lold)) 
	return(innerbody);
    
    oldloop = LOOP(CAR(lold));
    newloop = LOOP(CAR(lnew));

    index = loop_index(oldloop);
    oldindexvalue = (entity) GET_ENTITY_MAPPING(old_indexes, index);
    lnew_body = CONS(STATEMENT,
		    make_loop_nest_for_overlap(CDR(lold), CDR(lnew), CDR(lbl),
					       new_indexes, old_indexes, 
					       innerbody),
		    NIL);

    /* ??? should also look in lbl */
    compute_index = (oldindexvalue!=(entity)HASH_UNDEFINED_VALUE) &&
	variable_used_in_statement_p(index, innerbody);
    
    /* if the index value is needed, the increment is added
     */
    if (compute_index)
	lnew_body = CONS(STATEMENT, make_increment_statement(index), 
			lnew_body);

    loop_body(newloop) = make_block_statement(lnew_body);
    lnew_loop = CONS(STATEMENT,
		     make_stmt_of_instr(make_instruction(is_instruction_loop,
							 newloop)),
		     NIL);

    /*
     * i = initial_old_value 
     * DO i' = ...
     *   i = i + 1
     *   body
     * ENDDO
     */

    if (compute_index)
	lnew_loop = 
	    CONS(STATEMENT,
		 make_assign_statement(entity_to_expression(index),
				       entity_to_expression(oldindexvalue)),
		 lnew_loop);

    /* copy the non perfectly nested parts if needed
     */
    l = CONSP(CAR(lbl));
    if (!ENDP(l))
    {
	statement
	    s;
	instruction
	    i;
	list
	    lpre = NIL,
	    lpost = NIL;
	bool 
	    pre = TRUE;

	for(; !ENDP(l); l=CDR(l))
	{
	    s = STATEMENT(CAR(l));
	    i = statement_instruction(s);
	    
	    /*  switch from pre to post.
	     */
	    if (instruction_loop_p(i) && instruction_loop(i)==oldloop)
		pre = FALSE;
	    else
		if (pre)
		    lpre = CONS(STATEMENT, copy_statement(s), lpre);
		else
		    lpost = CONS(STATEMENT, copy_statement(s), lpost);
	}

	/* the swith must have been encountered */
	assert(!pre);
	    
	lnew_loop = gen_nconc(gen_nreverse(lpre),
		    gen_nconc(lnew_loop,
			      gen_nreverse(lpost)));
    }

    return(make_block_statement(lnew_loop));
}

bool generate_optimized_code_for_loop_nest(innerbody, pstat,
					   Wa, Ra, Ro, 
					   lWa, lRa, lRo)
statement innerbody, *pstat;
list Wa, Ra, Ro, lWa, lRa, lRo;
{
    syntax
	the_written_syntax = SYNTAX(CAR(Wa));
    reference
	the_written_reference = syntax_reference(the_written_syntax);
    entity
	array = reference_variable(the_written_reference);
    int
	an = load_hpf_number(array);
    entity_mapping
	new_indexes = MAKE_ENTITY_MAPPING(),
	old_indexes = MAKE_ENTITY_MAPPING();
    list
	boundcomp = NIL,
	newloops = NIL;
    statement
	newnest = NULL;
    range rg;
    expression
	lb, ub;
    entity
	index, newindex, newlobnd, newupbnd, oldidxvl;
    loop l, nl;
    int p, dim;

    MAPL(cl,
     {
	 l = LOOP(CAR(cl));
	 index = loop_index(l);
	 dim = which_array_dimension(the_written_reference, index);

	 if (ith_dim_distributed_p(array, dim, &p))
	 {
	     /* new bounds to compute, and so on */
	     rg = loop_range(l);
	     lb = copy_expression(range_lower(rg));
	     ub = copy_expression(range_upper(rg));
	     
	     newindex = NewTemporaryVariable(node_module, 
					     MakeBasic(is_basic_int));
	     newlobnd = NewTemporaryVariable(node_module, 
					     MakeBasic(is_basic_int));
	     newupbnd = NewTemporaryVariable(node_module, 
					     MakeBasic(is_basic_int));
	     oldidxvl = NewTemporaryVariable(node_module, 
					     MakeBasic(is_basic_int));

	     boundcomp = 
		 gen_nconc(boundcomp,
			   CONS(STATEMENT,
				statement_compute_bounds(newlobnd,
							 newupbnd,
							 oldidxvl,
							 lb, ub, an, p),
				NIL));

	     newloops = 
		 gen_nconc(newloops,
		   CONS(LOOP,
			make_loop_skeleton(newindex, 
					   entity_to_expression(newlobnd),
					   entity_to_expression(newupbnd)),
			NIL));

	     SET_ENTITY_MAPPING(new_indexes, index, newindex);
	     SET_ENTITY_MAPPING(old_indexes, index, oldidxvl);
	 }
	 else
	 {
	     nl = make_loop(loop_index(l),
			    loop_range(l),
			    statement_undefined,
			    loop_label(l),
			    make_execution(is_execution_sequential, UU),
			    NIL);
/* ??? there is a core dump on the second free, when executed, in test 37~;
	     free_execution(loop_execution(l));
	     free_loop(l);
*/
	     newloops = gen_nconc(newloops, CONS(LOOP, nl, NIL));
	 }
     },
	 lloop);

    update_indices_for_local_computation(new_indexes, Wa, lWa);
    update_indices_for_local_computation(new_indexes, Ra, lRa);
    update_indices_for_local_computation(new_indexes, Ro, lRo);
    
    /*
     * and now generates the code...
     */
    initialize_variable_used_map_for_current_loop_nest(innerbody);
    /*initialize_variable_used_map_for_statement(innerbody);*/

    if (hpfc_overlap_kill_unused_scalars(innerbody))
    {
	close_variable_used_map_for_statement();
	initialize_variable_used_map_for_current_loop_nest(innerbody);
	/* initialize_variable_used_map_for_statement(innerbody);*/
    }

    newnest = make_loop_nest_for_overlap(lloop, newloops, lblocks,
					 new_indexes, old_indexes, 
					 innerbody);
    close_variable_used_map_for_statement();

    
    (*pstat) = make_block_statement(gen_nconc(boundcomp,
					      CONS(STATEMENT, newnest,
						   NIL)));

    return(TRUE);
}

/*  To Kill scalar definitions within the generated code
 *  recognize if only one reference. 
 */
static bool hpfc_killed_scalar;

static void hpfc_overlap_kill_unused_scalars_rewrite(stat)
statement stat;
{
    instruction
	i = statement_instruction(stat);
    expression
	e = expression_undefined;
    entity
	var = entity_undefined;

    if (!instruction_assign_p(i)) return;

    e = EXPRESSION(CAR(call_arguments(instruction_call(i))));

    assert(expression_reference_p(e));

    var = reference_variable(syntax_reference(expression_syntax(e)));

    debug(5, "hpfc_overlap_kill_unused_scalars_rewrite",
	  "considering definition of %s (statement 0x%x)\n",
	  entity_name(var), stat);

    if (entity_integer_scalar_p(var) &&
	load_entity_variable_used(var)==1)
    {
	debug(3, "hpfc_overlap_kill_unused_scalars_rewrite",
	      "killing definition of %s (statement 0x%x)\n",
	      entity_name(var), stat);

	hpfc_killed_scalar = TRUE;
	statement_instruction(stat) =  /* ??? memory leak */
	    make_continue_instruction();
    }
}

/*   true if one statement was killed
 */
bool hpfc_overlap_kill_unused_scalars(stat)
statement stat;
{
    assert(get_variable_used_map()!=hash_table_undefined);

    hpfc_killed_scalar = FALSE;

    gen_recurse(stat,
		statement_domain,
		gen_true,
		hpfc_overlap_kill_unused_scalars_rewrite);

    return(hpfc_killed_scalar);
}

/* int which_array_dimension(r, e)
 *
 * returns the dimension of reference on which index entity e is used
 */
int which_array_dimension(r, e)
reference r;
entity e;
{
    int
	dim = 1;
    list
	li = reference_indices(r);
    Variable
	v = (Variable) e;

    MAPL(ce,
     {
	 normalized
	     n = expression_normalized(EXPRESSION(CAR(ce)));

	 if (normalized_linear_p(n) &&
	     (vect_coeff(v, (Pvecteur)normalized_linear(n)) != 0))
	     return(dim);

	 dim++;
     },
	 li);

    return(-1);
}

loop make_loop_skeleton(newindex, lower_expression, upper_expression)
entity newindex;
expression lower_expression, upper_expression;
{
    return(make_loop(newindex, 
		     make_range(lower_expression, 
				upper_expression,				
				int_to_expression(1)),
		     statement_undefined, /* statement is not yet defined */
		     entity_empty_label(),
		     make_execution(is_execution_sequential, UU),
		     NIL));
}

void update_indices_for_local_computation(new_indexes, Ref, lRef)
entity_mapping new_indexes;
list Ref, lRef;
{
    list
	lr = Ref,
	lkv = lRef;

    for ( ; (lr!=NIL) ; lr=CDR(lr), lkv=CDR(lkv))
    {
	int 
	    dim = 1;
	syntax 
	    s = SYNTAX(CAR(lr));
	reference
	    r = syntax_reference(s);
	entity
	    array = reference_variable(r);
	list
	    l1 = CONSP(CAR(lkv)),
	    lk = CONSP(CAR(l1)),
	    li = reference_indices(r),
	    lv = CONSP(CAR(CDR(l1))),
	    li2 = NIL;

	for ( ; (lk!=NIL) ; lk=CDR(lk), li=CDR(li), lv=CDR(lv))
	{
	    expression
		indice = EXPRESSION(CAR(li));
	    Pvecteur
		v = PVECTOR(CAR(lv));
	    access
		ac = INT(CAR(lk));

	    /*
	     * caution: only distributed dimensions indexes are modified
	     * other have to remain untouched...
	     * ??? aligned star is missing
	     */
	    switch (access_tag(ac))
	    {
	    case aligned_shift: /* find the new index of the loop */
	    {
		Pvecteur
		    vindex = the_index_of_vect(v);
		entity
		    oldindex = (entity) var_of(vindex),
		    newindex = (entity) GET_ENTITY_MAPPING(new_indexes, 
							   oldindex);
		int
		    shift = (int) vect_coeff(TSHIFTV, v);

		if (shift==0)
		{
		    li2 = gen_nconc(li2,
				    CONS(EXPRESSION,
					 entity_to_expression(newindex),
					 NIL));
		}
		else
		{
		    li2 = 
			gen_nconc(li2,
				  CONS(EXPRESSION,
				       MakeBinaryCall
				       (CreateIntrinsic((shift>0)?
							(PLUS_OPERATOR_NAME):
							(MINUS_OPERATOR_NAME)),
					entity_to_expression(newindex),
					int_to_expression(abs(shift))),
				       NIL));
		}

		break;
	    }
	    case aligned_constant: /* compute the local indice */
	    {
		int
		    tval = vect_coeff(TEMPLATEV, v);
		
		li2 = gen_nconc(li2,
				CONS(EXPRESSION,
				     int_to_expression
				     (template_cell_local_mapping(array, 
								  dim, 
								  tval)),
				     NIL));
		break;
	    }
	    case aligned_affine:
	    case aligned_star:
		pips_error("update_indices_for_local_computation",
			   "part of that function not implemented yet\n");
		break;
	    default: /* ??? nothing is changed */
		li2 = gen_nconc(li2, CONS(EXPRESSION, indice, NIL));
		break;
	    }
	    dim++;
	}

	reference_indices(r) = li2;

	ifdebug(8)
	 {
	     fprintf(stderr,
	      "[update_indices_for_local_computation]\nnew reference is:\n");
	     print_reference(r);
	     fprintf(stderr, "\n");
	 }

    }
    
}

statement make_increment_statement(index)
entity index;
{
    return(make_assign_statement
	   (entity_to_expression(index),
	    MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME),
			   entity_to_expression(index),
			   int_to_expression(1))));
}

/* bool variable_used_in_statement_p(ent, stat)
 *
 * not 0 if ent is referenced in statement stat.
 * yes, I know, proper effects may be called several
 * times for the same statement...
 *
 * ??? I should have used cumulated/proper effects to be computed on
 * the statement being generated, but It would not have been as easy
 * to compute and to use.
 */
static statement 
  current_variable_used_statement = statement_undefined;

static void variable_used_rewrite(r)
reference r;
{
    entity 
	v = reference_variable(r);
    int
	n = load_entity_variable_used(v);

    store_entity_variable_used(v, int_undefined_p(n) ? 1 : n+1);
}

void initialize_variable_used_map_for_statement(stat)
statement stat;
{
    make_variable_used_map();
    current_variable_used_statement = stat; 

    gen_recurse(stat,
		reference_domain,
		gen_true,
		variable_used_rewrite);
}

void initialize_variable_used_map_for_current_loop_nest(inner_body)
statement inner_body;
{
    list
	ll=lloop, 
	lb=lblocks;
    statement
	s;
    instruction
	i;
    loop
	l;

    make_variable_used_map();     
    current_variable_used_statement = inner_body;

    gen_recurse(inner_body,
		reference_domain,
		gen_true,
		variable_used_rewrite);

    for (; !ENDP(ll); ll=CDR(ll), lb=CDR(lb))
    {
	l = LOOP(CAR(ll));

	MAPL(cs,
	 {
	     s = STATEMENT(CAR(cs));
	     i = statement_instruction(s);
	     
	     if (!(instruction_loop_p(i) && l==i))
		 gen_recurse(s,
			     reference_domain,
			     gen_true,
			     variable_used_rewrite);

	 },
	     CONSP(CAR(lb)));
    }
}

void close_variable_used_map_for_statement()
{
    free_variable_used_map();
    current_variable_used_statement = statement_undefined;
}

bool variable_used_in_statement_p(ent, stat)
entity ent;
statement stat;
{
    bool
	result;

    assert(stat == current_variable_used_statement);

    result = (!entity_variable_used_undefined_p(ent));
    
    return(result);
}

/* syntax choose_one_syntax_in_references_list(pls)
 *
 * one of the syntax is chosen from the list. The "larger one".
 * and the list is given back, the chosen syntax as first element.
 */
syntax choose_one_syntax_in_references_list(pls)
list *pls;
{
    list
	cls = *pls,
	nls = NIL;
    syntax
	chosen = SYNTAX(CAR(cls));
    int 
	chosen_distribution = 
	    number_of_distributed_dimensions
		(reference_variable(syntax_reference(chosen)));

    MAPL(cs,
     {
	 syntax
	     current = SYNTAX(CAR(cs));
	 int
	     current_distribution = 
		 number_of_distributed_dimensions
		     (reference_variable(syntax_reference(current)));

	 if (current_distribution > chosen_distribution)
	 {
	     nls = CONS(SYNTAX, chosen, nls);
	     chosen = current;
	     chosen_distribution = current_distribution;
	 }
	 else
	     nls = CONS(SYNTAX, current, nls);
     },
	 CDR(cls));
	 
    gen_free_list(cls);
    *pls = CONS(SYNTAX, chosen, nls);

    debug(7, "choose_one_syntax_in_references_list",
	  "reference to %s chosen, %d dimensions\n",
	  entity_name(reference_variable(syntax_reference(chosen))),
	  chosen_distribution);

    return(chosen);
}

int number_of_distributed_dimensions(a)
entity a;
{
    int
	p = -1,
	ndim = NumberOfDimension(a),
	i = 1,
	n = 0;

    for (i=1 ; i<=ndim ; i++)
	if (ith_dim_distributed_p(a, i, &p)) n++;
   
    return(n);
}

/* bool on_same_proc_p(t1, t2, temp, dim)
 *
 * true if the given template elements on the specified dimension
 * are mapped on the same processor.
 */
bool on_same_proc_p(t1, t2, template, dim)
int t1, t2;
entity template;
int dim;
{
    int p;

    return(processor_number(template, dim, t1, &p) ==
	   processor_number(template, dim, t2, &p));
}


/* That is all
 */
