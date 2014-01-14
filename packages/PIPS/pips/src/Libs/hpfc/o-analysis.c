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
/* Overlap Analysis Module for HPFC
 * 
 * Fabien Coelho, August 1993
 */

#include "defines-local.h"
#include "access_description.h"

#include "effects-generic.h"

static list lblocks = NIL, lloop  = NIL;

GENERIC_LOCAL_FUNCTION(entity_variable_used, entity_int)

/* true if there is no cyclic distribution for the array
 */
bool block_distributed_p(entity array)
{
    int	dim = NumberOfDimension(array);
    tag n;

    for(; dim>0; dim--)
    {
	n = new_declaration_tag(array, dim);
	if ((n==is_hpf_newdecl_gamma) || (n==is_hpf_newdecl_delta))
	    /* distributed && (nd==is_hpf_newdecl_none)) ?
	     * ??? the case is not handled later on 
	     */
	    return(false);
    }

    return(true);
}

/* true if indices are constants or index
 */
static bool simple_indices_p(reference r)
{
    entity array = reference_variable(r);
    int	dim = 1;

    ifdebug(6) {
	pips_debug(6, "considering reference: ");
	print_reference(r);
	fprintf(stderr,"\n");
    }

    MAP(EXPRESSION, e,
     {
	 normalized  n = expression_normalized(e); 
	 int p;
	 bool b1 = ith_dim_distributed_p(array, dim, &p);
	 bool b2 = ((!b1) ? local_integer_constant_expression(e) : false);

	 pips_debug(7, "%s(DIM=%d), distributed %d, locally constant %d\n",
		    entity_name(array), dim, b1, b2);

	 if (!b2)
	 {
	 if (normalized_complex_p(n)) 
	     /* cannot decide, so it is supposed to be false */
	 {
	     pips_debug(7, "returning false (complex)\n");
	     return false; 
	 }
	 else
	 {
	     Pvecteur v = (Pvecteur) normalized_linear(n);
	     int s = vect_size(v);
	     
	     if (s>1) 
	     {
		 ifdebug(7) {
		     pips_debug(7, "returning false, vect size %d>1\n", s);
		     vect_debug(v);
		 }
		 return false;
	     }
	     
	     if ((s==1) && 
		 (!entity_loop_index_p((entity)v->var)) &&
		 (value_zero_p(vect_coeff(TCST, v))))
	     {
		 pips_debug(7, "returning false (not simple)\n");
		 return false;
	     }
	     else
	     if (entity_loop_index_p((entity)v->var))
	     {
		 /* ??? checks that there is a shift alignment, 
		  * what shouldn't be necessary...
		  */
		 alignment al = FindArrayDimAlignmentOfArray(array, dim);
		 int rate = al==alignment_undefined?
		     0: HpfcExpressionToInt(alignment_rate(al)) ;

		 if (rate!=0 && rate!=1)
		 {
		     pips_debug(7, "returning false (stride)\n");
		     return(false);
		 }
	     }
	 }
         }
	 
	 dim++;
     },
	 reference_indices(r));
    
    pips_debug(7, "returning TRUE!\n");
    return true;
}

/* true if references are aligned or, for constants, on the same processor...
 */
static bool 
aligned_p(
    reference r1,
    reference r2,
    list lvref,
    list lkref)
{
    entity e2 = reference_variable(r2),	template = array_to_template(e2);
    list lv = lvref, lk = lkref;
    bool result = true;
    int i = 1 ;

    pips_debug(7, "arrays %s and %s\n",
	       entity_name(reference_variable(r1)), entity_name(e2));

    for ( ; (lk!=NIL) ; POP(lv), POP(lk))
    {
	tag t = access_tag(INT(CAR(lk)));
	Pvecteur v = (Pvecteur) PVECTOR(CAR(lv));
	Value vt = vect_coeff(TEMPLATEV, v),
	      vd = vect_coeff(DELTAV, v),
	      vs = vect_coeff(TSHIFTV, v);
	int p,
	    tpldim = template_dimension_of_array_dimension(e2, i),
	    tpl = VALUE_TO_INT(vt),
	    dlt = VALUE_TO_INT(vd),
	    tsh = VALUE_TO_INT(vs);

	if ((t==not_aligned) ||
	    ((t==aligned_constant) && 
	     (processor_number(template, tpldim, tpl, &p)!=
	      processor_number(template, tpldim, tpl-dlt, &p))) ||
	    ((t==aligned_shift) && (tsh!=0)))
	    return false;

	i++;
    }
    
    return result;
}

/* true if the given template elements on the specified dimension
 * are mapped on the same processor.
 */
static bool on_same_proc_p(t1, t2, template, dim)
int t1, t2;
entity template;
int dim;
{
    int p;
    return(processor_number(template, dim, t1, &p) ==
	   processor_number(template, dim, t2, &p));
}

/* every thing should be manageable, i.e.
 * ??? removed: no star in the dimensions , 
 * and the width has to be accepted...
 */
static bool 
message_manageable_p(
    entity array,
    list lpref,
    list lkref)
{
    list lp = NIL, lk = NIL;
    int	i;

    for(i=1, lk=lkref, lp=lpref ; lk!=NIL ; lk=CDR(lk), lp=CDR(lp))
    {
	tag ta = access_tag(INT(CAR(lk)));
	Pvecteur v = (Pvecteur) PVECTOR(CAR(lp));
	Value vs = vect_coeff(TSHIFTV, v),
	      vd = vect_coeff(DELTAV, v),
	      vt = vect_coeff(TEMPLATEV, v);
	int p = 0,
	    shift = VALUE_TO_INT(vs),
	    dlt = VALUE_TO_INT(vd),
	    t2 = VALUE_TO_INT(vt);

	if ((ta==not_aligned) ||
	    /*(ta==local_star) ||*/
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

	    return(false);
	}

	i++;
    }

    if (!block_distributed_p(array)) return(false);

    /*
     * here the overlap is accepted, and stored
     *
     * ??? this should be done elsewhere, because too much overlap
     * may be too much memory, allocated... in generate_one_message()?
     */


    for(i=1, lk=lkref, lp=lpref ; lk!=NIL ; lk=CDR(lk), lp=CDR(lp))
    {
	tag ta = access_tag(INT(CAR(lk)));
	Value vs = vect_coeff(TSHIFTV, (Pvecteur) PVECTOR(CAR(lp)));
	int shift = VALUE_TO_INT(vs);

	if ((ta==aligned_shift) && (shift!=0))
	    set_overlap(array, i, (shift<0)?(0):(1), abs(shift));

	i++;
    }

    return(true); /* accepted! */
}

/*
bool statically_decidable_loops(l)
list l;
{
    range r = ((ENDP(l))?(NULL):(loop_range(LOOP(CAR(l)))));

    return((ENDP(l))?
	   (true):
	   (expression_integer_constant_p(range_lower(r)) &&
	    expression_integer_constant_p(range_upper(r)) &&
	    expression_integer_constant_p(range_increment(r)) &&
	    (HpfcExpressionToInt(range_increment(r))==1) &&
	    statically_decidable_loops(CDR(l))));
}
*/

/*   generate the call to the dynamic loop bounds computation
 */
static statement 
statement_compute_bounds(
    entity newlobnd,
    entity newupbnd, 
    entity oldidxvl, 
    expression lb,
    expression ub,
    int an,
    int dp)
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

	hpfc_killed_scalar = true;
	statement_instruction(stat) =  /* ??? memory leak */
	    make_continue_instruction();
    }
}

/*   true if one statement was killed
 */
static bool hpfc_overlap_kill_unused_scalars(statement stat)
{
    message_assert("defined", !entity_variable_used_undefined_p());

    hpfc_killed_scalar = false;

    gen_recurse(stat, statement_domain,	gen_true,
		hpfc_overlap_kill_unused_scalars_rewrite);

    return(hpfc_killed_scalar);
}

/* returns the dimension of reference on which index entity e is used
 */
static int which_array_dimension(r, e)
reference r;
entity e;
{
    int	dim = 1;
    list li = reference_indices(r);
    Variable v = (Variable) e;

    MAP(EXPRESSION, e,
    {
	normalized n = expression_normalized(e);
	
	if (normalized_linear_p(n) &&
	    (vect_coeff(v, (Pvecteur)normalized_linear(n)) != 0))
	    return(dim);
	
	dim++;
    },
	li);

    return(-1);
}

static loop 
make_loop_skeleton(
    entity newindex,
    expression lower_expression, 
    expression upper_expression)
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

static void 
update_indices_for_local_computation(
    entity_mapping new_indexes,
    list Ref,
    list lRef)
{
    list lr = Ref, lkv = lRef;

    for ( ; (lr!=NIL) ; lr=CDR(lr), lkv=CDR(lkv))
    {
	int dim = 1;
	syntax s = SYNTAX(CAR(lr));
	reference r = syntax_reference(s);
	entity array = reference_variable(r);
	list
	    l1 = CONSP(CAR(lkv)),
	    lk = CONSP(CAR(l1)),
	    li = reference_indices(r),
	    lv = CONSP(CAR(CDR(l1))),
	    li2 = NIL;

	for ( ; (lk!=NIL) ; POP(lk), POP(li), POP(lv))
	{
	    expression indice = EXPRESSION(CAR(li));
	    Pvecteur v = (Pvecteur) PVECTOR(CAR(lv));
	    access ac = INT(CAR(lk));

	    /* caution: only distributed dimensions indexes are modified
	     * other have to remain untouched...
	     * ??? aligned star is missing
	     */
	    switch (access_tag(ac))
	    {
	    case aligned_shift: /* find the new index of the loop */
	    {
		Pvecteur vindex = the_index_of_vect(v);
		entity
		    oldindex = (entity) var_of(vindex),
		    newindex = (entity) GET_ENTITY_MAPPING(new_indexes, 
							   oldindex);
		Value shift = vect_coeff(TSHIFTV, v);

		if (value_zero_p(shift))
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
				       (entity_intrinsic(value_pos_p(shift)?
							(PLUS_OPERATOR_NAME):
							(MINUS_OPERATOR_NAME)),
					entity_to_expression(newindex),
					Value_to_expression(value_abs(shift))),
				       NIL));
		}

		break;
	    }
	    case aligned_constant: /* compute the local indice */
	    {
		Value vval = vect_coeff(TEMPLATEV, v);
		int tval = VALUE_TO_INT(vval);
		
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
		pips_internal_error("part of that function not implemented yet");
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

static statement make_increment_statement(index)
entity index;
{
    return(make_assign_statement
	   (entity_to_expression(index),
	    MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
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
    entity v = reference_variable(r);    

    if (bound_entity_variable_used_p(v))
	update_entity_variable_used(v, load_entity_variable_used(v)+1);
    else
	store_entity_variable_used(v, 1);
}

static void 
initialize_variable_used_map_for_current_loop_nest(
    statement inner_body)
{
    list ll=lloop, lb=lblocks;
    instruction i;
    loop l;

    init_entity_variable_used();     
    current_variable_used_statement = inner_body;

    gen_recurse(inner_body, reference_domain, gen_true,	variable_used_rewrite);

    for (; !ENDP(ll); ll=CDR(ll), lb=CDR(lb))
    {
	l = LOOP(CAR(ll));

	MAP(STATEMENT, s,
	{
	    i = statement_instruction(s);
	    
	    if (!(instruction_loop_p(i) && l==instruction_loop(i)))
		gen_recurse(s,
			    reference_domain,
			    gen_true,
			    variable_used_rewrite);
	    
	},
	    CONSP(CAR(lb)));
    }
}

static void close_variable_used_map_for_statement()
{
    close_entity_variable_used();
    current_variable_used_statement = statement_undefined;
}

static bool variable_used_in_statement_p(ent, stat)
entity ent;
statement stat;
{
    message_assert("current statement", stat==current_variable_used_statement);
    return bound_entity_variable_used_p(ent);
}

static int number_of_distributed_dimensions(a)
entity a;
{
    int p = -1, ndim = NumberOfDimension(a), i = 1, n = 0;

    for (i=1 ; i<=ndim ; i++)
	if (ith_dim_distributed_p(a, i, &p)) n++;
   
    return(n);
}

/* one of the syntax is chosen from the list. The "larger one".
 * and the list is given back, the chosen syntax as first element.
 */
static syntax choose_one_syntax_in_references_list(pls)
list *pls;
{
    list cls = *pls, nls = NIL;
    syntax chosen = SYNTAX(CAR(cls));
    int chosen_distribution = 
	    number_of_distributed_dimensions
		(reference_variable(syntax_reference(chosen)));

    MAP(SYNTAX, current,
    {
	int current_distribution = 
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

static statement 
make_loop_nest_for_overlap(
    list lold,
    list lnew,
    list lbl,
    entity_mapping new_indexes,
    entity_mapping old_indexes, 
    statement innerbody)
{
    entity index, oldindexvalue;
    loop oldloop, newloop;
    list l, lnew_loop = NIL, lnew_body = NIL;
    bool compute_index = false;

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
		     instruction_to_statement(make_instruction(is_instruction_loop,
							 newloop)),
		     NIL);

    /* i = initial_old_value 
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
	    pre = true;

	for(; !ENDP(l); l=CDR(l))
	{
	    s = STATEMENT(CAR(l));
	    i = statement_instruction(s);
	    
	    /*  switch from pre to post.
	     */
	    if (instruction_loop_p(i) && instruction_loop(i)==oldloop)
		pre = false;
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

static bool 
generate_optimized_code_for_loop_nest(
    statement innerbody, 
    statement *pstat,
    syntax the_computer_syntax,
    list Wa,
    list Ra,
    list Ro,
    list lWa,
    list lRa,
    list lRo)
{
    reference the_computer_reference = syntax_reference(the_computer_syntax);
    entity array = reference_variable(the_computer_reference);
    int an = load_hpf_number(array);
    entity_mapping
	new_indexes = MAKE_ENTITY_MAPPING(),
	old_indexes = MAKE_ENTITY_MAPPING();
    list boundcomp = NIL, newloops = NIL;
    statement newnest = NULL;
    range rg;
    expression lb, ub;
    entity index, newindex, newlobnd, newupbnd, oldidxvl;
    loop nl;
    int p, dim;

    MAP(LOOP, l,
     {
	 index = loop_index(l);
	 dim = which_array_dimension(the_computer_reference, index);

	 if (ith_dim_distributed_p(array, dim, &p))
	 {
	     statement bc;

	     /* new bounds to compute, and so on */
	     rg = loop_range(l);
	     lb = copy_expression(range_lower(rg));
	     ub = copy_expression(range_upper(rg));
	     
	     newindex = make_new_scalar_variable(node_module, 
					     MakeBasic(is_basic_int));
	     newlobnd = make_new_scalar_variable(node_module, 
					     MakeBasic(is_basic_int));
	     newupbnd = make_new_scalar_variable(node_module, 
					     MakeBasic(is_basic_int));
	     oldidxvl = make_new_scalar_variable(node_module, 
					     MakeBasic(is_basic_int));
         AddEntityToCurrentModule(newindex);
         AddEntityToCurrentModule(newlobnd);
         AddEntityToCurrentModule(newupbnd);
         AddEntityToCurrentModule(oldidxvl);

	     bc = statement_compute_bounds
		 (newlobnd, newupbnd, oldidxvl, lb, ub, an, p);

	     /* constant new loop bounds are computed on entry
	      * in the subroutine.
	      */
	     if (expression_integer_constant_p(lb) && 
		 expression_integer_constant_p(ub))
		 hpfc_add_ahead_of_node_code(bc);
	     else
		 boundcomp = gen_nconc(boundcomp, CONS(STATEMENT, bc, NIL));

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
    
    /* and now generates the code...
     */
    initialize_variable_used_map_for_current_loop_nest(innerbody);

    if (hpfc_overlap_kill_unused_scalars(innerbody))
    {
	close_variable_used_map_for_statement();
	initialize_variable_used_map_for_current_loop_nest(innerbody);
    }

    newnest = make_loop_nest_for_overlap(lloop, newloops, lblocks,
					 new_indexes, old_indexes, 
					 innerbody);
    close_variable_used_map_for_statement();

    
    (*pstat) = make_block_statement(gen_nconc(boundcomp,
					      CONS(STATEMENT, newnest,
						   NIL)));

    return(true);
}

/* must clear everything before returning in Overlap_Analysis...
 */
#define RETURN(x) \
{ pips_debug(9, "returning %d from line %d\n", x, __LINE__);\
  gen_free_list(Wa); gen_free_list(lWa); gen_free_list(Ra);\
  gen_free_list(lRa); gen_free_list(Ro); gen_free_list(lRo);\
  gen_free_list(Rrt); gen_free_list(lblocks); gen_free_list(lloop);\
  gen_free_list(W); gen_free_list(R); gen_free_list(lw); gen_free_list(lr);\
  reset_hpfc_current_statement(); reset_current_loops(); return x;}

/* check conditions and compile...
 */
bool Overlap_Analysis(stat, pstat)
statement stat, *pstat;
{
    list lw = NIL, lr = NIL, Ra = NIL, Ro = NIL, Rrt = NIL,
	lWa = NIL, lRa = NIL, lRo = NIL, W  = NIL, Wa = NIL, 
        Wrt = NIL, lvect = NIL, lkind = NIL, R=NIL;
    syntax the_computer_syntax = syntax_undefined;
    reference the_computer_reference = reference_undefined;
    statement innerbody, messages_stat, newloopnest;
    bool computer_is_written = true;

    DEBUG_STAT(9, "considering statement", stat);

    set_hpfc_current_statement(stat);
    set_current_loops(stat);

    lblocks = NIL,
    lloop = NIL;
    innerbody = parallel_loop_nest_to_body(stat, &lblocks, &lloop);

    FindRefToDistArrayInStatement(stat, &lw, &lr);

    /* keeps only written references of which dimensions are block distributed,
     * and indices simple enough (=> normalization of loops may be usefull).
     * ??? bug: should also search for A(i,i) things that are forbidden...
     */
    MAP(SYNTAX, s,
    {
	reference r = syntax_reference(s);
	entity array = reference_variable(r);
	
	if ((block_distributed_p(array)) && 
	    (simple_indices_p(r)) && (!replicated_p(array)))
	    W = CONS(SYNTAX, s, W);
	else
	    Wrt = CONS(SYNTAX, s, Wrt);
    },
	lw);

    pips_debug(9, "choosing computer\n");

    if (W) /* ok distributed variable written ! */
    {
	the_computer_syntax = choose_one_syntax_in_references_list(&W);
	the_computer_reference = syntax_reference(the_computer_syntax);
	Wa = CONS(SYNTAX, the_computer_syntax, NIL);
    }
    else  /* must chose the computer among read references! */
    {
	computer_is_written = false;

	MAP(SYNTAX, s,
	{
	    reference r = syntax_reference(s);
	    entity array = reference_variable(r);
	
	    if ((block_distributed_p(array)) && 
		(simple_indices_p(r)) && (!replicated_p(array)))
		R = CONS(SYNTAX, s, R);
	},
	    lr);

	if (R) 
	{
	    the_computer_syntax = choose_one_syntax_in_references_list(&R);
	    the_computer_reference = syntax_reference(the_computer_syntax);
	    Ra = CONS(SYNTAX, the_computer_syntax, NIL);
	}
	else
	    RETURN(false); 
    }

    if (!align_check(the_computer_reference,
		     the_computer_reference, &lvect, &lkind))
	pips_internal_error("no self alignment!");

    if (computer_is_written)
      lWa = CONS(LIST, CONS(LIST, lkind, CONS(LIST, lvect, NIL)), NIL);
    else
      lRa = CONS(LIST, CONS(LIST, lkind, CONS(LIST, lvect, NIL)), NIL);

    pips_debug(9, "checking alignments\n");

    MAP(SYNTAX, s,
    {
	reference r = syntax_reference(s);
	if (the_computer_reference==r) 
	  continue;
	if (align_check(the_computer_reference, r, &lvect, &lkind))
	{
	    if (aligned_p(the_computer_reference, r, lvect, lkind))
	    {
		Wa = gen_nconc(Wa, CONS(SYNTAX, s, NIL));
		lWa = gen_nconc(lWa, CONS(LIST,
					  CONS(LIST, lkind, 
					       CONS(LIST, lvect, NIL)),
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
	W);

    pips_debug(5, "Wa length is %zd (%zd), Wrt lenght is %zd\n",
	       gen_length(Wa), gen_length(lWa), gen_length(Wrt));

    if (gen_length(Wrt)!=0) 
	RETURN(false);

    /* Now, we have the following situation:
     * Wa: set of aligned written refs, the first of which is ``the'' ref.
     */
    MAP(SYNTAX, s,
     {
	 reference r = syntax_reference(s);
	 entity array = reference_variable(r);
	 list lvect = NIL;
	 list lkind = NIL;

	 if (the_computer_reference==r) continue;

	 pips_debug(6, "dealing with reference of array %s\n",
		    entity_name(array));

	 ifdebug(6)
	 {
	     fprintf(stderr, "[Overlap_Analysis]\nreference is:\n");
	     print_reference(r);
	     fprintf(stderr, "\n");
	 }

	 if (align_check(the_computer_reference, r, &lvect, &lkind))
	 {
	     if (aligned_p(the_computer_reference, r, lvect, lkind))
	     {
		 Ra = gen_nconc(Ra, CONS(SYNTAX, s, NIL));
		 lRa = gen_nconc(lRa, CONS(LIST,
					   CONS(LIST, lkind,
						CONS(LIST, lvect, NIL)),
					   NIL));
	     }
	     else
	     if (message_manageable_p(array, lvect, lkind))
	     {
		 Ro = gen_nconc(Ro, CONS(SYNTAX, s, NIL));
		 lRo = gen_nconc(lRo, CONS(LIST,
					   CONS(LIST, lkind, 
						CONS(LIST, lvect, NIL)),
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

    debug(5, "Overlap_Analysis",
	  "Ra length is %d, Ro length is %d, Rrt lenght is %d\n",
	  gen_length(Ra), gen_length(Ro), gen_length(Rrt));

    if (gen_length(Rrt)!=0) 
	RETURN(false);

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
    if (!generate_optimized_code_for_loop_nest
	(innerbody, &newloopnest, the_computer_syntax, 
	 Wa, Ra, Ro, lWa, lRa, lRo))
	RETURN(false);

    DEBUG_STAT(9, entity_name(node_module), newloopnest);

    (*pstat) =
	make_block_statement
	    (CONS(STATEMENT, messages_stat,
	     CONS(STATEMENT,
		  loop_nest_guard(newloopnest,
				  the_computer_reference,
		  CONSP(CAR(CONSP(CAR(computer_is_written? lWa: lRa)))),
		  CONSP(CAR(CDR(CONSP(CAR(computer_is_written? lWa: lRa)))))),
		  NIL)));

    DEBUG_STAT(8, entity_name(node_module), *pstat);

    RETURN(true);
}

/* That is all
 */
