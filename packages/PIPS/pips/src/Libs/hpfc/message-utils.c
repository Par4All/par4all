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
/* Message Utilities
 * 
 * Fabien Coelho, August 1993
 */

#include "defines-local.h"
#include "access_description.h"

/* returns the index of an affine vector
 */
Pvecteur the_index_of_vect(v0)
Pvecteur v0;
{
    Pvecteur v1 = vect_del_var(v0, TCST);

    vect_erase_var(&v1, DELTAV);
    vect_erase_var(&v1, TEMPLATEV);
    vect_erase_var(&v1, TSHIFTV);

    assert(vect_size(v1)==1);

    return v1;
}

list add_to_list_of_ranges_list(l, r)
list l;
range r;
{
    if (ENDP(l)) /* first time */
	return CONS(LIST, CONS(RANGE, r, NIL), NIL);

    /* else */
    MAPL(cc,
     {
	 list lr = CONSP(CAR(cc));
	 assert(!ENDP(lr));
	 lr = gen_nconc(lr, CONS(RANGE, r, NIL));
     },
	 l);

    return(l);
}

/* ??? the complexity of this function could be greatly improved
 */
list dup_list_of_ranges_list(l)
list l;
{
    list result = NIL;

    MAPL(cc,
     {
	 list lr = CONSP(CAR(cc));
	 list lrp = NIL;
	 MAP(RANGE, r, lrp = gen_nconc(lrp, CONS(RANGE, r, NIL)), lr);
	 assert(gen_length(lrp)==gen_length(lr));
	 result = gen_nconc(result, CONS(LIST, lrp, NIL));
     },
	 l);

    assert(gen_length(l)==gen_length(result));
    return result;
}

/* ??? the complexity of this function could be improved greatly...
 */
list dup_list_of_Pvecteur(l)
list l;
{
    list result = NIL;

    MAPL(cv,
     {
	 Pvecteur v = (Pvecteur) PVECTOR(CAR(cv));
	 result = gen_nconc(result, CONS(PVECTOR, (VECTOR) vect_dup(v), NIL));
     },
	 l);

    return result;
}

/* caution, is initial list is destroyed.
 */
list add_elem_to_list_of_Pvecteur(l, var, val)
list l;
int var, val;
{
    list result = NIL;

    if (ENDP(l))
	return CONS(PVECTOR, (VECTOR) vect_new((Variable) (intptr_t)var,
					       int_to_value(val)), NIL);

    if ((var==0) || (val==0))
	return l;

    /* else */

    MAPL(cv,
     {
	 Pvecteur v = (Pvecteur) PVECTOR(CAR(cv));
	 pips_debug(9, "size of vector %p is %d\n", v, vect_size(v));
	 vect_add_elem(&v, (Variable) (intptr_t) var, int_to_value(val));
	 result = gen_nconc(result, CONS(PVECTOR, (VECTOR) v, NIL));
     },
	 l);

    gen_free_list(l);

    return result;
}
    
/* range complementary_range(array, dim, r)
 */
range complementary_range(array, dim, r)
entity array;
int dim;
range r;
{
    entity newarray = load_new_node(array);
    dimension d = FindIthDimension(newarray, dim);
    int	rlo = HpfcExpressionToInt(range_lower(r)),
	rup = HpfcExpressionToInt(range_upper(r)),
	rin = HpfcExpressionToInt(range_increment(r)),
	dlo = HpfcExpressionToInt(dimension_lower(d)),
	dup = HpfcExpressionToInt(dimension_upper(d));

    assert(rin==1);

    if (dup==rup)
	return(make_range(int_to_expression(dlo),
			  int_to_expression(rlo-1),
			  int_to_expression(1)));

    if (dlo==rlo)
	return(make_range(int_to_expression(rup+1),
			  int_to_expression(dup),
			  int_to_expression(1)));

    pips_internal_error("well, there is a problem here around");

    return(range_undefined); /* just to avoid a gcc warning */
}

/* list generate_message_from_3_lists(array, lcontent, lneighbour, ldomain)
 */
list generate_message_from_3_lists(array, lcontent, lneighbour, ldomain)
entity array;
list lcontent, lneighbour, ldomain;
{
    list lc = lcontent, ln = lneighbour, ld = ldomain, lm = NIL;
    int len = gen_length(lcontent);

    assert(len==gen_length(lneighbour) && len==gen_length(ldomain));

    for ( ; lc!=NIL ; lc=CDR(lc), ln=CDR(ln), ld=CDR(ld))
    {
	lm = CONS(MESSAGE,
		  make_message(array,
			       CONSP(CAR(lc)),
			       (Pvecteur)PVECTOR(CAR(ln)),
			       CONSP(CAR(ld))),
		  lm);			       
    }

    return(lm);
}

bool empty_section_p(lr)
list lr;
{
    return((ENDP(lr))?
	   (false):
	   (empty_range_p(RANGE(CAR(lr))) || empty_section_p(CDR(lr))));
}

bool empty_range_p(r)
range r;
{
    int lo, up;

    /*   if we cannot decide, the range is supposed not to be empty
     */
    if (! (hpfc_integer_constant_expression_p(range_lower(r), &lo) &&
	   hpfc_integer_constant_expression_p(range_upper(r), &up)))
	return false; 
    else
	return lo>up;
}

char *sprint_lrange(str, l)
string str;
list l;
{
    string s = str;
    bool firstrange = true;

    MAP(RANGE, r,
    {
	if (!firstrange)
	{
	    sprintf(s, ", "); 
	    s += strlen(s);
	}
	else
	    firstrange = false;

	sprint_range(s, r);
	s += strlen(s);
    },
	l);

    return str;
}

char *sprint_range(string str, range r)
{
    int lo, up, in;
    bool
	blo = hpfc_integer_constant_expression_p(range_lower(r), &lo),
	bup = hpfc_integer_constant_expression_p(range_upper(r), &up),
	bin = hpfc_integer_constant_expression_p(range_increment(r), &in);

    if (blo && bup && bin)
    {
	if (in==1)
	{
	    if (lo==up)
		sprintf(str, "%d", lo);
	    else
		sprintf(str, "%d:%d", lo, up);
	}
	else
	    sprintf(str, "%d:%d:%d", lo, up, in);
    }
    else
	sprintf(str, "X");

    return str+strlen(str);
}

list compute_receive_content(array, lr, v)
entity array;
list lr;
Pvecteur v;
{
    list content = NIL, l = lr;
    int i = 1;

    assert(NumberOfDimension(array)==gen_length(lr));

    for (; l; i++, POP(l))
    {
	int procdim = 0;
	bool distributed_dim = ith_dim_distributed_p(array, i, &procdim);
	Value vn = distributed_dim? 
	    vect_coeff((Variable) (intptr_t)procdim, v): VALUE_ZERO;
	int neighbour = VALUE_TO_INT(vn);
	range r = RANGE(CAR(l));

	if (neighbour!=0)
	{
	    entity newarray = load_new_node(array);
	    dimension nadim = FindIthDimension(newarray, i);
	    expression incr  = range_increment(r);
	    int lom1  = HpfcExpressionToInt(dimension_lower(nadim))-1,
		upp1  = HpfcExpressionToInt(dimension_upper(nadim))+1,
		width = (HpfcExpressionToInt(range_upper(r)) -
			 HpfcExpressionToInt(range_lower(r)));

	    content =
		gen_nconc(content,
			  CONS(RANGE,
			       (neighbour<0)?
			       make_range(int_to_expression(upp1),
					  int_to_expression(upp1+width),
					  incr):
			       make_range(int_to_expression(lom1-width),
					  int_to_expression(lom1),
					  incr),
			       NIL));
	}
	else
	{
	    content = gen_nconc(content, CONS(RANGE, r, NIL)); /* shared! */
	}				
    }

    return(content);
}

list compute_receive_domain(lr, v)
list lr;
Pvecteur v;
{
    list l = lr, domain = NIL;
    int i = 1;
    
    for ( ; l!=NIL ; i++, l=CDR(l))
    {
	range r = RANGE(CAR(l));
	Value vn = vect_coeff((Variable) (intptr_t)i, v);
	int neighbour = VALUE_TO_INT(vn);

	if (neighbour==0)
	    domain = gen_nconc(domain, CONS(RANGE, r, NIL)); /* shared! */
	else
	{
	    int lo = HpfcExpressionToInt(range_lower(r)),
		up = HpfcExpressionToInt(range_upper(r));

	    domain = 
		gen_nconc(domain, 
			  CONS(RANGE, 
			       make_range(int_to_expression(lo+neighbour),
					  int_to_expression(up+neighbour),
					  range_increment(r)), 
			       NIL));
	}	    
    }
    
    return(domain);
}

bool larger_message_in_list(m, l)
message m;
list l;
{
    MAP(MESSAGE, mp, if (message_larger_p(mp, m)) return true, l);
    return(false);
}

/* bool message_larger_p(m1, m2)
 *
 * true if m1>=m2... (caution, it is only a partial order)
 */
bool message_larger_p(m1, m2)
message m1, m2;
{
    if (message_array(m1)!=message_array(m2))
	return(false);

    if (value_ne(vect_coeff(TCST, (Pvecteur) message_neighbour(m1)),
		 vect_coeff(TCST, (Pvecteur) message_neighbour(m2))))
	return(false);

    /*
     * same array and same destination, let's look at the content and domain...
     */

    return(lrange_larger_p(message_content(m1), message_content(m2)) &&
	   lrange_larger_p(message_dom(m1), message_dom(m2)));
}

bool lrange_larger_p(lr1, lr2)
list lr1, lr2;
{
    list l1 = lr1, l2 = lr2;

    assert(gen_length(lr1) == gen_length(lr2));
    
    for ( ; l1; POP(l1), POP(l2))
    {
	range r1 = RANGE(CAR(l1)), r2 = RANGE(CAR(l2));
	int lo1, up1, in1, lo2, up2, in2;
	bool
	    blo1 = hpfc_integer_constant_expression_p(range_lower(r1), &lo1),
	    bup1 = hpfc_integer_constant_expression_p(range_upper(r1), &up1),
	    bin1 = hpfc_integer_constant_expression_p(range_increment(r1), &in1),
	    blo2 = hpfc_integer_constant_expression_p(range_lower(r2), &lo2),
	    bup2 = hpfc_integer_constant_expression_p(range_upper(r2), &up2),
	    bin2 = hpfc_integer_constant_expression_p(range_increment(r2), &in2),
	    bcst = (blo1 && bup1 && bin1 && blo2 && bup2 && bin2);

	if (!bcst) /* can look for formal equality... */
	{
	    return(expression_equal_p(range_lower(r1), range_lower(r2)) &&
		   expression_equal_p(range_upper(r1), range_upper(r2)) &&
		   expression_equal_p(range_increment(r1), range_increment(r2)));
	}

	if (in1!=in2) return(false);
	
	/* ??? something more intelligent could be expected */
	if ((in1!=1) && ((lo1!=lo2) || (up1!=up2))) 
	    return(false);

        if ((in1==1) && ((lo1>lo2) || (up1<up2)))
	    return(false);
    }

    pips_debug(7, "returning TRUE\n");

    return true;
}

/****************************************************************** RANGES */

list 
array_ranges_to_template_ranges(
    entity array,
    list lra)
{
    align a = load_hpf_alignment(array);
    entity template = align_template(a);
    list la = align_alignment(a), lrt = NIL;
    int i = 1;
    
    for (i=1 ; i<=NumberOfDimension(template) ; i++)
    {
	alignment al = FindAlignmentOfTemplateDim(la, i);
	int arraydim = alignment_undefined_p(al)? -1: alignment_arraydim(al);

	if (arraydim==-1) /* scratched */
	{
	    dimension d = FindIthDimension(template, i);
	    lrt = gen_nconc(lrt,
               CONS(RANGE,
		    make_range(copy_expression(dimension_lower(d)),
			       copy_expression(dimension_upper(d)),
			       int_to_expression(1)), NIL));
	} 
	else if (arraydim==0) /* replication */
	{
	    expression b = alignment_constant(al);

	    lrt = gen_nconc(lrt,
			    CONS(RANGE,
				 make_range(b, b, int_to_expression(1)),
				 NIL));
	}
	else
	{
	    range rg = RANGE(gen_nth(arraydim-1, lra));

	    if (expression_undefined_p(range_lower(rg)))
	    {
		/* non distributed dimension, so not used...
		 */
		lrt = gen_nconc(lrt, CONS(RANGE, 
					  make_range(expression_undefined,
						     expression_undefined,
						     expression_undefined),
					  NIL));
	    }
	    else
	    {
		int
		    a  = HpfcExpressionToInt(alignment_rate(al)),
		    b  = HpfcExpressionToInt(alignment_constant(al)),
		    lb, ub, in;
		expression tl, tu, ti;

		lb = HpfcExpressionToInt(range_lower(rg));
		ub = HpfcExpressionToInt(range_upper(rg));
		in = HpfcExpressionToInt(range_increment(rg));
		tl = int_to_expression(a*lb+b);
		tu = int_to_expression(a*ub+b);
		ti = int_to_expression(a*in);

		lrt = gen_nconc(lrt,
				CONS(RANGE,
				     make_range(tl, tu, ti),
				     NIL));
	    }
	}
    }
    
    return(lrt);	    
}

list template_ranges_to_processors_ranges(template, lrt)
entity template;
list lrt;
{
    distribute d = load_hpf_distribution(template);
    entity proc = distribute_processors(d);
    list ld = distribute_distribution(d), lrp = NIL;
    int	i = 1;

    for (i=1 ; i<=NumberOfDimension(proc) ; i++)
    {
	int tdim = 0;
	distribution di = FindDistributionOfProcessorDim(ld, i, &tdim);
	style s = distribution_style(di);
	range rg = RANGE(gen_nth(tdim-1, lrt));
	int p  = 0,
	    tl = HpfcExpressionToInt(range_lower(rg)),
	    tu = HpfcExpressionToInt(range_upper(rg)),
	    n  = HpfcExpressionToInt(distribution_parameter(di));

	switch (style_tag(s))
	{
	case is_style_block:
	{
	    expression
	      pl = int_to_expression(processor_number(template, tdim, tl, &p)),
	      pu = int_to_expression(processor_number(template, tdim, tu, &p));

	    /* ??? another increment could be computed?
	     */
	    lrp = gen_nconc(lrp,
			    CONS(RANGE,
				 make_range(pl, pu, int_to_expression(1)),
				 NIL));
	    break;
	}
	case is_style_cyclic:
	{
	    dimension dp = FindIthDimension(proc, i);
	    int	sz = dimension_size(dp);
	    expression
	      pl = int_to_expression(processor_number(template, tdim, tl, &p)),
	      pu = int_to_expression(processor_number(template, tdim, tu, &p));

	    if (((tu-tl)>n*sz) || (pl>pu) || ((pl==pu) && ((tu-tl)>n)))
		lrp = gen_nconc(lrp,
			    CONS(RANGE,
				 make_range(dimension_lower(dp),
					    dimension_upper(dp),
					    int_to_expression(1)),
				 NIL));
	    else
		lrp = gen_nconc(lrp,
				CONS(RANGE,
				     make_range(pl, pu,
						int_to_expression(1)),
				     NIL));
	    
	    break;
	}
	case is_style_none:
	    pips_internal_error("unexpected none style distribution");
	    break;
	default:
	    pips_internal_error("unexpected style tag (%d)",
		       style_tag(s));
	}
    }
    return(lrp);
}

list array_access_to_array_ranges(r, lkref, lvref)
reference r;
list lkref, lvref;
{
    entity array = reference_variable(r);
    int dim = 1;
    list li  = reference_indices(r), lk  = lkref, lv  = lvref, lra = NIL,
	ldim = variable_dimensions(type_variable(entity_type(array)));

    pips_debug(7, "considering array %s\n", entity_name(array));

    for (; li; POP(li), POP(lk), POP(lv), POP(ldim), dim++)
    {
	access a = INT(CAR(lk));
	Pvecteur v = (Pvecteur) PVECTOR(CAR(lv));
	normalized n = expression_normalized(EXPRESSION(CAR(li)));
	
	pips_debug(8, "DIM=%d[%d]\n", dim, a);

	switch (access_tag(a))
	{
	case local_form_cst:
	{
	    /* this dimension shouldn't be used, 
	     * so I put there whatever I want...
	     * ??? mouais, the information is in the vector, I think.
	     */
	    lra = gen_nconc(lra,
			    CONS(RANGE,
				 make_range(expression_undefined,
					    expression_undefined,
					    expression_undefined),
				 NIL));
	    
	    break;
	}
	case aligned_constant:
	case local_constant:
	{
	    Value tc = vect_coeff(TCST, v);
	    expression arraycell = Value_to_expression(tc);

	    lra = gen_nconc(lra,
			    CONS(RANGE,
				 make_range(arraycell, arraycell,
					    int_to_expression(1)),
				 NIL));
	    break;
	}
	case aligned_shift:
	case local_shift:
	{
	    range
		/* ??? should check that it is ok */
		rg = loop_index_to_range
		    ((entity) var_of(vect_del_var(v, TCST)));
	    expression
		rl = range_lower(rg),
		ru = range_upper(rg),
		in = range_increment(rg),
		al = expression_undefined,
		au = expression_undefined;
	    Value vdt =  vect_coeff(TCST, (Pvecteur) normalized_linear(n));
	    int lb = -1, ub = -1, dt = VALUE_TO_INT(vdt);

	    if (expression_integer_constant_p(rl))
		lb = HpfcExpressionToInt(rl),
		al = int_to_expression(lb-dt);
	    else
		al = copy_expression(dimension_lower(DIMENSION(CAR(ldim))));

	    if (expression_integer_constant_p(ru))
		ub = HpfcExpressionToInt(ru),
		au = int_to_expression(ub-dt);
	    else
		au = copy_expression(dimension_upper(DIMENSION(CAR(ldim))));

	    lra = gen_nconc(lra, CONS(RANGE, make_range(al, au, in), NIL));
	    break;
	}
	case aligned_affine:
	case local_affine:
	{
	    Pvecteur v2 = vect_del_var(v, TCST);
	    entity index = (entity) var_of(v2);
	    range rg = loop_index_to_range(index);
	    Value vdt =  vect_coeff(TCST, (Pvecteur) normalized_linear(n));
	    int	rate = val_of(v2),
	        in = HpfcExpressionToInt(range_increment(rg)),
		lb = HpfcExpressionToInt(range_lower(rg)),
		ub = HpfcExpressionToInt(range_upper(rg)),
	        dt = VALUE_TO_INT(vdt);
	    expression
		ai = int_to_expression(rate*in),
		al = int_to_expression(rate*lb-dt),
		au = int_to_expression(rate*ub-dt);

	    lra = gen_nconc(lra, CONS(RANGE, make_range(al, au, ai), NIL));
	    break;
	}
	default:
	    pips_internal_error("unexpected but maybe legal access");
	    break;
	}
    }
    
    return(lra);
}

/***************************************************************** GUARDS */
 
statement generate_guarded_statement(stat, proc, lr)
statement stat;
entity proc;
list lr;
{
    expression guard;

    return((make_guard_expression(proc, lr, &guard))?
	   (instruction_to_statement
	    (make_instruction(is_instruction_test,
			      make_test(guard,
					stat,
					make_continue_statement
					(entity_empty_label()))))):
	   stat);
}

/* bool make_guard_expression(proc, lr, pguard)
 *
 * compute the expression for the processors ranges lr guard.
 * return true if not empty, false if empty.
 */
bool make_guard_expression(proc, lrref, pguard)
entity proc;
list lrref;
expression *pguard;
{
    int	i, len = -1;
    expression procnum = int_to_expression(load_hpf_number(proc));
    list lr = lrref, conjonction = NIL;
    
    for (i=1 ; i<=NumberOfDimension(proc) ; i++, lr=CDR(lr))
    {
	 range rg = RANGE(CAR(lr));
	 expression
	     rloexpr = range_lower(rg),
	     rupexpr = range_upper(rg);
	 dimension d = FindIthDimension(proc, i);
	 int
	     lo  = HpfcExpressionToInt(dimension_lower(d)),
	     up  = HpfcExpressionToInt(dimension_upper(d)),
	     sz  = up-lo+1,
	     rlo = HpfcExpressionToInt(rloexpr),
	     rup = HpfcExpressionToInt(rupexpr);

	 if (rlo>rup) /* empty match case */
	 {
	     (*pguard) = entity_to_expression(entity_intrinsic(".FALSE."));
	     return(true); /* ??? memory leak with the content of conjonction */
	 }

	 if ((rlo==rup) && (sz!=1) && (rlo>=lo) && (rup<=up))
	 {
	     /* MYPOS(i, procnum).EQ.nn
	      */
	     conjonction = 
		CONS(EXPRESSION,
		     eq_expression(make_mypos_expression(i, procnum), rloexpr),
		     conjonction);
	 }
	 else
	 {
	     if (rlo>lo)
	     {
		 /* MYPOS(i, procnum).GE.(rloexpr)
		  */
		 conjonction =
		   CONS(EXPRESSION,
		     ge_expression(make_mypos_expression(i, procnum), rloexpr),
			conjonction);
	     }

	     if (rup<up)
	     {
		 /* MYPOS(i, procnum).LE.(rupexpr)
		  */
		 conjonction =
		   CONS(EXPRESSION,
		     le_expression(make_mypos_expression(i, procnum), rupexpr),
			conjonction);
	     }
	 }
     }
    
    /* use of conjonction 
     */

    len = gen_length(conjonction);

    if (len==0) /* no guard */
    {
	(*pguard) = expression_undefined;
	return(false);
    }
    else
    {
	(*pguard) = expression_list_to_conjonction(conjonction);
	gen_free_list(conjonction);
	return(true);
    }
}

expression make_mypos_expression(i, exp)
int i;
expression exp;
{
    return(reference_to_expression
	   (make_reference(hpfc_name_to_entity(MYPOS),
			   CONS(EXPRESSION, int_to_expression(i),
			   CONS(EXPRESSION, exp,
				NIL)))));
}

statement loop_nest_guard(stat, r, lkref, lvref)
statement stat;
reference r;
list lkref, lvref;
{
    entity
	array    = reference_variable(r),
	template = array_to_template(array),
	proc     = template_to_processors(template);
    list
	lra = array_access_to_array_ranges(r, lkref, lvref),
	lrt = array_ranges_to_template_ranges(array, lra),
	lrp = template_ranges_to_processors_ranges(template, lrt);
    statement
	result = generate_guarded_statement(stat, proc, lrp);

    gen_free_list(lra); /* ??? memory leak of some expressions */
    gen_free_list(lrt);
    gen_free_list(lrp);

    return(result);
}

/*   That is all
 */
