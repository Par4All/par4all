/*
 * Message Utilities
 * 
 * Fabien Coelho, August 1993
 */

/*
 * included files, from C libraries, newgen and pips libraries.
 */

#include <stdio.h>
#include <string.h>

extern int      fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

#include "misc.h"
#include "ri-util.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * Pvecteur the_index_of_vect(v0)
 *
 * returns the index of an affine vector
 */
Pvecteur the_index_of_vect(v0)
Pvecteur v0;
{
    Pvecteur
	v1 = vect_del_var(v0, TCST),
	v2 = vect_del_var(v1, DELTAV),
	v3 = vect_del_var(v2, TEMPLATEV),
	v4 = vect_del_var(v3, TSHIFTV);

    pips_assert("the_index_of_vect", (vect_size(v4)==1));

    vect_rm(v1);
    vect_rm(v2);
    vect_rm(v3);

    return(v4);
}

/*
 * list add_to_list_of_ranges_list(l, r)
 */
list add_to_list_of_ranges_list(l, r)
list l;
range r;
{
    if (ENDP(l)) /* first time */
	return(CONS(CONSP, CONS(RANGE, r, NIL), NIL));

    /* else */
    MAPL(cc,
     {
	 list
	     lr = CONSP(CAR(cc));

	 pips_assert("add_to_list_of_ranges_list", (!ENDP(lr)));

	 lr = gen_nconc(lr, CONS(RANGE, r, NIL));
     },
	 l);

    return(l);
}

/*
 * list dup_list_of_ranges_list(l)
 *
 * ??? the complexity of this function could be greatly improved
 */
list dup_list_of_ranges_list(l)
list l;
{
    list
	result = NIL;
    
    MAPL(cc,
     {
	 list
	     lr = CONSP(CAR(cc));
	 list
	     lrp = NIL;

	 MAPL(cr,
	  {
	      range
		  r = RANGE(CAR(cr));

	      lrp = gen_nconc(lrp, CONS(RANGE, r, NIL));
	  },
	      lr);

	 pips_assert("dup_list_of_ranges_list",
		     gen_length(lrp)==gen_length(lr));

	 result = gen_nconc(result, CONS(CONSP, lrp, NIL));
     },
	 l);

    pips_assert("dup_list_of_ranges_list", 
		gen_length(l)==gen_length(result));

    return(result);
}

/*
 * list dup_list_of_Pvecteur(l)
 *
 * ??? the complexity of this function could be improved greatly...
 */
list dup_list_of_Pvecteur(l)
list l;
{
    list
	result = NIL;

    MAPL(cv,
     {
	 Pvecteur
	     v = PVECTOR(CAR(cv));

	 result = gen_nconc(result, CONS(PVECTOR, vect_dup(v), NIL));
     },
	 l);

    return(result);
}

/*
 * list add_elem_to_list_of_Pvecteur(l, var, val)
 *
 * caution, is initial list is destroied.
 */
list add_elem_to_list_of_Pvecteur(l, var, val)
list l;
int var, val;
{
    list
	result = NIL;

    if (ENDP(l)) 
	return(CONS(PVECTOR, vect_new(var, val), NIL));

    if ((var==0) || (val==0))
	return(l);

    /* else */

    MAPL(cv,
     {
	 Pvecteur
	     v = PVECTOR(CAR(cv));

	 debug(9, "add_elem_to_list_of_Pvecteur",
	       "size of vector %x is %d\n", (int) v, vect_size(v));

	 vect_add_elem(&v, var, val);

	 result = gen_nconc(result, CONS(PVECTOR, v, NIL));
     },
	 l);

    gen_free_list(l);

    return(result);
}
    

/*
 * range complementary_range(array, dim, r)
 */
range complementary_range(array, dim, r)
entity array;
int dim;
range r;
{
    entity
	newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
    dimension
	d = FindIthDimension(newarray, dim);
    int
	rlo = HpfcExpressionToInt(range_lower(r)),
	rup = HpfcExpressionToInt(range_upper(r)),
	rin = HpfcExpressionToInt(range_increment(r)),
	dlo = HpfcExpressionToInt(dimension_lower(d)),
	dup = HpfcExpressionToInt(dimension_upper(d));

    pips_assert("complementary_range", (rin==1));

    if (dup==rup)
	return(make_range(int_to_expression(dlo),
			  int_to_expression(rlo-1),
			  int_to_expression(1)));

    if (dlo==rlo)
	return(make_range(int_to_expression(rup+1),
			  int_to_expression(dup),
			  int_to_expression(1)));

    pips_error("complementary_range",
	       "well, there is a problem here around\n");

    return(range_undefined); /* just to avoid a gcc warning */
}

/*
 * list generate_message_from_3_lists(array, lcontent, lneighbour, ldomain)
 *
 */
list generate_message_from_3_lists(array, lcontent, lneighbour, ldomain)
entity array;
list lcontent, lneighbour, ldomain;
{
    list
	lc = lcontent,
	ln = lneighbour,
	ld = ldomain,
	lm = NIL;
    int
	len = gen_length(lcontent);

    pips_assert("generate_message_from_3_lists",
		((len==gen_length(lneighbour)) && (len==gen_length(ldomain))));

    for ( ; lc!=NIL ; lc=CDR(lc), ln=CDR(ln), ld=CDR(ld))
    {
	lm = CONS(MESSAGE,
		  make_message(array,
			       CONSP(CAR(lc)),
			       PVECTOR(CAR(ln)),
			       CONSP(CAR(ld))),
		  lm);			       
    }

    return(lm);
}

/*
 * bool empty_section_p(lr)
 */
bool empty_section_p(lr)
list lr;
{
    return((ENDP(lr))?
	   (FALSE):
	   (empty_range_p(RANGE(CAR(lr))) || empty_section_p(CDR(lr))));
}

/*
 * bool empty_range_p(r)
 */
bool empty_range_p(r)
range r;
{
    int 
	lo, up;

    /* if we cannot decide, the range is supposed not to be empty */

    if (! (hpfc_integer_constant_expression_p(range_lower(r), &lo) &&
	   hpfc_integer_constant_expression_p(range_upper(r), &up)))
	return(FALSE); 
    else
	return(lo > up);
}


/*
 * void sprint_lrange(str, l)
 */
char *sprint_lrange(str, l)
string str;
list l;
{
    string
	s = str;
    bool
	firstrange = TRUE;

    MAPL(cr,
     {
	 range
	     r = RANGE(CAR(cr));

	 if (!firstrange)
	     s += strlen(sprintf(s, ", "));

	 firstrange = FALSE;
	 s += strlen(sprint_range(s, r));
     },
	 l);

    return(str);
}


/*
 * void sprint_range(s, r)
 */
char *sprint_range(str, r)
string str;
range r;
{
    int
	lo, up, in;
    bool
	blo = hpfc_integer_constant_expression_p(range_lower(r), &lo),
	bup = hpfc_integer_constant_expression_p(range_upper(r), &up),
	bin = hpfc_integer_constant_expression_p(range_increment(r), &in);

    if (blo && bup && bin)
    {
	if (in==1)
	    if (lo==up)
		return(sprintf(str, "%d", lo));
	    else
		return(sprintf(str, "%d:%d", lo, up));
	else
	    return(sprintf(str, "%d:%d:%d", lo, up, in));
    }
    else
	return(sprintf(str, "X"));
}

/*
 * list compute_receive_content(array, lr, v)
 */
list compute_receive_content(array, lr, v)
entity array;
list lr;
Pvecteur v;
{
    list
	content = NIL,
	l = lr;
    int
	i = 1;

    pips_assert("compute_receive_content",
		(NumberOfDimension(array)==gen_length(lr)));

    for ( ; l!=NIL ; i++, l=CDR(l))
    {
	int 
	    procdim = 0;
	bool
	    distributed_dim = ith_dim_distributed_p(array, i, &procdim);
	int 
	    neighbour = ((distributed_dim)?((int) vect_coeff(procdim, v)):(0));
	range
	    r = RANGE(CAR(l));

	if (neighbour!=0)
	{
	    entity
		newarray = (entity) GET_ENTITY_MAPPING(oldtonewnodevar, array);
	    dimension
		nadim = FindIthDimension(newarray, i);
	    expression
		incr  = range_increment(r);
	    int
		lom1  = HpfcExpressionToInt(dimension_lower(nadim))-1,
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

/*
 * list compute_receive_domain(lr, v)
 */
list compute_receive_domain(lr, v)
list lr;
Pvecteur v;
{
    list
	l = lr,
	domain = NIL;
    int
	i = 1;
    
    for ( ; l!=NIL ; i++, l=CDR(l))
    {
	range
	    r = RANGE(CAR(l));
	int
	    neighbour = (int) vect_coeff(i, v);

	if (neighbour==0)
	    domain = gen_nconc(domain, CONS(RANGE, r, NIL)); /* shared! */
	else
	{
	    int
		lo = HpfcExpressionToInt(range_lower(r)),
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

/*
 * bool larger_message_in_list(m, l)
 */
bool larger_message_in_list(m, l)
message m;
list l;
{
    MAPL(cm,
     {
	 message
	     mp = MESSAGE(CAR(cm));

	 if (message_larger_p(mp, m))
	     return(TRUE);
     },
	 l);

    return(FALSE);
}

/* 
 * bool message_larger_p(m1, m2)
 *
 * true if m1>=m2... (caution, it is only a partial order)
 */
bool message_larger_p(m1, m2)
message m1, m2;
{
    if (message_array(m1)!=message_array(m2))
	return(FALSE);

    if ((int) vect_coeff(TCST, (Pvecteur) message_neighbour(m1))!=
	(int) vect_coeff(TCST, (Pvecteur) message_neighbour(m2)))
	return(FALSE);

    /*
     * same array and same destination, let's look at the content and domain...
     */

    return(lrange_larger_p(message_content(m1), message_content(m2)) &&
	   lrange_larger_p(message_dom(m1), message_dom(m2)));
}

/*
 * bool lrange_larger_p(lr1, lr2)
 */
bool lrange_larger_p(lr1, lr2)
list lr1, lr2;
{
    list
	l1 = lr1,
	l2 = lr2;

    pips_assert("lrange_larger_p", gen_length(lr1) == gen_length(lr2));
    
    for ( ; l1!=NIL ; l1=CDR(l1), l2=CDR(l2))
    {
	range
	    r1 = RANGE(CAR(l1)),
	    r2 = RANGE(CAR(l2));
	int
	    lo1 = -1, 
	    up1 = -1, 
	    in1 = -1, 
	    lo2 = -1, 
	    up2 = -1, 
	    in2 = -1;
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

	if (in1!=in2) return(FALSE);
	
	/* ??? something more intelligent could be expected */
	if ((in1!=1) && ((lo1!=lo2) || (up1!=up2))) 
	    return(FALSE);

        if ((in1==1) && ((lo1>lo2) || (up1<up2)))
	    return(FALSE);
    }

    debug(7, "lrange_larger_p", "returning TRUE\n");

    return(TRUE);
}
