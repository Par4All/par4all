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
/* Messages handling
 *
 * Fabien Coelho, August 1993
 */

#include "defines-local.h"
#include "access_description.h"

/*  ????
 */
static expression safe_static_domain_bound(array, dim, e, shift, lower)
entity array;
int dim;
expression e;
int shift;
bool lower;
{
    expression result;
    int l=0, u=0, tmp=-1;

    if (hpfc_integer_constant_expression_p(e, &tmp))
	result = int_to_expression(tmp+shift);
    else
    {
	get_entity_dimensions(array, dim, &l, &u);
	result = int_to_expression(lower ? l : u);
    }

    return(result);
}

/* message generate_one_message(array, li, lk, lv)
 *
 * generate one message after the reference given,
 */
static message generate_one_message(array, li, lk, lv)
entity array;
list li, lk, lv;
{
    int i   = 1, len = gen_length(li);
    list lip  = li, lkp  = lk, lvp  = lv, lr   = NIL, ldom = NIL;
    entity newarray = load_new_node(array);
    Pvecteur procv = VECTEUR_NUL;
    
    assert(len==gen_length(lk) && len==gen_length(lv));

    for ( ; i<=len ; i++, lip=CDR(lip), lkp=CDR(lkp), lvp=CDR(lvp))
    {
	tag at = access_tag(INT(CAR(lkp)));
	Pvecteur v  = (Pvecteur) PVECTOR(CAR(lvp));
	expression indice = EXPRESSION(CAR(lip)); 
	dimension newdim = FindIthDimension(newarray, i);
	int
	    arraylo = HpfcExpressionToInt(dimension_lower(newdim)),
	    arrayup = HpfcExpressionToInt(dimension_upper(newdim));

	pips_debug(9, "array %s, dimension %d, access considered is %d\n",
		   entity_local_name(newarray), i, at);

	switch (at)
	{
	case aligned_constant:
	{
	    /* I think I don't need to store the to where value...
	     * that's rather interesting since I do not know where I could
	     * put it...
	     */
	    Value cst = vect_coeff(TCST, v), vt = vect_coeff(TEMPLATEV, v);
	    int tpl = VALUE_TO_INT(vt),
	        localarraycell = template_cell_local_mapping(array, i, tpl);

	    /* no neighbour concerned */
	    /* content: */
	    lr = gen_nconc(lr,
			   CONS(RANGE,
				make_range(int_to_expression(localarraycell),
					   int_to_expression(localarraycell),
					   int_to_expression(1)),
				NIL));
	    /* domain: */
	    ldom = gen_nconc(ldom,
			     CONS(RANGE,
				  make_range(Value_to_expression(cst),
					     Value_to_expression(cst),
					     int_to_expression(1)),
				  NIL));
	    break;
	}
	case aligned_shift: 
	    /* 
	     * note that rate==1 
	     */
	{
	    Pvecteur vindex = the_index_of_vect(v);
	    entity index = (entity) var_of(vindex);
	    range rg = loop_index_to_range(index);
	    Value vs = vect_coeff(TSHIFTV, v), vi = vect_coeff(TCST, v);
	    int	procdim,
	    	shift = VALUE_TO_INT(vs),
		ishft = VALUE_TO_INT(vi),
		n     = DistributionParameterOfArrayDim(array, i, &procdim);
	    expression dom_lb, dom_ub;

	    debug(10, "generate_one_message",
		  "n = %d, just to avoid a gcc warning:-)\n", n);
	    
	    if (shift!=0)
	    {
		/* neighbour to send to */
		procv = vect_add(procv, 
				 vect_new((Variable) (intptr_t)procdim, 
					  (shift<0)?(VALUE_ONE):(VALUE_MONE)));
		
		/* content, on the sender point of view... */
		if (shift<0)
		{
		    lr = gen_nconc
			(lr,
			 CONS(RANGE,
			      make_range(int_to_expression(arrayup+shift+1),
					 int_to_expression(arrayup),
					 int_to_expression(1)),
			      NIL));
		}
		else /* (shift>0) */
		{
		    lr = gen_nconc
			(lr,
			 CONS(RANGE,
			      make_range(int_to_expression(arraylo),
					 int_to_expression(arraylo+shift-1),
					 int_to_expression(1)),
			      NIL));
		}
	    }
	    else /* shift == 0 */
	    {
		/* content: send it all? */
		lr = gen_nconc(lr,
			       CONS(RANGE,
				    make_range(int_to_expression(arraylo),
					       int_to_expression(arrayup),
					       int_to_expression(1)),
				    NIL));
	    }

	    /* assert(gen_consistent_p(rg)); */

	    dom_lb = safe_static_domain_bound(array, i, 
					      range_lower(rg), ishft, true);
	    dom_ub = safe_static_domain_bound(array, i, 
					      range_upper(rg), ishft, false);
	    
	    /* domain */
	    ldom = 
		gen_nconc(ldom,
			  CONS(RANGE,
			       make_range(dom_lb,
					  dom_ub,
					  int_to_expression(1)), /* ??? why 1 */
			       NIL));
	    
	    break;
	}
	case local_constant:
	{
	    Value
		/* ??? bug if there is a declaration shift... */
		arraycell = vect_coeff(TCST, v);

	    /* no neighbour concerned */
	    /* content: */
	    lr = gen_nconc(lr,
			   CONS(RANGE,
				make_range(Value_to_expression(arraycell),
					   Value_to_expression(arraycell),
					   int_to_expression(1)),
				NIL));
	    /* domain: */
	    ldom = gen_nconc(ldom,
			     CONS(RANGE,
				  make_range(Value_to_expression(arraycell),
					     Value_to_expression(arraycell),
					     int_to_expression(1)),
				  NIL));
	    break;
	}
	case local_shift:
	{
	    Pvecteur vindex = the_index_of_vect(v);
	    entity index = (entity) var_of(vindex);
	    range rg = loop_index_to_range(index);
	    Value vs = vect_coeff(TCST, v);
	    int shift = VALUE_TO_INT(vs),
		ishft = shift,
		lb    = HpfcExpressionToInt(range_lower(rg)),
		ub    = HpfcExpressionToInt(range_upper(rg));
	    
	    /* no neighbour */
	    /* content: all shifted space indexed */
	    /* ??? false if declaration shift */
	    lr = 
		gen_nconc(lr,
			  CONS(RANGE,
			       make_range(int_to_expression(lb+shift),
					  int_to_expression(ub+shift),
					  int_to_expression(1)),
			       NIL));

	    /* domain */
	    ldom = 
		gen_nconc(ldom,
			  CONS(RANGE,
			       make_range(int_to_expression(lb+ishft),
					  int_to_expression(ub+ishft),
					  int_to_expression(1)),/* ??? why 1 */
			       NIL));
	    

	    break;
	}
	case local_affine:
	{
	    Pvecteur
		vindex = the_index_of_vect(v);
	    entity
		index = (entity) var_of(vindex);
	    range
		rg = loop_index_to_range(index);
	    Value vs = vect_coeff(TCST, v);
	    int
		rate  = VALUE_TO_INT(val_of(vindex)),
		shift = VALUE_TO_INT(vs),
		ishft = shift,
		in    = HpfcExpressionToInt(range_increment(rg)),
		lb    = HpfcExpressionToInt(range_lower(rg)),
		ub    = HpfcExpressionToInt(range_upper(rg));
	    
	    /* no neighbour */
	    /* content: all shifted space indexed */
	    /* ??? false if declaration shift */
	    lr = gen_nconc(lr,
			   CONS(RANGE,
				make_range(int_to_expression(rate*lb+shift),
					   int_to_expression(rate*ub+shift),
					   int_to_expression(in*rate)),
				    NIL));

	    /* domain */
	    ldom = gen_nconc(ldom,
			     CONS(RANGE,
				  make_range(int_to_expression(rate*lb+ishft),
					     int_to_expression(rate*ub+ishft),
					     int_to_expression(in*rate)), 
				  NIL));
	    

	    break;
	}
	case local_form_cst:
	{
	    /* no neighbour concerned */
	    /* content: */
	    lr = gen_nconc(lr,
			   CONS(RANGE,
				make_range(indice,
					   indice,
					   int_to_expression(1)),
				NIL));
	    /* domain: */
	    ldom = gen_nconc(ldom,
			     CONS(RANGE,
				  make_range(indice,
					     indice,
					     int_to_expression(1)),
				  NIL));
	    break;
	}
	default:
	    pips_internal_error("access tag not welcomed here (%d)", at);
	    break;
	}
    }	    
    
    return(make_message(array, lr, procv, ldom));
}

/* list messages_generation(Ro, lRo)
 *
 * first kind of messages generation 
 * (the overlap management is handled in message_manageable_p())
 */
static list messages_generation(Ro, lRo)
list Ro, lRo;
{
    list
	lm = NIL,
	lr = NIL,
	lp = NIL;
   
    for (lr=Ro, lp=lRo ; lr!=NIL ; lr=CDR(lr), lp=CDR(lp))
    {
	reference
	    r = syntax_reference(SYNTAX(CAR(lr)));
	entity
	    array = reference_variable(r);
	list
	    li  = reference_indices(r),
	    lkv = CONSP(CAR(lp)),
	    lk  = CONSP(CAR(lkv)),
	    lv  = CONSP(CAR(CDR(lkv)));
	int
	    len = gen_length(li);

	assert(len==gen_length(lk) && len==gen_length(lv));

	lm = CONS(MESSAGE,
		  generate_one_message(array, li, lk, lv),
		  lm);	  
    }

    return(lm);
}

static list atomize_one_message(m)
message m;
{
    entity
	array = message_array(m);
    list
	content = message_content(m),
	domain  = message_dom(m),
	lneighbour = NIL,
	lcontent   = NIL,
	ldomain    = NIL,
	cnt = NIL, 
	dom = NIL,
	lm = NIL;
    Pvecteur
	v = (Pvecteur) message_neighbour(m);
    int
	i = 1;
    
    ifdebug(9)
    {
	fprintf(stderr, "[atomize_one_message] ");
	fprint_message(stderr, m);
    }
    
    for (cnt=content, dom=domain ; cnt!=NIL ; cnt=CDR(cnt), dom=CDR(dom), i++)
    {
	int procdim;
	range rcnt = RANGE(CAR(cnt)), rdom = RANGE(CAR(dom));
	bool distributed = ith_dim_distributed_p(array, i, &procdim);
	Value vn = distributed? vect_coeff((Variable) (intptr_t)procdim, v):  VALUE_ZERO;
	int neighbour = VALUE_TO_INT(vn);

	debug(9, "atomize_one_message", "dimension %d\n", i);

	if (neighbour==0)
	{
	    lcontent   = add_to_list_of_ranges_list(lcontent, rcnt);
	    ldomain    = add_to_list_of_ranges_list(ldomain, rdom);
	    lneighbour = add_elem_to_list_of_Pvecteur(lneighbour, 0, 0);
	}
	else /* let's count in binary mode */
	{
	    list
		lc = dup_list_of_ranges_list(lcontent),
		ld = dup_list_of_ranges_list(ldomain),
		ln = dup_list_of_Pvecteur(lneighbour);
	    
	    lcontent = 
		add_to_list_of_ranges_list(lcontent,
					   complementary_range(array, i, rcnt));
	    ldomain    = add_to_list_of_ranges_list(ldomain, rdom);
	    lneighbour = add_elem_to_list_of_Pvecteur(lneighbour, 0, 0);

	    lc = add_to_list_of_ranges_list(lc, rcnt);
	    ld = add_to_list_of_ranges_list(ld, rdom);
	    ln = add_elem_to_list_of_Pvecteur(ln, procdim, neighbour);

	    lcontent   = gen_nconc(lcontent, lc);
	    ldomain    = gen_nconc(ldomain, ld);
	    lneighbour = gen_nconc(lneighbour, ln);
	}	
    }

    /*
     * message generation from the lists
     */

    lm = generate_message_from_3_lists(array, lcontent, lneighbour, ldomain);

    gen_free_list(lcontent);
    gen_free_list(lneighbour);
    gen_free_list(ldomain);

    return(lm);
}

static list messages_atomization(lm1)
list lm1;
{
    list lm2 = NIL;

    MAP(MESSAGE, m, lm2 = gen_nconc(atomize_one_message(m), lm2), lm1);

    return lm2;
}

static list keep_non_empty_messages_with_destination(l)
list l;
{
    list result = NIL;

    MAP(MESSAGE, m,
     {
	 if ((!VECTEUR_NUL_P((Pvecteur) message_neighbour(m))) &&
	     (!empty_section_p(message_content(m))))
	     result = CONS(MESSAGE, m, result);
	 /* ??? else memory leak */
     },
	 l);

    return(result);
}

static list keep_non_empty_domain_messages(l)
list l;
{
    list
	result = NIL;
    
    MAP(MESSAGE, m,
     {
	 if (!empty_section_p(message_dom(m)))
	     result = CONS(MESSAGE, m, result);
	 /* ??? else memory leak */
     },
	 l);

    return(result);
}

/* message shape_one_message(m)
 *
 * caution, rate==1 is assumed.
 */
static message shape_one_message(m)
message m;
{
    entity
	array = message_array(m);
    list 
	lr = message_content(m),
	ld = message_dom(m),
	lnewdomain = NIL;
    int
	procdim = 0,
	i = 1;

    for ( ; lr!=NIL ; i++, lr=CDR(lr), ld=CDR(ld))
    {
	range
	    d = RANGE(CAR(ld)),
	    r = RANGE(CAR(lr));

	if (ith_dim_distributed_p(array, i, &procdim)) /* ??? is that all ? */
	{
	    int
		newdlo = 0,
		newdup = 0,
		dlo = HpfcExpressionToInt(range_lower(d)),
		dup = HpfcExpressionToInt(range_upper(d)),
		rlo = HpfcExpressionToInt(range_lower(r)),
		rup = HpfcExpressionToInt(range_upper(r)),
		n = DistributionParameterOfArrayDim(array, i, &procdim),
		localdlo = global_array_cell_to_local_array_cell(array, i, dlo),
		localdup = global_array_cell_to_local_array_cell(array, i, dup);
	    
	    assert(rlo<=rup); /* message content isn't empty */
	    
	    /*
	     * I thought a long time about it, just for these two formulas:
	     */
	    newdlo = dlo+(rlo-localdlo)+((localdlo>rup)?(n):(0));
	    newdup = dup+(rup-localdup)-((localdup<rlo)?(n):(0));
	    
	    lnewdomain = gen_nconc(lnewdomain,
				   CONS(RANGE,
					make_range(int_to_expression(newdlo),
						   int_to_expression(newdup),
						   int_to_expression(1)),
					NIL));
	}
	else
	{
	    lnewdomain = gen_nconc(lnewdomain, CONS(RANGE, d, NIL));
	}					
    }
    
    gen_free_list(message_dom(m));
    message_dom(m) = lnewdomain;
    
    return(m);
}

static list messages_shaping(l)
list l;
{
    list result = NIL;
    
    MAP(MESSAGE, m, result = CONS(MESSAGE, shape_one_message(m), result), l);

    return(result);
}

static message one_message_guards_and_neighbour(m)
message m;
{
    Pvecteur
	v = (Pvecteur) message_neighbour(m);
    entity
	array     = message_array(m),
	template  = array_to_template(array),
	processor = template_to_processors(template);
    list
	lra = message_dom(m),
	lrt = array_ranges_to_template_ranges(array, lra),
	lrp = template_ranges_to_processors_ranges(template, lrt);
    int
	i = 1, t = 0, procndim = NumberOfDimension(processor);
    Value vt;

    gen_free_list(lra);
    gen_free_list(lrt);

    message_dom(m) = lrp;

    /* now the neighbour computation in the linearized processor
     * arrangment named NODETIDS() of common /HPFC_PARAM/.
     * the code is similar to the runtime support function HPFC_PROCLID().
     */

    assert(procndim>=1);

    vt = vect_coeff((Variable) 1, v);
    t = VALUE_TO_INT(vt);
    for (i=2 ; i<=NumberOfDimension(processor) ; i++)
    {
	Value vi = vect_coeff((Variable) (intptr_t) i, v);
	t = t*SizeOfIthDimension(processor, i) + VALUE_TO_INT(vi) ;
    }

    message_neighbour_(m) =
	newgen_Pvecteur(vect_add(v, vect_new(TCST, int_to_value(t))));

    return(m);
}

static list messages_guards_and_neighbour(l)
list l;
{
    list result = NIL;

    MAP(MESSAGE, m,
	result = CONS(MESSAGE, one_message_guards_and_neighbour(m), result),
	l);
    
    return result;
}


static message one_receive_message(send)
message send;
{
    entity
	array = message_array(send);
    Pvecteur
	sendneighbour = (Pvecteur) message_neighbour(send),
	neighbour = vect_new(TCST, 
			     value_uminus(vect_coeff(TCST, sendneighbour)));
    list
	content = compute_receive_content(array, message_content(send), 
					  sendneighbour),
	domain  = compute_receive_domain(message_dom(send), sendneighbour);

    return(make_message(array, content, neighbour, domain));
}

/* list receive_messages_generation(lms)
 *
 * this function must warranty that all messages send will be received,
 * so the messages are symetric.
 */
static list receive_messages_generation(lms)
list lms;
{
    list lmr = NIL;

    MAP(MESSAGE, send,
    {
	lmr = gen_nconc(lmr, CONS(MESSAGE,
				  one_receive_message(send),
				  NIL));
    },
	lms);

    return(lmr);
}

/* statement st_one_message(m, bsend)
 *
 * this function will have to be modified in order to include 
 * message coalescing and aggregation. Here, a direct call to
 * a function that packs and passes the message is made.
 */
static statement st_one_message(m, bsend)
message m;
bool bsend;
{
    char buffer[100], *buf = buffer;
    entity array     = message_array(m),
	   processor = array_to_processors(array);
    list content = message_content(m),
	 domain  = message_dom(m);
    Value vn = vect_coeff(TCST, (Pvecteur) message_neighbour(m));
    int neighbour = VALUE_TO_INT(vn);
    statement
	cmpneighbour = st_compute_neighbour(neighbour),
	pass = (bsend ? st_send_to_neighbour() : st_receive_from_neighbour()),
	pack = st_generate_packing(array, content, bsend);
    list interm = CONS(STATEMENT, 
		      cmpneighbour,
		      (bsend ?
		       CONS(STATEMENT, pack, CONS(STATEMENT, pass, NIL)) :
		       CONS(STATEMENT, pass, CONS(STATEMENT, pack, NIL))));
    statement
	result = generate_guarded_statement(make_block_statement(interm),
					    processor,
					    domain);
    
    /* comment generation to improve readibility of the code
     */
    sprintf(buf, "! %s(", entity_local_name(processor));
    buf += strlen(buf);
    sprint_lrange(buf, domain);
    buf += strlen(buf);
    sprintf(buf, ") %s %s(", bsend ? "send" : "receive",
	    entity_local_name(array));
    buf += strlen(buf);
    sprint_lrange(buf, content);
    buf += strlen(buf);
    sprintf(buf, ") %s (%s%d)\n", bsend ? "to" : "from", 
	    neighbour>0 ? "+" : "-", abs(neighbour));
    buf += strlen(buf);

    insert_comments_to_statement(result, buffer);
    return result;
}

static list generate_the_messages(lm, bsend)
list lm;
bool bsend;
{
    list l = NIL;

    MAP(MESSAGE, m, l = CONS(STATEMENT, st_one_message(m, bsend), l), lm);

    return l;
}

/* list remove_stammering_messages(lm)
 *
 * remove messages that are contained in another message
 */
static list remove_stammering_messages(lm)
list lm;
{
    list kept = NIL;

    MAPL(cm,
    {
	message m = MESSAGE(CAR(cm));

	if (!larger_message_in_list(m, kept) &&
	    !larger_message_in_list(m, CDR(cm)))
	    kept = CONS(MESSAGE, m, kept);
    },
	lm);

    return kept;
}

/* every required conditions are supposed to be verified in this function.
 */
statement messages_handling(Ro, lRo)
list Ro, lRo;
{
    list
	lm1  = NIL,
	lm2  = NIL,
	lm2p = NIL,
	lm3  = NIL,
	lm3p = NIL,
	lm4  = NIL,
	lms  = NIL,
	lmr  = NIL,
	lsend = NIL,
	lreceive = NIL;
    int
	len = gen_length(Ro);
    
    assert(len==gen_length(lRo) && len>=1);

    /* first kind of messages generation
     *
     * a message is 
     * an array,
     * a mixed content on the local array,
     * a set of neighbours (in a Pvecteur) and
     * an array section concerned.
     */
    lm1 = messages_generation(Ro, lRo);

    debug(6, "messages_handling", "lm1 length is %d\n", gen_length(lm1));
    ifdebug(8)
    {
	fprint_lmessage(stderr, lm1);
    }

    /*
     * second kind of messages generation
     *
     * messages are atomized to real messages:
     * a message is 
     * an array, 
     * a content (on the local array),
     * a neighbour to send to (in a Pvecteur),
     * a domain of the array concerned.
     */
    lm2 = messages_atomization(lm1);
    gen_free_list(lm1);

    debug(6, "messages_handling",
	  "lm2 length is %d\n", gen_length(lm2));
    ifdebug(8)
    {
	fprint_lmessage(stderr, lm2);
    }

    lm2p = keep_non_empty_messages_with_destination(lm2);
    gen_free_list(lm2);

    debug(6, "messages_handling",
	  "lm2p length is %d\n", gen_length(lm2p));
    ifdebug(8)
    {
	fprint_lmessage(stderr, lm2p);
    }


    /*
     * third kind of messages generation
     *
     * the domain is restricted to what is necessary for the
     * given message, and this on every distributed dimension...
     * this is important for the guards generation later on.
     */
    lm3 = messages_shaping(lm2p);
    gen_free_list(lm2p);

    debug(6, "messages_handling",
	  "lm3 length is %d\n", gen_length(lm3));
    ifdebug(8)
    {
	fprint_lmessage(stderr, lm3);
    }


    lm3p = keep_non_empty_domain_messages(lm3);
    gen_free_list(lm3);

    debug(6, "messages_handling", "lm3p length is %d\n", gen_length(lm3p));
    ifdebug(8) fprint_lmessage(stderr, lm3p);

    /*
     * fourth kind of messages generation
     * 
     * the array section domain is translated into a processors section,
     * and the neighbour shift is computed for the linearized processors 
     * arrangement considered in the runtime resolution.
     */
    lm4 = messages_guards_and_neighbour(lm3p);
    gen_free_list(lm3p);
    
    debug(6, "messages_handling", "lm4 length is %d\n", gen_length(lm4));
    ifdebug(8) fprint_lmessage(stderr, lm4);

    /* here should be performed some message coalescing and/or aggregation:
     * a first simple version could check for messages sent twice, and so on.
     */
    hpfc_warning("messages coalescing and aggregation not implemented\n");

    lms = remove_stammering_messages(lm4);

    /*  RECEIVE 
     */
    lmr = receive_messages_generation(lms);

    assert(gen_length(lmr)==gen_length(lms));

    ifdebug(8)
    {
	fprintf(stderr, "[message handling] lmr and lms\n");
	fprint_lmessage(stderr, lmr);
	fprint_lmessage(stderr, lms);
    }

    lsend = generate_the_messages(lms, SEND);
    lreceive = generate_the_messages(lmr, RECEIVE);

    return(make_block_statement(gen_nconc(lsend, lreceive)));    
}

/*   that is all
 */
