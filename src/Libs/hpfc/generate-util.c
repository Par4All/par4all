/* HPFC module by Fabien COELHO
 *
 * $RCSfile: generate-util.c,v $ version $Revision$
 * ($Date: 1995/04/21 10:28:19 $, ) 
 */

#include "defines-local.h"

entity CreateIntrinsic(string name); /* in syntax */

/* builds a statement
 *   VAR_i = MYPOS(i, proc_number) // i=1 to proc dimension
 */
statement define_node_processor_id(proc, creation)
entity proc;
entity (*creation)(/* int */);
{
    int i=0, procn = load_hpf_number(proc);
    expression proce = int_to_expression(procn); /* ??? gonna be shared! */
    entity dummy;
    list mypos_indices, ls = NIL;
    reference mypos;

    for(i=NumberOfDimension(proc); i>=1; i--)
	dummy = creation(i),
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
 * the normalized field which is the expression that is going to
 * be used to define the variable.
 */
statement generate_deducables(le)
list le;
{
    list rev = gen_nreverse(gen_copy_seq(le)), ls = NIL;

    MAPL(ce,
     {
	 expression e = EXPRESSION(CAR(ce));
	 entity var = reference_variable(expression_reference(e));
	 Pvecteur v = vect_dup(normalized_linear(expression_normalized(e)));
	 int coef = vect_coeff((Variable) var, v);

	 assert(abs(coef)==1);

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

static list hpfc_gen_n_vars_expr(creation, number)
entity (*creation)();
int number;
{
    list result = NIL;
    assert(number>=0 && number<=7);

    for(; number>0; number--)
	result = CONS(EXPRESSION, entity_to_expression(creation(number)),
		      result);

    return(result);
}

expression make_reference_expression(e, creation)
entity e, (*creation)();
{
    return(reference_to_expression(make_reference(e,
	   hpfc_gen_n_vars_expr(creation, NumberOfDimension(e)))));
}

/* the following functions generate the statements to appear in
 * the I/O loop nest.
 */
statement hpfc_initsend()
{
    /* 2 args to pvmfinitsend
     */
    return
	(hpfc_make_call_statement
	     (hpfc_name_to_entity(PVM_INITSEND), 
	      CONS(EXPRESSION, MakeCharacterConstantExpression("PVMRAW"),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(BUFID)),
		   NIL))));
}

statement hpfc_packing(array, creation, pack)
entity array;
entity (*creation)();
bool pack;
{
    return
	(hpfc_make_call_statement
	     (hpfc_name_to_entity(pack ? PVM_PACK : PVM_UNPACK),
	      CONS(EXPRESSION, pvm_what_option_expression(array),
	      CONS(EXPRESSION, make_reference_expression(array, creation),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, int_to_expression(1),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(INFO)),
		   NIL)))))));
}

list make_list_of_constant(val, number)
int val, number;
{
    list l=NIL;
    int i;
    assert(number>=0);

    for(i=1; i<=number; i++)
	l = CONS(EXPRESSION, make_integer_constant_expression(val), l);

    return(l);
}

/* expr = expr + 2
 */
statement hpfc_add_2(exp)
expression exp;
{
    entity plus = CreateIntrinsic(PLUS_OPERATOR_NAME);
    return(make_assign_statement
	   (expression_dup(exp), 
	    MakeBinaryCall(plus, exp, int_to_expression(2))));

}

statement hpfc_message(tid, channel, send)
expression tid, channel;
bool send;
{
    expression 
	third = entity_to_expression(hpfc_name_to_entity(send ? INFO : BUFID));
    entity pvmf = hpfc_name_to_entity(send ? PVM_SEND : PVM_RECV);

    return(make_block_statement
	   (CONS(STATEMENT, hpfc_make_call_statement(pvmf,
				   CONS(EXPRESSION, tid,
				   CONS(EXPRESSION, channel,
				   CONS(EXPRESSION, third,
					NIL)))),
	    CONS(STATEMENT, hpfc_add_2(copy_expression(channel)),
		 NIL))));				    
}

#define psi(i) entity_to_expression(creation(i))

/* statement st_compute_lid(proc)
 *
 *       T_LID=CMP_LID(pn, pi...)
 */
statement hpfc_compute_lid(lid, proc, creation)
entity lid, proc;
entity (*creation)();
{
    int     ndim = NumberOfDimension(proc);

    if (!get_bool_property("HPFC_EXPAND_CMPLID"))
    {
	entity cmp_lid = hpfc_name_to_entity(CMP_LID);
	
	return(make_assign_statement
   	       (entity_to_expression(lid),
		make_call_expression
		(cmp_lid,
		 CONS(EXPRESSION, 
		      int_to_expression(load_hpf_number(proc)),
		      gen_nconc(hpfc_gen_n_vars_expr(creation, ndim),
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
	
	for(i=2; i<=ndim; i++)
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

/* that is all
 */
