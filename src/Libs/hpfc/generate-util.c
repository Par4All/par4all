/* HPFC module by Fabien COELHO
 *
 * $RCSfile: generate-util.c,v $ version $Revision$
 * ($Date: 1995/09/12 14:26:27 $, ) 
 */

#include "defines-local.h"

entity CreateIntrinsic(string name); /* in syntax */

/* builds a statement
 *   VAR_i = MYPOS(i, proc_number) // i=1 to proc dimension
 */
statement 
define_node_processor_id(entity proc,
			 entity (*creation)(int))
{
    int i, procn = load_hpf_number(proc);
    list /* of expression */ mypos_indices,
         /* of statement */  ls = NIL;
    reference mypos;
    entity dummy;

    for(i=NumberOfDimension(proc); i>0; i--)
    {
	dummy = creation(i);
	mypos_indices = CONS(EXPRESSION, int_to_expression(i),
			CONS(EXPRESSION, int_to_expression(procn), 
			     NIL));
	mypos = make_reference(hpfc_name_to_entity(MYPOS), mypos_indices);
	ls = CONS(STATEMENT,
		  make_assign_statement(entity_to_expression(dummy),
					reference_to_expression(mypos)),
		  ls);
    }
				       
    return make_block_statement(ls);
}

/* statement generate_deducables(list le)
 *
 * the le list of expression is used to generate the deducables.
 * The fields of interest are the variable which is referenced and 
 * the normalized field which is the expression that is going to
 * be used to define the variable.
 */
statement 
generate_deducables(list le)
{
    list rev = gen_nreverse(gen_copy_seq(le)), ls = NIL;

    MAP(EXPRESSION, e,
    {
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
    return make_block_statement(ls);
}

list /* of expression */
hpfc_gen_n_vars_expr(entity (*creation)(), int number)
{
    list result = NIL;
    assert(number>=0 && number<=7);

    for(; number>0; number--)
	result = CONS(EXPRESSION, entity_to_expression(creation(number)),
		      result);

    return result;
}

expression 
make_reference_expression(entity e,
			  entity (*creation)(int))
{
    return reference_to_expression(make_reference(e,
	   hpfc_gen_n_vars_expr(creation, NumberOfDimension(e))));
}

/* the following functions generate the statements to appear in
 * the I/O loop nest.
 */
statement 
set_logical(entity log, bool val)
{
    return make_assign_statement
	(entity_to_expression(log),
	 make_call_expression(MakeConstant
	      (val ? ".TRUE." : ".FALSE.", is_basic_logical),
			      NIL));
}

statement 
hpfc_add_n(entity var, int n)
{
    return make_assign_statement
	(entity_to_expression(var),
	 MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME),
			entity_to_expression(var), int_to_expression(n)));
}

/* expr = expr + 2
 */
statement 
hpfc_add_2(exp)
expression exp;
{
    entity plus = CreateIntrinsic(PLUS_OPERATOR_NAME);
    return(make_assign_statement
	   (expression_dup(exp), 
	    MakeBinaryCall(plus, exp, int_to_expression(2))));

}

statement 
hpfc_message(tid, channel, send)
expression tid, channel;
bool send;
{
    expression third = 
	entity_to_expression(hpfc_name_to_entity(send ? INFO : BUFID));
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

/* returns if (LAZY_{SEND,RECV}) then
 */
statement 
hpfc_lazy_guard(bool snd, statement then)
{
    entity decision = hpfc_name_to_entity(snd ? LAZY_SEND : LAZY_RECV);
    return test_to_statement
     (make_test(entity_to_expression(decision), then, make_empty_statement()));
}

/* IF (LAZY_snd) THEN 
 *   PVMFsnd()
 *   LAZY_snd = FALSE // if receive
 * ENDIF
 */
static statement 
hpfc_lazy_message(expression tid, 
		  expression channel, 
		  bool snd)
{
    entity decision = hpfc_name_to_entity(snd ? LAZY_SEND : LAZY_RECV);
    statement 
	comm = hpfc_message(tid, channel, snd),
	then = snd ? comm : 
	    make_block_statement(CONS(STATEMENT, comm,
				 CONS(STATEMENT, set_logical(decision, FALSE),
				      NIL))) ;
    
    return hpfc_lazy_guard(snd, then);
}

statement 
hpfc_generate_message(entity ld, 
		      bool send, 
		      bool lazy)
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

    return lazy ? hpfc_lazy_message(tid, chn, send) : 
	          hpfc_message(tid, chn, send);
}

statement 
hpfc_initsend(lazy)
bool lazy;
{
    statement init;

    /* 2 args to pvmfinitsend
     */
    init = hpfc_make_call_statement
	     (hpfc_name_to_entity(PVM_INITSEND), 
	      CONS(EXPRESSION, MakeCharacterConstantExpression("PVMRAW"),
	      CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(BUFID)),
		   NIL)));

    return lazy ? make_block_statement
	(CONS(STATEMENT, init,
         CONS(STATEMENT, set_logical(hpfc_name_to_entity(LAZY_SEND), FALSE),
	      NIL))) :
		  init ;
}

/****************************************************************** PACKING */

/* returns the buffer entity for array
 */
entity 
hpfc_buffer_entity(entity array)
{
    return hpfc_name_to_entity(concatenate
			       (pvm_what_options(entity_basic(array)),
				BUFFER_SUFFIX, NULL));
}

expression
hpfc_buffer_reference(entity array, entity index)
{
    return reference_to_expression
	(make_reference(hpfc_buffer_entity(array),
         CONS(EXPRESSION, entity_to_expression(index), 
	      NIL)));
}

/* returns PVMF(un)pack(..., array(creation), 1, 1, HPFC_INFO)
 */
static statement 
hpfc_pvm_packing(entity array,
		 entity (*creation)(int), 
		 bool pack)

{
    return hpfc_make_call_statement
	(hpfc_name_to_entity(pack ? PVM_PACK : PVM_UNPACK),
	 CONS(EXPRESSION, pvm_what_option_expression(array),
         CONS(EXPRESSION, make_reference_expression(array, creation),
	 CONS(EXPRESSION, int_to_expression(1),
	 CONS(EXPRESSION, int_to_expression(1),
	 CONS(EXPRESSION, entity_to_expression(hpfc_name_to_entity(INFO)),
		   NIL))))));
}

/* array(creation) = buffer(current++) or inverse...
 */
static statement 
hpfc_buffer_packing(entity array,
		    entity (*creation)(), 
		    bool pack)

{
    entity index = hpfc_name_to_entity(BUFFER_INDEX);
    expression
	array_ref = make_reference_expression(array, creation),
	buffer_ref = hpfc_buffer_reference(array, index);
    statement
	increment = hpfc_add_n(index, 1),
	assignment = make_assign_statement(pack ? buffer_ref : array_ref,
					   pack ? array_ref : buffer_ref);
    
    return make_block_statement(CONS(STATEMENT, increment,
				CONS(STATEMENT, assignment,
				     NIL)));
				     
}

/* returns an packing call for hpfc, that (un)pack array(creation).
 */
statement
hpfc_packing(entity array,
	     entity (*creation)(), 
	     bool pack)
{
    return /* get_bool_property(HPFC_USE_BUFFERS) ?
	hpfc_buffer_packing(array, creation, pack) : */
        hpfc_pvm_packing(array, creation, pack);
}

/* the lazy issues.
 * note that target processors should be known 
 * to generate the appropriate broadcast?
 */
statement 
hpfc_lazy_packing(array, lid, creation, pack, lazy)
entity array, lid;
entity (*creation)();
bool pack, lazy;
{
    statement pack_stmt = hpfc_packing(array, creation, pack);

    return lazy ? (pack ? make_block_statement
       (CONS(STATEMENT, pack_stmt,
	CONS(STATEMENT, set_logical(hpfc_name_to_entity(LAZY_SEND), TRUE),
	     NIL))) :
		   make_block_statement
       (CONS(STATEMENT, hpfc_generate_message(lid, FALSE, TRUE),
	CONS(STATEMENT, pack_stmt,
	     NIL)))) : pack_stmt ;
}

list /* of expression */
make_list_of_constant(val, number)
int val, number;
{
    list l=NIL;

    assert(number>=0);
    for(; number; number--)
	l = CONS(EXPRESSION, int_to_expression(val), l);

    return l;
}

#define psi(i) entity_to_expression(creation(i))

/* statement st_compute_lid(proc)
 *
 *       T_LID=CMP_LID(pn, pi...)
 */
statement 
hpfc_compute_lid(entity lid,
		 entity proc, 
		 entity (*creation)(int))
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
