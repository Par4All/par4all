/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
 /*
  * PACKAGE MOVEMENTS
  *
  * Corinne Ancourt  - September 1991
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* #include "values.h" */
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "matrice.h"
#include "tiling.h"
#include "movements.h"
#include "misc.h"
#include "text-util.h"
#include "parser_private.h"
#include "polyedre.h"

void wp65_debug_print_text(entity m, statement s)
{
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    print_text(stderr, text_statement(m, 0, s, NIL));
    debug_off();
}

void wp65_debug_print_module(entity m, statement s)
{
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    text t = text_module(m, s);
    print_text(stderr, t);
    free_text(t);
    debug_off();
}

extern Value offset_dim1;
extern Value offset_dim2;

bool
variable_in_declaration_module_p(m, v)
entity m;
entity v;
{
    value val = entity_initial(m);
    code c = value_code(val);
    cons *d, *cp1;
    d = code_declarations(c);

    if (d == NIL) return(false) ;
    for (cp1 = d ; !ENDP(cp1) ; cp1 = CDR( cp1 ))  {
	if (strcmp(entity_local_name(ENTITY(CAR(cp1))),
		   entity_local_name(v)) == 0)
	    return(true);
    }
    return (false);
}

static
entity
find_entity(entity module, Pvecteur pv,string st)
{
  entity  new_ind;
  string name;
  new_ind=gen_find_tabulated(
      concatenate(entity_local_name(module),
		  MODULE_SEP_STRING,
		  entity_local_name((entity) vecteur_var(pv)),
		  st, (char *) NULL),
      entity_domain);
 
    if (new_ind == entity_undefined) {
	name = strdup(concatenate(entity_local_name((entity) vecteur_var(pv)),
				  st, NULL));
	new_ind = make_scalar_integer_entity(name, entity_local_name(module));
	AddEntityToDeclarations( new_ind,module);
	free(name);
    }
  return(new_ind);
}


entity
find_operator(entity module, string oper, string str )
{
    entity operator;
    string name =  concatenate(TOP_LEVEL_MODULE_NAME,
			       MODULE_SEP_STRING, 
			       module_local_name(module),
			       "_",oper, "_",str,NULL);

    if ((operator = gen_find_tabulated(name, entity_domain))
	== entity_undefined)
	operator=make_entity(strdup(name),
			     make_type(is_type_functional,
				       make_functional(NIL,
						       make_type(is_type_void,
								 UU))),
			     make_storage_rom(),
			     make_value(is_value_unknown,NIL));
    return(operator);
}

statement
make_statement_operator(entity oper,cons * args)
{
  return  make_statement (entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  string_undefined,
			  make_instruction(is_instruction_call,
					   make_call (oper, args)),
			  NIL,
			  NULL,
			  empty_extensions (), make_synchronization_none());
}



/* statement make_movements_loop_body_wp65(module,receive_code,
 *                                         ent,local_indices,var_id,
 *                                         sc_neg,sc_pos,index_base,rank,
 *                                         number_of_lower_bounds,
 *                                         number_of_upper_bounds)
 *
 * This function generates the loop body of the movement code. In the case of
 * bank code generation the loop body  must be :
 *
 *     O1 = .....
 *     O2 = .....
 *     IF (O1.LE.O2) THEN
 *     BANK_/WP65_send/receive_nb_bytes(Prod_id,ES_A(L,O1),O2-O1+1)
 * ES_A is the emulated shared variable given as entity ent.
 * Prod_id is the Processeur id.  given as a Pvecteur in var_id.
 * O and L are the local indices for the bank passed like Pbase in
 * local_indices.
 *
 * In the case of engine code generation the loop body  must be :
 *
 *    LI1 = ....
 *    LI2 = ....
 *    IF (LI1.LE.LI2) THEN
 *     BANK_/WP65_send/receive_nb_bytes(Bank_id,L_A(LJ,LI1),LI2-LI1+1)
 * L_A is the local variable given as entity in ent.
 * Bank_id is the bank id, given as Pvecteur in var_id.
 * LJ, and LI are the local indices passed like Pbase in local_indices
 */
statement
make_movements_loop_body_wp65(module,receive_code,ent,local_indices,var_id,sc_neg,sc_pos,index_base,rank,number_of_lower_bounds,number_of_upper_bounds)
entity module;
bool receive_code;      /* is true if the code is generated for receive */
entity ent;                /* corresponds to  the shared entiy if bank_code
                              and to the local entity otherwise */
Pbase local_indices;       /* correspond to O,L if bank_code and to LI,LJ
			      otherwise */
Pbase var_id;              /* corresponds to the Pvecteur belonging Prod_id
			      if bank_code and Bank_id otherwise */
Psysteme sc_neg,sc_pos;
Pbase index_base;
int rank;
int number_of_lower_bounds,number_of_upper_bounds;
{
    expression lower_bound,upper_bound;
    expression expr_ind1,expr_ind2,expr,expr2,exp_ent,expr_cond;
    entity new_ind1,new_ind2, mod;
    entity operator_assign,operator_minus,operator_plus,operator_le,
    operator_receive,operator_send;
    string lower_or_equal;
    statement stat,stat1,stat2,lbody;
    Pvecteur  pvt,ofs = local_indices;
    type tp = entity_type(ent);
    Value  pmin,pmax;
    int nb_bytes = 0;
    text t; 
    test test1;
     cons * args, * args2, * lex2, * lex3;
    char *str1;
    Psysteme sctmp=NULL;
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"make_movements_loop_body_wp65","begin\n");

    operator_assign=
	gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						ASSIGN_OPERATOR_NAME ),
			   entity_domain);

    operator_minus=
	gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						MINUS_OPERATOR_NAME ),
			   entity_domain);
    operator_plus=
	gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						PLUS_OPERATOR_NAME ),
			   entity_domain);

    if (type_variable_p(tp)) {
	variable var = type_variable(tp);
	basic b = variable_basic(var);
	nb_bytes = SizeOfElements(b);
    }

    str1=i2a(nb_bytes);
    operator_receive = find_operator(module, "RECEIVE",str1);
    operator_send = find_operator(module,"SEND",str1);
    free(str1);
  
    ofs = local_indices;

    /* create the new indices new_ind1 et new_ind2 corresponding to
       LI1 et LI2 when the code is generated for engines
       and O1 et O2 when code is generated for banks
       vecteur_var(ofs) is respectivly LI or O  */



    new_ind1 = find_entity(module, ofs,SUFFIX_FOR_TEMP_VAR1_IN_INNER_LOOP);
    expr_ind1 = make_vecteur_expression(vect_new((char *) new_ind1,
						 vecteur_val(ofs)));

    new_ind2=  find_entity(module, ofs,SUFFIX_FOR_TEMP_VAR2_IN_INNER_LOOP);
    expr_ind2 = make_vecteur_expression(vect_new((char *) new_ind2,
						 vecteur_val(ofs)));

    /* build the expression     new_ind2 - new_ind1+1 */

    lex2 = CONS(EXPRESSION,expr_ind1,NIL);
    expr = make_op_expression(operator_minus,
			      CONS(EXPRESSION,expr_ind2,lex2));

    lex2 = CONS(EXPRESSION,int_to_expression(1),NIL);
    expr2 = make_op_expression(operator_plus,
			       CONS(EXPRESSION,expr,lex2));

    /* build the list of expressions :
       Prod_id,ES_A(L,O1),O2-O1+1  for bank case and
       Bank_id,L_A(LJ,LI1),LI2-LI1+1 for engine case
      
       */
 
    args = CONS(EXPRESSION,expr2,NIL);;
    pvt =vect_new((char *) new_ind1, vecteur_val(ofs));
    vect_add_elem(&pvt,TCST,offset_dim1);
 
    expr_ind1 = make_vecteur_expression(pvt);	

    pvt =(!VECTEUR_NUL_P(ofs->succ))  ?
	vect_add(vect_new(vecteur_var(ofs->succ), VALUE_ONE),
		 vect_new(TCST,offset_dim2)):
		     vect_new(TCST,offset_dim2);
    args2 = CONS(EXPRESSION,
		 make_vecteur_expression(pvt),
		 NIL);
    args2 = CONS(EXPRESSION,expr_ind1,args2);
    exp_ent = make_expression(make_syntax(is_syntax_reference,
					  make_reference(ent,
							 args2
							 )),
			      normalized_undefined);
    args = CONS(EXPRESSION,exp_ent,args);
    args = CONS(EXPRESSION,make_vecteur_expression(vect_dup(var_id)),args);

    /* generate the send or the receive call */

    stat =(receive_code) ? make_statement_operator(operator_receive,args)
	: make_statement_operator(operator_send,args);

    /* build the test around stat */

    sctmp = sc_dup(sc_pos);
    sctmp = sc_append(sctmp,sc_neg);


    (void)sc_minmax_of_variable(sctmp, vecteur_var(ofs), &pmin, &pmax);
    ifdebug(4) {
	fprint_string_Value(stderr, "borne min ", pmin);
	fprint_string_Value(stderr, ", borne sup ", pmax);
	fprintf(stderr,"\n");
    }
    /* if (pmin == INT_MIN || pmax == INT_MAX || pmax > pmin) { */
	new_ind1=find_entity(module, ofs,SUFFIX_FOR_TEMP_VAR1_IN_INNER_LOOP);

	expr_ind1 = make_vecteur_expression(vect_new((char *) new_ind1,
						     vecteur_val(ofs)));

	new_ind2= find_entity(module,ofs,SUFFIX_FOR_TEMP_VAR2_IN_INNER_LOOP);
	expr_ind2 = make_vecteur_expression(vect_new((char *) new_ind2,
						     vecteur_val(ofs)));


	lower_or_equal = concatenate(TOP_LEVEL_MODULE_NAME,
				     MODULE_SEP_STRING,".LE.", NULL);

	if ((operator_le =
	     gen_find_tabulated(lower_or_equal, entity_domain))
	    == entity_undefined)
	    operator_le = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
					     lower_or_equal);
	expr_cond = MakeBinaryCall(operator_le,expr_ind1,expr_ind2);

	test1 =  make_test(expr_cond,stat,make_continue_statement(entity_empty_label()));
	stat = test_to_statement(test1);
    /*    }*/
    /* build the whole code:

       O1 = .....
       O2 = .....
       IF (O1.LE.O2) THEN
       BANK_/WP65_send/receive(Prod_id,ES_A(L,O1),O2-O1+1)  for bank case
       or:
       LI1 = ....
       LI2 = ....
       IF (LI1.LE.LI2) THEN
       BANK_/WP65_send/receive(Bank_id,L_A(LJ,LI1),LI2-LI1+1) for engine case        */
    expr_ind1 = make_vecteur_expression(vect_new((char *) new_ind1,
						 vecteur_val(ofs)));
    expr_ind2 = make_vecteur_expression(vect_new((char *) new_ind2,
						 vecteur_val(ofs)));
    lower_bound = lower_bound_generation(sc_neg,index_base,
					 number_of_lower_bounds,
					 rank);
    lex2 = CONS(EXPRESSION,lower_bound,NIL);
    stat1 = make_statement_operator(operator_assign,
				    CONS(EXPRESSION,expr_ind1,lex2));

    upper_bound = upper_bound_generation(sc_pos,index_base,
					 number_of_upper_bounds,
					 rank);

    lex2 =  CONS(EXPRESSION,upper_bound,NIL);
    stat2 = make_statement_operator(operator_assign,
				    CONS(EXPRESSION,expr_ind2,lex2));
    lex2 = CONS(STATEMENT,stat,NIL);
    lex3 = CONS(STATEMENT,stat2,lex2);

    lbody = make_block_statement(CONS(STATEMENT,stat1,lex3));
    ifdebug(8) {
	mod = local_name_to_top_level_entity(entity_local_name(module));
	t = text_statement(mod, 2, lbody, NIL);
	print_text(stderr,t);
    }
    debug(8,"make_movements_loop_body_wp65","end\n");
    debug_off();


    return (lbody);

}

/* statement make_datum_movement(module,receive_code,ent,
 *                                               local_indices,var_id)
 *
 * This  function generates the loop body of the movement code. In the case of
 * bank code generation the loop body  must be :
 *
 *    BANK_/WP65_ send/receive_nb_bytes(ES_A,O,1,L,Prod_id)
 * ES_A is the emulated shared variable given as entity ent.
 * Prod_id is the Processeur id.  given as a Pvecteur in var_id.
 * O and L are the local indices for the bank passed like Pbase in
 * local_indices.
 *
 * In the case of engine code generation the loop body  must be :
 *
 *     BANK_/WP65_send/receive_nb_bytes(L_A,LI,1,LJ,Bank_id)
 * L_A is the local variable given as entity in ent.
 * Bank_id is the bank id, given as Pvecteur in var_id.
 * LJ, and LI are the local indices passed like Pbase in local_indices
 */
statement make_datum_movement(module,receive_code,ent,local_indices,var_id)
entity module;
bool receive_code;      /* is true if the code is generated for receive */
entity ent;                /* corresponds to  the shared entiy if bank_code
                              and to the local entity otherwise */
Pbase local_indices;       /* correspond to O,L if bank_code and to LJ,LI
			      otherwise */
Pbase var_id;              /* corresponds to the Pvecteur belonging Prod_id
			      if bank_code and Bank_id otherwise */
{
 
    expression exp_ent;
    statement lbody;
    Pvecteur ofs = local_indices;
    Pvecteur pvt =VECTEUR_NUL;
    entity operator_receive,operator_send; 
    cons * args, * args2;
    type tp = entity_type(ent);
    int nb_bytes = 0;
    char *str1;
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"make_datum_movement","begin\n");

    if (type_variable_p(tp)) {
	variable var = type_variable(tp);
	basic b = variable_basic(var);
	nb_bytes = SizeOfElements(b);
    }

    str1=i2a(nb_bytes);
    operator_receive = find_operator(module, "RECEIVE",str1);
    operator_send = find_operator(module,"SEND",str1);
    free(str1);
  
    /* build the list of expressions :
       Prod_id,ES_A(L,O),1  for bank case and
       Bank_id,L_A(LJ,LI),1 for engine case
       */
 
    args = CONS(EXPRESSION,int_to_expression(1),NIL);
    pvt =(!VECTEUR_NUL_P(ofs->succ))  ?
	vect_add(vect_new(vecteur_var(ofs->succ),VALUE_ONE),
		 vect_new(TCST,offset_dim2)):
		     vect_new(TCST,offset_dim2);

   args2 = CONS(EXPRESSION,make_vecteur_expression(pvt),NIL);
    pvt =vect_new(vecteur_var(ofs),VALUE_ONE);
    vect_add_elem(&pvt,TCST,offset_dim1);
    args2 = CONS(EXPRESSION,make_vecteur_expression(pvt),
		 args2);

    exp_ent = make_expression(make_syntax(is_syntax_reference,
					  make_reference(ent,
							 args2
							 )),
			      normalized_undefined);	

    args = CONS(EXPRESSION,exp_ent,args);
    args = CONS(EXPRESSION,make_vecteur_expression(vect_dup(var_id)),args);
    /* generate the send or the receive call */
 
    lbody = (receive_code) ?
	make_statement_operator(operator_receive,args):
	    make_statement_operator(operator_send,args);
	
    debug(8,"make_datum_movement","end\n");
    debug_off();
    return (lbody);

}

/* statement make_movement_scalar_wp65(receive_code,r)
 *
 * This  function generates the loop body of the movement code. In the case of
 * bank code generation the loop body  must be :
 *
 *     call BANK_/WP65_send/receive_nb_bytes(S)
 *
 * In the case of engine code generation the loop body  must be :
 *
 *     call BANK_/WP65_send/receive_nb_bytes(S)
 *
 *   where nb_bytes is the number of bytes needed for the variable location
 */
statement make_movement_scalar_wp65(module,receive_code,r, var_id)
entity module;
bool receive_code;      /* is true if the code is generated for receive */
reference r;                /* corresponds to scalaire entity */
entity var_id;
{
 
    statement lbody;
    cons * args;
    expression expr;
    list lexp1;
    entity
	operator_receive, operator_send,
	var = reference_variable(r); 
    type t;
    int nb_bytes=4;		/* nb_bytes is the number of bytes
				   needed for the variable location  */
    basic bas;
    char *str1;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    pips_debug(8, "begin\n");
    pips_debug(8, "considering %s\n", entity_local_name(var));

    t = entity_type(var);

    if (type_variable_p(t)) {
	variable var = type_variable(t);
	bas = variable_basic(var);
	nb_bytes = SizeOfElements(bas);
    }
    str1=i2a(nb_bytes);
    operator_receive = find_operator(module, "RECEIVE",str1);
    operator_send = find_operator(module,"SEND",str1);
    free(str1);
  
    /* build the  expression :     S    */
    /*    args = CONS(EXPRESSION,int_expr(nb_bytes),NIL);*/
 
    lexp1= CONS(EXPRESSION,int_to_expression(1),NIL);
    expr = make_expression(make_syntax(is_syntax_reference,r
				       ),normalized_undefined);
    args = CONS(EXPRESSION,expr,lexp1);
    args = CONS(EXPRESSION,
		make_vecteur_expression(vect_new((Variable)var_id,
						 VALUE_ONE)),args);
    /* generate the send or the receive call */

    lbody =  (receive_code) ?
	make_statement_operator(operator_receive,args):
	    make_statement_operator(operator_send,args);
    ifdebug(9) {
	pips_debug(9, "returning :\n");
	wp65_debug_print_text(entity_undefined, lbody);
    }

    pips_debug(8,"end\n");
    debug_off();
    return (lbody);

}




