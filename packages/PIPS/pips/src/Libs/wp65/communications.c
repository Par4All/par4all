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
/*  Computation of communications needed  
 * for generating distributed code in PUMA
 *
 * Corinne Ancourt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "matrice.h"
#include "tiling.h"
#include "database.h"
#include "text.h"

#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "misc.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "resources.h"
#include "movements.h"
#include "constants.h"
#include "wp65.h"

/* This function associates to each variable in the call statement  the statement 
 * where it should be communicated  (this statement is often external to the call)
*/

void 
call_instruction_to_communications(
    statement s,
    statement st_level1,
    statement st_level2,
    list *lwr,
    list *lwr_local, 
    statement_mapping *fetch_map,
    statement_mapping *store_map, 
    hash_table r_to_ud,
    list *lpv) 
{
    if(assignment_statement_p(s)) {
	reference r;
	entity rv;
	list lexpr =
	    call_arguments(instruction_call(statement_instruction(s)));
	/* first reference in statement s */
	bool first_reference = true;

	for(; !ENDP(lexpr); POP(lexpr)) {
	    expression e = EXPRESSION(CAR(lexpr));
	    list lr = expression_to_operand_list(e, NIL);
	    if (lr != NIL) {
		list consr = lr;
		ifdebug(2) {
		    (void) fprintf(stderr, "reference list:");
		    print_reference_list(lr);
		    (void) fprintf(stderr, "first_reference=%s\n",
				   bool_to_string(first_reference));
		}
		for(consr = lr; !ENDP(consr) ; POP(consr)) {
		    r = REFERENCE(CAR(consr));
		    rv = reference_variable(r);
		    if (first_reference) {
			ifdebug(2) {
			    (void) fprintf(stderr,"list lwr_local:");
			    reference_list_print(*lwr_local);}
			reference_list_update(lwr_local,r);
			update_map(*store_map,st_level2,r);
			first_reference = false;
		    }
		    else {
			 if(!entity_is_argument_p(rv, *lpv)) {
			   debug(8,"loop_nest_to_local_variables",
			   "Variable %s is not private\n",  
				 entity_local_name(rv));
			ifdebug(2) { 
			    (void) fprintf(stderr,"list lwr_local:");
			    reference_list_print(*lwr_local);
			    (void) fprintf(stderr,"list lwr : ");
			    reference_list_print(*lwr);} 

			if (reference_scalar_p(r)) {
			    if (!reference_in_list_p(r,*lwr_local))
			    update_map(*fetch_map,st_level1,r); 
			}
			else {
			    if (array_indice_in_list_p(r,*lwr_local))
				update_map(*fetch_map,s,r);
			    else if (array_indice_in_list_p(r,*lwr))
				update_map(*fetch_map,st_level2,r);
			    else update_map(*fetch_map,st_level1,r);
			}
		       }
		    }
		}
	    }
	}
    }
    else { 
	ifdebug(9) 
	    (void) fprintf(stderr, 
			   "call-communications - not assignment\n");
    }
}

/* This function associates to each variable in the loop the statement 
 * where it should be communicated  (this statement may be external to the loop).
*/


void loop_instruction_to_communications(statement stmt, statement st_level1, 
					statement st_level2, list *lwr, list *lwr_local, 
					statement_mapping *fetch_map,statement_mapping *store_map,
					hash_table r_to_ud,list *lpv)
{
    instruction inst = statement_instruction(stmt);
    statement b = loop_body(instruction_loop(inst));
    instruction inst2=  (instruction) statement_instruction(b);
    reference_list_add(lwr, lwr_local);
    reference_list_update(lwr,
			  make_reference(loop_index(instruction_loop(inst)),
					 NIL));
    *lwr_local = NIL; 
    *lpv=loop_locals(instruction_loop(inst));

    switch(instruction_tag(inst2)) {
    case is_instruction_block: 
	st_level2 = STATEMENT(CAR(instruction_block(inst)));
	break;
    case is_instruction_call: 
	st_level2 =b;
	break;
    default:
	(void) fprintf(stderr,
		       "loop_instruction_to_communications: non implemented case \n");
	break;
    }
    statement_to_communications(b,st_level1, st_level2,
			     lwr,lwr_local,
			     fetch_map,store_map,r_to_ud,lpv);

}

/* This function associates to each variable in stmt  the 
 *  statement where it should  be communicated.
 * The lwr list corresponds to the list of variables that have 
 * been updated before the current statement bloc. 
 * The lwr_local list corresponds to the list of variables 
 * that are updated in the current statement bloc.
*/

void statement_to_communications(statement stmt, statement st_level1, statement st_level2, 
				 list * lwr,list * lwr_local,
				 statement_mapping *fetch_map,
				 statement_mapping * store_map,
				 hash_table r_to_ud, list *lpv)
{
    instruction inst = statement_instruction(stmt);
    
    debug(8, "statement_to_communications", "begin with tag %d\n", 
	  instruction_tag(inst));

    switch(instruction_tag(inst)) {
    case is_instruction_block: {
	ifdebug(7) 
	    (void) fprintf(stderr,
			   "statement_to_communications-instruction block- begin\n");
	st_level2 = STATEMENT(CAR(instruction_block(inst)));
	MAPL( sts, {
	    statement s = STATEMENT(CAR(sts));
	    statement_to_communications(s,st_level1, st_level2,
					lwr,lwr_local,
					fetch_map,store_map, 
					r_to_ud,lpv);
	}, instruction_block(inst));
	ifdebug(7) 
	    (void) fprintf(stderr,
			   "statement_to_communications-instruction block- end\n");
	break;
    }
    case is_instruction_test:
	(void) fprintf(stderr,"not implemented\n");
	break;
    case is_instruction_loop: {
	ifdebug(7)
	    (void) fprintf(stderr,
			   "statement_to_communications-instruction loop- begin\n");
	loop_instruction_to_communications(stmt, stmt, stmt, 
					   lwr, lwr_local,
					   fetch_map,store_map,
					   r_to_ud,lpv);
	reference_list_add(lwr, lwr_local);
	ifdebug(7)
	    (void) fprintf(stderr,
			   "statement_to_communications-instruction loop- end\n");
	break;
    }
    case is_instruction_call: {
	ifdebug(7)
	    (void) fprintf(stderr,"statement_to_communications-instruction call- begin\n");
	call_instruction_to_communications(stmt, st_level1, st_level2, 
					   lwr, lwr_local,
					   fetch_map,store_map,r_to_ud,lpv);
	ifdebug(7)
	    (void) fprintf(stderr,"statement_to_communications-instruction call- end\n");
	break; 
    }
    case is_instruction_goto: {
	pips_internal_error("Unexpected goto");
	break;}
    case is_instruction_unstructured: {
	pips_internal_error("Sorry: unstructured not implemented");
	break;}
    default: 
	pips_internal_error("Bad instruction tag");
    }
}


/* This function associates to each variable  the statement in l where it should be communicated 
 * Fecth_map contains for each statement the list of variables having to be communicated 
 * before its execution. 
 * Store_map contains for each statement the list of variables having to be communicated 
 * after its execution.
*/

void 
compute_communications(list l, statement_mapping *fetch_map,statement_mapping *store_map) 
{
    list lwr=NIL;			/* list of written variables */
    list lwr_local= NIL;		/* list of variables written in a local
					   instruction block */
    list lpv = NIL;			/* list of privates variables */
    hash_table r_to_ud1 = hash_table_make(hash_pointer,0);
    statement first_stmt = STATEMENT(CAR(l));
    MAPL(pm,{
	statement s1 = STATEMENT(CAR(pm));  
	statement_to_communications(s1,first_stmt,s1,&lwr, &lwr_local,
				    fetch_map,store_map, r_to_ud1,&lpv);
    },l);

}

static list 
constant_symbolic_communication(
    entity compute_or_memory_module,list lrefs,
    bool load_code,entity var_id) 
{
    /* bool load_code  is true if the generated computational code 
       must be a RECEIVE, false if it  must be a SEND*/

    list lrs;
    list ccode = NIL; /* movements for the scalar variables 
			 for the compute module or  the memory module */
    for (lrs =lrefs ; !ENDP(lrs) ; POP(lrs)) {
	reference r = REFERENCE(CAR(lrs));
	statement sblock=
	    make_movement_scalar_wp65(compute_or_memory_module,load_code,
				      r,var_id);
	ccode = gen_nconc(ccode,CONS(STATEMENT, sblock, NIL));
    } 
    return ccode;
}

void 
include_constant_symbolic_communication(
    entity compute_or_memory_module,list lrefs,
    bool load_code,statement computational_or_emulator,
    entity var_id) 
{ 
    instruction i;
    list ccode = constant_symbolic_communication(compute_or_memory_module,lrefs, 
						   load_code,var_id); 
    /* Add data movements to the appropriated module */
    i = statement_instruction(computational_or_emulator);
    instruction_block(i) = gen_nconc(instruction_block(i), ccode);
}

static list 
array_indices_communication(
    entity compute_or_memory_module,
    Pbase bank_indices, 
    int bn,
    int ls,
    list lrefs,
    bool load_code,
    entity var_id,
    Pbase loop_indices,
    tiling tile, 
    Pvecteur tile_delay, 
    Pvecteur tile_indices, 
    Pvecteur tile_local_indices)
{
    list gcode =NIL;
    list icode = NIL;
    list ind=NIL;
    list lrs = NIL;
    statement stat;
    Pbase bas_var[3];
    Pvecteur pv1,pv2,pv,pvi;
    int i,j;
    Value coef;
   
    /* movements for the scalar variables 
       for the compute module or  the memory module */
    bas_var[1] =  VECTEUR_NUL;
    bas_var[2] = VECTEUR_NUL;
    for (lrs =  lrefs;  !ENDP(lrs); POP(lrs)) { 
	reference r = REFERENCE(CAR(lrs));
	Variable vbank,vligne,vofs;
	expression exp1,exp2,exp3,exp4;
	list lex2;
	type t; 
	Value ms=VALUE_ZERO;
	for(ind= reference_indices(r),i=1 ; !ENDP(ind);POP(ind),i++) { 
	    expression e = EXPRESSION(CAR(ind));
	    normalized norm = NORMALIZE_EXPRESSION(e);
	    if (normalized_linear_p(norm))
		bas_var[i]=vect_dup((Pvecteur) normalized_linear(norm));
	    else {  normalized norm=NormalizeSyntax(expression_syntax(e));
		    bas_var[i]=vect_dup((Pvecteur) normalized_linear(norm));
		    fprintf(stderr,
			    "[array_indices_communication ERROR]--> NON LINEAR funct.\n");
		}
	}

 	t = entity_type(reference_variable(r));
	if (type_variable_p(t)) {
	    variable var = type_variable(t);
	    cons * dims = variable_dimensions(var);
	    dimension dim1 = DIMENSION(CAR(dims));
	    expression lower= dimension_lower(dim1);
	    normalized norm1 = NORMALIZE_EXPRESSION(lower);
	    expression upper= dimension_upper(dim1);
	    normalized norm2 = NORMALIZE_EXPRESSION(upper);
	    Value min_ms = VALUE_ZERO;
	    Value max_ms = VALUE_ZERO;
	    if (normalized_linear_p(norm1) && normalized_linear_p(norm2)) {
		min_ms=vect_coeff(TCST,(Pvecteur) normalized_linear(norm1));
		max_ms=vect_coeff(TCST,(Pvecteur) normalized_linear(norm2));
	    }
	    ms = value_minus(max_ms,min_ms);
	    value_increment(ms);
	}
	
	vbank = (Variable) var_id;
	vligne = vecteur_var(bank_indices->succ);
	vofs = vecteur_var(bank_indices->succ->succ);
	if (VECTEUR_NUL_P(bas_var[2]))
	    bas_var[2] = vect_new(TCST,VALUE_ONE);
	for (i=1;i<=2;i++) {	
	    for (pvi = loop_indices,j=1; !VECTEUR_NUL_P(pvi); 
		 pvi = pvi->succ,j++) {
		if (value_notzero_p(coef = vect_coeff(pvi->var,
						      bas_var[i]))) 
		{ 
		    pv = make_loop_indice_equation
			(loop_indices,tile, tile_delay,
			 tile_indices,tile_local_indices,j);
		  vect_add_elem(&bas_var[i],pvi->var,value_uminus(coef)); 
		
		  bas_var[i] =vect_add(bas_var[i],vect_multiply(pv,coef));
		}
	    }
	}
	
	/* building equality L = [(var1 -1) +(var2-1)*ms]/bn*ls] */
	pv1 = vect_dup(bas_var[2]);
	pv1 = vect_multiply(pv1,ms);
	pv2 = vect_dup(bas_var[1]);
	pv2 = vect_add(pv2,pv1);
	vect_add_elem(&pv2,TCST,value_plus(value_uminus(ms),VALUE_MONE));
	exp1= make_vecteur_expression(pv2);
	exp2 = int_to_expression(bn*ls);
	lex2 =CONS(EXPRESSION,exp2,NIL);
	exp3 = make_div_expression(exp1,lex2);	
	exp4 = make_vecteur_expression(vect_new(vligne,VALUE_ONE));
	stat = make_assign_statement(exp4,exp3);

	icode = CONS(STATEMENT,stat,NIL);
	gcode = gen_nconc(gcode,icode);
	/* building equality B = [(var1 -1) +(var2-1)*ms -bn*ls*L ]/ls+1 */
	pv1 = vect_dup(bas_var[2]);
	pv1 = vect_multiply(pv1,ms);
	pv2 = vect_dup(bas_var[1]);
	pv2 = vect_add(pv2,pv1);
	vect_add_elem(&pv2,vligne,int_to_value((-bn*ls)));
	vect_add_elem(&pv2,TCST,value_plus(value_uminus(ms),VALUE_MONE));
	exp1= make_vecteur_expression(pv2);
	exp2 = int_to_expression(ls);
	lex2 =CONS(EXPRESSION,exp2,NIL);
	exp3 = make_div_expression(exp1,lex2);	
	exp4 = make_vecteur_expression(vect_new(vbank,VALUE_ONE));
	stat = make_assign_statement(exp4,exp3);
	icode = CONS(STATEMENT,stat,NIL);
	gcode = gen_nconc(gcode,icode);
	/* building equality O = [(var1 -1) +(var2-1)*ms -bn*ls*L -ls*B */
	pv1 = vect_dup(bas_var[2]);
	pv1 = vect_multiply(pv1,ms);
	pv2 = vect_dup(bas_var[1]);
	pv2 = vect_add(pv2,pv1);
	vect_add_elem(&pv2,vligne,int_to_value((-bn*ls)));
	vect_add_elem(&pv2,vbank,int_to_value((-ls)));
	vect_add_elem(&pv2,TCST,value_plus(value_uminus(ms),VALUE_MONE));
	exp1= make_vecteur_expression(pv2);
	exp4 = make_vecteur_expression(vect_new(vofs,VALUE_ONE));
	stat = make_assign_statement(exp4,exp1);
	icode = CONS(STATEMENT,stat,NIL);
	gcode = gen_nconc(gcode,icode);
    }
    return (gcode);
}


static list 
array_scalar_access_to_compute_communication(
    entity compute_module,Pbase bank_indices,
    int bn,int ls,list lt,
    bool load_code,entity proc_id, 
    entity var_id,bool fully_sequential,
    Pbase loop_indices,tiling tile,Pvecteur tile_delay, 
    Pvecteur tile_indices,Pvecteur tile_local_indices)
{
   
    list icode,lst;
    icode  = array_indices_communication(compute_module,bank_indices,
					 bn,ls,lt,
					 load_code,var_id,loop_indices,tile,
					 tile_delay,
					 tile_indices,tile_local_indices);
  lst = CONS(REFERENCE,make_reference(var_id,NIL),
		    CONS(REFERENCE, 
			 make_reference((entity) bank_indices->succ->var,
					NIL),
			 CONS(REFERENCE, 
			      make_reference((entity) bank_indices->succ->succ->var,NIL),
			      NIL)));

    if (!fully_sequential) { 
	/*    Generation of the compute code corresponding to the transfer 
	      of a scalar array element :
	      L = (-101+100*L_J+P)/400
	      X3 = (-101+100*L_J+P-400*L)/100
	      O = -101+100*L_J+P-400*L-100*X3
	      DOALL BANK_ID = 0, 3
	      CALL WP65_SEND_4(BANK_ID, X3, 1)
	      CALL WP65_SEND_4(BANK_ID, L, 1)
	      CALL WP65_SEND_4(BANK_ID, O, 1)
	      ENDDO
	      CALL WP65_RECEIVE_4(X3, L_B_0_0(P,L_J), 1)
	      */
	entity ent1 = make_new_module_variable(compute_module,100);

	list ccode= constant_symbolic_communication(compute_module,lst,
						    !load_code,ent1);
	range looprange = make_range(int_to_expression(0),
				     int_to_expression(bn-1),
				     int_to_expression(1));
	statement loopbody = make_block_statement(ccode);
	entity looplabel = make_loop_label(9000, compute_module);
	   
	loop newloop = make_loop(ent1, 
				 looprange,
				 loopbody,
				 looplabel, 
				 make_execution(is_execution_parallel,UU),
				 NIL);

	statement  stat = loop_to_statement(newloop);
    AddEntityToDeclarations(ent1,compute_module);
	icode = gen_nconc(icode,CONS(STATEMENT,stat,NIL));
	ccode = constant_symbolic_communication(compute_module,lt,load_code,var_id);
	icode = gen_nconc(icode,ccode);  
	return icode;
    }
    else {

	list ccode = constant_symbolic_communication(compute_module,lst,!load_code,proc_id);
	ccode =  gen_nconc(icode,ccode);
	icode = constant_symbolic_communication(compute_module,lt,load_code,proc_id);
	ccode =  gen_nconc(ccode,icode);
	return ccode;
    }
}
 
list array_scalar_access_to_bank_communication(entity memory_module,Pbase  bank_indices, 
					       int bn,int ls, list lt,bool load_code,
					       entity proc_id,entity var_id,bool fully_sequential)
{   
  
    reference ref1 = make_reference((entity) bank_indices->succ->var,NIL);
    reference ref2 =  make_reference((entity) bank_indices->succ->succ->var,NIL);
    list lst = CONS(REFERENCE,make_reference(var_id,NIL),
		    CONS(REFERENCE,ref1, CONS(REFERENCE,ref2,NIL)));
    if (!fully_sequential) {
	/* 
	   Generation of the emulated shared memory code corresponding to 
	   the transfer of a scalar array element :
	   
	   CALL BANK_RECEIVE_4(PROC_ID, X1, 1)
	   CALL BANK_RECEIVE_4(PROC_ID, L, 1)
	   CALL BANK_RECEIVE_4(PROC_ID, O, 1)
	   IF (BANK_ID = X1) THEN
	   CALL BANK_SEND_4(PROC_ID, ES_B(O,L), 1)
	   ENDIF
	   */
	list ccode= constant_symbolic_communication(memory_module,
						    lst,!load_code,proc_id); 
	list tcode = constant_symbolic_communication(memory_module,
						     lt,load_code,proc_id);
	expression exp1 =make_vecteur_expression(vect_new(bank_indices->var,
							  VALUE_ONE));
	expression exp2 = make_vecteur_expression(vect_new((Variable) var_id,
							   VALUE_ONE));
	expression test_bound = MakeBinaryCall(entity_intrinsic(EQUAL_OPERATOR_NAME),exp1,exp2);
	statement testbody = make_block_statement(tcode);
	test tst =  make_test(test_bound,testbody,
			      make_continue_statement(entity_empty_label()));
	statement stat = test_to_statement(tst);
	return (gen_nconc(ccode,CONS(STATEMENT,stat,NIL)));
    } 
    else { 
	entity  ent1 = make_new_module_variable(memory_module,100);
	list ccode= constant_symbolic_communication(memory_module,lst,
						    !load_code,(entity) bank_indices->var); 
	list icode= constant_symbolic_communication(memory_module,lt,
						    load_code,ent1);
	range looprange = make_range(int_to_expression(0),
				     int_to_expression(bn-1),
				     int_to_expression(1));
	statement loopbody = make_block_statement(icode);
	entity looplabel = make_loop_label(9000, memory_module);
	
	loop newloop = make_loop(ent1,looprange, loopbody, looplabel, 
				 make_execution(is_execution_parallel,UU), NIL);

	statement  stat = loop_to_statement(newloop);
	AddEntityToDeclarations(ent1,memory_module);
	icode = gen_nconc(ccode,CONS(STATEMENT,stat,NIL));
	return icode;
    }
}

static list 
build_esv_list(list lt, hash_table v_to_esv, Pbase bank_indices)
{
    list newlt=NIL;
    MAPL(ref, {
	Variable var1 = (Variable) reference_variable(REFERENCE(CAR(ref)));
	entity esv = (entity) hash_get(v_to_esv, (char *) var1); 
	expression expr1 = make_vecteur_expression
	    (vect_new((char *) bank_indices->succ->var, VALUE_ONE));
	expression expr2 = make_vecteur_expression
	    (vect_new((char *) bank_indices->succ->succ->var, VALUE_ONE));
	list args = CONS(EXPRESSION,expr2, CONS(EXPRESSION,expr1,NIL));
	reference ref1 =  make_reference((Variable) esv,args);
    newlt=CONS(REFERENCE,ref1,newlt);
    },lt);
    return gen_nreverse(newlt);
 
}

static void insert_array_scalar_access_movement(entity compute_module,entity memory_module,
				    Pbase  bank_indices, int bn,int ls,entity proc_id,entity ent1,
				   list lt, statement stat,bool load,hash_table v_to_esv,
				   list *new_slst,list *new_blist, bool fully_sequential,
						Pbase loop_indices,tiling  tile,Pvecteur tile_delay, 
				      Pvecteur tile_indices,  Pvecteur tile_local_indices)
{
    list ccode ;
    ifdebug(8) {
	fprintf(stderr,
		" communication to be inserted at run time stat no %"PRIdPTR": ",
		statement_number(stat));
	reference_list_print(lt); 
   
	       }
  
    /* creation d'une nouvelle entite pour servir de temporaire au 
       numero de banc memoire contenant la variable scalaire */
    ccode =array_scalar_access_to_compute_communication(compute_module, 
							bank_indices,bn,
							ls,lt,load,proc_id,
							ent1,fully_sequential,
							loop_indices,tile, tile_delay,tile_indices,
							tile_local_indices); 

    *new_slst = gen_nconc(*new_slst,ccode); 
    lt = build_esv_list(lt,v_to_esv,bank_indices);
    ccode =array_scalar_access_to_bank_communication(memory_module, bank_indices,bn,
						     ls,lt,!load, proc_id,
						     ent1,fully_sequential);

    *new_blist = gen_nconc(*new_blist,ccode);
}  
	    

void insert_run_time_communications(entity compute_module,entity memory_module,
				    Pbase  bank_indices, int bn,int ls,entity proc_id,
				   list list_statement_block,
				    statement_mapping fetch_map,statement_mapping store_map,
				    list *new_slst,list *new_blist,hash_table v_to_esv,
				    bool fully_sequential,
				    Pbase loop_indices,tiling  tile,Pvecteur tile_delay, 
				      Pvecteur tile_indices,  Pvecteur tile_local_indices)
{
    int nbcall=0; 
    entity ent1=entity_undefined; 
    
    MAPL(st1, 
     { instruction inst = statement_instruction(STATEMENT(CAR(st1)));
       switch(instruction_tag(inst)) {
       case is_instruction_block: {
	  
	   list new_slst1 = NIL; 
	   nbcall = 0; 
	   MAPL(st2, {
	       insert_run_time_communications(compute_module,memory_module, 
					      bank_indices,bn,ls,proc_id,
					      CONS(STATEMENT,
						   STATEMENT(CAR(st2)),NIL),
					      fetch_map,store_map,
					      &new_slst1,new_blist,
					      v_to_esv,fully_sequential,
					      loop_indices,tile, tile_delay,
					      tile_indices,tile_local_indices);
	   

},
		instruction_block(inst));  
	   instruction_block(inst)= new_slst1;  
	   break; }
       case is_instruction_loop: {
	   statement lbody = loop_body(instruction_loop(inst));
	   statement sbody = make_block_statement(CONS(STATEMENT,
						       lbody,NIL));
	   cons *nbody=(instruction_call_p(statement_instruction(lbody))) ? 
	       CONS(STATEMENT,sbody,NIL) : 
		   CONS(STATEMENT,lbody,NIL);
	   insert_run_time_communications(compute_module,memory_module, 
					  bank_indices,bn,ls,proc_id,
					  nbody,fetch_map,store_map,new_slst,
					  new_blist,
					  v_to_esv,fully_sequential,
					  loop_indices,tile, tile_delay,
					  tile_indices,tile_local_indices);
	    *new_slst = gen_nconc(*new_slst,
				  CONS(STATEMENT,STATEMENT(CAR(st1)),NIL)); 
	   break;
       }
       case is_instruction_call: {
	   list  lt;
	 
	   if ((lt= (list) 
		GET_STATEMENT_MAPPING(fetch_map,STATEMENT(CAR(st1))))
	       != (list) HASH_UNDEFINED_VALUE && nbcall ) { 

	       ent1 = make_new_module_variable(compute_module,100);
	       AddEntityToDeclarations(ent1,compute_module);
	       
	       insert_array_scalar_access_movement(compute_module,memory_module,bank_indices,
						   bn,ls,proc_id,ent1,lt,STATEMENT(CAR(st1)),true,
						   v_to_esv,new_slst,new_blist,fully_sequential,
						   loop_indices,tile, tile_delay,tile_indices,
						   tile_local_indices
						   );
	   }
	   if ((lt=(list) GET_STATEMENT_MAPPING(store_map,
						STATEMENT(CAR(st1))))
	       != (list) HASH_UNDEFINED_VALUE && nbcall )
	   { 
	       ent1 = make_new_module_variable(compute_module,100);
	       AddEntityToDeclarations(ent1,compute_module);
	   
	       insert_array_scalar_access_movement(compute_module,memory_module,bank_indices,
						   bn,ls,proc_id,ent1,lt,STATEMENT(CAR(st1)),false,
						   v_to_esv,new_slst,new_blist,fully_sequential,
						   loop_indices,tile, tile_delay,tile_indices,
						   tile_local_indices);
	   }
	   *new_slst = gen_nconc(*new_slst,CONS(STATEMENT,STATEMENT(CAR(st1)),NIL));
	   nbcall ++;
	   break;
       }
       default:
	   break;
       }
   }, list_statement_block);
}

bool 
test_run_time_communications(list list_statement_block,
				    statement_mapping fetch_map,statement_mapping store_map)
{
    bool ok ;
    int nbcall =0;
    MAPL(st1, 
     { instruction inst = statement_instruction(STATEMENT(CAR(st1)));
       switch(instruction_tag(inst)) {
       case is_instruction_block: { 
	   ok = false;
	   MAPL(st2, {
	       ok = (ok) ? ok : 
		   test_run_time_communications(CONS(STATEMENT,
						     STATEMENT(CAR(st2)),
						     NIL),
						fetch_map,store_map);
	   }, 
	       instruction_block(inst));
	   return ok;
	   break; }
       case is_instruction_loop: {
	   return(test_run_time_communications(CONS(STATEMENT,
						    loop_body(instruction_loop(inst)),
						    NIL),
					       fetch_map,store_map));
	   break;}
       case is_instruction_call: {
	   list  lt = (list) GET_STATEMENT_MAPPING(fetch_map,STATEMENT(CAR(st1))); 
	   if (lt != (list) HASH_UNDEFINED_VALUE) {
	       nbcall ++;
	   return (true);  
	       } else
		   return(false);
	   break;
       }
       default:
	   break;
       }
   }, 
       list_statement_block);
 /* just to avoid warning */
 return (true);
}

