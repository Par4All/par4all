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
/* Code Generation for Distributed Memory Machines
  *
  * Higher level functions
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin, Corinne Ancourt
  * 1991
  */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "matrice.h"
#include "tiling.h"
#include "database.h"
#include "text.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "movements.h"
#include "rice.h"
#include "constants.h"

#include "ricedg.h" 
#include "wp65.h"

static list implied_do_range_list=NIL;
static int loop_nest_dim =0;
static list loops_of_nest=NIL;

static void 
add_range_in_list(range r)
{
implied_do_range_list = (implied_do_range_list==NIL) ?
    CONS(RANGE,r,NIL):
	gen_nconc(implied_do_range_list, CONS(RANGE,r,NIL));
}
static void 
implied_do_ranges(statement s)
{
 gen_recurse(s,range_domain,gen_true,add_range_in_list);
}

static void 
update_loop_nest_dim(loop l)
{
   
    statement  s = loop_to_statement(l);
    loops_of_nest = CONS(STATEMENT,s,loops_of_nest);
    loop_nest_dim ++;
}

static void 
compute_loop_nest_dim(statement l)
{
    loop_nest_dim = 0;
    gen_recurse(l,loop_domain,gen_true,update_loop_nest_dim);
}

/* Function used during code generation for non-perfectly 
   nested loops. The loops of inner nests are executed sequentially. 
   No tiling is applied on these inner loops. Thus, the new loop
   bounds do not change, except that there are shift -1 to deal with 
   Fortran declarations.
 */
static void 
reduce_loop_bound(loop l)
{
    range r = loop_range(l);
    Pvecteur  vlow = VECTEUR_NUL,vup = VECTEUR_NUL;
    expression low = range_lower(r);
    expression up = range_upper(r);
    normalized low_norm = NORMALIZE_EXPRESSION(low);
    normalized up_norm = NORMALIZE_EXPRESSION(up);
    
    if (normalized_linear_p(low_norm) && normalized_linear_p(up_norm)){
	vlow = (Pvecteur) normalized_linear(low_norm);	
	vup = (Pvecteur) normalized_linear(up_norm);
	vect_add_elem(&vlow,TCST, VALUE_MONE);
	vect_add_elem(&vup,TCST, VALUE_MONE);
	range_lower(r)= make_vecteur_expression(vlow);
	range_upper(r)= make_vecteur_expression(vup);
    }
    else (void) fprintf(stderr,"non linear loop bounds-cannot be reduced\n");
}

static void
reduce_loop_bound_for_st(statement stmp)
{
 gen_recurse(stmp,loop_domain,gen_true,reduce_loop_bound);

}
bool 
list_of_calls_p(list lsb)
{
    list pl;
    for(pl=lsb; 
	pl!=NIL &&  statement_call_p(STATEMENT(CAR(pl))); 
	pl = CDR(pl));
    return(pl==NIL);
}

entity
ith_index_of_ref(reference r, int level)
{
    entity result = entity_undefined;
    list  ith_index=reference_indices(r);
    expression exp_ind = EXPRESSION(gen_nth(level,ith_index));
    syntax sy = expression_syntax(exp_ind);
    if (syntax_reference_p(sy)) 
	result = reference_variable(syntax_reference(sy));
    return (result);
} 

loop
ith_loop_in_loop_nest(statement s1,int level)
{
    instruction inst = statement_instruction(s1);
    loop loop1;
    if (instruction_block_p(inst)) { 
	statement st1 = STATEMENT(CAR(instruction_block(inst)));
	inst = statement_instruction(st1);
    }
    loop1 = instruction_loop(inst);
    return((level ==1) ? loop1 :
		ith_loop_in_loop_nest(loop_body(loop1),level -1));

}



void 
instruction_to_wp65_code(entity module, list l, graph dg,int pn,int bn,int ls,int pd,
			      entity proc_id, entity proc_id_mm,Pbase bank_indices,
			      hash_table v_to_esv,hash_table v_to_nlv,
			      entity compute_module, statement computational,
			      entity memory_module, statement emulator, 
			      statement_mapping fetch_map,
			      statement_mapping store_map)
{

    statement mod_stat/*,cms*/;
    //entity cme;
    debug_on("WP65_DEBUG_LEVEL");

    /* FI: the semantics has been changed: the next two get_xxx() cannot/should not return
     * something undefined.
     */
    /* if ((cme = get_current_module_entity()) == entity_undefined)  */
    /* FI: already done somewhere else */
    /* set_current_module_entity(local_name_to_top_level_entity(module_local_name(module)));  */
    /* if ((cms = get_current_module_statement()) == statement_undefined) */
    /* FI: shouldn't the current statement be retrieved at the same time as the current module? */
    set_current_module_statement( (statement)
				  db_get_memory_resource(DBR_CODE, 
							 module_local_name(module),
							 true) ); 
    mod_stat = get_current_module_statement();
    MAPL(pm,{
	statement s1 = STATEMENT(CAR(pm));  
	instruction inst = statement_instruction(s1);
	ifdebug(9) 
	    (void) fprintf(stderr,
			   "instruction_to_wp65_code-instruction- begin\n");
	switch(instruction_tag(inst)) {
	    
	case is_instruction_block:{
	    instruction_to_wp65_code(module,instruction_block(inst), 
				     dg, pn, bn, ls, pd,
				     proc_id, proc_id_mm, bank_indices,
				     v_to_esv, v_to_nlv,
				     compute_module, computational,
				     memory_module, emulator, 
				     fetch_map,store_map);
	    
	    break;}
	    case is_instruction_test: {
		(void) fprintf(stderr,
			    "instruction_to_wp65_code-Sorry:test not implemented\n");
		break;}
	    case is_instruction_goto: {
		(void) fprintf(stderr,
			       "instruction_to_wp65_cod -Sorry:goto implemented\n");
		break;}
	    case is_instruction_loop: {
		loop_nest_to_wp65_code(module, s1, dg, pn, bn, ls, pd,
				       proc_id, proc_id_mm, bank_indices,
				       v_to_esv, v_to_nlv,
				       compute_module, computational,
				       memory_module, emulator,
				       fetch_map,store_map,mod_stat);
		break;}
	    case is_instruction_call:  {     
		if (!return_statement_p(s1)) {
  
		     call_to_wp65_code(s1,compute_module,
				      memory_module,
				      (entity) bank_indices->var,proc_id,
				      computational,emulator, fetch_map,
				      store_map,v_to_esv);
		}
		break;
	    } 
	    case is_instruction_unstructured: {
		pips_internal_error("Sorry: unstructured not implemented");
		break;}
	    default: 
		(void) fprintf(stderr, 
			       " instruction_to_wp65_code- bad instruction tag \n");
		break;
	    }
	ifdebug(9) 
	    (void) fprintf(stderr,
			   "instruction_to_wp65_code-instruction_end \n");
    },l);
    /* reset_current_module_entity(); */
    reset_current_module_statement(); 
    debug_off();
}




void 
call_to_wp65_code(statement s, entity compute_module, entity memory_module,
		  entity bank_id, entity proc_id,
		  statement computational,statement emulator, 
		  statement_mapping fetch_map,  
		  statement_mapping store_map, hash_table v_to_esv)
{ 
    list lrefs;
    bool load_code = true;
    instruction i; 
    call c = instruction_call(statement_instruction(s));
    /* To deal with implied_do and I/Os */


    if (strcmp(entity_local_name(call_function(c)), "WRITE") == 0) {
	generate_io_wp65_code(s,s,v_to_esv,false);
	i = statement_instruction(emulator);
	instruction_block(i) = gen_nconc(instruction_block(i), 
					 CONS(STATEMENT, 
					      copy_statement(s), NIL));
    }
     else { /* communications for variables having to be 
	      loaded in local memories  for assignments */
	if ((lrefs = (list) GET_STATEMENT_MAPPING(fetch_map,s))
	    != (list) HASH_UNDEFINED_VALUE) { 
	    ifdebug(9) { 
		(void) fprintf(stderr,
			       "Vars having to be loaded for stat %"PRIdPTR":\n",
			       statement_number(s));
		reference_list_print(lrefs); }    
	    include_constant_symbolic_communication(compute_module,lrefs,
						    load_code,computational,
						    proc_id);
	    include_constant_symbolic_communication(memory_module,lrefs,
						    !load_code,emulator,
						    bank_id);
	} 
	i = statement_instruction(computational);
	instruction_block(i) = gen_nconc(instruction_block(i), 
					 CONS(STATEMENT, copy_statement(s), NIL));
	 /* communications for variables having to be 
	      stored in global memory  for assignments */
	if ((lrefs = (list) GET_STATEMENT_MAPPING(store_map,s))
	    != (list) HASH_UNDEFINED_VALUE) {
	    load_code = false;
	    ifdebug(9) {
		(void) fprintf(stderr,
			       "Vars having to be stored for stat %"PRIdPTR":\n",
			       statement_number(s));
		reference_list_print(lrefs); }
	    include_constant_symbolic_communication(compute_module,lrefs,
						    load_code,computational,proc_id);
	    include_constant_symbolic_communication(memory_module,lrefs,
						    !load_code,emulator,bank_id);
	} 
     }
}

/* This function extracts from an implied_do expression the 
   reference having to be computed or printed */

expression 
ref_in_implied_do(expression exp)
{
    call c2 = syntax_call(expression_syntax(exp));
    list  call_args = call_arguments(c2); 
    expression last_call = EXPRESSION(CAR(gen_last(call_args)));
    syntax s = expression_syntax(last_call);
    
    return((syntax_call_p(s)
	    && (strcmp(entity_local_name(call_function(syntax_call(s))),
		       "IMPLIED-DO") == 0)) ? 
	   ref_in_implied_do(last_call):
	   last_call );
}

/* This function translates a reference in I/O statement into 
   its corresponding emulated shared memory reference */

reference
translate_IO_ref(call c, hash_table v_to_esv, bool loop_or_call_print)
{
    list pio,pc = NIL;
    bool iolist_reached = false;
    expression exp; 
    syntax s;
    reference  result=reference_undefined;
    if (same_string_p(entity_local_name(call_function(c)), "WRITE")) {
	pio = pc = call_arguments(c);  
	/* scan the argument list till IOLIST' arguments*/
	while (!ENDP(pio) && (!iolist_reached)) {
	    call c1;
	    expression arg;
	    s = expression_syntax(EXPRESSION(CAR(pio)));
	    c1=syntax_call(s);
	    arg = EXPRESSION(CAR(CDR(pio)));
	    
	    if (((strcmp(entity_local_name(call_function(c1)),"FMT=")==0) &&
		 (strcmp(words_to_string(words_expression(arg,NIL)),"*")==0))
		||((strcmp(entity_local_name(call_function(c1)),"UNIT=")==0) 
		   &&
		   (strcmp(words_to_string(words_expression(arg,NIL)),"*")==0)))
		pio = CDR(CDR(pio));
	    else
		if (strcmp(entity_local_name(call_function(c1)),
			   "IOLIST=")==0) {
		    iolist_reached = true;
		    pio = CDR(pio);
		}
	}
	exp = EXPRESSION(CAR(pio));
	/* implied-do case: the array reference is the first argument 
	 in the last argument list of the implied do call */
	s=expression_syntax(exp);
	if (syntax_call_p(s) && 
	    (strcmp(entity_local_name(call_function(syntax_call(s))),
	       "IMPLIED-DO") == 0)) {
	    exp = ref_in_implied_do(exp);
	    s=expression_syntax(exp);
	}
	if (syntax_reference_p(s)) {
	    syntax sy1 = expression_syntax(exp);
	    reference ref1 = syntax_reference(sy1);
	    list indic = gen_full_copy_list(reference_indices(ref1));
	    entity ent1=(entity)hash_get(v_to_esv,
					 (char *) reference_variable(ref1));
	    reference newr = make_reference(ent1,indic);
	    expression_syntax(exp)= make_syntax(is_syntax_reference,newr);
	    result = ref1;
	}
    }  
    else
	pips_user_error("function calls are not handled in this version\n");
    return result;
}



statement
generate_io_wp65_code(statement s1,statement body,hash_table v_to_esv,bool loop_or_call_print)
{
    
    list rvld,pl;
    statement result = s1;
    instruction inst = statement_instruction(body);
    call c;
    reference r;
 
    entity  rv,esv_ref;
    type rvt;
    int nb_dim,i;
    
    /* we know that it is a loop nest containing a PRINT function 
     The loop body contains block statement in case of CONTINUE 
     loop nest style*/
    statement_ordering(s1)=STATEMENT_ORDERING_UNDEFINED;
    if (instruction_block_p(inst)) {
	list b = instruction_block(inst);
	c = instruction_call(statement_instruction(STATEMENT(CAR(b))));
    }
    else 
	c= instruction_call(inst);
     r = translate_IO_ref(c,v_to_esv,loop_or_call_print);
    rv = reference_variable(r);
    esv_ref = (entity) hash_get(v_to_esv,(char *) rv); 
    rvt = entity_type(esv_ref);
    rvld = variable_dimensions(type_variable(rvt));
    nb_dim = gen_length((list) rvld);
    
    if (loop_or_call_print) /* loop nest belonging IO statement */
	for (i=1; i<=nb_dim; i++) {
	    list ldim = gen_nthcdr(i-1,rvld);
	    expression low = dimension_lower(DIMENSION(CAR(ldim)));
	    expression up = dimension_upper(DIMENSION(CAR(ldim)));
	    range looprange = make_range(low, up,
				   int_to_expression(1));
	    loop loopi = ith_loop_in_loop_nest(result,i);
	    loop_range(loopi) = looprange;
	}
    else { /* implied_do case */
	implied_do_range_list = NIL;
	implied_do_ranges(result);
	implied_do_range_list = gen_nreverse(implied_do_range_list);
	 for (i=1,pl = implied_do_range_list; !ENDP(pl); POP(pl),i++) {
	    range r1 = RANGE(CAR(pl));
	    list ldim = gen_nthcdr(i-1,rvld); 
	    expression low = copy_expression(
		dimension_lower(DIMENSION(CAR(ldim))));
	    expression up = copy_expression(
		dimension_upper(DIMENSION(CAR(ldim))));
	    range_lower(r1)=low; 
	    range_upper(r1)=up; 
	} 
    }
    return(result);
}

/* Test if the statement  resulting  from the 
   perfectly_loop_nest_to_body function contains at first call an io */
bool
io_loop_nest_p(statement st)
{
    instruction inst = statement_instruction(st);
    call c;
    if (instruction_block_p(inst)) {
	list b = instruction_block(inst);
	c = instruction_call(statement_instruction(STATEMENT(CAR(b))));
    }
    else 
	c= instruction_call(inst);
    return (strcmp(entity_local_name(call_function(c)), "WRITE") == 0) ;
}



void 
loop_nest_movement_generation(
    entity module,
    statement loop_nest,
    int pn,
    int bn,
    int ls,
    int pd,
    entity proc_id, 
    entity proc_id_mm,
    Pbase bank_indices, 
    hash_table v_to_esv,
    hash_table v_to_nlv, 
    entity compute_module,
    statement computational, 
    entity memory_module,
    statement emulator, 
    statement_mapping fetch_map,
    statement_mapping store_map,
    statement mod_stat,
    bool fully_parallel,
    Psysteme sc_tile, 
    Pbase initial_basis,
    Pbase local_basis,
    Pbase local_basis2, 
    Pbase tile_basis_in_tile_basis,
    Pbase tile_basis_in_tile_basis2,
    Pbase loop_body_indices,
    list lpv,
    list * lb,
    list *blb,
    list *sb,
    list *bsb,
    int first_parallel_level,
    int last_parallel_level,
    hash_table llv_to_lcr,
    hash_table r_to_llv, 
    hash_table v_to_lllv,
    hash_table r_to_ud) 
{ 
    
    list fetch_data_list =NIL; 
    list store_data_list = NIL; 
    list fetch_reference_list = NIL; 
    list store_reference_list = NIL;
    debug_on("WP65_DEBUG_LEVEL");
    /* the list of data having to be store into the global memory 
       is the concatenation  of store_map sets of the internal loops */
     
    concat_data_list(&fetch_data_list,&fetch_reference_list,
		     loop_nest,fetch_map,fully_parallel); 
    concat_data_list(&store_data_list , &store_reference_list ,
		     loop_nest,store_map,fully_parallel);   
    MAPL(r1,{  entity rv = (entity) reference_variable(REFERENCE(CAR(r1)));
	       if(!entity_is_argument_p(rv, lpv))
		   make_all_movement_blocks(
		       module,compute_module,memory_module,
		       rv,fetch_reference_list,
		       llv_to_lcr, v_to_lllv,r_to_ud, v_to_esv,
		       pn,bn, ls,
		       sc_tile, initial_basis, local_basis,
		       proc_id, bank_indices,loop_body_indices,
		       lb, blb,is_action_read,first_parallel_level,
					    last_parallel_level);
	   },
	 fetch_data_list);
    /* update of variable names according to the module of appartenance */
    sc_variables_rename(sc_tile,local_basis2,local_basis, (string(*)(void*))entity_local_name);
    sc_variables_rename(sc_tile, tile_basis_in_tile_basis2, 
			tile_basis_in_tile_basis,(string(*)(void*))entity_local_name);
    MAPL(r1,{ 
	reference rf = REFERENCE(CAR(r1));
	entity rv = (entity) reference_variable(rf);
	if (!reference_scalar_p(rf)  && !entity_is_argument_p(rv, lpv)) 
	    make_all_movement_blocks(module,compute_module,memory_module,
				     rv,store_reference_list ,
				     llv_to_lcr, v_to_lllv, r_to_ud, v_to_esv,
				     pn,bn, ls,
				     sc_tile, initial_basis, local_basis,
				     proc_id, bank_indices, loop_body_indices,
				     sb, bsb, is_action_write, 
				     first_parallel_level,
				     last_parallel_level);
    },
	 store_data_list);
    debug_off();
}


void 
loop_nest_to_wp65_code(
    entity module,
    statement loop_nest,
    graph dg, 
    int pn,
    int bn,
    int ls,
    int pd, 
    entity proc_id,
    entity proc_id_mm,
    Pbase bank_indices,
    hash_table v_to_esv,
    hash_table v_to_nlv,
    entity compute_module,
    statement computational, 
    entity memory_module,
    statement emulator,
    statement_mapping fetch_map,
    statement_mapping store_map,
    statement mod_stat)
{

    Psysteme iteration_domain = sc_new();  
    Psysteme  iteration_domain2 = sc_new();
    Psysteme sc_tile = SC_UNDEFINED;
    tiling tile;
    Pvecteur tile_delay = VECTEUR_NUL;
    Pvecteur pv;
    int it;
    list lb = NIL;			/* load block */
    list blb = NIL;			/* bank load block */
    list sb = NIL;			/* store block */
    list bsb = NIL;			/* bank store block */
    list cb = NIL;			/* compute block */
    /* list of local variables to list of conflicting references */
    hash_table llv_to_lcr = hash_table_make(hash_pointer, 0);
    /* inverse table: list of local variables to use for a given reference */
    hash_table r_to_llv = hash_table_make(hash_pointer, 0);
    /* variable to list of list of local variables: the main list
       is an image of connected components of the dependence graph filtered
       for a given variable, v; the low-level list is a function of
       the number of pipeline stages */
    hash_table v_to_lllv = hash_table_make(hash_pointer,0);
    /* reference to use-def usage ; use and def are encoded as action_is_read
       and action_is_write */
    hash_table r_to_ud = hash_table_make(hash_pointer,0);

    /* pipelining offset; no pipelining is implemented yet */
    Pbase offsets = VECTEUR_NUL;
    statement body;

    /* private variables for the loop nest */
    list lpv = loop_locals(instruction_loop(statement_instruction(loop_nest)));

    /* local variables */
    list cbl, embl = NIL;
    statement cs, ems,bs;
    statement io_st=statement_undefined;
    instruction i;
    list store_data_list = NIL; 
    list store_reference_list = NIL;
    Pbase initial_basis = BASE_NULLE;
    Pbase full_initial_basis = BASE_NULLE;
    Pbase local_basis = BASE_NULLE;
    Pbase tile_basis_in_initial_basis = BASE_NULLE;
    Pbase tile_basis_in_tile_basis = BASE_NULLE;
    Pbase local_basis2 = BASE_NULLE; 
    Pbase initial_basis2= BASE_NULLE; 
    Pbase tile_basis_in_tile_basis2 = BASE_NULLE; 
    Pbase loop_body_indices = BASE_NULLE;
    int i1,lpl, loop_nest_dimt;
    int first_parallel_level=1;
    int last_parallel_level, perfect_nested_loop_size; 
    bool loop_carried_dep[11];  
    bool fully_parallel;
    bool fully_sequential=true;
    int   nested_level2,nested_level=0;
    list list_statement_block=NIL;
    list list_statement_block2=NIL;
    list new_compute_lst = NIL;
    list new_bank_lst = NIL;
    instruction binst,binst2;
    bool io_statementp=false;

    debug_on("WP65_DEBUG_LEVEL");
    debug(5,"loop_nest_to_wp65_code", "begin\n");

    for (i1=1;i1<=10;i1++)  loop_carried_dep[i1] = false;
    compute_loop_nest_dim(loop_nest); 
    loop_nest_dimt = loop_nest_dim;
   

    fully_parallel =  full_parallel_loop_nest_p(mod_stat,loop_nest,
					      loop_nest_dimt,dg,
                                                loop_carried_dep);
    find_iteration_domain(loop_nest,&iteration_domain, &initial_basis,
			  &nested_level,&list_statement_block, &binst);
    full_initial_basis = base_dup(initial_basis); 
    perfect_nested_loop_size= vect_size(initial_basis);

    /* pour eviter tous les problemes de tiling sur les nids de boucles
       internes a des nids de boucles mal imbriques on execute les boucles
       internes en sequentiel. Il faudrait autrement s'assurer que l'on a
       les meme indices de boucles en interne pour le tiling et que les
       bornes sont identiques (meme restrictions que pour du loop fusion) */
  
    for (i1=perfect_nested_loop_size+1;i1<=10;i1++)  
	loop_carried_dep[i1] = true;
    assert(perfect_nested_loop_size <=10);

    last_parallel_level =perfect_nested_loop_size+1; 
    if (!fully_parallel) {  
	for (i1=1; i1<=loop_nest_dimt && (loop_carried_dep[i1] == true);
	     i1++);
	first_parallel_level = i1; 
	for (i1=first_parallel_level; 
	     i1<=loop_nest_dimt && (loop_carried_dep[i1] == false);i1++);
	last_parallel_level = i1-1; 
	for (it=1, pv=initial_basis;
	     it <perfect_nested_loop_size &&  it<last_parallel_level;
	     it++, pv=pv->succ);
	loop_body_indices = base_dup(pv->succ);
	ifdebug(4) {
	    (void) fprintf(stderr,"first_parallel_level :%d, last_parallel_level %d\n",
			   first_parallel_level,last_parallel_level);
	    (void) fprintf(stderr,"\nLoop body basis - loop_nest:");
	    base_fprint(stderr, loop_body_indices, (string(*)(void*))entity_local_name);
  	    (void) fprintf(stderr,"\nInitial basis - loop_nest:");
	    base_fprint(stderr, initial_basis, (string(*)(void*))entity_local_name);
	}
    }

    /* creation of new indices */
    create_tile_basis(module,compute_module,memory_module, initial_basis,
		      &tile_basis_in_initial_basis,
		      &tile_basis_in_tile_basis,
		      &local_basis, 
		      &tile_basis_in_tile_basis2,
		      &local_basis2);
    lpl = (fully_parallel) ? last_parallel_level-1:last_parallel_level;
    tile = loop_nest_to_tile(iteration_domain, ls, 
			     initial_basis, first_parallel_level,lpl,
			     perfect_nested_loop_size); 
      
    sc_tile = loop_bounds_to_tile_bounds(iteration_domain,initial_basis, 
					 tile, tile_delay, 
					 tile_basis_in_tile_basis, 
					 local_basis);
  
    ifdebug(8) { 
	fprintf(stderr,"loop body \n");
	MAP(STATEMENT, s, wp65_debug_print_text(module, s), 
	    list_statement_block); 
	     }
    if (list_of_calls_p(list_statement_block)) {
	body = perfectly_nested_loop_to_body(loop_nest);
 
     if (io_loop_nest_p(body)) {
	 io_st = generate_io_wp65_code(loop_nest,body,v_to_esv,true);
     io_statementp = true;}
     else {

	/* a modifie pour tenir compte des dimensions reelles des domaines */

	loop_nest_to_local_variables(module, compute_module, memory_module, 
				     llv_to_lcr, r_to_llv, v_to_lllv, 
				     r_to_ud, v_to_esv, v_to_nlv,
				     lpv, body, initial_basis, dg, 
				     bn, ls, pd, 
				     tile);
	ifdebug(4) fprint_wp65_hash_tables(stderr, llv_to_lcr, r_to_llv, v_to_lllv, 
					   r_to_ud, v_to_esv);
  
	loop_nest_movement_generation(module,loop_nest,pn,bn, ls, pd,proc_id,proc_id_mm,
				      bank_indices,v_to_esv,v_to_nlv,compute_module,
				      computational, memory_module,emulator,fetch_map,
				      store_map,mod_stat,fully_parallel,sc_tile, 
				      full_initial_basis,local_basis, local_basis2,
				      tile_basis_in_tile_basis, tile_basis_in_tile_basis2, 
				      loop_body_indices,lpv,&lb,&blb,&sb,
				      &bsb,first_parallel_level, last_parallel_level, 
				      llv_to_lcr, r_to_llv, v_to_lllv,r_to_ud ); 
}    
    }
    else {
	Pbase tbib2 = BASE_NULLE;
	Pbase tbtl3 = BASE_NULLE;
	Pbase tbtl4 = BASE_NULLE;
	Pbase lba3 = BASE_NULLE; 
	Pbase lba4 = BASE_NULLE; 
	Pbase lba5 = BASE_NULLE; 
	Pbase lba6 = BASE_NULLE; 
	Pbase tbtl5 = BASE_NULLE;
	Pbase tbtl6 = BASE_NULLE;
	Psysteme sc_tile2 = SC_UNDEFINED;
	tiling tile2;
 
	MAPL(lsb,
	 {
	     statement stmp =  STATEMENT(CAR(lsb)); 
	     if (!continue_statement_p(stmp)) {
		 instruction_block(binst) = CONS(STATEMENT,stmp,NIL);
		 ifdebug(8) 
		 {
		     fprintf(stderr,"generation des transferts pour \n");
		     wp65_debug_print_text(module, loop_nest);
		 }
		 body =  perfectly_nested_loop_to_body(loop_nest);
		 find_iteration_domain(loop_nest,&iteration_domain2,
				       &initial_basis2,&nested_level2,
				       &list_statement_block2,&binst2);
		 if (loop_nest_dimt > perfect_nested_loop_size) {
		     loop_body_indices = vect_dup(initial_basis2);
		     for ( pv=initial_basis; pv!= NULL;
			vect_chg_coeff(&loop_body_indices,pv->var,VALUE_ZERO),
			  pv =pv->succ);
		     full_initial_basis=base_reversal(vect_add(initial_basis,
							       loop_body_indices));
		     /* normalement full_initila_basis == initial_basis2*/
		}
		 ifdebug(2) {
		     (void) fprintf(stderr,"full basis\n");
		     base_fprint(stderr, initial_basis2, 
				 (string(*)(void*))entity_local_name);
		     (void)  fprintf(stderr,"full iteration domain\n");
		     sc_fprint(stderr,iteration_domain2, 
			       (string(*)(void*))entity_local_name);
		 }
		 create_tile_basis(module,compute_module,memory_module, loop_body_indices,
				   &tbib2, &tbtl3, &lba3, &tbtl4, &lba4);
		 tbtl5= base_reversal(vect_add(tile_basis_in_tile_basis, tbtl3));
		 tbtl6= base_reversal(vect_add(tile_basis_in_tile_basis2, tbtl4));
		 //Pbase tbib3=base_reversal(vect_add(tile_basis_in_initial_basis, tbib2));
		 lba5= base_reversal(vect_add( local_basis, lba3));
		 lba6= base_reversal(vect_add( local_basis2, lba4));
		 /* a modifie pour tenir compte des dimensions reelles des domaines */
		 tile2 = loop_nest_to_tile(iteration_domain2, ls, initial_basis2, 
					   first_parallel_level,lpl,
					   perfect_nested_loop_size); 
		 sc_tile2=loop_bounds_to_tile_bounds(iteration_domain2,initial_basis2, 
						     tile2, tile_delay,tbtl5,lba5);

		 loop_nest_to_local_variables(module, compute_module, memory_module, 
					      llv_to_lcr, r_to_llv, v_to_lllv, 
					      r_to_ud, v_to_esv, v_to_nlv,
					      lpv, body, initial_basis2, dg, 
					      bn, ls, pd,tile2);
		 ifdebug(4) fprint_wp65_hash_tables(stderr, llv_to_lcr, 
						    r_to_llv, v_to_lllv, 
						    r_to_ud, v_to_esv);
		 loop_nest_movement_generation(
		     module,stmp,pn,bn, ls, pd,proc_id,
		     proc_id_mm, bank_indices,v_to_esv,
		     v_to_nlv,compute_module, computational,
		     memory_module,emulator,fetch_map,
		     store_map, mod_stat,fully_parallel,
		     sc_tile2,initial_basis2,lba5,lba6,tbtl5,
		     tbtl6,loop_body_indices,lpv,&lb,&blb,&sb,
		     &bsb,first_parallel_level, 
		     last_parallel_level, llv_to_lcr, 
		     r_to_llv, v_to_lllv,r_to_ud ); 
		 reduce_loop_bound_for_st(stmp);
	     } 
		
	 },
	     list_statement_block); 
    }	 
    fully_sequential = (first_parallel_level >last_parallel_level); 

    if (io_statementp)  {
	cs = statement_undefined;
	ems =io_st;
    }
    else {
	insert_run_time_communications(compute_module, memory_module,
				       bank_indices,bn,ls,proc_id,
				       list_statement_block,fetch_map,
				       store_map,&new_compute_lst,
				       &new_bank_lst,v_to_esv,
				       fully_sequential,
				       initial_basis,tile, tile_delay, 
				       tile_basis_in_tile_basis, 
				       local_basis);

	instruction_block(binst)= new_compute_lst;
	body = instruction_to_statement(binst);
	ifdebug(8) {
	    fprintf(stderr,"loop body \n");
	    wp65_debug_print_text(module, body); 
	    fprintf(stderr,"base_initiale 1\n");
	    vect_fprint(stderr,initial_basis,(string(*)(void*))entity_local_name);
	}
	
    sc_variables_rename(sc_tile,local_basis,local_basis2, (string(*)(void*))entity_local_name);
    sc_variables_rename(sc_tile, tile_basis_in_tile_basis, 
			tile_basis_in_tile_basis2,(string(*)(void*))entity_local_name);  

    cb = make_compute_block(compute_module, body, offsets, r_to_llv,tile, 
			    initial_basis, local_basis2,
			    tile_basis_in_tile_basis2,
			    tile_basis_in_initial_basis,
			    iteration_domain,first_parallel_level,
			    last_parallel_level);
      

    if (new_bank_lst != NIL) {
	bs = make_scanning_over_one_tile(module, 
					 make_block_statement(new_bank_lst),
					 tile, initial_basis, local_basis,
					 tile_basis_in_tile_basis,
					 tile_basis_in_initial_basis,
					 iteration_domain,
					 first_parallel_level,
					 last_parallel_level);
	new_bank_lst = CONS(STATEMENT,bs,NIL);
    }
	/* put together the different pieces of code as two lists of
	   statements */
	cbl = gen_nconc(lb, gen_nconc(cb ,sb));
	bsb = gen_nconc(new_bank_lst, bsb);
	embl =gen_nconc(blb, bsb);
	
	/* add the scanning over the tiles around them to build a proper stmt */
	cs = make_scanning_over_tiles(compute_module, cbl, proc_id, pn,tile, 
				      initial_basis, 
				      tile_basis_in_tile_basis2,
				      tile_basis_in_initial_basis,
				      iteration_domain,first_parallel_level,
				      last_parallel_level);
	ems = make_scanning_over_tiles(memory_module, embl, proc_id_mm, 
				       pn, tile, initial_basis, 
				       tile_basis_in_tile_basis, 
				       tile_basis_in_initial_basis,
				       iteration_domain,first_parallel_level,
				       last_parallel_level);
	
	if (fully_sequential) {
	    range looprange = make_range(int_to_expression(0),
					 int_to_expression(pn-1),
					 int_to_expression(1));
	    entity looplabel = make_loop_label(9000, 
					       compute_module);
	    loop newloop = make_loop(proc_id, 
				     looprange,
				     ems,
				     looplabel, 
				     make_execution(is_execution_parallel,
						    UU),
				     NIL);
	    
	    ems = loop_to_statement(newloop);
	}
	/* update computational and emulator with cs and ems */
    }
    i = statement_instruction(computational);
    if (cs != statement_undefined) 
	instruction_block(i) = gen_nconc(instruction_block(i),CONS(STATEMENT, cs, NIL));
    i = statement_instruction(emulator);
    instruction_block(i) = gen_nconc(instruction_block(i),CONS(STATEMENT, ems, NIL));
    
    ifdebug(5) {
	(void) fprintf(stderr,
		       "Vars having to be stored into global memory:\n");
	reference_list_print(store_data_list);
	reference_list_print(store_reference_list);
    }
    gen_map((gen_iter_func_t)reference_scalar_defined_p, store_data_list);
    
      MAPL(r1,{ 
	reference rf = REFERENCE(CAR(r1));
	if (reference_scalar_p(rf)) {
	    include_constant_symbolic_communication(compute_module,CONS(REFERENCE,rf,NIL),
						    false,computational,proc_id);
	    include_constant_symbolic_communication(memory_module,CONS(REFERENCE,rf,NIL),
						    true,emulator,(entity)bank_indices->var);
	}
    },
	 store_data_list); 
    hash_table_free(llv_to_lcr);
    hash_table_free(r_to_llv);
    hash_table_free(v_to_lllv);
    hash_table_free(r_to_ud);
    
    debug(5,"loop_nest_to_wp65_code", "end\n");
    debug_off();
}




/* generates all data movements related to entity v, loads or stores
   depending on use_def */
void 
make_all_movement_blocks(initial_module,compute_module,memory_module, v,map,
			 llv_to_lcr, v_to_lllv,r_to_ud, v_to_esv,
			 pn,bn, ls,
			 iteration_domain, initial_basis, local_basis,
			 proc_id, bank_indices,  loop_body_indices,
			 pmb, pbmb,
			 use_def,first_parallel_level,last_parallel_level)
entity initial_module;
entity compute_module;
entity memory_module;
entity v;
list map;
hash_table llv_to_lcr;
hash_table v_to_lllv;
hash_table r_to_ud;
hash_table v_to_esv;
int pn;
int bn;
int ls;
Psysteme iteration_domain;
Pbase initial_basis;
Pbase local_basis;
entity proc_id;
Pbase bank_indices;  
Pbase loop_body_indices;
list * pmb; /* movement blocks */
list * pbmb; /* bank movement blocks */
tag use_def;
int first_parallel_level,last_parallel_level;
{
    list lllv = (list) hash_get(v_to_lllv, (char *) v);
    entity esv = (entity) hash_get(v_to_esv, (char *) v);

    debug(8,"make_all_movement_blocks", "begin\n");
    for(; !ENDP(lllv); POP(lllv)) {
	/* FI: depending on the pipeline stage, a specific lv should be 
	   chosen; by default, let's take the first one */
	list llv = LIST(CAR(lllv));
	entity lv = ENTITY(CAR(llv));
	list lr = (list) hash_get(llv_to_lcr, (char *) llv);
	bool proper_tag = false;

	for(; !ENDP(lr) && !proper_tag ; POP(lr)) {
	    reference r = REFERENCE(CAR(lr));
	    if( reference_in_list_p(r,map) && 
	       (intptr_t) hash_get(r_to_ud, r) == (intptr_t)use_def) {
		statement mbs;	/* statement for one movement block */
		statement bmbs;	 /* statement for one bank movement block */

		proper_tag = true;
		switch(use_def) {
		case is_action_read:
		    make_load_blocks(initial_module,compute_module,
				     memory_module, v, esv, lv,
				     lr, r_to_ud, iteration_domain,
				     initial_basis, bank_indices,
				     local_basis, loop_body_indices,
				     proc_id, pn,bn, ls, 
				     &mbs, &bmbs,first_parallel_level,
				     last_parallel_level);

		    break;
		case is_action_write:
		    make_store_blocks(initial_module,compute_module,
				      memory_module, v, esv, lv,
				      lr, r_to_ud, iteration_domain,
				      initial_basis, 
				      bank_indices,local_basis,
				      loop_body_indices,
				      proc_id, pn,bn, ls, 
				      &mbs, &bmbs,first_parallel_level,
				      last_parallel_level);
		    break;
		default:
		    pips_internal_error("unexpected use-def = %d", use_def);
		}

		ifdebug(9) {
		    pips_debug(9, "mbs=\n");
		    wp65_debug_print_text(compute_module, mbs);
		    pips_debug(9, "bmbs=\n");
		    wp65_debug_print_text(compute_module, mbs);
		}

		*pmb = gen_nconc(*pmb, CONS(STATEMENT, mbs, NIL));
		*pbmb = gen_nconc(*pbmb, CONS(STATEMENT, bmbs, NIL));
	    }
	}
    }

    debug(8,"make_all_movement_blocks", "end\n");
}



void 
search_parallel_loops( mod_stat,loop_statement,dg,loop_carried_dep)
statement mod_stat;
statement loop_statement;
graph dg;
bool loop_carried_dep[11];
{
    cons *pv1, *ps, *pc;
    set region = region_of_loop(loop_statement);;

    set_enclosing_loops_map( loops_mapping_of_statement( mod_stat) );

    for (pv1 = graph_vertices(dg); !ENDP(pv1); pv1 = CDR(pv1)) {
	vertex v1 = VERTEX(CAR(pv1));
	statement s1 = vertex_to_statement(v1);
	list loops1 = load_statement_enclosing_loops(s1);

	if (set_belong_p(region,(char *) s1))  {
	    for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
		successor su = SUCCESSOR(CAR(ps));
		vertex v2 = successor_vertex(su);
		statement s2 = vertex_to_statement(v2);
		list loops2 = load_statement_enclosing_loops(s2);
		dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
		int nbrcomloops = FindMaximumCommonLevel(loops1, loops2);
		
		for (pc = dg_arc_label_conflicts(dal); 
		     !ENDP(pc); pc = CDR(pc)) {
		    conflict c = CONFLICT(CAR(pc));

		    if(conflict_cone(c) != cone_undefined) {
			cons * lls = cone_levels(conflict_cone(c));
			cons *llsred =NIL;
			MAPL(pl,{ 
			    int level = INT(CAR(pl));
			    if ((level <= nbrcomloops) 
				      && !ignore_this_conflict(v1,v2,c,level))
				llsred = gen_nconc(llsred, CONS(INT, level,  NIL));
			}, lls); 
			if (llsred != NIL)  
			    MAPL(pl, 
			     { loop_carried_dep[INT(CAR(pl))] = true;
			   }, llsred);
		    }
		}
	    }
	}
    } 
  reset_enclosing_loops_map(); 
}

bool 
full_parallel_loop_nest_p(statement mod_stat,statement loop_stmt, 
			  int nest_dim,graph dg, bool *loop_carried_dep)
{
    int i; 
    search_parallel_loops(mod_stat,loop_stmt, dg,loop_carried_dep);
    for (i=1; i<= nest_dim && loop_carried_dep[i]== false; i++);
    return( (i>nest_dim) ? true : false);
}

