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
  * File: wp65.c
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin, Corinne Ancourt, Lei Zhou
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
#include "properties.h"
#include "prettyprint.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "movements.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "top-level.h"
#include "control.h"

#include "wp65.h"

DEFINE_CURRENT_MAPPING(fetch,list)
DEFINE_CURRENT_MAPPING(store,list) 


static void unary_into_binary_ref(reference ref)
{
    list lt; 
    if (gen_length((lt = reference_indices(ref))) ==1) {
	expression expr = int_to_expression(1);
	reference_indices(ref)= CONS(EXPRESSION, EXPRESSION(CAR(lt)),
				     CONS(EXPRESSION,expr,NIL));
    }
}

static void translate_unary_into_binary_ref(statement stat)
{
    gen_recurse(stat, reference_domain, gen_true, unary_into_binary_ref);
    ifdebug(8) {  
	entity module =get_current_module_entity();
	fprintf(stderr,"statement without unary references \n");
	wp65_debug_print_text(module,stat);
    }

} 


entity 
MakeEntityFunction(string sname)
{
    entity f = make_empty_function(sname, MakeIntegerResult(), make_language_unknown());
    return f;
}

#if 0
static void print_ref(reference r)
{
  fprintf(stderr, "reference to %s is %p\n", 
	  entity_name(reference_variable(r)), r);
}

static void print_eff(effect e)
{
  print_ref(effect_any_reference(e));
}

static void debug_refs(gen_chunk *x)
{
  gen_multi_recurse(x, 
		    reference_domain, gen_true, print_ref,
		    effect_domain, gen_true, print_eff,
		    NULL);
}
#endif

bool 
wp65(string input_module_name)
{
    entity module = module_name_to_entity(input_module_name);
    /* Let's modify the old code instead of copy it but do not tell
       pipsdbm; else we would get a *copy* of the code, not consistent
       with the dependence graph */
    statement s;
    entity compute_module = entity_undefined;
    statement computational = statement_undefined;
    entity memory_module = entity_undefined;
    statement emulator = statement_undefined;
    int pn, bn, ls, pd = PIPELINE_DEPTH;
    graph dg;
    string ppp;

    s = (statement) db_get_memory_resource(DBR_CODE, input_module_name,true);
    dg = (graph) db_get_memory_resource(DBR_DG, input_module_name, true);
    set_ordering_to_statement(s);
    debug_on("WP65_DEBUG_LEVEL");
    debug(8, "wp65", "begin\n");

    get_model(&pn, &bn, &ls);
    ifdebug(1) model_fprint(stderr, pn, bn ,ls);

    regions_init();
    set_current_module_entity(module);
    ppp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
    set_string_property(PRETTYPRINT_PARALLEL, "doall");

    /*    fprintf(stderr,"refs du code\n"); debug_refs(s);
	  fprintf(stderr,"refs du dg\n"); debug_refs(dg); */

    translate_unary_into_binary_ref(s);      

    /* fprintf(stderr,"refs du code\n"); debug_refs(s); 
       fprintf(stderr,"refs du dg\n"); debug_refs(dg); */

    module_to_wp65_modules(module, s, dg,
			   pn, bn, ls, pd, 
			   &compute_module, &computational, 
			   &memory_module, &emulator);
    
    (void) variable_declaration_coherency_p(compute_module,computational);
    (void) variable_declaration_coherency_p(memory_module,emulator);

  
    /* Put final code for the computational module in a text resource 
       of the database */
    init_prettyprint(empty_text);
    make_text_resource(input_module_name,
		       DBR_WP65_COMPUTE_FILE, 
		       WP65_COMPUTE_EXT,
		       text_module(compute_module,computational));
    close_prettyprint();

    /* Put final code for the memory module in a text resource 
       of the database */
    make_text_resource(input_module_name,
		       DBR_WP65_BANK_FILE, 
		       WP65_BANK_EXT,
		       text_module(memory_module,emulator));

    debug(8, "wp65", "end\n");

    /* reset_current_module_statement(); */
    reset_ordering_to_statement();
    reset_current_module_entity();
    set_string_property(PRETTYPRINT_PARALLEL, ppp); free(ppp);
    debug_off();

return true;

}

void module_to_wp65_modules(module, module_code, dg, 
			    pn, bn, ls, pd,
			    pcompute_module, pcomputational,
			    pmemory_module, pemulator)
entity module;
statement module_code;
graph dg; /* dependence graph */
int pn;   /* processor_number */
int bn;   /* bank number */
int ls;   /* line size   */
int pd;   /* pipeline depth */
entity * pcompute_module;
statement * pcomputational;
entity * pmemory_module;
statement * pemulator;
{
    string compute_module_name;
    string memory_module_name;
    instruction i = instruction_undefined;
    list l = list_undefined;
    entity proc_id = entity_undefined;
    entity proc_id_mm = entity_undefined;
    entity bank_id = entity_undefined;
    entity bank_line = entity_undefined;
    entity bank_offset = entity_undefined;
    Pbase bank_indices = BASE_NULLE;
    statement fs = statement_undefined;

    /* variable to emulated shared variable */
    hash_table v_to_esv = hash_table_make(hash_pointer,0);
    /* To establish an occurence numbering accross loop nests */
    hash_table v_to_nlv = hash_table_make(hash_pointer, 0);

    entity compute_module;
    statement computational;
    entity memory_module;
    statement emulator;
    statement_mapping fetch_map= MAKE_STATEMENT_MAPPING();
    statement_mapping store_map= MAKE_STATEMENT_MAPPING();
    entity div;

    debug(6,"module_to_wp65_modules","begin\n");

    /*       Generate two new modules, compute module and memory module
     */
    compute_module_name = strdup(COMPUTE_ENGINE_NAME);
    compute_module = make_empty_subroutine(compute_module_name,copy_language(module_language(module)));

    module_functional_parameters(compute_module)
	= CONS(PARAMETER, make_parameter(MakeTypeVariable(make_basic_int(4), NIL),
					 make_mode(is_mode_reference, UU),
					 make_dummy_unknown()),
	       NIL);

    memory_module_name = strdup(BANK_NAME);
    memory_module = make_empty_subroutine(memory_module_name,copy_language(module_language(module)));

    module_functional_parameters(memory_module)
	= CONS(PARAMETER, 
	       make_parameter(MakeTypeVariable(make_basic_int(
							  4), NIL),
			      make_mode(is_mode_reference, UU),
			      make_dummy_unknown()),
	       NIL);
    computational = make_block_statement(NIL);
    emulator = make_block_statement(NIL);

    div =MakeEntityFunction("idiv");
    AddEntityToDeclarations(div,compute_module);
    AddEntityToDeclarations(div,memory_module);

  
    /* Generate scalar variables that are going to be used all over
       the compute and memory module
       */
    proc_id_mm = make_scalar_integer_entity(PROCESSOR_IDENTIFIER,
					    entity_local_name(memory_module));
    AddEntityToDeclarations( proc_id_mm,memory_module);

    proc_id = make_scalar_integer_entity(PROCESSOR_IDENTIFIER,
					 entity_local_name(compute_module));
    entity_storage(proc_id) = make_storage(is_storage_formal, 
					   make_formal(compute_module, 1));
    AddEntityToDeclarations( proc_id,compute_module);

    bank_id = make_scalar_integer_entity(BANK_IDENTIFIER,
					 entity_local_name(compute_module));
    AddEntityToDeclarations( bank_id,compute_module);

    bank_id = make_scalar_integer_entity(BANK_IDENTIFIER,
					 entity_local_name(memory_module));
    entity_storage(bank_id) = make_storage(is_storage_formal, 
					   make_formal(memory_module, 1)); 
    AddEntityToDeclarations( bank_id,memory_module);

    bank_line = make_scalar_integer_entity(BANK_LINE_IDENTIFIER,
					   entity_local_name(compute_module)); 
    AddEntityToDeclarations( bank_line,compute_module); 

    bank_line = make_scalar_integer_entity(BANK_LINE_IDENTIFIER,
					   entity_local_name(memory_module));
    AddEntityToDeclarations( bank_line,memory_module);

    bank_offset = make_scalar_integer_entity(BANK_OFFSET_IDENTIFIER,
					     entity_local_name(memory_module));
    AddEntityToDeclarations( bank_offset,memory_module);
    bank_offset = make_scalar_integer_entity(BANK_OFFSET_IDENTIFIER,
					     entity_local_name(compute_module));
    AddEntityToDeclarations( bank_offset,compute_module);
    /* variables related to bank emulation are put together in one basis
       to decrease the number of parameters; they are used as such when
       liner systems of constraints are built */
    bank_indices = 
	vect_add_variable(vect_add_variable(vect_add_variable(BASE_NULLE,
							      (char*) 
							      bank_offset),
					    (char *) bank_line),
			  (char *) bank_id);

    ifdebug(6) {
	(void) fprintf(stderr,"Bank indices:\n");
	vect_fprint(stderr, bank_indices, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,"Code for the computational module:\n");
	wp65_debug_print_module(compute_module,computational);
	(void) fprintf(stderr,"Code for the memory module:\n");
	wp65_debug_print_module(memory_module,emulator);
    }


    /* skip a potential useless unstructured */
    i = statement_instruction(module_code);
    if (instruction_unstructured_p(i)) {
	unstructured u = instruction_unstructured(i);
	control c = unstructured_control(u);
	i = statement_instruction(control_statement(c));
    }

    /* generate code for each loop nest in the module and append it
       to computational and emulator */
    l = instruction_block(i); 

    compute_communications(l,&fetch_map,&store_map);
  
    instruction_to_wp65_code(module,l, dg, 
			     pn, bn, ls, pd,
			     proc_id, proc_id_mm, bank_indices,
			     v_to_esv, v_to_nlv,
			     compute_module, computational,
			     memory_module, emulator, fetch_map,store_map);

    i = statement_instruction(computational);
    instruction_block(i) = gen_nconc(instruction_block(i), 
				     CONS(STATEMENT, 
					  make_return_statement(compute_module), 
					  NIL));
    i = statement_instruction(emulator);
    instruction_block(i) = gen_nconc(instruction_block(i), 
				     CONS(STATEMENT, 
					  make_return_statement(memory_module),
					  NIL));

  
    fs = STATEMENT(CAR(instruction_block(statement_instruction(computational))));
    statement_comments(fs) = 
	strdup(concatenate("\nC     WP65 DISTRIBUTED CODE FOR ",
			   module_local_name(module), "\n",
			   (string_undefined_p(statement_comments(fs)) 
			    ? NULL :statement_comments(fs)),
			   NULL));
    module_reorder(fs);
    fs = STATEMENT(CAR(instruction_block(statement_instruction(emulator))));
    statement_comments(fs) =
	strdup(concatenate("\nC     BANK DISTRIBUTED CODE FOR ",
			   module_local_name(module), "\n",
			   (string_undefined_p(statement_comments(fs)) 
			    ? NULL :statement_comments(fs)),
			   NULL));
    module_reorder(fs);
    /* kill_statement_number_and_ordering(computational);
    kill_statement_number_and_ordering(emulator);*/
    

    ifdebug(1) {
	(void) fprintf(stderr,"Final code for the computational module:\n");
	wp65_debug_print_module(compute_module,computational);
	(void) fprintf(stderr,"Final code for the memory module:\n");
	wp65_debug_print_module(memory_module,emulator);
    }

    hash_table_free(v_to_esv);
    hash_table_free(v_to_nlv);
   
    /* return results */
    * pcompute_module = compute_module;
    * pcomputational = computational;
    * pmemory_module = memory_module;
    * pemulator = emulator;

    debug(6,"module_to_wp65_modules","end\n");

}


/* Ignore this function: debugging purposes only */
void fprint_wp65_hash_tables(fd, llv_to_lcr, r_to_llv, v_to_lllv, r_to_ud,
			     v_to_esv)
FILE * fd;
hash_table llv_to_lcr;
hash_table r_to_llv;
hash_table v_to_lllv;
hash_table r_to_ud;
hash_table v_to_esv;
{
    fputs("\nKey mappings for WP65:\n\n", fd);

    fputs("Mapping llv_to_lcr from list of local variables to conflicting references:\n", 
	  fd);
    HASH_MAP(llv, lcr,
	 {
	     list llvl=(list) llv;
	  list lcrl= (list) lcr;
	   for(; !ENDP(llvl); POP(llvl)) {
	       entity lv = ENTITY(CAR(llvl));
	       fputs(entity_local_name(lv), fd);
	       if(ENDP(CDR(llvl)))
		   (void) putc(' ', fd);
	       else
		   (void) putc(',', fd);
	   }
	  fputs("\t->\t",fd);
	  for(; !ENDP(lcrl); POP(lcrl)) {
	      reference r = REFERENCE(CAR(lcrl));
	      print_words(fd, words_reference(r, NIL));
	      if(ENDP(CDR(lcrl)))
		  (void) putc('\n', fd);
	      else
		  (void) putc(',', fd);
	  }
      },
	     llv_to_lcr);

    fputs("\nMapping r_to_llv from a reference to a list of local variables:\n", 
	  fd);
    HASH_MAP(r, llv,
	 {  list llvl=(list) llv;
	   print_words(fd, words_reference((reference) r, NIL));
	     fputs("\t->\t",fd);
	     for(; !ENDP(llvl); POP(llvl)) {
		 entity lv = ENTITY(CAR(llvl));
		 fputs(entity_local_name(lv), fd);
		 if(ENDP(CDR(llvl)))
		     (void) putc(' ', fd);
		 else
		     (void) putc(',', fd);
	     }
	     (void) putc('\n', fd);
	 },
	     r_to_llv);

    fputs("\nMapping v_to_lllv from variables to lists of lists of local variables:\n",
	  fd);
    HASH_MAP(v, lllv,
	 {  list lllvl=(list)lllv;
	   (void) fprintf(fd,"%s\t->\t", entity_name((entity) v));
	   for(; !ENDP(lllvl); POP(lllvl)) {
	       list llv = LIST(CAR(lllvl));
	       (void) putc('(',fd);
	       for(; !ENDP(llv); POP(llv)) {
		   fputs(entity_local_name(ENTITY(CAR(llv))), fd);
		   if(ENDP(CDR(llv)))
		       (void) putc(')', fd);
		   else
		       (void) putc(',', fd);
	       }
	       if(ENDP(CDR(lllvl)))
		   (void) putc('\n', fd);
	       else
		   (void) putc(',',fd);
	   }
       },
	     v_to_lllv);


    fputs("\nMapping r_to_ud from references to use-def:\n", fd);
    HASH_MAP(r, use_def,
	 {
	   print_words(fd, words_reference((reference) r, NIL));
	     fputs("\t->\t",fd);
	     fputs(((intptr_t) use_def == (intptr_t)is_action_read) ? "use\n" : "def\n", fd);
       },
	     r_to_ud);


    fputs("\nMapping v_to_esv from variables to emulated shared variables:\n",
	  fd);
    HASH_MAP(v, esv,
	 {
	   (void) fprintf(fd,"%s\t->\t", entity_name((entity) v));
	   (void) fprintf(fd,"%s\n", entity_name((entity) esv));
       },
	     v_to_esv);

    (void) putc('\n', fd);
}


bool wp65_conform_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    if (instruction_unstructured_p(i)) {
	/* there should be only one instruction: do not put a STOP in the
	   source file */
	unstructured u = instruction_unstructured(i);
	control c = unstructured_control(u);
	if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
	    i = statement_instruction(control_statement(c));
	}
	else {
	    debug(1,"wp65_conform_p",
		  "program body is an unstructured with at least two nodes\n");
	    return false;
	}
    }

    if(!instruction_block_p(i)) {
	debug(1,"wp65_conform_p","program body is not a block\n");
	return false;
    }
    else {
	list ls = instruction_block(i);
	MAPL(pm,{
	    statement s1 = STATEMENT(CAR(pm));
	    if(!assignment_statement_p(s1) && !perfectly_nested_loop_p(s1)) {
		if(!stop_statement_p(s1) && !return_statement_p(s1)) {
		    debug(1,"wp65_conform_p",
			  "program body contains a non-perfectly nested loop\n");
		    return false;
		}
	    }
	},ls);
    }
    return true;
}
