 /* Code Generation for Distributed Memory Machines
  *
  * Build the iteration domain and the basis associated to a loop nest
  *
  * File: find_iteration_domain.c
  *
  * PUMA, ESPRIT contract 2701
  * Corinne Ancourt
  * 1994
  */

#include <stdio.h>

#include "genC.h"

#include "linear.h"

#include "ri.h"
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
#include "resources.h"
#include "pipsdbm.h"
#include "rice.h"
#include "movements.h"

#include "constants.h"
#include "wp65.h"

extern Psysteme loop_iteration_domaine_to_sc();

void find_iteration_domain(s, sc, basis, nested_level,
                                    list_statement_block, inst)
statement s;
Psysteme *sc; 
Pbase *basis;
int * nested_level ;
list *list_statement_block;
instruction * inst;
{
    list list_loop_statement=NIL;
  
    debug(8, "find_iteration_domain", "begin\n");
    iteration_domain_from_statement(&list_loop_statement,s,
                                    nested_level,
                                    list_statement_block,inst);
    compute_iteration_domain(list_loop_statement,sc,basis);
	ifdebug(8) {
	    (void) fprintf(stderr,"[find_iteration_domain] initial basis \n");
	    vect_fprint(stderr,*basis,entity_local_name);
	    sc_fprint(stderr,*sc,entity_local_name);
	}
    debug(8, "find_iteration_domain", "end\n");
}



void compute_iteration_domain(list_loop_statement,sc,basis)
list  list_loop_statement;
Psysteme *sc; 
Pbase *basis;
{

    Psysteme sci;
    Pbase base_index = BASE_NULLE;
 
    /* computation of the list of loop indices base_index 
       and of the iteration domain sci*/
       
    debug(8,"compute_iteration_domain","begin\n");

    sci = loop_iteration_domaine_to_sc(list_loop_statement, &base_index);
    sci->base = base_reversal(sci->base);
    ifdebug(8) { (void) fprintf(stderr,"compute_iteration_domain\n");
		 vect_fprint(stderr,base_index,entity_local_name);
	     }
    *sc = sci;
    *basis = base_index;
    debug(8,"compute_iteration_domain","end\n");
}

void iteration_domain_from_statement(list_loop_statement, s,nested_level, blocks,inst)
list *list_loop_statement;
statement s;
int * nested_level;
list *blocks;
instruction *inst;
{
    instruction i;
    cons *b;
    loop l;
   debug(8, "iteration_domain_from_statement", "begin\n");
	
    i = statement_instruction(s);
    switch (instruction_tag(i)) {

    case is_instruction_loop:
	l = instruction_loop(i);
	*list_loop_statement = CONS (STATEMENT,s,*list_loop_statement);
	iteration_domain_from_statement(list_loop_statement,loop_body(l),
                                        nested_level, 
                                        blocks,inst);
	break;

    case is_instruction_block: {
        int nbl = 0;
        boolean simple_block = FALSE;
	b= instruction_block(i);
        nbl = gen_length((list) b);
        simple_block = (nbl==1 
                        || (nbl ==2 && continue_statement_p(STATEMENT(CAR(CDR(b)))))) 
            ? TRUE : FALSE;
        
	if (simple_block && instruction_loop_p(statement_instruction(STATEMENT(CAR(b)))))
                iteration_domain_from_statement(list_loop_statement,
						STATEMENT(CAR(b))
						,nested_level, blocks,inst);
	else { 
            *nested_level = gen_length(*list_loop_statement);
	    *inst = i;
            *blocks = b;
            }
      
	break;
    }
    case is_instruction_call: {
	/* case where there is a unique assignment in do-enddo loop nest*/
	 *nested_level = gen_length(*list_loop_statement);
	 *inst =make_instruction_block(CONS(STATEMENT,s,NIL));  
	 *blocks =CONS(STATEMENT,s,NIL);
        return;
	   }

    case is_instruction_test:
	 return;
	
    case is_instruction_unstructured:
	 return;
	
    case is_instruction_goto:
	pips_error("search_array_from_statement",
		   "unexpected goto in code");
    default:
	pips_error("search_array_from_statement",
		   "unexpected tag");
    }
    
    debug(8, "search_array_from_statement", "end\n");
}
