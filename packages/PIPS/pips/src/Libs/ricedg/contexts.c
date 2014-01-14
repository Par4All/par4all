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
#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"

#include "misc.h"

#include "ri-util.h"
#include "effects-util.h"

static Psysteme make_context_of_loop(stat)
statement stat;
{
    loop il = statement_loop(stat);
    Psysteme cil = sc_new();
    entity i = loop_index(il);
    range r = loop_range(il);
    expression lb = range_lower(r), ub = range_upper(r);
    normalized nl, nu;
    int inc;

   

    nl = NORMALIZE_EXPRESSION(lb);
    nu = NORMALIZE_EXPRESSION(ub);

    if ((inc = loop_increment_value(il)) == 0) {/* it isn't constant. ignore
                                                 this context */ 
	return(cil);
    }
    else {
	if (inc < 0) {
	    normalized n = nl;
	    nl = nu;
	    nu = n;
	}

	if (normalized_linear_p(nl)) {
	
	    Pvecteur pv, pv1;
	    Pcontrainte pc;

	    pv = (Pvecteur) normalized_linear(nl);
	    pv1 = vect_dup(pv);                  /* par yy */
	    vect_add_elem(&pv1, (Variable) i, -1);
	    pc =  contrainte_make(pv1);
	    sc_add_ineg(cil, pc);
	}

	if (normalized_linear_p(nu)) {

	    Pvecteur pv, pv1;
	    Pcontrainte pc;

	    pv = (Pvecteur) normalized_linear(nu);
	    pv1 = vect_dup(pv);                  /* par yy */
	    (void) vect_multiply(pv1, -1);
	    vect_add_elem(&pv1, (Variable) i, 1);
	    pc =  contrainte_make(pv1);
	    sc_add_ineg(cil, pc);
	}
	
	ifdebug(5) {
	    fprintf(stderr, "Execution context of loop %td is:\n", 
		statement_number(stat));
	    sc_fprint(stderr, cil, (get_variable_name_t) entity_local_name);
	}

	return(cil);
    }
}




static void contexts_mapping_of_statement(m, c, s)
statement_mapping m;
Psysteme c;
statement s;
{
    instruction i = statement_instruction(s);

    SET_STATEMENT_MAPPING(m, s, c);

    switch(instruction_tag(i)) {

      case is_instruction_block:
	MAPL(ps, {
	    contexts_mapping_of_statement(m, c, STATEMENT(CAR(ps)));
	}, instruction_block(i));
	break;

      case is_instruction_loop: {
	  Psysteme nc = make_context_of_loop(s);

	  contexts_mapping_of_statement(m, 
					sc_intersection(nc, nc, c), 
					loop_body(instruction_loop(i)));
	  break;
      }

      /* The next two cases are added for the dependence test 
         include IF  (by Yi-Qing 12/92)*/ 
      case is_instruction_test: {
	 contexts_mapping_of_statement(m, c, test_true(instruction_test(i))); 
	 contexts_mapping_of_statement(m, c, test_false(instruction_test(i)));
	 break;
      }

      case is_instruction_unstructured: {
	  unstructured u = instruction_unstructured(i);
	  cons *blocs = NIL ;

	  CONTROL_MAP(ct, {
	      contexts_mapping_of_statement(m, c, control_statement(ct));
	  }, unstructured_control(u), blocs);
	  
	  gen_free_list( blocs ); 
	  break;
      }
      case is_instruction_whileloop:
      case is_instruction_call:
      case is_instruction_goto:	
	break;

      default:
	pips_internal_error("unexpected tag %d", instruction_tag(i));
    }
}



statement_mapping contexts_mapping_of_nest(stat)
statement stat;
{
    statement_mapping contexts_map;

    pips_assert("contexts_mapping_of_nest", statement_loop_p(stat));

    contexts_map = MAKE_STATEMENT_MAPPING();

    contexts_mapping_of_statement(contexts_map, NIL, stat);

    ifdebug(8) {
	STATEMENT_MAPPING_MAP(st, context, {
	    statement stp = (statement) st;

	    if (statement_call_p(stp)) {
		fprintf(stderr, "Execution context of statement %td :\n", 
			statement_number(stp));
		sc_fprint(stderr, (Psysteme) context, (get_variable_name_t) entity_local_name);
	    }
	}, contexts_map);
    }

    return(contexts_map);
}










