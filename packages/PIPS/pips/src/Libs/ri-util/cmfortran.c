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
/* Prettyprinter for CM FORTRAN loops.

   There are memory leaks here since a new expression is constructed.

*/

#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "text-util.h"
#include "ri.h"

#include "ri-util.h"

/*======================================================================*/
cons *words_cmf_loop_range(obj)
range obj;
{
    cons *pc;
    call c = syntax_call(expression_syntax(range_increment(obj)));

    pc = words_subexpression(range_lower(obj), 0, true, NIL);
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_subexpression(range_upper(obj), 0, true, NIL));
    if (/*  expression_constant_p(range_increment(obj)) && */
	 strcmp( entity_local_name(call_function(c)), "1") == 0 )
	return(pc);
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(range_increment(obj), NIL));

    return(pc);
}


/*======================================================================*/
text text_loop_cmf(module, label, margin, obj, n, lr, lidx)
     entity module;
     const char* label;
     int margin;
     loop obj;
     int n;
     list lr, lidx;
{
  text result_text = text_undefined;
  instruction i;
  entity idx;
  range r;

  i = statement_instruction(loop_body(obj));
  idx = loop_index(obj);
  r = loop_range(obj);

  lr = gen_nconc(lr, CONS(RANGE, r, NIL));
  lidx = gen_nconc(lidx, CONS(ENTITY, idx, NIL));

  if(!instruction_assign_p(i)) {
    if(instruction_loop_p(i)) {
      result_text = text_loop_cmf(module, label, margin,
				  instruction_loop(i), n, lr, lidx);
    }
  }
  else {
    list pc, lli, llr;
    unformatted u;

    pc = CHAIN_SWORD(NIL, "FORALL(");
    for(lli = lidx, llr = lr; !ENDP(lli); POP(lli), POP(llr)) {
      pc = CHAIN_SWORD(pc, entity_local_name(ENTITY(CAR(lli))));
      pc = CHAIN_SWORD(pc, " = ");
      pc = gen_nconc(pc, words_cmf_loop_range(RANGE(CAR(llr))));
      if(CDR(lli) != NIL)
	pc = CHAIN_SWORD(pc, ", ");
    }
    pc = CHAIN_SWORD(pc, ") ");
    pc = gen_nconc(pc, words_call(instruction_call(i), 0, true, true, NIL));
    u = make_unformatted(strdup(label), n, margin, pc) ;
    result_text = make_text(CONS(SENTENCE,
				 make_sentence(is_sentence_unformatted,
					       u),
				 NIL));
  }
  return(result_text);
}

