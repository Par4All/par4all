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
/* Prettyprinter for CRAFT loops.

   There are memory leaks here since a new expression is constructed.

*/

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "misc.h"
#include "properties.h"

#include "ri-util.h"
/*=======================================================================*/
/* void rewrite_modulo_expression(exp):
 *
 * filter on the call "ca". It removes all the call to the modulo
 * function.
 * 
 * AP 94/12/20 */

static void rewrite_modulo_expression(exp)
expression exp;
{
  syntax sy;
  call ca;
  entity func;
  list args;
  expression first_arg;

  sy = expression_syntax(exp);
  if(syntax_tag(sy) == is_syntax_call) {
    ca = syntax_call(sy);
    func = call_function(ca);
    if(strcmp(entity_local_name(func), MODULO_OPERATOR_NAME) == 0) {
      args = call_arguments(ca);
      if(gen_length(args) != 2)
	user_error("rewrite_modulo_expression",
		   "\nA modulo not with exactly 2 arguments\n");
      else {
	first_arg = EXPRESSION(CAR(args));
	expression_syntax(exp) = expression_syntax(first_arg);
      }
    }
  }
}

/*======================================================================*/
expression remove_modulo(exp)
expression exp;
{
  gen_recurse(exp, expression_domain, gen_true, rewrite_modulo_expression);

  return(exp);
}

/*======================================================================*/
text text_loop_craft(module, label, margin, obj, n, lr, lidx)
     entity module;
     const char* label;
     int margin;
     loop obj;
     int n;
     list lr, lidx;
{
  text result_text = make_text(NIL);
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
      result_text = text_loop_craft(module, label, margin,
				  instruction_loop(i), n, lr, lidx);
    }
  }
  else {
    list pc, lli, llr, args, new_lli = NIL;
    unformatted u;
    int c;
    char *comment;
    expression lhs_exp = expression_undefined;
    syntax lhs_sy;
    
    args = call_arguments(instruction_call(i));
    if(!ENDP(args))
      lhs_exp = copy_expression(EXPRESSION(CAR(args)));
    else
      user_error("text_loop_craft",
		 "Call to an assign with no argument\n");

    lhs_sy = expression_syntax(lhs_exp);
    if(syntax_tag(lhs_sy) != is_syntax_reference)
      user_error("text_loop_craft", "\n An lhs expression not a ref\n");
    else {
      lli = reference_indices(syntax_reference(lhs_sy));
      for(; !ENDP(lli); POP(lli)) {
	new_lli = gen_nconc(new_lli, CONS(EXPRESSION,
					  remove_modulo(EXPRESSION(CAR(lli))),
					  NIL));
      }
      reference_indices(syntax_reference(lhs_sy)) = new_lli;
    }

    comment = (char*) malloc(64);
    sprintf(comment,"CDIR$ DOSHARED(");
    for(lli = lidx; !ENDP(lli); POP(lli)) {
      sprintf(comment, "%s%s", comment,
	      entity_local_name(ENTITY(CAR(lli))));
      if(CDR(lli) != NIL)
	sprintf(comment, "%s, ", comment);
    }
    sprintf(comment, "%s) ON %s\n", comment,
	    words_to_string(words_expression(lhs_exp, NIL)));
    ADD_SENTENCE_TO_TEXT(result_text, make_sentence(is_sentence_formatted,
						    comment));

    for(lli = lidx, llr = lr, c = 0; !ENDP(lli); POP(lli), POP(llr), c++) {
      pc = CHAIN_SWORD(NIL, "DO " );
      pc = CHAIN_SWORD(pc, entity_local_name(ENTITY(CAR(lli))));
      pc = CHAIN_SWORD(pc, " = ");
      pc = gen_nconc(pc, words_loop_range(RANGE(CAR(llr)), NIL));
      u = make_unformatted(strdup(label), n,
			   margin+c*INDENTATION, pc);
      ADD_SENTENCE_TO_TEXT(result_text,
			   make_sentence(is_sentence_unformatted, u));
    }
    MERGE_TEXTS(result_text, text_statement(module, margin+c*INDENTATION,
					    loop_body(obj), NIL));

    for(c = gen_length(lidx)-1; c > -1; c--) {
      ADD_SENTENCE_TO_TEXT(result_text,
			   MAKE_ONE_WORD_SENTENCE(margin+c*INDENTATION,
						  "ENDDO"));
    }
  }
  return(result_text);
}
