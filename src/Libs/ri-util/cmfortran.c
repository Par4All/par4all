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
#include "misc.h"

#include "ri-util.h"

/*======================================================================*/
cons *words_cmf_loop_range(obj)
range obj;
{
    cons *pc;
    call c = syntax_call(expression_syntax(range_increment(obj)));

    pc = words_subexpression(range_lower(obj), 0, TRUE);
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_subexpression(range_upper(obj), 0, TRUE));
    if (/*  expression_constant_p(range_increment(obj)) && */
	 strcmp( entity_local_name(call_function(c)), "1") == 0 )
	return(pc);
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(range_increment(obj)));

    return(pc);
}


/*======================================================================*/
text text_loop_cmf(module, label, margin, obj, n, lr, lidx)
     entity module;
     string label;
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
    pc = gen_nconc(pc, words_call(instruction_call(i), 0, TRUE, TRUE));
    u = make_unformatted(strdup(label), n, margin, pc) ;
    result_text = make_text(CONS(SENTENCE,
				 make_sentence(is_sentence_unformatted,
					       u),
				 NIL));
  }
  return(result_text);
}

