/*
 * $Id$
 *
 * $Log: prettyprint.c,v $
 * Revision 1.86  1997/10/24 16:29:11  coelho
 * bug-- : external declaration if not yet parsed.
 *
 * Revision 1.85  1997/10/23 11:37:58  irigoin
 * Detection of last statement added.
 *
 * Revision 1.84  1997/10/08 08:41:37  coelho
 * management of saved variable fixed.
 *
 * Revision 1.83  1997/10/08 06:04:49  coelho
 * dim or save variables are ok.
 *
 * Revision 1.82  1997/09/19 17:33:57  coelho
 * SAVE & dims fixed again...
 *
 * Revision 1.81  1997/09/19 17:17:06  coelho
 * SAVE and dimensions
 *
 * Revision 1.80  1997/09/19 16:34:09  irigoin
 * Bug fix in text-entity_declaration: SAVE statements were not generated any
 * longer
 *
 * Revision 1.79  1997/09/19 07:50:09  coelho
 * hpf directive with !.
 *
 * Revision 1.78  1997/09/17 13:47:58  coelho
 * complex 8/16 distinction.
 *
 * Revision 1.77  1997/09/17 12:56:57  coelho
 * implicit DCMPLX ignored.
 *
 * Revision 1.76  1997/09/16 11:50:55  coelho
 * implied complex is hidden.
 *
 * Revision 1.75  1997/09/16 07:58:28  coelho
 * more debug, and print dimension of SAVEd args?
 *
 * Revision 1.74  1997/09/15 14:28:53  coelho
 * fixes for new COMMON prefix.
 *
 * Revision 1.73  1997/09/15 11:58:20  coelho
 * initial value may be undefined from wp65... guarded.
 *
 * Revision 1.72  1997/09/15 09:31:54  coelho
 * declaration regeneration~: data / blockdata fixes.
 *
 * Revision 1.71  1997/09/13 16:04:01  coelho
 * cleaner.
 *
 * Revision 1.70  1997/09/13 15:37:49  coelho
 * fixed a bug that added a blank line in the regenerated declarations.
 * basic block data recognition added.
 * integer data initializations also added.
 * many functions moved as static.
 *
 * Revision 1.69  1997/09/03 14:44:07  coelho
 * no assert.
 *
 * Revision 1.67  1997/08/04 16:56:08  coelho
 * back to initial, because  declarations are not atomic as I thought.
 *
 * Revision 1.65  1997/07/24 15:10:08  keryell
 * Assert added to insure no attribute on a sequence statement.
 *
 * Revision 1.64  1997/07/22 11:27:42  keryell
 * %x -> %p formats.
 *
 * Revision 1.63  1997/06/02 06:52:55  coelho
 * rcs headers, plus fixed commons pp for hpfc vs regions.
 *
 */

#ifndef lint
char lib_ri_util_prettyprint_c_rcsid[] = "$Header: /home/data/tmp/PIPS/pips_data/trunk/src/Libs/ri-util/RCS/prettyprint.c,v 1.86 1997/10/24 16:29:11 coelho Exp $";
#endif /* lint */
 /*
  * Prettyprint all kinds of ri related data structures
  *
  *  Modifications:
  * - In order to remove the extra parentheses, I made the several changes:
  * (1) At the intrinsic_handler, the third term is added to indicate the
  *     precendence, and accordingly words_intrinsic_precedence(obj) is built
  *     to get the precedence of the call "obj".
  * (2) words_subexpression is created to distinguish the
  *     words_expression.  It has two arguments, expression and
  *     precedence. where precedence is newly added. In case of failure
  *     of words_subexpression , that is, when
  *     syntax_call_p is FALSE, we use words_expression instead.
  * (3) When words_call is firstly called , we give it the lowest precedence,
  *        that is 0.
  *    Lei ZHOU           Nov. 4, 1991
  *
  * - Addition of CMF and CRAFT prettyprints. Only text_loop() has been
  * modified.
  *    Alexis Platonoff, Nov. 18, 1994

  * - Modifications of sentence_area to deal with  the fact that
  *   " only one appearance of a symbolic name as an array name in an 
  *     array declarator in a program unit is permitted."
  *     (Fortran standard, number 8.1, line 40) 
  *   array declarators now only appear with the type declaration, not with the
  *   area. - BC - october 196.
  * - Modification of text_entity_declaration to ensure that the OUTPUT of PIPS
  *   can also be used as INPUT; in particular, variable declarations must 
  *   appear
  *   before common declarations. BC.
  * - neither are DATA statements for non integers (FI/FC)
  * - Also, EQUIVALENCE statements are not generated for the moment. BC.
  *     Thay are now??? FC?
  */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"
#include "prettyprint.h"

/* Define the markers used in the raw unstructured output when the
   PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH property is true: */
#define PRETTYPRINT_UNSTRUCTURED_BEGIN_MARKER "\200Unstructured"
#define PRETTYPRINT_UNSTRUCTURED_END_MARKER "\201Unstructured End"
#define PRETTYPRINT_UNSTRUCTURED_ITEM_MARKER "\202Unstructured Item"
#define PRETTYPRINT_UNSTRUCTURED_SUCC_MARKER "\203Unstructured Successor ->"
#define PRETTYPRINT_UNREACHABLE_EXIT_MARKER "\204Unstructured Unreachable"



/********************************************************************* MISC */

text empty_text(entity e, int m, statement s)
{ return( make_text(NIL));}

static text (*text_statement_hook)(entity, int, statement) = empty_text;

void init_prettyprint( hook )
text (*hook)(entity, int, statement) ;
{
    /* checks that the prettyprint hook was actually reset...
     */
    pips_assert("prettyprint hook not set", text_statement_hook==empty_text);
    text_statement_hook = hook ;
}

/* because some prettyprint functions may be used for debug, so
 * the last hook set by somebody may have stayed there although
 * being non sense...
 */
void close_prettyprint()
{
    text_statement_hook = empty_text;
}


/********************************************************************* WORDS */

static int words_intrinsic_precedence(call);
static int intrinsic_precedence(string);

static list 
words_parameters(entity e)
{
    list pc = NIL;
    type te = entity_type(e);
    functional fe;
    int nparams, i;

    pips_assert("words_parameters", type_functional_p(te));

    fe = type_functional(te);
    nparams = gen_length(functional_parameters(fe));

    for (i = 1; i <= nparams; i++) {
	entity param = find_ith_parameter(e, i);

	if (pc != NIL) {
	    pc = CHAIN_SWORD(pc, ",");
	}

	pc = CHAIN_SWORD(pc, entity_local_name(param));
    }

    return(pc);
}


/* This function is added by LZ
 * 21/10/91
 * extended to cope with PRETTYPRINT_HPFC 
 */

static list 
words_dimension(dimension obj)
{
    list pc;

    pc = words_expression(dimension_lower(obj));
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(dimension_upper(obj)));

    return(pc);
}

/* some compilers don't like dimensions that are declared twice.
 * this is the case of g77 used after hpfc. thus I added a
 * flag not to prettyprint again the dimensions of common variables. FC.
 *
 * It is in the standard that dimensions cannot be declared twice in a 
 * single module. BC.
 */
list 
words_declaration(
    entity e,
    bool prettyprint_common_variable_dimensions_p)
{
    list pl = NIL;
    pl = CHAIN_SWORD(pl, entity_local_name(e));

    if (type_variable_p(entity_type(e)))
    {
	if (prettyprint_common_variable_dimensions_p || 
	    !(variable_in_common_p(e) || variable_static_p(e)))
	{
	    if (variable_dimensions(type_variable(entity_type(e))) != NIL) 
	    {
		list dims = variable_dimensions(type_variable(entity_type(e)));
	
		pl = CHAIN_SWORD(pl, "(");

		MAPL(pd, 
		{
		    pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd))));
		    if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, ",");
		}, 
		    dims);
	
		pl = CHAIN_SWORD(pl, ")");
	    }
	}
    }
    
    attach_declaration_to_words(pl, e);

    return(pl);
}

static list 
words_constant(constant obj)
{
    list pc;

    pc=NIL;

    if (constant_int_p(obj)) {
	pc = CHAIN_IWORD(pc,constant_int(obj));
    }
    else {
	pips_internal_error("unexpected tag");
    }

    return(pc);
}

static list 
words_value(value obj)
{
    list pc;

    if (value_symbolic_p(obj)) {
	pc = words_constant(symbolic_constant(value_symbolic(obj)));
    }
    else if (value_constant(obj)) {
	pc = words_constant(value_constant(obj));
    }
    else {
	pips_internal_error("unexpected tag");
	pc = NIL;
    }

    return(pc);
}

static list 
words_basic(basic obj)
{
    list pc = NIL;

    if (basic_int_p(obj)) {
	pc = CHAIN_SWORD(pc,"INTEGER*");
	pc = CHAIN_IWORD(pc,basic_int(obj));
    }
    else if (basic_float_p(obj)) {
	pc = CHAIN_SWORD(pc,"REAL*");
	pc = CHAIN_IWORD(pc,basic_float(obj));
    }
    else if (basic_logical_p(obj)) {
	pc = CHAIN_SWORD(pc,"LOGICAL*");
	pc = CHAIN_IWORD(pc,basic_logical(obj));
    }
    else if (basic_overloaded_p(obj)) {
	pc = CHAIN_SWORD(pc,"OVERLOADED");
    }
    else if (basic_complex_p(obj)) {
	pc = CHAIN_SWORD(pc,"COMPLEX*");
	pc = CHAIN_IWORD(pc,basic_complex(obj));
    }
    else if (basic_string_p(obj)) {
	pc = CHAIN_SWORD(pc,"STRING*(");
	pc = gen_nconc(pc, words_value(basic_string(obj)));
	pc = CHAIN_SWORD(pc,")");
    }
    else {
	pips_error("words_basic", "unexpected tag");
    }

    return(pc);
}

/* exported for craft 
 */
list 
words_loop_range(range obj)
{
    list pc;
    call c = syntax_call(expression_syntax(range_increment(obj)));

    pc = words_subexpression(range_lower(obj), 0);
    pc = CHAIN_SWORD(pc,", ");
    pc = gen_nconc(pc, words_subexpression(range_upper(obj), 0));
    if (/*  expression_constant_p(range_increment(obj)) && */
	 strcmp( entity_local_name(call_function(c)), "1") == 0 )
	return(pc);
    pc = CHAIN_SWORD(pc,", ");
    pc = gen_nconc(pc, words_expression(range_increment(obj)));

    return(pc);
}

list /* of string */ 
words_range(range obj)
{
    list pc = NIL ;

    /* if undefined I print a star, why not!? */
    if (expression_undefined_p(range_lower(obj)))
	return CONS(STRING, MAKE_SWORD("*"), NIL);
    /* else */
    pc = CHAIN_SWORD(pc,"(/I,I=");
    pc = gen_nconc(pc, words_expression(range_lower(obj)));
    pc = CHAIN_SWORD(pc,",");
    pc = gen_nconc(pc, words_expression(range_upper(obj)));
    pc = CHAIN_SWORD(pc,",");
    pc = gen_nconc(pc, words_expression(range_increment(obj)));
    pc = CHAIN_SWORD(pc,"/)") ;

    return(pc);
}

/* exported for expression.c
 */
list 
words_reference(reference obj)
{
    list pc = NIL;
    string begin_attachment;
    
    entity e = reference_variable(obj);

    pc = CHAIN_SWORD(pc, entity_local_name(e));
    begin_attachment = STRING(CAR(pc));
    
    if (reference_indices(obj) != NIL) {
	pc = CHAIN_SWORD(pc,"(");
	MAPL(pi, {
	    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(pi)), 0));
	    if (CDR(pi) != NIL)
		pc = CHAIN_SWORD(pc,",");
	}, reference_indices(obj));
	pc = CHAIN_SWORD(pc,")");
    }
    attach_reference_to_word_list(begin_attachment, STRING(CAR(gen_last(pc))),
				  obj);

    return(pc);
}

static list 
words_label_name(string s)
{
    return(CHAIN_SWORD(NIL, local_name(s)+strlen(LABEL_PREFIX))) ;
}

list 
words_regular_call(call obj)
{
    list pc = NIL;

    entity f = call_function(obj);
    value i = entity_initial(f);
    type t = entity_type(f);
    
    if (call_arguments(obj) == NIL) {
	if (type_statement_p(t))
	    return(CHAIN_SWORD(pc, entity_local_name(f)+strlen(LABEL_PREFIX)));
	if (value_constant_p(i)||value_symbolic_p(i))
	    return(CHAIN_SWORD(pc, entity_local_name(f)));
    }

    if (type_void_p(functional_result(type_functional(t)))) {
	pc = CHAIN_SWORD(pc, "CALL ");
    }

    /* the implied complex operator is hidden... [D]CMPLX_(x,y) -> (x,y)
     */
    if (!ENTITY_IMPLIED_CMPLX_P(f) && !ENTITY_IMPLIED_DCMPLX_P(f))
	pc = CHAIN_SWORD(pc, entity_local_name(f));

    if( !ENDP( call_arguments(obj))) {
	pc = CHAIN_SWORD(pc, "(");
	MAPL(pa, {
	    pc = gen_nconc(pc, words_expression(EXPRESSION(CAR(pa))));
	    if (CDR(pa) != NIL)
		    pc = CHAIN_SWORD(pc, ", ");
	}, call_arguments(obj));
	pc = CHAIN_SWORD(pc, ")");
    }
    else if(!type_void_p(functional_result(type_functional(t)))) {
	pc = CHAIN_SWORD(pc, "()");
    }
    return(pc);
}

static list 
words_assign_op(call obj, int precedence)
{
    list pc = NIL;
    list args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);

    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(args)),  prec));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(CDR(args))), prec));

    return(pc);
}

static list 
words_substring_op(call obj, int precedence)
{
  /* The substring function call is reduced to a syntactic construct */
    list pc = NIL;
    expression r = expression_undefined;
    expression l = expression_undefined;
    expression u = expression_undefined;
    /* expression e = EXPRESSION(CAR(CDR(CDR(CDR(call_arguments(obj)))))); */
    int prec = words_intrinsic_precedence(obj);

    pips_assert("words_substring_op", gen_length(call_arguments(obj)) == 3 || 
		gen_length(call_arguments(obj)) == 4);

    r = EXPRESSION(CAR(call_arguments(obj)));
    l = EXPRESSION(CAR(CDR(call_arguments(obj))));
    u = EXPRESSION(CAR(CDR(CDR(call_arguments(obj)))));

    pc = gen_nconc(pc, words_subexpression(r,  prec));
    pc = CHAIN_SWORD(pc, "(");
    pc = gen_nconc(pc, words_subexpression(l, prec));
    pc = CHAIN_SWORD(pc, ":");
    pc = gen_nconc(pc, words_subexpression(u, prec));
    pc = CHAIN_SWORD(pc, ")");

    return(pc);
}

static list 
words_assign_substring_op(call obj, int precedence)
{
  /* The assign substring function call is reduced to a syntactic construct */
    list pc = NIL;
    expression e = expression_undefined;
    int prec = words_intrinsic_precedence(obj);

    pips_assert("words_substring_op", gen_length(call_arguments(obj)) == 4);

    e = EXPRESSION(CAR(CDR(CDR(CDR(call_arguments(obj))))));

    pc = gen_nconc(pc, words_substring_op(obj,  prec));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_subexpression(e, prec));

    return(pc);
}

static list 
words_nullary_op(call obj, int precedence)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));

    return(pc);
}

static list 
words_io_control(list *iol, int precedence)
{
    list pc = NIL;
    list pio = *iol;

    while (pio != NIL) {
	syntax s = expression_syntax(EXPRESSION(CAR(pio)));
	call c;

	if (! syntax_call_p(s)) {
	    pips_error("words_io_control", "call expected");
	}

	c = syntax_call(s);

	if (strcmp(entity_local_name(call_function(c)), "IOLIST=") == 0) {
	    *iol = CDR(pio);
	    return(pc);
	}

	if (pc != NIL)
	    pc = CHAIN_SWORD(pc, ",");
	
	pc = CHAIN_SWORD(pc, entity_local_name(call_function(c)));
	pc = gen_nconc(pc, words_expression(EXPRESSION(CAR(CDR(pio)))));

	pio = CDR(CDR(pio));
    }

    if (pio != NIL)
	    pips_error("words_io_control", "bad format");

    *iol = NIL;

    return(pc);
}

static list 
words_implied_do(call obj, int precedence)
{
    list pc = NIL;

    list pcc;
    expression index;
    syntax s;
    range r;

    pcc = call_arguments(obj);
    index = EXPRESSION(CAR(pcc));

    pcc = CDR(pcc);
    s = expression_syntax(EXPRESSION(CAR(pcc)));
    if (! syntax_range_p(s)) {
	pips_error("words_implied_do", "range expected");
    }
    r = syntax_range(s);

    pc = CHAIN_SWORD(pc, "(");
    MAPL(pcp, {
	pc = gen_nconc(pc, words_expression(EXPRESSION(CAR(pcp))));
	if (CDR(pcp) != NIL)
	    pc = CHAIN_SWORD(pc, ",");
    }, CDR(pcc));
    pc = CHAIN_SWORD(pc, ", ");

    pc = gen_nconc(pc, words_expression(index));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_loop_range(r));
    pc = CHAIN_SWORD(pc, ")");

    return(pc);
}

static list 
words_unbounded_dimension(call obj, int precedence)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

static list 
words_list_directed(call obj, int precedence)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

static list 
words_io_inst(call obj, int precedence)
{
    list pc = NIL;
    list pcio = call_arguments(obj);
    list pio_write = pcio;
    boolean good_fmt = FALSE;
    bool good_unit = FALSE;
    bool iolist_reached = FALSE;
    bool complex_io_control_list = FALSE;
    list fmt_words = NIL;
    list unit_words = NIL;
    string called = entity_local_name(call_function(obj));
    
    /* AP: I try to convert WRITE to PRINT. Three conditions must be
       fullfilled. The first, and obvious, one, is that the function has
       to be WRITE. Secondly, "FMT" has to be equal to "*". Finally,
       "UNIT" has to be equal either to "*" or "6".  In such case,
       "WRITE(*,*)" is replaced by "PRINT *,". */
    /* GO: Not anymore for UNIT=6 leave it ... */
      while ((pio_write != NIL) && (!iolist_reached)) {
	syntax s = expression_syntax(EXPRESSION(CAR(pio_write)));
	call c;
	expression arg = EXPRESSION(CAR(CDR(pio_write)));

	if (! syntax_call_p(s)) {
	    pips_error("words_io_inst", "call expected");
	}

	c = syntax_call(s);
	if (strcmp(entity_local_name(call_function(c)), "FMT=") == 0) {
	   good_fmt = strcmp
	       (STRING(CAR(fmt_words = words_expression(arg))), "*")==0;
	   pio_write = CDR(CDR(pio_write));
	}
	else if (strcmp(entity_local_name(call_function(c)), "UNIT=") == 0) {
	    good_unit = strcmp
		(STRING(CAR(unit_words = words_expression(arg))), "*")==0;
	    pio_write = CDR(CDR(pio_write));
	}
	else if (strcmp(entity_local_name(call_function(c)), "IOLIST=") == 0) {
	  iolist_reached = TRUE;
	  pio_write = CDR(pio_write);
	}
	else {
	    complex_io_control_list = TRUE;
	  pio_write = CDR(CDR(pio_write));
	}
      }

    if (good_fmt && good_unit && same_string_p(called, "WRITE"))
    {
	/* WRITE (*,*) -> PRINT * */

       if (pio_write != NIL) /* WRITE (*,*) pio -> PRINT *, pio */
       {
          pc = CHAIN_SWORD(pc, "PRINT *, ");
       }
       else     /* WRITE (*,*)  -> PRINT *  */
       {
          pc = CHAIN_SWORD(pc, "PRINT * ");
       }
       
       pcio = pio_write;
    }
    else if (good_fmt && good_unit && same_string_p(called, "READ"))
    {
       /* READ (*,*) -> READ * */
	
       if (pio_write != NIL) /* READ (*,*) pio -> READ *, pio */
       {
          pc = CHAIN_SWORD(pc, "READ *, ");
       }
       else   /* READ (*,*)  -> READ *  */
       {
          pc = CHAIN_SWORD(pc, "READ * ");
       }
       pcio = pio_write;
    }	
    else if(!complex_io_control_list) {
	pips_assert("A unit must be defined", !ENDP(unit_words));
      pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
      pc = CHAIN_SWORD(pc, " (");
      pc = gen_nconc(pc, unit_words);
      if(!ENDP(fmt_words)) {
	  pc = CHAIN_SWORD(pc, ", ");
	  pc = gen_nconc(pc, fmt_words);
      }
      pc = CHAIN_SWORD(pc, ") ");
      pcio = pio_write;
    }
    else {
      pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
      pc = CHAIN_SWORD(pc, " (");
      /* FI: missing argument; I use "precedence" because I've no clue;
         see LZ */
      pc = gen_nconc(pc, words_io_control(&pcio, precedence));
      pc = CHAIN_SWORD(pc, ") ");
      /* 
	free_words(unit_words);
	free_words(fmt_words);
      */
    }

    /* because the "IOLIST=" keyword is embedded in the list
       and because the first IOLIST= has already been skipped,
       only odd elements are printed */
    MAPL(pp, {
	pc = gen_nconc(pc, words_expression(EXPRESSION(CAR(pp))));

	if (CDR(pp) != NIL) {
	    POP(pp);
	    if(pp==NIL) 
		pips_error("words_io_inst","missing element in IO list");
	    pc = CHAIN_SWORD(pc, ", ");
	}
    }, pcio);
    return(pc) ;
}

static list 
null(call obj, int precedence)
{
    return(NIL);
}

static list
words_prefix_unary_op(call obj, int precedence)
{
    list pc = NIL;
    expression e = EXPRESSION(CAR(call_arguments(obj)));
    int prec = words_intrinsic_precedence(obj);

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
    pc = gen_nconc(pc, words_subexpression(e, prec));

    return(pc);
}

static list 
words_unary_minus(call obj, int precedence)
{
    list pc = NIL;
    expression e = EXPRESSION(CAR(call_arguments(obj)));
    int prec = words_intrinsic_precedence(obj);

    if ( prec < precedence )
	pc = CHAIN_SWORD(pc, "(");
    pc = CHAIN_SWORD(pc, "-");
    pc = gen_nconc(pc, words_subexpression(e, prec));
    if ( prec < precedence )
	pc = CHAIN_SWORD(pc, ")");

    return(pc);
}

list /* of string */
words_goto_label(string tlabel)
{
    list pc = NIL;
    if (strcmp(tlabel, RETURN_LABEL_NAME) == 0) {
	pc = CHAIN_SWORD(pc, RETURN_FUNCTION_NAME);
    }
    else {
	pc = CHAIN_SWORD(pc, "GOTO ");
	pc = CHAIN_SWORD(pc, tlabel);
    }
    return pc;
}

/* 
 * If the infix operator is either "-" or "/", I perfer not to delete 
 * the parentheses of the second expression.
 * Ex: T = X - ( Y - Z ) and T = X / (Y*Z)
 *
 * Lei ZHOU       Nov. 4 , 1991
 */
static list 
words_infix_binary_op(call obj, int precedence)
{
    list pc = NIL;
    list args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);
    list we1 = words_subexpression(EXPRESSION(CAR(args)), prec);
    list we2;

    if ( strcmp(entity_local_name(call_function(obj)), "/") == 0 )
	we2 = words_subexpression(EXPRESSION(CAR(CDR(args))), 100);
    else if ( strcmp(entity_local_name(call_function(obj)), "-") == 0 ) {
	expression exp = EXPRESSION(CAR(CDR(args)));
	if ( expression_call_p(exp) &&
	     words_intrinsic_precedence(syntax_call(expression_syntax(exp))) >= 
	     intrinsic_precedence("*") )
	    /* precedence is greter than * or / */
	    we2 = words_subexpression(exp, prec);
	else
	    we2 = words_subexpression(exp, 100);
    }
    else
	we2 = words_subexpression(EXPRESSION(CAR(CDR(args))), prec);

    
    if ( prec < precedence )
	pc = CHAIN_SWORD(pc, "(");
    pc = gen_nconc(pc, we1);
    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
    pc = gen_nconc(pc, we2);
    if ( prec < precedence )
	pc = CHAIN_SWORD(pc, ")");

    return(pc);
}

/* precedence needed here
 * According to the Precedence of Operators 
 * Arithmetic > Character > Relational > Logical
 * Added by Lei ZHOU    Nov. 4,91
 */
struct intrinsic_handler {
    char * name;
    list (*f)();
    int prec;
} tab_intrinsic_handler[] = {
    {"**", words_infix_binary_op, 30},

    {"//", words_infix_binary_op, 30},

    {"--", words_unary_minus, 25},

    {"*", words_infix_binary_op, 21},
    {"/", words_infix_binary_op, 21},

    {"+", words_infix_binary_op, 20},
    {"-", words_infix_binary_op, 20},


    {".LT.", words_infix_binary_op, 15},
    {".GT.", words_infix_binary_op, 15},
    {".LE.", words_infix_binary_op, 15},
    {".GE.", words_infix_binary_op, 15},
    {".EQ.", words_infix_binary_op, 15},
    {".NE.", words_infix_binary_op, 15},

    {".NOT.", words_prefix_unary_op, 9},

    {".AND.", words_infix_binary_op, 8},

    {".OR.", words_infix_binary_op, 6},

    {".EQV.", words_infix_binary_op, 3},
    {".NEQV.", words_infix_binary_op, 3},

    {"=", words_assign_op, 1},

    {"WRITE", words_io_inst, 0},
    {"READ", words_io_inst, 0},
    {"PRINT", words_io_inst, 0},
    {"OPEN", words_io_inst, 0},
    {"CLOSE", words_io_inst, 0},
    {"INQUIRE", words_io_inst, 0},
    {"BACKSPACE", words_io_inst, 0},
    {"REWIND", words_io_inst, 0},
    {"ENDFILE", words_io_inst, 0},
    {"IMPLIED-DO", words_implied_do, 0},

    {RETURN_FUNCTION_NAME, words_nullary_op, 0},
    {"PAUSE", words_nullary_op, 0},
    {"STOP", words_nullary_op, 0},
    {"CONTINUE", words_nullary_op, 0},
    {"END", words_nullary_op, 0},
    {FORMAT_FUNCTION_NAME, words_prefix_unary_op, 0},
    {UNBOUNDED_DIMENSION_NAME, words_unbounded_dimension, 0},
    {LIST_DIRECTED_FORMAT_NAME, words_list_directed, 0},

    {SUBSTRING_FUNCTION_NAME, words_substring_op, 0},
    {ASSIGN_SUBSTRING_FUNCTION_NAME, words_assign_substring_op, 0},

    {NULL, null, 0}
};

static list 
words_intrinsic_call(call obj, int precedence)
{
    struct intrinsic_handler *p = tab_intrinsic_handler;
    char *n = entity_local_name(call_function(obj));

    while (p->name != NULL) {
	if (strcmp(p->name, n) == 0) {
	    return((*(p->f))(obj, precedence));
	}
	p++;
    }

    return(words_regular_call(obj));
}

static int 
intrinsic_precedence(string n)
{
    struct intrinsic_handler *p = tab_intrinsic_handler;

    while (p->name != NULL) {
	if (strcmp(p->name, n) == 0) {
	    return(p->prec);
	}
	p++;
    }

    return(0);
}

static int
words_intrinsic_precedence(call obj)
{
    char *n = entity_local_name(call_function(obj));

    return(intrinsic_precedence(n));
}

/* exported for cmfortran.c
 */
list 
words_call(
    call obj,
    int precedence)
{
    list pc;

    entity f = call_function(obj);
    value i = entity_initial(f);
    
    pc = (value_intrinsic_p(i)) ? words_intrinsic_call(obj, precedence) : 
	                          words_regular_call(obj);

    return(pc);
}

/* exported for expression.c 
 */
list 
words_syntax(syntax obj)
{
    list pc;

    if (syntax_reference_p(obj)) {
	pc = words_reference(syntax_reference(obj));
    }
    else if (syntax_range_p(obj)) {
	pc = words_range(syntax_range(obj));
    }
    else if (syntax_call_p(obj)) {
	pc = words_call(syntax_call(obj), 0);
    }
    else {
	pips_error("words_syntax", "tag inconnu");
	pc = NIL;
    }

    return(pc);
}

/* this one is exported. */
list /* of string */
words_expression(expression obj)
{
    return words_syntax(expression_syntax(obj));
}

/* exported for cmfortran.c 
 */
list 
words_subexpression(
    expression obj,
    int precedence)
{
    list pc;
    
    if ( expression_call_p(obj) )
	pc = words_call(syntax_call(expression_syntax(obj)), precedence);
    else 
	pc = words_syntax(expression_syntax(obj));
    
    return pc;
}


/**************************************************************** SENTENCE */


/* We have no way to distinguish between the SUBROUTINE and PROGRAM
 * They two have almost the same properties.
 * For the time being, especially for the PUMA project, we have a temporary
 * idea to deal with it: When there's no argument(s), it should be a PROGRAM,
 * otherwise, it should be a SUBROUTINE. 
 * Lei ZHOU 18/10/91
 *
 * correct PROGRAM and SUBROUTINE distinction added, FC 18/08/94
 * approximate BLOCK DATA / SUBROUTINE distinction also added. FC 09/97
 */
static sentence 
sentence_head(entity e)
{
    list pc = NIL;
    type te = entity_type(e);
    functional fe;
    type tr;
    list args = words_parameters(e);

    pips_assert("is functionnal", type_functional_p(te));

    fe = type_functional(te);
    tr = functional_result(fe);
    
    if (type_void_p(tr)) 
    {
	if (entity_main_module_p(e))
	    pc = CHAIN_SWORD(pc, "PROGRAM ");
	else
	{
	    if (entity_blockdata_p(e))
		pc = CHAIN_SWORD(pc, "BLOCKDATA ");
	    else
		pc = CHAIN_SWORD(pc, "SUBROUTINE ");
	}
    }
    else if (type_variable_p(tr)) {
	pc = gen_nconc(pc, words_basic(variable_basic(type_variable(tr))));
	pc = CHAIN_SWORD(pc, " FUNCTION ");
    }
    else {
	pips_internal_error("unexpected type for result\n");
    }
    pc = CHAIN_SWORD(pc, module_local_name(e));

    if ( !ENDP(args) ) {
	pc = CHAIN_SWORD(pc, "(");
	pc = gen_nconc(pc, args);
	pc = CHAIN_SWORD(pc, ")");
    }
    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

sentence 
sentence_tail(void)
{
    return(MAKE_ONE_WORD_SENTENCE(0, "END"));
}

sentence 
sentence_variable(entity e)
{
    list pc = NIL;
    type te = entity_type(e);

    pips_assert("sentence_variable", type_variable_p(te));

    pc = gen_nconc(pc, words_basic(variable_basic(type_variable(te))));
    pc = CHAIN_SWORD(pc, " ");

    pc = gen_nconc(pc, words_declaration(e, TRUE));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

static bool
empty_static_area_p(entity e)
{
    if (!static_area_p(e)) return FALSE;
    return ENDP(area_layout(type_area(entity_type(e))));
}

/*  special management of empty commons added.
 *  this may happen in the hpfc generated code.
 */
static sentence 
sentence_area(entity e, entity module, bool pp_dimensions)
{
    string area_name = module_local_name(e);
    type te = entity_type(e);
    list pc = NIL, entities = NIL;

    if (dynamic_area_p(e)) /* shouldn't get in? */
	return sentence_undefined;

    assert(type_area_p(te));

    if (!ENDP(area_layout(type_area(te))))
    {
	bool pp_hpfc = get_bool_property("PRETTYPRINT_HPFC");
	MAP(ENTITY, ee,
	    if (local_entity_of_module_p(ee, module) || pp_hpfc)
	        entities = CONS(ENTITY, ee, entities),
	    area_layout(type_area(te)));

	/*  the common is not output if it is empty
	 */
	if (!ENDP(entities))
	{
	    bool comma = FALSE;
	    bool is_save = static_area_p(e);

	    if (is_save)
	    {
		pc = CHAIN_SWORD(pc, "SAVE ");
	    }
	    else
	    {
		pc = CHAIN_SWORD(pc, "COMMON ");
		if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) != 0) 
		{
		    pc = CHAIN_SWORD(pc, "/");
		    pc = CHAIN_SWORD(pc, area_name);
		    pc = CHAIN_SWORD(pc, "/ ");
		}
	    }
	    entities = gen_nreverse(entities);
	    
	    MAP(ENTITY, ee, 
	     {
		 if (comma) pc = CHAIN_SWORD(pc, ",");
		 else comma = TRUE;
		 /* hpfc: dimension of common variables are specified
		  * within the COMMON, not with the type. this is just
		  * a personnal taste. FC.
		  */
		 pc = gen_nconc(pc, 
			words_declaration(ee, !is_save && pp_dimensions));
	     },
		 entities);

	    gen_free_list(entities);
	}
    }

    return make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc));
}

sentence 
sentence_goto_label(
    entity module,
    string label,
    int margin,
    string tlabel,
    int n)
{
    list pc = words_goto_label(tlabel);

    return(make_sentence(is_sentence_unformatted, 
	    make_unformatted(label?strdup(label):NULL, n, margin, pc)));
}

static sentence 
sentence_goto(
    entity module,
    string label,
    int margin,
    statement obj,
    int n)
{
    string tlabel = entity_local_name(statement_label(obj)) + 
	           strlen(LABEL_PREFIX);
    return sentence_goto_label(module, label, margin, tlabel, n);
}

static sentence 
sentence_basic_declaration(entity e)
{
    list decl = NIL;
    basic b = entity_basic(e);

    pips_assert("b is defined", !basic_undefined_p(b));

    decl = CHAIN_SWORD(decl, basic_to_string(b));
    decl = CHAIN_SWORD(decl, " ");
    decl = CHAIN_SWORD(decl, entity_local_name(e));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, decl)));
}

static sentence 
sentence_external(entity f)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "EXTERNAL ");
    pc = CHAIN_SWORD(pc, entity_local_name(f));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

static sentence 
sentence_symbolic(entity f)
{
    list pc = NIL;
    value vf = entity_initial(f);
    expression e = symbolic_expression(value_symbolic(vf));

    pc = CHAIN_SWORD(pc, "PARAMETER (");
    pc = CHAIN_SWORD(pc, entity_local_name(f));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_expression(e));
    pc = CHAIN_SWORD(pc, ")");

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

/* why is it assumed that the constant is an int ??? 
 */
static sentence 
sentence_data(entity e)
{
    list pc = NIL;
    constant c;

    if (! value_constant_p(entity_initial(e)))
	return(sentence_undefined);

    c = value_constant(entity_initial(e));

    if (! constant_int_p(c))
	return(sentence_undefined);

    pc = CHAIN_SWORD(pc, "DATA ");
    pc = CHAIN_SWORD(pc, entity_local_name(e));
    pc = CHAIN_SWORD(pc, " /");
    pc = CHAIN_IWORD(pc, constant_int(c));
    pc = CHAIN_SWORD(pc, "/");

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}


/********************************************************************* TEXT */

#define ADD_WORD_LIST_TO_TEXT(t, l)\
  if (!ENDP(l)) ADD_SENTENCE_TO_TEXT(t,\
	        make_sentence(is_sentence_unformatted, \
			      make_unformatted(NULL, 0, 0, l)));

/* We add this function to cope with the declaration
 * When the user declare sth. there's no need to declare sth. for the user.
 * When nothing is declared ( especially there is no way to know whether it's 
 * a SUBROUTINE or PROGRAM). We will go over the entire module to find all the 
 * variables and declare them properly.
 * Lei ZHOU 18/10/91
 *
 * the float length is now tested to generate REAL*4 or REAL*8.
 * ??? something better could be done, printing "TYPE*%d".
 * the problem is that you cannot mix REAL*4 and REAL*8 in the same program
 * Fabien Coelho 12/08/93 and 15/09/93
 *
 * pf4 and pf8 distinction added, FC 26/10/93
 *
 * Is it really a good idea to print overloaded type variables~? FC 15/09/93
 * PARAMETERS added. FC 15/09/93
 *
 * typed PARAMETERs FC 13/05/94
 * EXTERNALS are missing: added FC 13/05/94
 *
 * Bug: parameters and their type should be put *before* other declarations
 *      since they may use them. Changed FC 08/06/94
 *
 * COMMONS are also missing:-) added, FC 19/08/94
 *
 * updated to fully control the list to be used.
 */
/* hook for commons, when not generated...
 */
static string default_common_hook(entity module, entity common)
{
    return strdup(concatenate
        ("common to include: ", entity_local_name(common), "\n", NULL));
}

static string (*common_hook)(entity, entity) = default_common_hook; 
void set_prettyprinter_common_hook(string(*f)(entity,entity)){ common_hook=f;}
void reset_prettyprinter_common_hook(){ common_hook=default_common_hook;}

/* debugging for equivalences */
#define EQUIV_DEBUG 8

static void 
equiv_class_debug(list l_equiv)
{
    if (ENDP(l_equiv))
	fprintf(stderr, "<none>");
    MAP(ENTITY, equiv_ent,
	{
	    fprintf(stderr, " %s", entity_local_name(equiv_ent));
	}, l_equiv);
    fprintf(stderr, "\n");
}


/* static int equivalent_entity_compare(entity *ent1, entity *ent2)
 * input    : two pointers on entities.
 * output   : an integer for qsort.
 * modifies : nothing.
 * comment  : this is a comparison function for qsort; the purpose
 *            being to order a list of equivalent variables.
 * algorithm: If two variables have the same offset, the longest 
 * one comes first; if they have the same lenght, use a lexicographic
 * ordering.
 * author: bc.
 */
static int
equivalent_entity_compare(entity *ent1, entity *ent2)
{
    int result;
    int offset1 = ram_offset(storage_ram(entity_storage(*ent1)));
    int offset2 = ram_offset(storage_ram(entity_storage(*ent2)));
    Value size1, size2;
    
    result = offset1 - offset2;

    /* pips_debug(1, "entities: %s %s\n", entity_local_name(*ent1),
	  entity_local_name(*ent2)); */
    
    if (result == 0)
    {
	/* pips_debug(1, "same offset\n"); */
	size1 = ValueSizeOfArray(*ent1);
	size2 = ValueSizeOfArray(*ent2);
	result = value_compare(size2,size1);
	
	if (result == 0)
	{
	    /* pips_debug(1, "same size\n"); */
	    result = strcmp(entity_local_name(*ent1), entity_local_name(*ent2));
	}
    }

    return(result);
}

/* static text text_equivalence_class(list  l_equiv)
 * input    : a list of entities representing an equivalence class.
 * output   : a text, which is the prettyprint of this class.
 * modifies : sorts l_equiv according to equivalent_entity_compare.
 * comment  : partially associated entities are not handled. 
 * author   : bc.
 */
static text
text_equivalence_class(list /* of entities */ l_equiv)
{
    text t_equiv = make_text(NIL);
    list lw = NIL;
    list l1, l2;
    entity ent1, ent2;
    int offset1, offset2;
    Value size1, size2, offset_end1;
    boolean first;

    if (ENDP(l_equiv)) return(t_equiv);

    /* FIRST, sort the list by increasing offset from the beginning of
       the memory suite. If two variables have the same offset, the longest 
       one comes first; if they have the same lenght, use a lexicographic
       ordering */
    ifdebug(EQUIV_DEBUG)
    {
	pips_debug(1, "equivalence class before sorting:\n");
	equiv_class_debug(l_equiv);
    }
    
    gen_sort_list(l_equiv,equivalent_entity_compare);
	
    ifdebug(EQUIV_DEBUG)
    {
	pips_debug(1, "equivalence class after sorting:\n");
	equiv_class_debug(l_equiv);
    }
    
    /* THEN, prettyprint the sorted list*/	
    pips_debug(EQUIV_DEBUG,"prettyprint of the sorted list\n");
	
    /* We are sure that there is at least one equivalence */
    lw = CHAIN_SWORD(lw, "EQUIVALENCE");

    /* At each step of the next loop, we consider two entities
     * from the equivalence class. l1 points on the first entity list,
     * and l2 on the second one. If l2 is associated with l1, we compute
     * the output string, and l2 becomes the next entity. If l2 is not
     * associated with l1, l1 becomes the next entity, until it is 
     * associated with l1. In the l_equiv list, l1 is always before l2.
     */
    
    /* loop initialization */
    l1 = l_equiv;
    ent1 = ENTITY(CAR(l1));
    offset1 = ram_offset(storage_ram(entity_storage(ent1)));
    size1 = ValueSizeOfArray(ent1);
    l2 = CDR(l_equiv);
    first = TRUE;
    /* */
    
    while(!ENDP(l2))
    {
	ent2 = ENTITY(CAR(l2));
	offset2 = ram_offset(storage_ram(entity_storage(ent2)));
	
	pips_debug(EQUIV_DEBUG, "dealing with: %s %s\n",
		   entity_local_name(ent1),
		   entity_local_name(ent2));
	
	/* If the two variables have the same offset, their
	 * first elements are equivalenced. 
	 */
	if (offset1 == offset2)
	{
	    pips_debug(EQUIV_DEBUG, "easiest case: offsets are the same\n");
	    if (! first)
		lw = CHAIN_SWORD(lw, ",");
	    else
		first = FALSE;
	    lw = CHAIN_SWORD(lw, " (");
	    lw = CHAIN_SWORD(lw, entity_local_name(ent1));
	    lw = CHAIN_SWORD(lw, ",");
	    lw = CHAIN_SWORD(lw, entity_local_name(ent2));
	    lw = CHAIN_SWORD(lw, ")");		
	    POP(l2);
	}
	/* Else, we first check that there is an overlap */
	else 
	{
	    pips_assert("the equivalence class has been sorted\n",
			offset1 < offset2);
	    
	    size2 = ValueSizeOfArray(ent2);		
	    offset_end1 = value_plus(offset1, size1);
	    
	    /* If there is no overlap, we change the reference variable */
	    if (value_le(offset_end1,offset2))
	    {
		pips_debug(1, "second case: there is no overlap\n");
		POP(l1);
		ent1 = ENTITY(CAR(l1));
		offset1 = ram_offset(storage_ram(entity_storage(ent1)));
		size1 = ValueSizeOfArray(ent1);	
		if (l1 == l2) POP(l2);
	    }
	    
	    /* Else, we must compute the coordinates of the element of ent1
	     * which corresponds to the first element of ent2
	     */
	    else
	    {
		/* ATTENTION: Je n'ai pas considere le cas 
		 * ou il y a association partielle. De ce fait, offset
		 * est divisiable par size_elt_1. */
		static char buffer[10];
		int offset = offset2 - offset1;
		int rest;
		int current_dim;    
		int dim_max = NumberOfDimension(ent1);		    
		int size_elt_1 = SizeOfElements(
		    variable_basic(type_variable(entity_type(ent1))));
		list l_tmp = variable_dimensions
		    (type_variable(entity_type(ent1)));
		normalized nlo;
		Pvecteur pvlo;
		    
		pips_debug(EQUIV_DEBUG, "third case\n");
		pips_debug(EQUIV_DEBUG, 
			   "offset=%d, dim_max=%d, size_elt_1=%d\n",
			   offset, dim_max,size_elt_1);
				
		if (! first)
		    lw = CHAIN_SWORD(lw, ",");
		else
		    first = FALSE;
		lw = CHAIN_SWORD(lw, " (");
		lw = CHAIN_SWORD(lw, entity_local_name(ent1));
		lw = CHAIN_SWORD(lw, "(");
		
		pips_assert("partial association case not implemented:\n"
			    "offset % size_elt_1 == 0",
			    (offset % size_elt_1) == 0);
		
		offset = offset/size_elt_1;
		current_dim = 1;
		
		while (current_dim <= dim_max)
		{
		    dimension dim = DIMENSION(CAR(l_tmp));
		    int new_decl;
		    int size;
		    
		    pips_debug(EQUIV_DEBUG, "prettyprinting dimension %d\n",
			       current_dim);
		    size = SizeOfIthDimension(ent1, current_dim);
		    rest = (offset % size);
		    offset = offset / size;
		    nlo = NORMALIZE_EXPRESSION(dimension_lower(dim));
		    pvlo = normalized_linear(nlo);
		    
		    pips_assert("sg", vect_constant_p(pvlo));			
		    pips_debug(EQUIV_DEBUG,
			       "size=%d, rest=%d, offset=%d, lower_bound=%d\n",
			       size, rest, offset, VALUE_TO_INT(val_of(pvlo)));
		    
		    new_decl = VALUE_TO_INT(val_of(pvlo)) + rest;
		    buffer[0] = '\0';
		    sprintf(buffer+strlen(buffer), "%d", new_decl);		 
		    lw = CHAIN_SWORD(lw,strdup(buffer));			
		    if (current_dim < dim_max)
			lw = CHAIN_SWORD(lw, ",");
		    
		    POP(l_tmp);
		    current_dim++;
		    
		} /* while */
		
		lw = CHAIN_SWORD(lw, ")");	
		lw = CHAIN_SWORD(lw, ",");
		lw = CHAIN_SWORD(lw, entity_local_name(ent2));
		lw = CHAIN_SWORD(lw, ")");	
		POP(l2);
	    } /* if-else: there is an overlap */
	} /* if-else: not same offset */
    } /* while */
    ADD_WORD_LIST_TO_TEXT(t_equiv, lw);
    
    pips_debug(EQUIV_DEBUG, "end\n");
    return t_equiv;
}


/* text text_equivalences(entity module, list ldecl)
 * input    : the current module, and the list of declarations.
 * output   : a text for all the equivalences.
 * modifies : nothing
 * comment  :
 */
static text 
text_equivalences(entity module, list ldecl)
{
    list equiv_classes = NIL, l_tmp;
    text t_equiv_class;

    pips_debug(1,"begin\n");

    /* FIRST BUILD EQUIVALENCE CLASSES */

    pips_debug(EQUIV_DEBUG, "loop on declarations\n");
    /* consider each entity in the declaration */
    MAP(ENTITY, e,
    {
	/* but only variables which have a ram storage must be considered
	 */
	if (type_variable_p(entity_type(e)) && 
	    storage_ram_p(entity_storage(e)))
	{
	    list l_shared = ram_shared(storage_ram(entity_storage(e)));
	    
	    ifdebug(EQUIV_DEBUG)
	    {
		pips_debug(1, "considering entity: %s\n", 
			   entity_local_name(e));
		pips_debug(1, "shared variables:\n");
		equiv_class_debug(l_shared);
	    }
	    
	    /* If this variable is statically aliased */
	    if (!ENDP(l_shared))
	    {
		bool found = FALSE;
		list found_equiv_class = NIL;
		
		/* We first look in already found equivalence classes
		 * if there is already a class in which one of the
		 * aliased variables appears 
		 */
		MAP(LIST, equiv_class,
		{
		    ifdebug(EQUIV_DEBUG)
		    {
			pips_debug(1, "considering equivalence class:\n");
			equiv_class_debug(equiv_class);
		    }
		    
		    MAP(ENTITY, ent,
		    {
			if (variable_in_list_p(ent, equiv_class))
			{
			    found = TRUE;
			    found_equiv_class = equiv_class;
			    break;
			}
		    }, l_shared);
		    
		    if (found) break;			    
		}, equiv_classes);
		
		if (found)
		{
		    pips_debug(EQUIV_DEBUG, "already there\n");
		    /* add the entities of shared which are not already in 
		     * the existing equivalence class. Useful ??
		     */
		    MAP(ENTITY, ent,
		    {
			if(!variable_in_list_p(ent, found_equiv_class))
			    found_equiv_class =
				gen_nconc(found_equiv_class,
					  CONS(ENTITY, ent, NIL));
		    }, l_shared)
		}
		else
		{
		    list l_tmp = NIL;
		    pips_debug(EQUIV_DEBUG, "not found\n");
		    /* add the list of variables in l_shared; necessary 
		     * because variables may appear several times in 
		     * l_shared. */
		    MAP(ENTITY, shared_ent,
		    {
			if (!variable_in_list_p(shared_ent, l_tmp))
			    l_tmp = gen_nconc(l_tmp,
					      CONS(ENTITY, shared_ent,
						   NIL));
		    }, 
			l_shared);
		    equiv_classes =
			gen_nconc(equiv_classes, CONS(LIST, l_tmp, NIL));
		}
	    }
	}
    }, ldecl);
    
    ifdebug(EQUIV_DEBUG)
    {
	pips_debug(1, "final equivalence classes:\n");
	MAP(LIST, equiv_class,
	{
	    equiv_class_debug(equiv_class);
	},
	    equiv_classes);	
    }

    /* SECOND, PRETTYPRINT THEM */
    t_equiv_class = make_text(NIL); 
    MAP(LIST, equiv_class,
    {
	MERGE_TEXTS(t_equiv_class, text_equivalence_class(equiv_class));
    }, equiv_classes);
    
    /* AND FREE THEM */    
    for(l_tmp = equiv_classes; !ENDP(l_tmp); POP(l_tmp))
    {
	list equiv_class = LIST(CAR(l_tmp));
	gen_free_list(equiv_class);
	LIST(CAR(l_tmp)) = NIL;
    }
    gen_free_list(equiv_classes);
    
    /* THE END */
    pips_debug(EQUIV_DEBUG, "end\n");
    return(t_equiv_class);
}

/* returns the DATA initializations.
 * limited to integers, because I do not know where is the value
 * for other types...
 */
static text 
text_data(entity module, list /* of entity */ ldecl)
{
    list /* of sentence */ ls = NIL;

    MAP(ENTITY, e,
    {
	value v = entity_initial(e);
	if(!value_undefined_p(v) && 
	   value_constant_p(v) && constant_int_p(value_constant(v)))
	    ls = CONS(SENTENCE, sentence_data(e), ls);
    },
	ldecl);

    return make_text(ls);
}

static text 
text_entity_declaration(entity module, list ldecl)
{
    bool print_commons = get_bool_property("PRETTYPRINT_COMMONS");
    list before = NIL, area_decl = NIL, ph = NIL,
	pi = NIL, pf4 = NIL, pf8 = NIL, pl = NIL, 
	pc8 = NIL, pc16 = NIL, ps = NIL;
    text r, t_chars = make_text(NIL); 
    string pp_var_dim = get_string_property("PRETTYPRINT_VARIABLE_DIMENSIONS");
    bool pp_in_type = FALSE, pp_in_common = FALSE;
     
    /* where to put the dimensionn information.
     */
    if (same_string_p(pp_var_dim, "type"))
	pp_in_type = TRUE, pp_in_common = FALSE;
    else if (same_string_p(pp_var_dim, "common"))
	pp_in_type = FALSE, pp_in_common = TRUE;
    else 
	pips_internal_error("PRETTYPRINT_VARIABLE_DIMENSIONS=\"%s\""
			    " unexpected value\n", pp_var_dim);

    MAP(ENTITY, e,
    {
	type te = entity_type(e);
	bool func = 
	    type_functional_p(te) && storage_rom_p(entity_storage(e));
	value v = entity_initial(e);
	bool param = func && value_symbolic_p(v);
	bool external =     /* subroutines won't be declared */
	    (func && 
	     (value_code_p(v) || value_unknown_p(v) /* not parsed callee */) &&
	     !type_void_p(functional_result(type_functional(te))));
	bool area_p = type_area_p(te);
	bool var = type_variable_p(te);
	bool in_ram = storage_ram_p(entity_storage(e));
	
	pips_debug(3, "entity name is %s\n", entity_name(e));

	if (!print_commons && area_p && !SPECIAL_COMMON_P(e))
	{
	    area_decl = 
		CONS(SENTENCE, make_sentence(is_sentence_formatted,
					     common_hook(module, e)),
		     area_decl);
	}
	
	if (!print_commons && 
	    (area_p || (var && in_ram && 
	  !SPECIAL_COMMON_P(ram_section(storage_ram(entity_storage(e)))))))
	{
	    pips_debug(5, "skipping entity %s\n", entity_name(e));
	}
	else if (param || external)
	{
	    before = CONS(SENTENCE, sentence_basic_declaration(e), before);
	    if (param) {
		/*        PARAMETER
		 */
		pips_debug(7, "considered as a parameter\n");
		before = CONS(SENTENCE, sentence_symbolic(e), before);
	    } else {
		/*        EXTERNAL
		 */
		pips_debug(7, "considered as an external\n");
		before = CONS(SENTENCE, sentence_external(e), before);
	    }
	 }
	else if (area_p && !dynamic_area_p(e) && !empty_static_area_p(e))
	{
	    /*            AREAS: COMMONS and SAVEs
	     */	     
	    pips_debug(7, "considered as a regular common\n");
	    area_decl = CONS(SENTENCE, sentence_area(e, module, pp_in_common), 
			     area_decl);
	}
	else if (var)
	{
	    basic b = variable_basic(type_variable(te));
	    bool pp_dim = pp_in_type || variable_static_p(e);

	    pips_debug(7, "is a variable...\n");
	    
	    switch (basic_tag(b)) 
	    {
	    case is_basic_int:
		 /* simple integers are moved ahead...
		  */
		pips_debug(7, "is an integer\n");
		if (variable_dimensions(type_variable(te)))
		{
		    pi = CHAIN_SWORD(pi, pi==NIL ? "INTEGER " : ",");
		    pi = gen_nconc(pi, words_declaration(e, pp_dim)); 
		}
		else
		{
		    ph = CHAIN_SWORD(ph, ph==NIL ? "INTEGER " : ",");
		    ph = gen_nconc(ph, words_declaration(e, pp_dim)); 
		}
		break;
	    case is_basic_float:
		pips_debug(7, "is a float\n");
		switch (basic_float(b))
		{
		case 4:
		    pf4 = CHAIN_SWORD(pf4, pf4==NIL ? "REAL*4 " : ",");
		    pf4 = gen_nconc(pf4, words_declaration(e, pp_dim));
		    break;
		case 8:
		default:
		    pf8 = CHAIN_SWORD(pf8, pf8==NIL ? "REAL*8 " : ",");
		    pf8 = gen_nconc(pf8, words_declaration(e, pp_dim));
		    break;
		}
		break;			
	    case is_basic_complex:
		pips_debug(7, "is a complex\n");
		switch (basic_complex(b))
		{
		case 8:
		    pc8 = CHAIN_SWORD(pc8, pc8==NIL ? "COMPLEX*8 " : ",");
		    pc8 = gen_nconc(pc8, words_declaration(e, pp_dim));
		    break;
		case 16:
		default:
		    pc16 = CHAIN_SWORD(pc16, pc16==NIL ? "COMPLEX*16 " : ",");
		    pc16 = gen_nconc(pc16, words_declaration(e, pp_dim));
		    break;
		}
		break;
	    case is_basic_logical:
		pips_debug(7, "is a logical\n");
		pl = CHAIN_SWORD(pl, pl==NIL ? "LOGICAL " : ",");
		pl = gen_nconc(pl, words_declaration(e, pp_dim));
		break;
	    case is_basic_overloaded:
		/* nothing! some in hpfc I guess...
		 */
		break; 
	    case is_basic_string:
	    {
		value v = basic_string(b);
		pips_debug(7, "is a string\n");
		
		if (value_constant_p(v) && constant_int_p(value_constant(v)))
		{
		    int i = constant_int(value_constant(v));
		    
		    if (i==1)
		    {
			ps = CHAIN_SWORD(ps, ps==NIL ? "CHARACTER " : ",");
			ps = gen_nconc(ps, words_declaration(e, pp_dim));
		    }
		    else
		    {
			list chars=NIL;
			chars = CHAIN_SWORD(chars, "CHARACTER*");
			chars = CHAIN_IWORD(chars, i);
			chars = CHAIN_SWORD(chars, " ");
			chars = gen_nconc(chars, 
					  words_declaration(e, pp_dim));
			attach_declaration_size_type_to_words
			    (chars, "CHARACTER", i);
			ADD_WORD_LIST_TO_TEXT(t_chars, chars);
		    }
		}
		else if (value_unknown_p(v))
		{
		    list chars=NIL;
		    chars = CHAIN_SWORD(chars, "CHARACTER*(*) ");
		    chars = gen_nconc(chars, 
				      words_declaration(e, pp_dim));
		    attach_declaration_type_to_words
			(chars, "CHARACTER*(*)");
		    ADD_WORD_LIST_TO_TEXT(t_chars, chars);
		}
		else
		    pips_internal_error("unexpected value\n");
		break;
	    }
	    default:
		pips_internal_error("unexpected basic tag (%d)\n",
				    basic_tag(b));
	    }
	}
    }, ldecl);
    
    /* parameters must be kept in order
     * because that may depend one from the other, hence the reversion.
     */ 
    r = make_text(gen_nreverse(before));

    ADD_WORD_LIST_TO_TEXT(r, ph);
    attach_declaration_type_to_words(ph, "INTEGER");
    ADD_WORD_LIST_TO_TEXT(r, pi);
    attach_declaration_type_to_words(pi, "INTEGER");
    ADD_WORD_LIST_TO_TEXT(r, pf4);
    attach_declaration_type_to_words(pf4, "REAL*4");
    ADD_WORD_LIST_TO_TEXT(r, pf8);
    attach_declaration_type_to_words(pf8, "REAL*8");
    ADD_WORD_LIST_TO_TEXT(r, pl);
    attach_declaration_type_to_words(pl, "LOGICAL");
    ADD_WORD_LIST_TO_TEXT(r, pc8);
    attach_declaration_type_to_words(pc8, "COMPLEX*8");
    ADD_WORD_LIST_TO_TEXT(r, pc16);
    attach_declaration_type_to_words(pc16, "COMPLEX*16");
    ADD_WORD_LIST_TO_TEXT(r, ps);
    attach_declaration_type_to_words(ps, "CHARACTER");
    MERGE_TEXTS(r, t_chars);

    /* all about COMMON and SAVE declarations */
    MERGE_TEXTS(r, make_text(area_decl));

    /* and EQUIVALENCE statements... - BC -*/
    MERGE_TEXTS(r, text_equivalences(module, ldecl)); 

    /* what about DATA statements! FC */
    MERGE_TEXTS(r, text_data(module, ldecl));

    return r;
}

/* exported for hpfc.
 */
text 
text_declaration(entity module)
{
    return text_entity_declaration
	(module, code_declarations(entity_code(module)));
}

/* needed for hpfc 
 */
text 
text_common_declaration(common, module)
entity common, module;
{
    type t = entity_type(common);
    list ldecl;
    text result;

    pips_assert("indeed a common", type_area_p(t));

    ldecl = CONS(ENTITY, common, gen_copy_seq(area_layout(type_area(t))));
    result = text_entity_declaration(module, ldecl);

    gen_free_list(ldecl);
    return result;
}

static text 
text_block(
    entity module,
    string label,
    int margin,
    list objs,
    int n)
{
    text r = make_text(NIL);
    list pbeg, pend ;

    pend = NIL;

    if (ENDP(objs) && !get_bool_property("PRETTYPRINT_EMPTY_BLOCKS")) {
	return(r) ;
    }

    pips_assert("text_block", strcmp(label, "") == 0) ;
    
    if (get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
	get_bool_property("PRETTYPRINT_BLOCKS")) {
	unformatted u;
	
	if (get_bool_property("PRETTYPRINT_FOR_FORESYS")){
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  strdup("C$BB\n")));
	}
	else {
	    pbeg = CHAIN_SWORD(NIL, "BEGIN BLOCK");
	    pend = CHAIN_SWORD(NIL, "END BLOCK");
	    
	    u = make_unformatted(strdup("C"), n, margin, pbeg);
	    ADD_SENTENCE_TO_TEXT(r, 
				 make_sentence(is_sentence_unformatted, u));
	}
    }

    for (; objs != NIL; objs = CDR(objs)) {
	statement s = STATEMENT(CAR(objs));

	text t = text_statement(module, margin, s);
	text_sentences(r) = 
	    gen_nconc(text_sentences(r), text_sentences(t));
	text_sentences(t) = NIL;
	gen_free(t);
    }

    if (!get_bool_property("PRETTYPRINT_FOR_FORESYS") &&
			   (get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
			    get_bool_property("PRETTYPRINT_BLOCKS"))) {
	unformatted u;

	u = make_unformatted(strdup("C"), n, margin, pend);

	ADD_SENTENCE_TO_TEXT(r, 
			     make_sentence(is_sentence_unformatted, u));
    }
    return(r) ;
}

/* private variables.
 * modified 2-8-94 by BA.
 * extracted as a function and cleaned a *lot*, FC.
 */
static list /* of string */
loop_private_variables(loop obj)
{
    bool
	all_private = get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES"),
	hpf_private = get_bool_property("PRETTYPRINT_HPF"),
	some_before = FALSE;
    list l = NIL;

    /* comma-separated list of private variables. 
     * built in reverse order to avoid adding at the end...
     */
    MAP(ENTITY, p,
    {
	if((p!=loop_index(obj)) || all_private) 
	{
	    if (some_before) 
		l = CHAIN_SWORD(l, ",");
	    else
		some_before = TRUE; /* from now on commas, triggered... */

	    l = gen_nconc(l, words_declaration(p,TRUE));
	}
    }, 
	loop_locals(obj)) ; /* end of MAP */
    
    pips_debug(5, "#printed %d/%d\n", gen_length(l), 
	       gen_length(loop_locals(obj)));

    /* stuff around if not empty
     */
    if (l)
    {
	l = CONS(STRING, MAKE_SWORD(hpf_private ? "NEW(" : "PRIVATE "), l);
	if (hpf_private) CHAIN_SWORD(l, ")");
    }

    return l;
}

/* returns a formatted text for the HPF independent and new directive 
 * well, no continuations and so, but the directives do not fit the 
 * unformatted domain, because the directive prolog would not be well
 * managed there.
 */
#define HPF_DIRECTIVE "!HPF$ "

static text 
text_hpf_directive(
    loop obj,   /* the loop we're interested in */
    int margin) /* margin */
{
    list /* of string */ l = NIL, ln = NIL,
         /* of sentence */ ls = NIL;
    
    if (execution_parallel_p(loop_execution(obj)))
    {
	l = loop_private_variables(obj);
	ln = CHAIN_SWORD(ln, "INDEPENDENT");
	if (l) ln = CHAIN_SWORD(ln, ", ");
    }
    else if (get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES"))
	l = loop_private_variables(obj);
    
    ln = gen_nconc(ln, l);
    
    /* ??? directly put as formatted, doesn't matter?
     */
    if (ln) 
    {
	ls = CONS(SENTENCE, 
		  make_sentence(is_sentence_formatted, strdup("\n")), NIL);
	
	ln = gen_nreverse(ln);

	MAPL(ps, 
	{
	    ls = CONS(SENTENCE,
		make_sentence(is_sentence_formatted, STRING(CAR(ps))), ls);
	},
	     ln);
	
	for (; margin>0; margin--) /* margin managed by hand:-) */
	    ls = CONS(SENTENCE, make_sentence(is_sentence_formatted, 
					    strdup(" ")), ls);

	ls = CONS(SENTENCE, make_sentence(is_sentence_formatted, 
					  strdup(HPF_DIRECTIVE)), ls);

	gen_free_list(ln);
    }

    return make_text(ls);
}

text 
text_loop(
    entity module,
    string label,
    int margin,
    loop obj,
    int n)
{
    list pc = NIL;
    sentence first_sentence;
    unformatted u;
    text r = make_text(NIL);
    statement body = loop_body( obj ) ;
    entity the_label = loop_label(obj);
    string do_label = entity_local_name(the_label)+strlen(LABEL_PREFIX) ;
    bool structured_do = empty_local_label_name_p(do_label),
         doall_loop_p = FALSE,
         hpf_prettyprint = get_bool_property("PRETTYPRINT_HPF"),
         do_enddo_p = get_bool_property("PRETTYPRINT_DO_LABEL_AS_COMMENT"),
         all_private =  get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES");

    /* small hack to show the initial label of the loop to name it...
     */
    if(!structured_do && do_enddo_p)
    {
	ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
	  strdup(concatenate("!     INITIALLY: DO ", do_label, "\n", NULL))));
    }

    /* quite ugly management of other prettyprints...
     */
    switch(execution_tag(loop_execution(obj)) ) {
    case is_execution_sequential:
	doall_loop_p = FALSE;
	break ;
    case is_execution_parallel:
        if (get_bool_property("PRETTYPRINT_CMFORTRAN")) {
          text aux_r;
          if((aux_r = text_loop_cmf(module, label, margin, obj, n, NIL, NIL))
             != text_undefined) {
            MERGE_TEXTS(r, aux_r);
            return(r) ;
          }
        }
        if (get_bool_property("PRETTYPRINT_CRAFT")) {
          text aux_r;
          if((aux_r = text_loop_craft(module, label, margin, obj, n, NIL, NIL))
             != text_undefined) {
            MERGE_TEXTS(r, aux_r);
            return(r);
          }
        }
	if (get_bool_property("PRETTYPRINT_FORTRAN90") && 
	    instruction_assign_p(statement_instruction(body)) ) {
	    MERGE_TEXTS(r, text_loop_90(module, label, margin, obj, n));
	    return(r) ;
	}
	doall_loop_p = !get_bool_property("PRETTYPRINT_CRAY") &&
	    !get_bool_property("PRETTYPRINT_CMFORTRAN") &&
		!get_bool_property("PRETTYPRINT_CRAFT") && !hpf_prettyprint;
	break ;
    default:
	pips_error("text_loop", "Unknown tag\n") ;
    }

    /* HPF directives before the loop if required (INDEPENDENT and NEW)
     */
    if (hpf_prettyprint)
	MERGE_TEXTS(r, text_hpf_directive(obj, margin));

    /* LOOP prologue.
     */
    pc = CHAIN_SWORD(NIL, (doall_loop_p) ? "DOALL " : "DO " );
    
    if(!structured_do && !doall_loop_p && !do_enddo_p) {
	pc = CHAIN_SWORD(pc, concatenate(do_label, " ", NULL));
    }
    pc = CHAIN_SWORD(pc, entity_local_name(loop_index(obj)));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_loop_range(loop_range(obj)));
    u = make_unformatted(strdup(label), n, margin, pc) ;
    ADD_SENTENCE_TO_TEXT(r, first_sentence = 
			 make_sentence(is_sentence_unformatted, u));

    /* builds the PRIVATE scalar declaration if required
     */
    if(!ENDP(loop_locals(obj)) && (doall_loop_p || all_private)
       && !hpf_prettyprint) 
    {
	list /* of string */ lp = loop_private_variables(obj);

	if (lp) 
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
	        make_unformatted(NULL, 0, margin+INDENTATION, lp)));
    }

    /* loop BODY
     */
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, body));

    /* LOOP postlogue
     */
    if(structured_do || doall_loop_p || do_enddo_p ||
       get_bool_property("PRETTYPRINT_CRAY") ||
       get_bool_property("PRETTYPRINT_CRAFT") ||
       get_bool_property("PRETTYPRINT_CMFORTRAN"))
    {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDDO"));
    }

    attach_loop_to_sentence_up_to_end_of_text(first_sentence, r, obj);
    return r;
}

/* exported for unstructured.c 
 */
text 
init_text_statement(
    entity module,
    int margin,
    statement obj)
{
    instruction i = statement_instruction(obj);
    text r = make_text( NIL ) ;
    /* string comments = statement_comments(obj); */

    if (get_bool_property("PRETTYPRINT_ALL_EFFECTS")
	|| !((instruction_block_p(i) && 
	      !get_bool_property("PRETTYPRINT_BLOCKS")) || 
	     (instruction_unstructured_p(i) && 
	      !get_bool_property("PRETTYPRINT_UNSTRUCTURED")))) {
      /* FI: before calling the hook,
       * statement_ordering(obj) should be checked */
	r = (*text_statement_hook)( module, margin, obj );
	
	if (text_statement_hook != empty_text)
	    attach_decoration_to_text(r);
    }

    /*
    if (! string_undefined_p(comments)) {
	ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
					      comments));
    }
    */
    if (get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
	get_bool_property("PRETTYPRINT_STATEMENT_ORDERING")) {
	static char buffer[ 256 ] ;
	int so = statement_ordering(obj) ;

	if (!(instruction_block_p(statement_instruction(obj)) && 
	      (! get_bool_property("PRETTYPRINT_BLOCKS")))) {

	    if (so != STATEMENT_ORDERING_UNDEFINED) {
		sprintf(buffer, "C (%d,%d)\n", 
			ORDERING_NUMBER(so), ORDERING_STATEMENT(so)) ;
		ADD_SENTENCE_TO_TEXT(r, 
				     make_sentence(is_sentence_formatted, 
						   strdup(buffer))) ;
	    }
	    else {
		if(user_view_p())
	      ADD_SENTENCE_TO_TEXT(r, 
				   make_sentence(is_sentence_formatted, 
						   strdup("C (unreachable)\n")));
	    }
	}
    }
    return( r ) ;
}

static text 
text_logical_if(
    entity module,
    string label,
    int margin,
    test obj,
    int n)
{
    text r = make_text(NIL);
    list pc = NIL;
    statement tb = test_true(obj);
    instruction ti = statement_instruction(tb);
    call c = instruction_call(ti);

    pc = CHAIN_SWORD(pc, "IF (");
    pc = gen_nconc(pc, words_expression(test_condition(obj)));
    pc = CHAIN_SWORD(pc, ") ");
    pc = gen_nconc(pc, words_call(c, 0));

    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_unformatted, 
				       make_unformatted(strdup(label), n, 
							margin, pc)));

    return(r);
}

static text 
text_block_if(
    entity module,
    string label,
    int margin,
    test obj,
    int n)
{
    text r = make_text(NIL);
    list pc = NIL;
    statement test_false_obj;

    pc = CHAIN_SWORD(pc, "IF (");
    pc = gen_nconc(pc, words_expression(test_condition(obj)));
    pc = CHAIN_SWORD(pc, ") THEN");

    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_unformatted, 
				       make_unformatted(strdup(label), n, 
							margin, pc)));
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, 
				  test_true(obj)));

    test_false_obj = test_false(obj);
    if(statement_undefined_p(test_false_obj)){
	pips_error("text_test","undefined statement\n");
    }
    if (!statement_with_empty_comment_p(test_false_obj)
	||
	(!empty_statement_p(test_false_obj)
	 && !statement_continue_p(test_false_obj))
	||
	(empty_statement_p(test_false_obj)
	 && (get_bool_property("PRETTYPRINT_EMPTY_BLOCKS")))
	||
	(statement_continue_p(test_false_obj)
	 && (get_bool_property("PRETTYPRINT_ALL_LABELS")))) {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, 
				      test_false_obj));
    }

    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));

    return(r);
}

static text 
text_block_ifthen(
    entity module,
    string label,
    int margin,
    test obj,
    int n)
{
    text r = make_text(NIL);
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "IF (");
    pc = gen_nconc(pc, words_expression(test_condition(obj)));
    pc = CHAIN_SWORD(pc, ") THEN");

    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_unformatted, 
				       make_unformatted(strdup(label), n, 
							margin, pc)));
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, 
				  test_true(obj)));

    return(r);
}

static text 
text_block_else(
    entity module,
    string label,
    int margin,
    statement stmt,
    int n)
{    
    text r = make_text(NIL);

    if (!statement_with_empty_comment_p(stmt)
	||
	(!empty_statement_p(stmt)
	 && !statement_continue_p(stmt))
	||
	(empty_statement_p(stmt)
	 && (get_bool_property("PRETTYPRINT_EMPTY_BLOCKS")))
	||
	(statement_continue_p(stmt)
	 && (get_bool_property("PRETTYPRINT_ALL_LABELS")))) {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, stmt));
    }

    return r;
}

static text 
text_block_elseif(
    entity module,
    string label,
    int margin,
    test obj,
    int n)
{
    text r = make_text(NIL);
    list pc = NIL;
    statement tb = test_true(obj);
    statement fb = test_false(obj);

    /*
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, 
				  test_true(obj)));
				  */

    pc = CHAIN_SWORD(pc, "ELSEIF (");
    pc = gen_nconc(pc, words_expression(test_condition(obj)));
    pc = CHAIN_SWORD(pc, ") THEN");
    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_unformatted, 
				       make_unformatted(strdup(label), n, 
							margin, pc)));

    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, tb));

    if(statement_test_p(fb)) {
	MERGE_TEXTS(r, text_block_elseif(module, label, margin, 
					 statement_test(fb), n));
    }
    else {
	MERGE_TEXTS(r, text_block_else(module, label, margin, fb, n));
    }

    return(r);
}

static text 
text_test(
    entity module,
    string label,
    int margin,
    test obj,
    int n)
{
    text r = text_undefined;
    statement tb = test_true(obj);
    statement fb = test_false(obj);

    /* 1st case: one statement in the true branch => logical IF */
    if(nop_statement_p(fb)
       && statement_call_p(tb)
       && entity_empty_label_p(statement_label(tb))
       && empty_comments_p(statement_comments(tb))
       && !statement_continue_p(tb)
       && !get_bool_property("PRETTYPRINT_BLOCK_IF_ONLY")) {
	r = text_logical_if(module, label, margin, obj, n);
    }
    /* 2nd case: one test in the false branch => ELSEIF block */
    else if(statement_test_p(fb)
	    && empty_comments_p(statement_comments(fb))
	    && entity_empty_label_p(statement_label(fb))
	    && !get_bool_property("PRETTYPRINT_BLOCK_IF_ONLY")) {
	r = text_block_ifthen(module, label, margin, obj, n);
	MERGE_TEXTS(r, text_block_elseif
		    (module, label, margin, statement_test(fb), n));
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));

	/* r = text_block_if(module, label, margin, obj, n); */
    }
    else {
	r = text_block_if(module, label, margin, obj, n);
    }

    return r;
}

/* hook for adding something in the head. used by hpfc.
 * done so to avoid hpfc->prettyprint dependence in the libs. 
 * FC. 29/12/95.
 */
static string (*head_hook)(entity) = NULL;
void set_prettyprinter_head_hook(string(*f)(entity)){ head_hook=f;}
void reset_prettyprinter_head_hook(){ head_hook=NULL;}

static text 
text_instruction(
    entity module,
    string label,
    int margin,
    instruction obj,
    int n)
{
    text r=text_undefined, text_block(), text_unstructured() ;

    if (instruction_block_p(obj)) {
	r = text_block(module, label, margin, instruction_block(obj), n) ;
    }
    else if (instruction_test_p(obj)) {
	r = text_test(module, label, margin, instruction_test(obj), n);
    }
    else if (instruction_loop_p(obj)) {
	r = text_loop(module, label, margin, instruction_loop(obj), n);
    }
    else if (instruction_goto_p(obj)) {
	r = make_text(CONS(SENTENCE, 
			   sentence_goto(module, label, margin,
					 instruction_goto(obj), n), 
			   NIL));
    }
    else if (instruction_call_p(obj)) {
	unformatted u;
	sentence s;

	if (instruction_continue_p(obj) &&
	    empty_local_label_name_p(label) &&
	    !get_bool_property("PRETTYPRINT_ALL_LABELS")) {
	    debug(5, "text_instruction", "useless CONTINUE not printed\n");
	    r = make_text(NIL);
	}
	else {
	    u = make_unformatted(strdup(label), n, margin, 
				 words_call(instruction_call(obj), 0));

	    s = make_sentence(is_sentence_unformatted, u);

	    r = make_text(CONS(SENTENCE, s, NIL));
	}
    }
    else if (instruction_unstructured_p(obj)) {
	r = text_unstructured(module, label, margin, 
			      instruction_unstructured(obj), n) ;
    }
    else {
	pips_error("text_instruction", "unexpected tag");
    }

    return(r);
}

/* Handles all statements but tests that are nodes of an unstructured.
 * Those are handled by text_control.
 */
text 
text_statement(
    entity module,
    int margin,
    statement stmt)
{
    instruction i = statement_instruction(stmt);
    text r= make_text(NIL);
    text temp;
    string label = 
	    entity_local_name(statement_label(stmt)) + strlen(LABEL_PREFIX);
    string comments = statement_comments(stmt);

    pips_debug(2, "Begin for statement %s\n", statement_identification(stmt));

    if(statement_number(stmt)!=STATEMENT_NUMBER_UNDEFINED &&
       statement_ordering(stmt)==STATEMENT_ORDERING_UNDEFINED) {
      /* we are in trouble with some kind of dead (?) code... */
      pips_user_warning("I unexpectedly bumped into dead code?\n");
    }

    if (strcmp(label, RETURN_LABEL_NAME) == 0) {
	/* do not add a redundant RETURN before an END, unless required */
	if(get_bool_property("PRETTYPRINT_FINAL_RETURN")
	    || !last_statement_p(stmt)) {
	    /*
	    ADD_SENTENCE_TO_TEXT(temp,
				 MAKE_ONE_WORD_SENTENCE(margin,
							RETURN_FUNCTION_NAME));
	    */
	    sentence s = MAKE_ONE_WORD_SENTENCE(margin,
						RETURN_FUNCTION_NAME);
	    temp = make_text(CONS(SENTENCE, s ,NIL));
	}
	else {
	    temp = make_text(NIL);
	}
    }
    else {
	temp = text_instruction(module, label, margin, i,
				statement_number(stmt)) ;
    }

    if(!ENDP(text_sentences(temp))) {
	MERGE_TEXTS(r, init_text_statement(module, margin, stmt)) ;
	if (! string_undefined_p(comments)) {
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  comments));
	}
	MERGE_TEXTS(r, temp);
    }
    else {
	/* Preserve comments */
	if (! string_undefined_p(comments)) {
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  comments));
	}
    }

    attach_statement_information_to_text(r, stmt);

    ifdebug(1) if (instruction_sequence_p(i)) {
	pips_assert("This statement should be labelless, numberless"
		    " and commentless.",
		    statement_with_empty_comment_p(stmt)
		    && statement_number(stmt) == STATEMENT_NUMBER_UNDEFINED
		    && unlabelled_statement_p(stmt));
    }

    pips_debug(2, "End for statement %s\n", statement_identification(stmt));
       
    return(r);
}

/* Keep track of the last statement to decide if a final return can be omitted
 * or not. If no last statement can be found for sure, for instance because it
 * depends on the prettyprinter, last_statement is set to statement_undefined
 * which is safe.
 */
static statement last_statement = statement_undefined;

statement
find_last_statement(statement s)
{
    statement last = statement_undefined;

    pips_assert("statement is defined\n", !statement_undefined_p(s));

    if(block_statement_p(s)) {
	list ls = instruction_block(statement_instruction(s));

	last = (ENDP(ls)? statement_undefined : STATEMENT(CAR(gen_last(ls))));
    }
    else if(unstructured_statement_p(s)) {
	unstructured u = instruction_unstructured(statement_instruction(s));
	list trail = unstructured_to_trail(u);

	last = control_statement(CONTROL(CAR(trail)));

	gen_free_list(trail);
    }
    else if(statement_call_p(s)) {
	/* Hopefully it is a return statement.
	 * Since the semantics of STOP is ignored by the parser, a
	 * final STOp should be followed by a RETURN.
	 */
	last = s;
    }
    else {
	/* loop or test cannot be last statements of a module */
	last = statement_undefined;
    }

    /* recursive call */
    if(!statement_undefined_p(last)
       && (block_statement_p(last) || unstructured_statement_p(last))) {
	last = find_last_statement(last);
    }

    /* I had a lot of trouble writing the condition for this assert... */
    pips_assert("Last statement is either undefined or a call to return",
		statement_undefined_p(last) /*let's give up: it's always safe */
		|| !block_statement_p(s) /* not a block: any kind of statement can be found */
		||return_statement_p(last)); /* if a block, then a return */

    return last;
}

void
set_last_statement(statement s)
{
    statement ls = statement_undefined;

    pips_assert("last statement is undefined\n", statement_undefined_p(last_statement));

    ls = find_last_statement(s);

    last_statement = ls;
}

void
reset_last_statement()
{
    last_statement = statement_undefined;
}

bool
last_statement_p(statement s) {
    pips_assert("statement is defined\n", !statement_undefined_p(s));
    return s == last_statement;
}

/* function text text_module(module, stat)
 *
 * carefull! the original text of the declarations is used
 * if possible. Otherwise, the function text_declaration is called.
 */
text
text_module(entity module,
	    statement stat)
{
    text r = make_text(NIL);
    code c = entity_code(module);
    string s = code_decls_text(c);

    set_last_statement(stat);

    if ( strcmp(s,"") == 0 
	|| get_bool_property("PRETTYPRINT_ALL_DECLARATIONS") )
    {
	if (get_bool_property("PRETTYPRINT_HEADER_COMMENTS"))
	    /* Add the original header comments if any: */
	    ADD_SENTENCE_TO_TEXT(r, get_header_comments(module));
	
	ADD_SENTENCE_TO_TEXT(r, 
	   attach_head_to_sentence(sentence_head(module), module));
	if (head_hook) 
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
						  head_hook(module)));
	
	if (get_bool_property("PRETTYPRINT_HEADER_COMMENTS"))
	    /* Add the original header comments if any: */
	    ADD_SENTENCE_TO_TEXT(r, get_declaration_comments(module));
	
	MERGE_TEXTS(r, text_declaration(module));
    }
    else {
	ADD_SENTENCE_TO_TEXT(r, 
            attach_head_to_sentence(make_sentence(is_sentence_formatted, s),
				    module));
    }

    if (stat != statement_undefined) {
	MERGE_TEXTS(r, text_statement(module, 0, stat));
    }

    ADD_SENTENCE_TO_TEXT(r, sentence_tail());

    reset_last_statement();

    return(r);
}

text text_graph(), text_control() ;
string control_slabel() ;


void
output_a_graph_view_of_the_unstructured_successors(text r,
                                                   entity module,
                                                   int margin,
                                                   control c)
{                  
   add_one_unformated_printf_to_text(r, "%s %p\n",
                                     PRETTYPRINT_UNSTRUCTURED_ITEM_MARKER,
                                     c);

   if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH_VERBOSE")) {
      add_one_unformated_printf_to_text(r, "C Unstructured node %p ->", c);
      MAP(CONTROL, a_successor,
	  add_one_unformated_printf_to_text(r," %p", a_successor),
	  control_successors(c));
      add_one_unformated_printf_to_text(r,"\n");
   }

   MERGE_TEXTS(r, text_statement(module,
                                 margin,
                                 control_statement(c)));

   add_one_unformated_printf_to_text(r,
                                     PRETTYPRINT_UNSTRUCTURED_SUCC_MARKER);
   MAP(CONTROL, a_successor,
       {
          add_one_unformated_printf_to_text(r," %p", a_successor);
       },
          control_successors(c));
   add_one_unformated_printf_to_text(r,"\n");
}


bool
output_a_graph_view_of_the_unstructured_from_a_control(text r,
                                                       entity module,
                                                       int margin,
                                                       control begin_control,
                                                       control exit_control)
{
   bool exit_node_has_been_displayed = FALSE;
   list blocs = NIL;
   
   CONTROL_MAP(c,
               {
                  /* Display the statements of each node followed by
                     the list of its successors if any: */
                  output_a_graph_view_of_the_unstructured_successors(r,
                                                                     module,
                                                                     margin,
                                                                     c);
                  if (c == exit_control)
                     exit_node_has_been_displayed = TRUE;
               },
                  begin_control,
                  blocs);
   gen_free_list(blocs);

   return exit_node_has_been_displayed;
}

void
output_a_graph_view_of_the_unstructured(text r,
                                        entity module,
                                        string label,
                                        int margin,
                                        unstructured u,
                                        int num)
{
   bool exit_node_has_been_displayed = FALSE;
   control begin_control = unstructured_control(u);
   control end_control = unstructured_exit(u);

   add_one_unformated_printf_to_text(r, "%s %p end: %p\n",
                                     PRETTYPRINT_UNSTRUCTURED_BEGIN_MARKER,
                                     begin_control,
                                     end_control);
   exit_node_has_been_displayed =
      output_a_graph_view_of_the_unstructured_from_a_control(r,
                                                             module,
                                                             margin,
                                                             begin_control,
                                                             end_control);
   
   /* If we have not displayed the exit node, that mean that it is not
      connex with the entry node and so the code is
      unreachable. Anyway, it has to be displayed as for the classical
      Sequential View: */
   if (! exit_node_has_been_displayed) {
      /* Note that since the controlizer adds a dummy successor to the
         exit node, use
         output_a_graph_view_of_the_unstructured_from_a_control()
         instead of
         output_a_graph_view_of_the_unstructured_successors(): */
      output_a_graph_view_of_the_unstructured_from_a_control(r,
                                                             module,
                                                             margin,
                                                             end_control,
                                                             end_control);
      /* Even if the code is unreachable, add the fact that the
         control above is semantically related to the entry node. Add
         a dash arrow from the entry node to the exit node in daVinci,
         for example: */
      add_one_unformated_printf_to_text(r, "%s %p -> %p\n",
                                        PRETTYPRINT_UNREACHABLE_EXIT_MARKER,
                                        begin_control,
                                        end_control);
      if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH_VERBOSE"))
	  add_one_unformated_printf_to_text(r, "C Unreachable exit node (%p -> %p)\n",
					    begin_control,
					    end_control);
  }
   
   add_one_unformated_printf_to_text(r, "%s %p end: %p\n",
                                     PRETTYPRINT_UNSTRUCTURED_END_MARKER,
                                     begin_control,
                                     end_control);
}
