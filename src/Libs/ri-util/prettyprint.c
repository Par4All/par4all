/*
 * $Id$
 *
 * $Log: prettyprint.c,v $
 * Revision 1.124  1998/10/28 16:04:33  zory
 * prettyrpint for n-ary operators included
 * bug removed into fma prettyprint
 *
 * Revision 1.123  1998/09/22 07:25:42  zory
 * create prettyprint for specific operators that are used by the
 * OPTIMIZE_EXPRESSIONS transformation
 *
 * Revision 1.122  1998/09/17 12:06:49  keryell
 * Added attachment support for module call.
 *
 * Revision 1.121  1998/09/09 15:50:40  irigoin
 * Proper prettyprinting of ranges in expressions. Function text_loop() had
 * to be splitted into text_loop() and text_loop_default(). The later is
 * called from text_loop() and from text_loop_90() in fortran90.c. Sometimes
 * vector loops cannot be expressed as array assignments and a default loop
 * has to be printed. Function words_subscript_range() was added to print
 * ranges either as a vector constructor with an implied DO or as a triplet notation.
 *
 * Revision 1.120  1998/09/08 18:31:50  irigoin
 * Several quick bug fixes. Not a really consistent state for the Fortran 90
 * output but we want to make the Production version of PIPS consistent for
 * most outputs.
 *
 * Revision 1.119  1998/07/24 10:46:56  irigoin
 * Bug fix for unary minus
 *
 * Revision 1.118  1998/06/03 06:42:03  irigoin
 * Prettyprint of whileloop added
 *
 * Revision 1.117  1998/04/14 13:03:35  coelho
 * cleaner.
 *
 * Revision 1.116  1998/03/24 09:34:41  irigoin
 * Semantics of leftmost refined to distinguish between arithmetic
 * expressions and general expressions since leftmost is used to decide if a
 * unary minus requires parentheses or not. Basically, leftmost always
 * remains TRUE as long as an arithmetic expression has not been entered.
 * words_infix_binary_op() has been modified using the new constant
 * MINIMAL_ARITHMETIC_PRECEDENCE. It is assumed that arithmetic operators
 * have greater precedences than any other operator.
 *
 * Revision 1.115  1998/03/19 16:27:28  irigoin
 * leftmost argument added to handle unary minuses in front of expressions
 *
 * Revision 1.114  1998/03/10 16:48:01  irigoin
 * New property added to control parentheses:
 * PRETTYPRINT_ALL_PARENTHESES. Requested by Julien Zory.
 *
 * Revision 1.113  1998/03/08 20:43:28  irigoin
 * Improved pips error message
 *
 * Revision 1.112  1998/03/07 21:55:39  irigoin
 * Improved error message
 *
 * Revision 1.111  1998/03/05 14:03:11  irigoin
 * pips_assert() added in sentence_goto() to avoid the printout of a GOTO
 * without target label
 *
 * Revision 1.110  1998/02/11 21:23:07  ancourt
 * text_instruction static again :-)
 *
 * Revision 1.109  1998/02/11 20:23:06  ancourt
 * text_instruction deleting  the  static option
 *
 * Revision 1.108  1997/12/12 15:03:20  coelho
 * leaks--
 *
 * Revision 1.107  1997/12/12 14:51:11  coelho
 * leaks--
 *
 * Revision 1.106  1997/12/10 12:10:19  coelho
 * missing strdup fixed.
 *
 * Revision 1.105  1997/12/09 14:21:46  coelho
 * cleaner...
 *
 * Revision 1.104  1997/11/25 10:18:16  coelho
 * updates for add_to_current_line
 *
 * Revision 1.103  1997/11/22 12:16:49  coelho
 * OMP style prettyprint of parallel loops added.
 *
 * Revision 1.102  1997/11/21 13:33:38  coelho
 * type fixed.
 *
 * Revision 1.101  1997/11/21 13:19:23  coelho
 * string property driven prettyprint of parallel code.
 *
 * Revision 1.100  1997/11/20 16:05:44  keryell
 * Forgotten that there is IO instruction without any format...
 *
 * Revision 1.99  1997/11/20 14:11:45  keryell
 * Modified words_io_inst() to avoid using unnecessary word builders that
 * leads to memory leaks and Epips core dumps...
 *
 * Revision 1.98  1997/11/18 23:42:26  keryell
 * Fixed a memory leak in words_io_inst() with a nasty side effect that
 * leads to fail in Epips attacments...
 *
 * Revision 1.97  1997/11/12 15:30:10  coelho
 * cleaner...
 *
 * Revision 1.96  1997/11/08 17:25:10  coelho
 * extension for cloning (name different from actual module)
 *
 * Revision 1.95  1997/11/04 17:36:48  coelho
 * strdup of comments added.
 *
 * Revision 1.94  1997/11/04 12:46:09  coelho
 * unused function dropped.
 *
 * Revision 1.93  1997/11/03 14:27:38  coelho
 * debug on/off prettyprint debug level.
 *
 * Revision 1.92  1997/11/01 09:09:41  irigoin
 * assert transformed into a warning in find_last_statement() because people
 * do not have enough time to fix the problemes in Transformations and in the
 * polyhedral method. However Wp65 and hpfc have been fixed.
 *
 * Revision 1.91  1997/10/29 12:35:09  irigoin
 * Guard added to avoid calling find_last_statement() thru
 * set_last_statement() when it is not needed
 *
 * Revision 1.90  1997/10/28 15:01:13  coelho
 * much moved to declarations.c
 *
 * Revision 1.89  1997/10/28 14:07:55  coelho
 * prettyprint common together...
 *
 * Revision 1.88  1997/10/28 10:03:17  irigoin
 * assert added in text_statement() about the statement with label
 * "LABEL_RETURN_NAME"
 *
 * Revision 1.87  1997/10/27 15:31:03  coelho
 * drop overloaded externals...
 *
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
char lib_ri_util_prettyprint_c_rcsid[] = "$Header: /home/data/tmp/PIPS/pips_data/trunk/src/Libs/ri-util/RCS/prettyprint.c,v 1.124 1998/10/28 16:04:33 zory Exp $";
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

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "pipsdbm.h"

#include "misc.h"
#include "properties.h"
#include "prettyprint.h"

/* operator precedences are in the [0,100] range */

#define MAXIMAL_PRECEDENCE 100
#define MINIMAL_ARITHMETIC_PRECEDENCE 19

/* Define the markers used in the raw unstructured output when the
   PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH property is true: */
#define PRETTYPRINT_UNSTRUCTURED_BEGIN_MARKER "\200Unstructured"
#define PRETTYPRINT_UNSTRUCTURED_END_MARKER "\201Unstructured End"
#define PRETTYPRINT_UNSTRUCTURED_ITEM_MARKER "\202Unstructured Item"
#define PRETTYPRINT_UNSTRUCTURED_SUCC_MARKER "\203Unstructured Successor ->"
#define PRETTYPRINT_UNREACHABLE_EXIT_MARKER "\204Unstructured Unreachable"


/******************************************************************* STYLES */

static bool 
pp_style_p(string s)
{
    return same_string_p(get_string_property(PRETTYPRINT_PARALLEL), s);
}

#define pp_hpf_style_p() 	pp_style_p("hpf")
#define pp_f90_style_p() 	pp_style_p("f90")
#define pp_craft_style_p() 	pp_style_p("craft")
#define pp_cray_style_p() 	pp_style_p("cray")
#define pp_cmf_style_p()	pp_style_p("cmf")
#define pp_doall_style_p()	pp_style_p("doall")
#define pp_do_style_p()		pp_style_p("do")
#define pp_omp_style_p()	pp_style_p("omp")

/********************************************************************* MISC */

text empty_text(entity e, int m, statement s)
{ return make_text(NIL);}

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

/* exported for craft 
 */
list 
words_loop_range(range obj)
{
    list pc;
    call c = syntax_call(expression_syntax(range_increment(obj)));

    pc = words_subexpression(range_lower(obj), 0, TRUE);
    pc = CHAIN_SWORD(pc,", ");
    pc = gen_nconc(pc, words_subexpression(range_upper(obj), 0, TRUE));
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
    if (expression_undefined_p(range_lower(obj))) {
	pc = CONS(STRING, MAKE_SWORD("*"), NIL);
    }
    else {
	call c = syntax_call(expression_syntax(range_increment(obj)));

	  pc = CHAIN_SWORD(pc,"(/ (I,I=");
	  pc = gen_nconc(pc, words_expression(range_lower(obj)));
	  pc = CHAIN_SWORD(pc,",");
	  pc = gen_nconc(pc, words_expression(range_upper(obj)));
	if(strcmp( entity_local_name(call_function(c)), "1") != 0) {
	  pc = CHAIN_SWORD(pc,",");
	  pc = gen_nconc(pc, words_expression(range_increment(obj)));
	}
	  pc = CHAIN_SWORD(pc,") /)") ;
    }
    return pc;
}

/* FI: array constructor R433, p. 37 in Fortran 90 standard, can
   be used anywhere in arithmetic expressions whereas the triplet
   notation is restricted to subscript expressions. The triplet
   notation is used to define array sections (see R619, p. 64).
*/

list /* of string */ 
words_subscript_range(range obj)
{
    list pc = NIL ;

    /* if undefined I print a star, why not!? */
    if (expression_undefined_p(range_lower(obj))) {
	pc = CONS(STRING, MAKE_SWORD("*"), NIL);
    }
    else {
	call c = syntax_call(expression_syntax(range_increment(obj)));

	pc = gen_nconc(pc, words_expression(range_lower(obj)));
	pc = CHAIN_SWORD(pc,":");
	pc = gen_nconc(pc, words_expression(range_upper(obj)));
	if(strcmp( entity_local_name(call_function(c)), "1") != 0) {
	    pc = CHAIN_SWORD(pc,":");
	    pc = gen_nconc(pc, words_expression(range_increment(obj)));
	}
    }
    return pc;
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
	    expression subscript = EXPRESSION(CAR(pi));
	    syntax ssubscript = expression_syntax(subscript);

	    if(syntax_range_p(ssubscript)) {
		pc = gen_nconc(pc, words_subscript_range(syntax_range(ssubscript)));
	    }
	    else {
		pc = gen_nconc(pc, words_subexpression(subscript, 0, TRUE));
	    }

	    if (CDR(pi) != NIL)
		pc = CHAIN_SWORD(pc,",");
	}, reference_indices(obj));
	pc = CHAIN_SWORD(pc,")");
    }
    attach_reference_to_word_list(begin_attachment, STRING(CAR(gen_last(pc))),
				  obj);

    return(pc);
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


/* To deal with attachment on user module usage. */
list 
words_genuine_regular_call(call obj)
{
  list pc = words_regular_call(obj);
  
  if (call_arguments(obj) != NIL) {
    /* The call is not used to code a constant: */
    entity f = call_function(obj);
    type t = entity_type(f);
    /* The module name is the first one except if it is a procedure CALL. */
    if (type_void_p(functional_result(type_functional(t))))
      attach_regular_call_to_word(STRING(CAR(CDR(pc))), obj);
    else
      attach_regular_call_to_word(STRING(CAR(pc)), obj);
  }
  

  return pc;
}


static list 
words_assign_op(call obj, int precedence, bool leftmost)
{
    list pc = NIL, args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);

    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(args)), prec, TRUE));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(CDR(args))), prec, TRUE));

    return(pc);
}

static list 
words_substring_op(call obj, int precedence, bool leftmost)
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

    pc = gen_nconc(pc, words_subexpression(r,  prec, TRUE));
    pc = CHAIN_SWORD(pc, "(");
    pc = gen_nconc(pc, words_subexpression(l, prec, TRUE));
    pc = CHAIN_SWORD(pc, ":");
    pc = gen_nconc(pc, words_subexpression(u, prec, TRUE));
    pc = CHAIN_SWORD(pc, ")");

    return(pc);
}

static list 
words_assign_substring_op(call obj, int precedence, bool leftmost)
{
  /* The assign substring function call is reduced to a syntactic construct */
    list pc = NIL;
    expression e = expression_undefined;
    int prec = words_intrinsic_precedence(obj);

    pips_assert("words_substring_op", gen_length(call_arguments(obj)) == 4);

    e = EXPRESSION(CAR(CDR(CDR(CDR(call_arguments(obj))))));

    pc = gen_nconc(pc, words_substring_op(obj,  prec, TRUE));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_subexpression(e, prec, TRUE));

    return(pc);
}

static list 
words_nullary_op(call obj, int precedence, bool leftmost)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));

    return(pc);
}

static list 
words_io_control(list *iol, int precedence, bool leftmost)
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
words_implied_do(call obj, int precedence, bool leftmost)
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
words_unbounded_dimension(call obj, int precedence, bool leftmost)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

static list 
words_list_directed(call obj, int precedence, bool leftmost)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

static list 
words_io_inst(call obj,
	      int precedence, bool leftmost)
{
    list pc = NIL;
    list pcio = call_arguments(obj);
    list pio_write = pcio;
    boolean good_fmt = FALSE;
    bool good_unit = FALSE;
    bool iolist_reached = FALSE;
    bool complex_io_control_list = FALSE;
    expression fmt_arg = expression_undefined;
    expression unit_arg = expression_undefined;
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
	    /* Avoid to use words_expression(arg) because it set some
               attachments and unit_words may not be used
               later... RK. */
	    entity f;
	    /* The * format is coded as a call to "LIST_DIRECTED_FORMAT_NAME" function: */
	    good_fmt = syntax_call_p(expression_syntax(arg))
		&& value_intrinsic_p(entity_initial(f = 
                   call_function(syntax_call(expression_syntax(arg)))))
		    && (strcmp(entity_local_name(f),
			       LIST_DIRECTED_FORMAT_NAME)==0);
	    pio_write = CDR(CDR(pio_write));
	    /* To display the format later: */
	    fmt_arg = arg;
	}
	else if (strcmp(entity_local_name(call_function(c)), "UNIT=") == 0) {
	    /* Avoid to use words_expression(arg) because it set some
               attachments and unit_words may not be used
               later... RK. */
	    entity f;
	    /* The * format is coded as a call to "LIST_DIRECTED_FORMAT_NAME" function: */
	    good_unit = syntax_call_p(expression_syntax(arg))
		&& value_intrinsic_p(entity_initial(f = 
		    call_function(syntax_call(expression_syntax(arg)))))
		    && (strcmp(entity_local_name(f),
			       LIST_DIRECTED_FORMAT_NAME)==0);
	    /* To display the unit later: */
	    unit_arg = arg;
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

	if (pio_write != NIL)	/* WRITE (*,*) pio -> PRINT *, pio */
	{
	    pc = CHAIN_SWORD(pc, "PRINT *, ");
	}
	else			/* WRITE (*,*)  -> PRINT *  */
	{
	    pc = CHAIN_SWORD(pc, "PRINT * ");
	}
       
	pcio = pio_write;
    }
    else if (good_fmt && good_unit && same_string_p(called, "READ"))
    {
	/* READ (*,*) -> READ * */
	
	if (pio_write != NIL)	/* READ (*,*) pio -> READ *, pio */
	{
	    pc = CHAIN_SWORD(pc, "READ *, ");
	}
	else			/* READ (*,*)  -> READ *  */
	{
	    pc = CHAIN_SWORD(pc, "READ * ");
	}
	pcio = pio_write;
    }	
    else if (!complex_io_control_list) {
	list unit_words = words_expression(unit_arg);
	pips_assert("A unit must be defined", !ENDP(unit_words));
	pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
	pc = CHAIN_SWORD(pc, " (");
	pc = gen_nconc(pc, unit_words);
	
	if (!expression_undefined_p(fmt_arg)) {
	    /* There is a FORMAT: */
	    pc = CHAIN_SWORD(pc, ", ");
	    pc = gen_nconc(pc, words_expression(fmt_arg));
	}

	pc = CHAIN_SWORD(pc, ") ");
	pcio = pio_write;
    }
    else {
	pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
	pc = CHAIN_SWORD(pc, " (");
	/* FI: missing argument; I use "precedence" because I've no clue;
	   see LZ */
	pc = gen_nconc(pc, words_io_control(&pcio, precedence, leftmost));
	pc = CHAIN_SWORD(pc, ") ");
	/* 
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
		pips_internal_error("missing element in IO list");
	    pc = CHAIN_SWORD(pc, ", ");
	}
    }, pcio);
    return(pc) ;
}

static list 
null(call obj, int precedence, bool leftmost)
{
    return(NIL);
}

static list
words_prefix_unary_op(call obj, int precedence, bool leftmost)
{
    list pc = NIL;
    expression e = EXPRESSION(CAR(call_arguments(obj)));
    int prec = words_intrinsic_precedence(obj);

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
    pc = gen_nconc(pc, words_subexpression(e, prec, FALSE));

    return(pc);
}

static list 
words_unary_minus(call obj, int precedence, bool leftmost)
{
    list pc = NIL;
    expression e = EXPRESSION(CAR(call_arguments(obj)));
    int prec = words_intrinsic_precedence(obj);

    if ( prec < precedence || !leftmost)
	pc = CHAIN_SWORD(pc, "(");
    pc = CHAIN_SWORD(pc, "-");
    pc = gen_nconc(pc, words_subexpression(e, prec, FALSE));
    if ( prec < precedence || !leftmost)
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

/* EOLE : The multiply-add operator is used within the optimize
   transformation ( JZ - sept 98) - fma(a,b,c) -> (a + (b*c)) */
list /* of string */ 
eole_fma_specific_op(call obj, int precedence, bool leftmost){
  list /* of strings */ pc = NIL;
  list /* of expressions */ args = call_arguments(obj);

  int prec ;
  
  /* open parenthese one  */
  pc = CHAIN_SWORD(pc, "(");

  /* get precedence for add operator */
  prec = intrinsic_precedence("+");

  /* first argument */
  pc = gen_nconc(pc,words_subexpression(EXPRESSION(CAR(args)), prec, TRUE));
  
  /* add operator */
  pc = CHAIN_SWORD(pc,"+");

  /* open parenthese two */
  pc = CHAIN_SWORD(pc, "(");

  /* get precedence for multiply operator */
  prec = intrinsic_precedence("*");

   /* second argument */
  args = CDR(args);
  pc = gen_nconc(pc,words_subexpression(EXPRESSION(CAR(args)),prec,TRUE));

  /* multiply operator */
  pc = CHAIN_SWORD(pc,"*");

  /* third argument */
  args = CDR(args);
  pc = gen_nconc(pc,words_subexpression(EXPRESSION(CAR(args)),prec,FALSE));

  /* close parenthese two */
  pc = CHAIN_SWORD(pc, ")");

  /* close parenthese one  */
  pc = CHAIN_SWORD(pc,")");

  return pc;
}

/* Check if the given operator is associated with a special
    prettyprint. For instance, n-ary add and multiply operators which are
    used in the EOLE project use "+" and "*" prettyprints instead of the
    entity_local_name (JZ - sept 98) */
static string 
get_special_prettyprint_for_operator(call obj){

  static struct special_operator_prettyprint {
    char * name;
    char * op_prettyprint;
  } tab_operator_prettyprint[] = {
    {EOLE_SUM_OPERATOR_NAME,"+"},
    {EOLE_PROD_OPERATOR_NAME,"*"},
    {NULL,NULL}
  };
  int i = 0;
  string op_name; 

  /* get the entity name */
  op_name = entity_local_name(call_function(obj));

  while (tab_operator_prettyprint[i].name) {
    if (!strcmp(tab_operator_prettyprint[i].name,op_name))
      return tab_operator_prettyprint[i].op_prettyprint;
    else i++;
  }
  
  return op_name;
}

/* Extension of "words_infix_binary_op" function for nary operators used
   in the EOLE project - (since "nary" assumes operators with at least 2
   op)  - JZ (Oct. 98)*/

static list /* of string */
words_infix_nary_op(call obj, int precedence, bool leftmost)
{
  list /*of string*/ pc = NIL;
  list /* of expressions */ args = call_arguments(obj);
  
  /* get current operator precedence */
  int prec = words_intrinsic_precedence(obj);

  expression exp1 = EXPRESSION(CAR(args));
  expression exp2;

  list we1 = words_subexpression(exp1, prec, 
				 prec>=MINIMAL_ARITHMETIC_PRECEDENCE? leftmost: TRUE);
  list we2;

  /* open parenthese if necessary */
  if ( prec < precedence )
    pc = CHAIN_SWORD(pc, "(");
  pc = gen_nconc(pc, we1);

  /* reach the second arg */
  args = CDR(args);
  
  for(; args; args=CDR(args)) { /* for all args */
    exp2 = EXPRESSION(CAR(args));
    
    
    /* 
     * If the infix operator is either "-" or "/", I prefer not to delete 
     * the parentheses of the second expression.
     * Ex: T = X - ( Y - Z ) and T = X / (Y*Z)
     *
     * Lei ZHOU       Nov. 4 , 1991
     */
    if ( strcmp(entity_local_name(call_function(obj)), "/") == 0 )  /* divide operator */
      we2 = words_subexpression(exp2, MAXIMAL_PRECEDENCE, FALSE);
    else if ( strcmp(entity_local_name(call_function(obj)), "-") == 0 ) { /* minus operator */
      if ( expression_call_p(exp2) &&
	   words_intrinsic_precedence(syntax_call(expression_syntax(exp2))) >= 
	   intrinsic_precedence("*") )
	/* precedence is greater than * or / */
	we2 = words_subexpression(exp2, prec, FALSE);
      else
	we2 = words_subexpression(exp2, MAXIMAL_PRECEDENCE, FALSE);
    }
    else {
      we2 = words_subexpression(exp2, prec,
				prec<MINIMAL_ARITHMETIC_PRECEDENCE);
    }
  
    /* operator prettyprint */
    pc = CHAIN_SWORD(pc, get_special_prettyprint_for_operator(obj));

    pc = gen_nconc(pc, we2);
  }
  /* close parenthese if necessary */
  if ( prec < precedence )
    pc = CHAIN_SWORD(pc, ")");
  
  return(pc);
}


/* 
 * If the infix operator is either "-" or "/", I prefer not to delete 
 * the parentheses of the second expression.
 * Ex: T = X - ( Y - Z ) and T = X / (Y*Z)
 *
 * Lei ZHOU       Nov. 4 , 1991
 */
static list 
words_infix_binary_op(call obj, int precedence, bool leftmost)
{
    list pc = NIL;
    list args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);
    list we1 = words_subexpression(EXPRESSION(CAR(args)), prec, 
				   prec>=MINIMAL_ARITHMETIC_PRECEDENCE? leftmost: TRUE);
    list we2;

    if ( strcmp(entity_local_name(call_function(obj)), "/") == 0 )
	we2 = words_subexpression(EXPRESSION(CAR(CDR(args))), MAXIMAL_PRECEDENCE, FALSE);
    else if ( strcmp(entity_local_name(call_function(obj)), "-") == 0 ) {
	expression exp = EXPRESSION(CAR(CDR(args)));
	if ( expression_call_p(exp) &&
	     words_intrinsic_precedence(syntax_call(expression_syntax(exp))) >= 
	     intrinsic_precedence("*") )
	    /* precedence is greater than * or / */
	    we2 = words_subexpression(exp, prec, FALSE);
	else
	    we2 = words_subexpression(exp, MAXIMAL_PRECEDENCE, FALSE);
    }
    else {
	we2 = words_subexpression(EXPRESSION(CAR(CDR(args))), prec,
				  prec<MINIMAL_ARITHMETIC_PRECEDENCE);
    }

    
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
 *
 * A precedence is a integer in [0..MAXIMAL_PRECEDENCE]
 */

static struct intrinsic_handler {
    char * name;
    list (*f)();
    int prec;
} tab_intrinsic_handler[] = {
    {"**", words_infix_binary_op, 30},

    {"//", words_infix_binary_op, 30},

    /* The Fortran 77 standard does not allow x*-3 or x+-3, but this is dealt
    * with by argument leftmost, not by prorities.
    */
    {"--", words_unary_minus, 25},
    /* {"--", words_unary_minus, 19}, */

    {"*", words_infix_binary_op, 21},
    {"/", words_infix_binary_op, 21},

    {"+", words_infix_binary_op, 20},
    {"-", words_infix_binary_op, 20},

    /* Non-arithemtic operators have priorities lesser than MINIMAL_ARITHMETIC_PRECEDENCE
     * leftmost is restaured to true for unary minus.
     */

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

    /* These operators are used within the optimize transformation in
order to manipulate operators such as n-ary add and multiply or
multiply-add operators ( JZ - sept 98) */
    {EOLE_FMA_OPERATOR_NAME, eole_fma_specific_op , MINIMAL_ARITHMETIC_PRECEDENCE},
    {EOLE_SUM_OPERATOR_NAME, words_infix_nary_op, 20},
    {EOLE_PROD_OPERATOR_NAME, words_infix_nary_op, 21},
    {NULL, null, 0}
};

static bool precedence_p = TRUE;

static list 
words_intrinsic_call(call obj, int precedence, bool leftmost)
{
    struct intrinsic_handler *p = tab_intrinsic_handler;
    char *n = entity_local_name(call_function(obj));

    while (p->name != NULL) {
	if (strcmp(p->name, n) == 0) {
	    return((*(p->f))(obj, precedence, leftmost));
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
	if (strcmp(p->name, n) == 0)
	    return(p->prec);
	p++;
    }

    return 0;
}

static int
words_intrinsic_precedence(call obj)
{
    char *n = entity_local_name(call_function(obj));
    return intrinsic_precedence(n);
}

/* exported for cmfortran.c
 */
list 
words_call(
    call obj,
    int precedence,
    bool leftmost)
{
    list pc;
    entity f = call_function(obj);
    value i = entity_initial(f);
    pc = (value_intrinsic_p(i)) ? 
	(words_intrinsic_call(obj, 
			     (precedence_p||precedence<=1)? precedence : MAXIMAL_PRECEDENCE,
			     leftmost))
	: 
	words_genuine_regular_call(obj);
    return pc;
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
	pc = words_call(syntax_call(obj), 0, TRUE);
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
    int precedence,
    bool leftmost)
{
    list pc;
    
    if ( expression_call_p(obj) )
	pc = words_call(syntax_call(expression_syntax(obj)), precedence, leftmost);
    else 
	pc = words_syntax(expression_syntax(obj));
    
    return pc;
}


/**************************************************************** SENTENCE */

sentence 
sentence_tail(void)
{
    return(MAKE_ONE_WORD_SENTENCE(0, "END"));
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
    pips_assert("Legal label required", strlen(tlabel)!=0);
    return sentence_goto_label(module, label, margin, tlabel, n);
}


/********************************************************************* TEXT */

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


    if(!empty_local_label_name_p(label)) {
	pips_internal_error("Illegal label \"%s\". "
			    "Blocks cannot carry a label\n",
			    label);
    }
    
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
	free_text(t);
    }

    if (!get_bool_property("PRETTYPRINT_FOR_FORESYS") &&
	(get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
	 get_bool_property("PRETTYPRINT_BLOCKS"))) 
    {
	unformatted u = make_unformatted(strdup("C"), n, margin, pend);
	ADD_SENTENCE_TO_TEXT(r, 
			     make_sentence(is_sentence_unformatted, u));
    }
    return r;
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
	hpf_private = pp_hpf_style_p(),
	omp_private = pp_omp_style_p(),
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
	string private;
	if (hpf_private) private = "NEW(";
	else if (omp_private) private = "PRIVATE(";
	else private = "PRIVATE ";
	l = CONS(STRING, MAKE_SWORD(private), l);
	if (hpf_private || omp_private) CHAIN_SWORD(l, ")");
    }

    return l;
}

/* returns a formatted text for the HPF independent and new directive 
 * well, no continuations and so, but the directives do not fit the 
 * unformatted domain, because the directive prolog would not be well
 * managed there.
 */
static string
marged(
    string prefix,
    int margin)
{
    int len = strlen(prefix), i;
    string result = (string) malloc(strlen(prefix)+margin+1);
    strcpy(result, prefix);
    for (i=len; margin-->0;) 
	result[i++] = ' '; result[i]='\0';
    return result;
}

static text 
text_directive(
    loop obj,   /* the loop we're interested in */
    int margin,
    string basic_directive,
    string basic_continuation,
    string parallel)
{
    string
	dir = marged(basic_directive, margin),
	cont = marged(basic_continuation, margin);
    text t = make_text(NIL);
    char buffer[100]; /* ??? */
    list /* of string */ l = NIL;
    bool is_hpf = pp_hpf_style_p(), is_omp = pp_omp_style_p();

    /* start buffer */
    buffer[0] = '\0';
    
    if (execution_parallel_p(loop_execution(obj)))
    {
	add_to_current_line(buffer, dir, cont, t);
	add_to_current_line(buffer, parallel, cont, t);
	l = loop_private_variables(obj);
	if (l && is_hpf) 
	    add_to_current_line(buffer, ", ", cont, t);
    }
    else if (get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES"))
    {
	l = loop_private_variables(obj);
	if (l) 
	{
	    add_to_current_line(buffer, dir, cont, t);
	    if (is_omp) add_to_current_line(buffer, "DO", cont, t);
	}
    }
    
    if (strlen(buffer)>0)
	MAP(STRING, s, add_to_current_line(buffer, s, cont, t), l);

    /* what about reductions? should be associated to the ri somewhere.
     */
    close_current_line(buffer, t);
    free(dir);
    free(cont);
    return t;
}

#define HPF_SENTINEL 		"!HPF$"
#define HPF_DIRECTIVE 		HPF_SENTINEL " "
#define HPF_CONTINUATION 	HPF_SENTINEL "x"
#define HPF_INDEPENDENT 	"INDEPENDENT"

static text
text_hpf_directive(loop l, int m)
{
    return text_directive(l, m, "\n" HPF_DIRECTIVE, HPF_CONTINUATION,
			  HPF_INDEPENDENT);
}

#define OMP_SENTINEL 		"!$OMP"
#define OMP_DIRECTIVE 		OMP_SENTINEL " "
#define OMP_CONTINUATION 	OMP_SENTINEL "x"
#define OMP_PARALLELDO		"PARALLEL DO "

static text 
text_omp_directive(loop l, int m)
{
    return text_directive(l, m, "\n" OMP_DIRECTIVE, OMP_CONTINUATION,
			  OMP_PARALLELDO);
}

text 
text_loop_default(
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
    bool structured_do = empty_local_label_name_p(do_label);
    bool doall_loop_p = FALSE;
    bool hpf_prettyprint = pp_hpf_style_p();
    bool do_enddo_p = get_bool_property("PRETTYPRINT_DO_LABEL_AS_COMMENT");
    bool all_private =  get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES");

    if(execution_sequential_p(loop_execution(obj))) {
	doall_loop_p = FALSE;
    }
    else {
	doall_loop_p = pp_doall_style_p();
    }

    /* HPF directives before the loop if required (INDEPENDENT and NEW) */
    if (hpf_prettyprint)  MERGE_TEXTS(r, text_hpf_directive(obj, margin));
    /* idem if Open MP directives are required */
    if (pp_omp_style_p()) MERGE_TEXTS(r, text_omp_directive(obj, margin));

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
    if (structured_do || doall_loop_p || do_enddo_p ||
	pp_cray_style_p() || pp_craft_style_p() || pp_cmf_style_p())
    {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDDO"));
    }

    attach_loop_to_sentence_up_to_end_of_text(first_sentence, r, obj);

    return r;
}

text 
text_loop(
    entity module,
    string label,
    int margin,
    loop obj,
    int n)
{
    text r = make_text(NIL);
    statement body = loop_body( obj ) ;
    entity the_label = loop_label(obj);
    string do_label = entity_local_name(the_label)+strlen(LABEL_PREFIX) ;
    bool structured_do = empty_local_label_name_p(do_label);
    bool do_enddo_p = get_bool_property("PRETTYPRINT_DO_LABEL_AS_COMMENT");

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
	    MERGE_TEXTS(r, text_loop_default(module, label, margin, obj, n));
	break ;
    case is_execution_parallel:
        if (pp_cmf_style_p()) {
          text aux_r;
          if((aux_r = text_loop_cmf(module, label, margin, obj, n, NIL, NIL))
             != text_undefined) {
	      MERGE_TEXTS(r, aux_r);
          }
        }
        else if (pp_craft_style_p()) {
          text aux_r;
          if((aux_r = text_loop_craft(module, label, margin, obj, n, NIL, NIL))
             != text_undefined) {
            MERGE_TEXTS(r, aux_r);
          }
        }
	else if (pp_f90_style_p() && 
	    instruction_assign_p(statement_instruction(body)) ) {
	    MERGE_TEXTS(r, text_loop_90(module, label, margin, obj, n));
	}
	else {
	    MERGE_TEXTS(r, text_loop_default(module, label, margin, obj, n));
	}
	break ;
	default:
	pips_error("text_loop", "Unknown tag\n") ;
    }
    return r;
}

text 
text_whileloop(
    entity module,
    string label,
    int margin,
    whileloop obj,
    int n)
{
    list pc = NIL;
    sentence first_sentence;
    unformatted u;
    text r = make_text(NIL);
    statement body = whileloop_body( obj ) ;
    entity the_label = whileloop_label(obj);
    string do_label = entity_local_name(the_label)+strlen(LABEL_PREFIX) ;
    bool structured_do = empty_local_label_name_p(do_label);
    bool do_enddo_p = get_bool_property("PRETTYPRINT_DO_LABEL_AS_COMMENT");


    /* Show the initial label of the loop to name it...
     * FI: I believe this is useless for while loops since they cannot
     * be parallelized.
     */
    if(!structured_do && do_enddo_p)
    {
	ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
	  strdup(concatenate("!     INITIALLY: DO ", do_label, "\n", NULL))));
    }

    /* LOOP prologue.
     */
    pc = CHAIN_SWORD(NIL, "DO " );
    
    if(!structured_do && !do_enddo_p) {
	pc = CHAIN_SWORD(pc, concatenate(do_label, " ", NULL));
    }
    pc = CHAIN_SWORD(pc, "WHILE (");
    pc = gen_nconc(pc, words_expression(whileloop_condition(obj)));
    pc = CHAIN_SWORD(pc, ")");
    u = make_unformatted(strdup(label), n, margin, pc) ;
    ADD_SENTENCE_TO_TEXT(r, first_sentence = 
			 make_sentence(is_sentence_unformatted, u));

    /* loop BODY
     */
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, body));

    /* LOOP postlogue
     */
    if (structured_do) {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDDO"));
    }

    /* attach_loop_to_sentence_up_to_end_of_text(first_sentence, r, obj); */
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
    text r;

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
    else
	r  = make_text( NIL ) ;

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
    pc = gen_nconc(pc, words_call(c, 0, TRUE));

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
    text r = text_undefined;

    if (instruction_block_p(obj)) {
	r = text_block(module, label, margin, instruction_block(obj), n) ;
    }
    else if (instruction_test_p(obj)) {
	r = text_test(module, label, margin, instruction_test(obj), n);
    }
    else if (instruction_loop_p(obj)) {
	r = text_loop(module, label, margin, instruction_loop(obj), n);
    }
    else if (instruction_whileloop_p(obj)) {
	r = text_whileloop(module, label, margin, instruction_whileloop(obj), n);
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
				 words_call(instruction_call(obj), 0, TRUE));

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
    pips_debug(9, "statement_comments: --%s--\n", 
	       string_undefined_p(comments)? "<undef>": comments);

    if(statement_number(stmt)!=STATEMENT_NUMBER_UNDEFINED &&
       statement_ordering(stmt)==STATEMENT_ORDERING_UNDEFINED) {
	/* we are in trouble with some kind of dead (?) code... */
	pips_user_warning("I unexpectedly bumped into dead code?\n");
    }

    if (same_string_p(label, RETURN_LABEL_NAME)) 
    {
	pips_assert("Statement with return label must be a return statement",
		    return_statement_p(stmt));

	/* do not add a redundant RETURN before an END, unless requested */
	if(get_bool_property("PRETTYPRINT_FINAL_RETURN")
	   || !last_statement_p(stmt)) 
	{
	    sentence s = MAKE_ONE_WORD_SENTENCE(margin, RETURN_FUNCTION_NAME);
	    temp = make_text(CONS(SENTENCE, s ,NIL));
	}
	else {
	    temp = make_text(NIL);
	}
    }
    else
    {
	temp = text_instruction(module, label, margin, i,
				statement_number(stmt)) ;
    }

    /* note about comments: they are duplicated here, but I'm pretty
     * sure that the free is NEVER performed as it should. FC.
     */
    if(!ENDP(text_sentences(temp))) {
	MERGE_TEXTS(r, init_text_statement(module, margin, stmt)) ;
	if (! string_undefined_p(comments)) {
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  strdup(comments)));
	}
	MERGE_TEXTS(r, temp);
    }
    else {
	/* Preserve comments */
	if (! string_undefined_p(comments)) {
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  strdup(comments)));
	}
	free_text(temp);
    }

    attach_statement_information_to_text(r, stmt);

    ifdebug(1) {
	if (instruction_sequence_p(i)) {
	    if(!(statement_with_empty_comment_p(stmt)
		 && statement_number(stmt) == STATEMENT_NUMBER_UNDEFINED
		 && unlabelled_statement_p(stmt))) {
		user_log("Block statement %s\n"
			 "Block number=%d, Block label=\"%s\", block comment=\"%s\"\n",
			 statement_identification(stmt),
			 statement_number(stmt), label_local_name(statement_label(stmt)),
			 statement_comments(stmt));
		pips_error("text_statement", "This block statement should be labelless, numberless"
	
			   " and commentless.\n");
		}
	}
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

    pips_assert("statement is defined", !statement_undefined_p(s));

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

    /* Too many program transformations and syntheses violate the following assert */
    if(!(statement_undefined_p(last)
	 || !block_statement_p(s)
	 || return_statement_p(last))) {
	pips_user_warning("Last statement is not a RETURN!\n");
	last = statement_undefined;
    }

    /* I had a lot of trouble writing the condition for this assert... */
    pips_assert("Last statement is either undefined or a call to return",
	 statement_undefined_p(last) /* let's give up: it's always safe */
     || !block_statement_p(s) /* not a block: any kind of statement... */
		|| return_statement_p(last)); /* if a block, then a return */

    return last;
}

void
set_last_statement(statement s)
{
    statement ls = statement_undefined;
    pips_assert("last statement is undefined", 
		statement_undefined_p(last_statement));
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
text_named_module(
    entity name, /* the name of the module */
    entity module,
    statement stat)
{
    text r = make_text(NIL);
    code c = entity_code(module);
    string s = code_decls_text(c);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    /* This guard is correct but could be removed if find_last_statement()
     * were robust and/or if the internal representations were always "correct".
     * See also the guard for reset_last_statement()
     */
    if(!get_bool_property("PRETTYPRINT_FINAL_RETURN"))
	set_last_statement(stat);

    precedence_p = !get_bool_property("PRETTYPRINT_ALL_PARENTHESES");

    if ( strcmp(s,"") == 0 
	|| get_bool_property("PRETTYPRINT_ALL_DECLARATIONS") )
    {
	if (get_bool_property("PRETTYPRINT_HEADER_COMMENTS"))
	    /* Add the original header comments if any: */
	    ADD_SENTENCE_TO_TEXT(r, get_header_comments(module));
	
	ADD_SENTENCE_TO_TEXT(r, 
	   attach_head_to_sentence(sentence_head(name), module));
	
	if (head_hook) 
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
						  head_hook(module)));
	
	if (get_bool_property("PRETTYPRINT_HEADER_COMMENTS"))
	    /* Add the original header comments if any: */
	    ADD_SENTENCE_TO_TEXT(r, get_declaration_comments(module));
	
	MERGE_TEXTS(r, text_declaration(module));
    }
    else 
    {
	ADD_SENTENCE_TO_TEXT(r, 
            attach_head_to_sentence(make_sentence(is_sentence_formatted, 
						  strdup(s)),
				    module));
    }

    if (stat != statement_undefined) {
	MERGE_TEXTS(r, text_statement(module, 0, stat));
    }

    ADD_SENTENCE_TO_TEXT(r, sentence_tail());

    if(!get_bool_property("PRETTYPRINT_FINAL_RETURN"))
	reset_last_statement();

    debug_off();
    return(r);
}

text
text_module(
    entity module,
    statement stat)
{
    return text_named_module(module, module, stat);
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
