/* 	%A% ($Date: 1995/09/05 15:18:00 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
static char vcid[] = "%A% ($Date: 1995/09/05 15:18:00 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
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
  */
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "control.h"

/* Define the markers used in the raw unstructured output when the
   PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH property is true: */
#define PRETTYPRINT_UNSTRUCTURED_BEGIN_MARKER "\200Unstructured"
#define PRETTYPRINT_UNSTRUCTURED_END_MARKER "\201Unstructured End"
#define PRETTYPRINT_UNSTRUCTURED_ITEM_MARKER "\202Unstructured Item"
#define PRETTYPRINT_UNSTRUCTURED_SUCCESSOR_MARKER "\203Unstructured Successor ->"
#define PRETTYPRINT_UNSTRUCTURED_EDGE_MARKER "\201Unstructured Edge"
#define PRETTYPRINT_UNSTRUCTURED_EDGE_WITH_NODE_MARKER "\206Unstructured Edge With Node"
#define PRETTYPRINT_UNSTRUCTURED_SONS_BEGIN_MARKER "\204Unstructured Sons Begin"
#define PRETTYPRINT_UNSTRUCTURED_SONS_END_MARKER "\205Unstructured Sons End"


text empty_text( s )
statement s ;
{
    return( make_text( NIL )) ;
}

static text (*text_statement_hook)() = empty_text ;

void init_prettyprint( hook )
text (*hook)() ;
{
    text_statement_hook = hook ;
}

/* We have no way to distinguish between the SUBROUTINE and PROGRAM
 * They two have almost the same properties.
 * For the time being, especially for the PUMA project, we have a temporary
 * idea to deal with it: When there's no argument(s), it should be a PROGRAM,
 * otherwise, it should be a SUBROUTINE. 
 * Lei ZHOU 18/10/91
 *
 * correct PROGRAM and SUBROUTINE distinction added, FC 18/08/94
 */
sentence sentence_head(e)
entity e;
{
    cons *pc = NIL;
    type te = entity_type(e);
    functional fe;
    type tr;
    cons *args = words_parameters(e);

    pips_assert("sentence_head", type_functional_p(te));

    fe = type_functional(te);
    tr = functional_result(fe);

    
    if (type_void_p(tr)) 
    {
	/* the condition was ENDP(args) */
	pc = CHAIN_SWORD(pc, 
			 entity_main_module_p(e) ? 
			 "PROGRAM " : "SUBROUTINE ");
    }
    else if (type_variable_p(tr)) {
	pc = gen_nconc(pc, words_basic(variable_basic(type_variable(tr))));
	pc = CHAIN_SWORD(pc, " FUNCTION ");
    }
    else {
	pips_error("sentence_head", "unexpected type for result\n");
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

sentence sentence_tail()
{
    return(MAKE_ONE_WORD_SENTENCE(0, "END"));
}

sentence sentence_variable(e)
entity e;
{
    cons *pc = NIL;
    type te = entity_type(e);

    pips_assert("sentence_variable", type_variable_p(te));

    pc = gen_nconc(pc, words_basic(variable_basic(type_variable(te))));
    pc = CHAIN_SWORD(pc, " ");

    pc = gen_nconc(pc, words_declaration(e));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

/*  special management of empty commons added.
 *  this may happen in the hpfc generated code.
 */
sentence sentence_area(e, module)
entity e, module;
{
    string area_name = entity_local_name(e);
    type te = entity_type(e);
    bool prettyprint_hpfc = get_bool_property("PRETTYPRINT_HPFC");
    list 
	pc = NIL,
	entities = NIL;

    if (strcmp(area_name, STATIC_AREA_LOCAL_NAME) == 0 ||
	strcmp(area_name, DYNAMIC_AREA_LOCAL_NAME) == 0) 
	return(make_sentence(is_sentence_formatted, ""));

    assert(type_area_p(te));

    if (!ENDP(area_layout(type_area(te))))
    {
	MAPL(pee,
	 {
	     entity ee = ENTITY(CAR(pee));
	     if (local_entity_of_module_p(ee, module) || prettyprint_hpfc)
		 entities = CONS(ENTITY, ee, entities);
	 },
	     area_layout(type_area(te)));	     

	/*  the common is not output if it is empty
	 */
	if (!ENDP(entities))
	{
	    bool comma = FALSE;

	    pc = CHAIN_SWORD(pc, "COMMON ");
	    if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) != 0) 
	    {
		pc = CHAIN_SWORD(pc, "/");
		pc = CHAIN_SWORD(pc, area_name);
		pc = CHAIN_SWORD(pc, "/ ");
	    }
	    
	    entities = gen_nreverse(entities);
	    
	    MAPL(pee, 
	     {
		 entity ee = ENTITY(CAR(pee));
		 
		 if (comma) 
		     pc = CHAIN_SWORD(pc, ",");
		 else
		     comma = TRUE;
		 
		 pc = gen_nconc(pc, words_declaration(ee));
	     },
		 entities);

	    gen_free_list(entities);
	}
    }

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

sentence sentence_goto(module, margin, obj)
entity module;
int margin;
statement obj;
{
    string label = entity_local_name(statement_label(obj)) + 
	           strlen(LABEL_PREFIX);

    return( sentence_goto_label(module, margin, label)) ;
}

sentence sentence_goto_label(module, margin, label)
entity module;
int margin;
string label;
{
    cons *pc = NIL;

    if (strcmp(label, RETURN_LABEL_NAME) == 0) {
	pc = CHAIN_SWORD(pc, RETURN_FUNCTION_NAME);
    }
    else {
	pc = CHAIN_SWORD(pc, "GOTO ");
	pc = CHAIN_SWORD(pc, label);
    }

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, margin, pc)));
}

sentence sentence_basic_declaration(e)
entity e;
{
    list decl = NIL;
    basic b = entity_basic(e);

    pips_assert("sentence_basic_declaration", !basic_undefined_p(b));

    decl = CHAIN_SWORD(decl, basic_to_string(b));
    decl = CHAIN_SWORD(decl, " ");
    decl = CHAIN_SWORD(decl, entity_local_name(e));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, decl)));
}

sentence sentence_external(f)
entity f;
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "EXTERNAL ");
    pc = CHAIN_SWORD(pc, entity_local_name(f));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

sentence sentence_symbolic(f)
entity f;
{
    cons *pc = NIL;
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

sentence sentence_data(e)
entity e;
{
    cons *pc = NIL;
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
static text text_entity_declaration(module, ldecl)
entity module;
list ldecl;
{
    bool
	print_commons = get_bool_property("PRETTYPRINT_COMMONS");
    text 
	r = text_undefined;
    list
	before = NIL,
	after_before = NIL,
	pi = NIL,  
	pf4 = NIL, 
	pf8 = NIL, 
	pl = NIL, 
	pc = NIL,
	ps = NIL;

    MAPL(p,
     {
	 entity 
	     e = ENTITY(CAR(p));
	 type 
	     te = entity_type(e);
	 bool
	     func = 
		 type_functional_p(te) && storage_rom_p(entity_storage(e));
	 bool
	     param = func && value_symbolic_p(entity_initial(e));
	 bool
	     external =     /* subroutines won't be declared */
		 (func && 
		  value_code_p(entity_initial(e)) &&
		  !type_void_p(functional_result(type_functional(te))));
	 bool
	     common = type_area_p(te);
	 bool 
	     var = type_variable_p(te);
	 bool 
	     in_ram = storage_ram_p(entity_storage(e));
	 
	 debug(3, "text_declaration", "entity name is %s\n", entity_name(e));

	 if (!print_commons && common && !SPECIAL_COMMON_P(e))
	 {
	     after_before = 
		 CONS(SENTENCE,
		      make_sentence(is_sentence_formatted,
				    strdup(concatenate("common to include: ",
						       entity_local_name(e),
						       "\n",
						       NULL))),
		      after_before);
	 }

	 if (!print_commons && 
	     (common || 
	      (var && in_ram && 
	       !SPECIAL_COMMON_P(ram_section(storage_ram(entity_storage(e)))))))
	 {
	     debug(5, "text_declaration", 
		   "skipping entity %s\n", entity_name(e));
	 }
	 else if (param || external)
	 {
	     if (param) 
		 /*        PARAMETER
		  */
		 before = CONS(SENTENCE, sentence_symbolic(e), 
			       before);
	     else 
		 /*        EXTERNAL
		  */
		 before = CONS(SENTENCE, sentence_external(e), 
			       before);

	     before = CONS(SENTENCE, sentence_basic_declaration(e), 
			   before);
	 }
	 else if (common)
	 {
	     /*            COMMONS 
	      */
	     before = CONS(SENTENCE, sentence_area(e, module),
			   before);
	 }
	 else if (var)
	 {
	     basic
		 b = variable_basic(type_variable(te));
	     
	     switch ( basic_tag(b) ) 
	     {
	     case is_basic_int:
		 pi = CHAIN_SWORD(pi, pi==NIL ? "INTEGER " : ",");
		 pi = gen_nconc(pi, words_declaration(e)); 
		 break;
	     case is_basic_float:
		 switch (basic_float(b))
		 {
		 case 4:
		     pf4 = CHAIN_SWORD(pf4, pf4==NIL ? "REAL*4 " : ",");
		     pf4 = gen_nconc(pf4, words_declaration(e));
		     break;
		 case 8:
		 default:
		     pf8 = CHAIN_SWORD(pf8, pf8==NIL ? "REAL*8 " : ",");
		     pf8 = gen_nconc(pf8, words_declaration(e));
		     break;
		 }
		 break;			
	     case is_basic_logical:
		 pl = CHAIN_SWORD(pl, pl==NIL ? "LOGICAL " : ",");
		 pl = gen_nconc(pl, words_declaration(e));
		 break;
	     case is_basic_overloaded:
/*		 po = CHAIN_SWORD(po, po==NIL ? "OVERLOADED " : ",");
		 po = gen_nconc(po, words_declaration(e));*/
		 break; 
	     case is_basic_complex:
		 pc = CHAIN_SWORD(pc, pc==NIL ? "COMPLEX " : ",");
		 pc = gen_nconc(pc, words_declaration(e));
		 break;
	     case is_basic_string:
		 ps = CHAIN_SWORD(ps, ps==NIL ? "STRING  " : ",");
		 ps = gen_nconc(ps, words_declaration(e));
		 break;
	     default:
		 pips_error("text_declarations", 
			    "unexpected basic tag (%d)\n",
			    basic_tag(b));
	     }
	 }
     }, ldecl);

    r = make_text(gen_nconc(before, after_before));

    ADD_WORD_LIST_TO_TEXT(r, pi);
    ADD_WORD_LIST_TO_TEXT(r, pf4);
    ADD_WORD_LIST_TO_TEXT(r, pf8);
    ADD_WORD_LIST_TO_TEXT(r, pl);
    ADD_WORD_LIST_TO_TEXT(r, pc);
    ADD_WORD_LIST_TO_TEXT(r, ps);

    return (r);
}

text text_declaration(module)
entity module;
{
    return(text_entity_declaration(module,
				   code_declarations(entity_code(module))));
}

text text_common_declaration(common, module)
entity common, module;
{
    type 
	t = entity_type(common);
    list 
	ldecl = NIL;    
    text
	result = text_undefined;

    pips_assert("text_common_declaration", type_area_p(t));

    ldecl = CONS(ENTITY,
		 common,
		 gen_copy_seq(area_layout(type_area(t))));

    result = text_entity_declaration(module, ldecl);

    gen_free_list(ldecl);

    return(result);
}

text text_instruction(module, label, margin, obj, n)
entity module;
string label ;
int margin;
instruction obj;
int n ;
{
    text r=text_undefined, text_block(), text_unstructured() ;

    if (instruction_block_p(obj)) {
	r = text_block(module, label, margin, instruction_block(obj), n) ;
    }
    else if (instruction_test_p(obj)) {
	r = text_test(module, label, margin, instruction_test(obj),n);
    }
    else if (instruction_loop_p(obj)) {
	r = text_loop(module, label, margin, instruction_loop(obj),n);
    }
    else if (instruction_goto_p(obj)) {
	r = make_text(CONS(SENTENCE, 
			   sentence_goto(module, margin,
					 instruction_goto(obj)), 
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

text text_block(module, label, margin, objs, n)
entity module;
string label ;
int margin;
cons *objs;
int n;
{
    text r = make_text(NIL);
    cons *pbeg, *pend ;

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
	    pend = CHAIN_SWORD(NIL, "END   BLOCK");
	    
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

text text_loop(module, label, margin, obj, n)
entity module;
string label;
int margin;
loop obj;
int n ;
{
    text r = make_text(NIL);
    cons *pc ;
    unformatted u ;
    statement body = loop_body( obj ) ;
    string do_label = 
	    entity_local_name(loop_label( obj ))+strlen(LABEL_PREFIX) ;
    bool structured_do = empty_local_label_name_p( do_label );
    bool doall_loop_p = FALSE ;
    bool pp_all_priv_var_p = 
	get_bool_property("PRETTYPRINT_ALL_PRIVATE_VARIABLES");

    if(get_bool_property("PRETTYPRINT_DO_LABEL_AS_COMMENT")) {
	string st = concatenate("C     INITIALLY: DO ", do_label, NULL);

	ADD_SENTENCE_TO_TEXT(r, 
			     make_sentence(is_sentence_formatted, strdup(st)));
    }
    switch( execution_tag(loop_execution(obj)) ) {
    case is_execution_sequential:
	doall_loop_p = FALSE;
	break ;
    case is_execution_parallel:
        if (get_bool_property("PRETTYPRINT_CMFORTRAN")) {
          text aux_r;
          if((aux_r = text_loop_cmf(module, label, margin,
                                    obj, n, NIL, NIL))
             != text_undefined) {
            MERGE_TEXTS(r, aux_r);
            return(r) ;
          }
        }
        if (get_bool_property("PRETTYPRINT_CRAFT")) {
          text aux_r;
          if((aux_r = text_loop_craft(module, label, margin,
                                      obj, n, NIL, NIL))
             != text_undefined) {
            MERGE_TEXTS(r, aux_r);
            return(r);
          }
        }
	if (get_bool_property("PRETTYPRINT_FORTRAN90") && 
	    instruction_assign_p(statement_instruction(body)) ) {
	    MERGE_TEXTS(r, text_loop_90(module, label, margin, obj, n ));
	    return(r) ;
	}
	doall_loop_p = !get_bool_property("PRETTYPRINT_CRAY") &&
	    (!get_bool_property("PRETTYPRINT_CMFORTRAN")) &&
		(!get_bool_property("PRETTYPRINT_CRAFT"));
	break ;
    default:
	pips_error("text_loop", "Unknown tag\n") ;
    }
    pc = CHAIN_SWORD(NIL, (doall_loop_p) ? "DOALL " : "DO " );

    if(!structured_do && !doall_loop_p) {
	pc = CHAIN_SWORD(pc, concatenate(do_label, " ", NULL));
    }
    pc = CHAIN_SWORD(pc, entity_local_name(loop_index(obj)));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_loop_range(loop_range(obj)));
    u = make_unformatted(strdup(label), n, margin, pc) ;
    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted, u));


    /* private variables. modified 2-8-94 by BA */
    if( !ENDP(loop_locals(obj)) && (doall_loop_p || pp_all_priv_var_p) ) {
	int np = 0;

	pc = CHAIN_SWORD(NIL, "PRIVATE ") ;

	MAPL(ps, {
	    entity p = ENTITY(CAR(ps)) ;

	    if((p != loop_index(obj)) || pp_all_priv_var_p) {
		pc = CHAIN_SWORD(pc, entity_local_name(p));
		pc = CHAIN_SWORD(pc,
				 ( (ENDP(CDR(ps))) || (!pp_all_priv_var_p &&
						       (ENTITY(CAR(CDR(ps))) == loop_index(obj))
						       )
				  )?
				 "" : ",");
		np++;
	    }
	    else 
		pc = CHAIN_SWORD(pc,
				 ((ENDP(CDR(ps))) || 
				  (ENTITY(CAR(CDR(ps))) == loop_index(obj))
				  )?
				 "" : ",");

	}, loop_locals(obj)) ; /* end of MAPL */
	
	if(np > 0) {
	    u = make_unformatted(NULL, 0, margin+INDENTATION, pc) ;
	    ADD_SENTENCE_TO_TEXT(r, 
				 make_sentence(is_sentence_unformatted, u));
	}
    }
    MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, body));

    if(structured_do || 
       doall_loop_p || 
       get_bool_property("PRETTYPRINT_CRAY") ||
       get_bool_property("PRETTYPRINT_CRAFT") ||
       get_bool_property("PRETTYPRINT_CMFORTRAN")) {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDDO"));
    }
    return(r);
}

text init_text_statement(module, margin, obj)
entity module;
int margin;
statement obj;
{
    instruction i = statement_instruction(obj);
    text r = make_text( NIL ) ;
    string comments = statement_comments(obj);

    if (get_bool_property("PRETTYPRINT_ALL_EFFECTS")) {
	r = (*text_statement_hook)( module, margin, obj ) ;
    }
    else if ((instruction_block_p(i) && 
	      !get_bool_property("PRETTYPRINT_BLOCKS")) || 
	     (instruction_unstructured_p(i) && 
	      !get_bool_property("PRETTYPRINT_UNSTRUCTURED")))
	    ;
    else {
	r = (*text_statement_hook)( module, margin, obj ) ;
    }
    if (! string_undefined_p(comments)) {
	ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
					      comments));
    }
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
	}
    }
    return( r ) ;
}

text text_statement(module, margin, obj)
entity module;
int margin;
statement obj;
{
    instruction i = statement_instruction(obj);
    text r= make_text(NIL);
    text temp;
    string label = 
	    entity_local_name(statement_label(obj)) + strlen(LABEL_PREFIX);

    MERGE_TEXTS(r, init_text_statement(module, margin, obj)) ;

    if (strcmp(label, RETURN_LABEL_NAME) == 0) {
	/* do not add a redundant RETURN before an END:
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,
						       RETURN_FUNCTION_NAME));
						       */
	return(r);
    }
    temp = text_instruction(module, label, margin, i, statement_number(obj)) ;
    MERGE_TEXTS(r, temp);

    return(r);
}

text text_test(module, label, margin, obj,n)
entity module;
string label ;
int margin;
test obj;
int n;
{
    text r = make_text(NIL);
    cons *pc = NIL;
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
    if ((!empty_statement_p(test_false_obj)) &&
	(!statement_continue_p(test_false_obj))) {
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	MERGE_TEXTS(r, text_statement(module, margin+INDENTATION, 
				      test_false_obj));
      }

    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));

    return(r);
}

/*
 * function text text_module(module, stat)
 *
 * carefull! the original text of the declarations is used
 * if possible. Otherwise, the function text_declaration is called.
 */
text text_module(module, stat)
entity module;
statement stat;
{
   text r = make_text(NIL);
   code c = entity_code(module);
   string s = code_decls_text(c);

   if ( strcmp(s,"") == 0 
	|| get_bool_property("PRETTYPRINT_ALL_DECLARATIONS") ) {
      ADD_SENTENCE_TO_TEXT(r, sentence_head(module));
      MERGE_TEXTS(r, text_declaration(module));
   }
   else {
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, s));
   }

   if (stat != statement_undefined) {
      MERGE_TEXTS(r, text_statement(module, 0, stat));
   }

   ADD_SENTENCE_TO_TEXT(r, sentence_tail());

   return(r);
}

text text_graph(), text_control() ;
string control_slabel() ;


void
add_one_unformated_printf_to_text(text r,
                                  string a_format, ...)
{
   char buffer[200];
   
   va_list some_arguments;

   va_start(some_arguments, a_format);
   
   (void) vsprintf(buffer, a_format, some_arguments);
   ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
                                         strdup(buffer)));

   va_end(some_arguments);
}


void
output_a_graph_view_of_the_unstructured_successors(text r,
                                                   entity module,
                                                   int margin,
                                                   control c)
{                  
   char buffer[200];

   add_one_unformated_printf_to_text(r, "%s %#x\n",
                                     PRETTYPRINT_UNSTRUCTURED_ITEM_MARKER,
                                     (unsigned int) c);

   if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH")) {
      add_one_unformated_printf_to_text(r, "C Unstructured node %#x ->",
                                        (unsigned int) c);
      MAP(CONTROL, a_successor,
          {
             add_one_unformated_printf_to_text(r," %#x",
                                               (unsigned int) a_successor);
          },
             control_successors(c));
      add_one_unformated_printf_to_text(r,"\n");
   }

   MERGE_TEXTS(r, text_statement(module,
                                 margin,
                                 control_statement(c)));

   add_one_unformated_printf_to_text(r,
                                     PRETTYPRINT_UNSTRUCTURED_SUCCESSOR_MARKER);
   MAP(CONTROL, a_successor,
       {
          add_one_unformated_printf_to_text(r," %#x",
                                            (unsigned int) a_successor);
       },
          control_successors(c));
   add_one_unformated_printf_to_text(r,"\n");
}


void
output_a_graph_view_of_the_unstructured(text r,
                                        entity module,
                                        string label,
                                        int margin,
                                        unstructured u,
                                        int num)
{
   char buffer[200];
   list blocs = NIL;
   control begin_control = unstructured_control(u);
   control end_control = unstructured_exit(u);

   add_one_unformated_printf_to_text(r, "%s %#x end: %#x\n",
                                     PRETTYPRINT_UNSTRUCTURED_BEGIN_MARKER,
                                     (unsigned int) begin_control,
                                     (unsigned int) end_control);

   CONTROL_MAP(c,
               {
                  /* Display the statements of each node followed by
                     the list of its successors if any: */
                  output_a_graph_view_of_the_unstructured_successors(r,
                                                                     module,
                                                                     margin,
                                                                     c);
               },
                  begin_control,
                  blocs);
   gen_free_list(blocs);

   add_one_unformated_printf_to_text(r, "%s %#x end: %#x\n",
                                     PRETTYPRINT_UNSTRUCTURED_END_MARKER,
                                     (unsigned int) begin_control,
                                     (unsigned int) end_control);
}


/* TEXT_UNSTRUCTURED prettyprints the control graph CT (with label number
   NUM) from the MODULE at the current LABEL. If CEXIT == CT, then there
   is only one node and the goto+continue can be eliminated. */

text
text_unstructured(entity module,
                  string label,
                  int margin,
                  unstructured u, int num)
{
   text r ;
   hash_table labels = hash_table_make(hash_pointer, 0) ;
   set trail = set_make(set_pointer) ;
   control previous = control_undefined ;
   cons *blocs = NIL ;
   control cexit = unstructured_exit(u) ;
   control ct = unstructured_control(u) ;

   if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH"))
   {
      r = empty_text(NULL);
      
      output_a_graph_view_of_the_unstructured(r,
                                              module,
                                              label,
                                              margin,
                                              u,
                                              num);
   }
   else {      
      r = text_graph(module, margin, ct, &previous, trail, labels, cexit) ;
      MERGE_TEXTS(r, text_control(module, margin, cexit,
                                  &previous, set_make(set_pointer), labels,
                                  cexit)) ;
   }

   if( get_debug_level() == 9 ) {
      fprintf(stderr,"Unstructured %x (%x, %x)\n", 
              (unsigned int) u, (unsigned int) ct, (unsigned int) cexit ) ;
      CONTROL_MAP( n, {
         statement st = control_statement(n) ;

         fprintf(stderr, "\n%*sNode %x (%s)\n--\n", margin, "", 
                 (unsigned int) n, control_slabel(module, n, labels)) ;
         print_text(stderr, text_statement(module,margin,st));
         fprintf(stderr, "--\n%*sPreds:", margin, "");
         MAPL(ps,{fprintf(stderr,"%x ", (unsigned int) CONTROL(CAR(ps)));},
         control_predecessors(n));
         fprintf(stderr, "\n%*sSuccs:", margin, "") ;
         MAPL(ss,{fprintf(stderr,"%x ", (unsigned int) CONTROL(CAR(ss)));},
         control_successors(n));
         fprintf(stderr, "\n\n") ;
      }, ct , blocs) ;
      gen_free_list(blocs);
   }
   hash_table_free(labels) ;
   set_free(trail) ;
   return(r) ;
}

/* CONTROL_SLABEL returns a freshly allocated label name for the control
node C in the module M. H maps controls to label names. Computes a new
label name. */

string control_slabel(m, c, h)
entity m;
control c;
hash_table h;
{
    string l;

    if ((l = hash_get(h, (char *) c)) == HASH_UNDEFINED_VALUE) {
	statement st = control_statement(c) ;
	string label = entity_name( statement_label( st )) ;

	l = empty_label_p( label ) ? new_label_name(m) : label ;
	hash_put(h, (char *) c, strdup(l)) ;
    }
    pips_assert("control_slabel", strcmp(local_name(l), LABEL_PREFIX) != 0) ;
    pips_assert("control_slabel", strcmp(local_name(l), "") != 0) ;
    pips_assert("control_slabel", strcmp(local_name(l), "=") != 0) ;
    return(strdup(l));
}

/* ADD_CONTROL_GOTO adds to the text R a goto statement to the control
node SUCC from the current one OBJ in the MODULE and with a MARGIN.
LABELS maps control nodes to label names and SEENS (that links the
already prettyprinted node) is used to see whether a simple fall-through
wouldn't do. */

static void add_control_goto(module, margin, r, obj, 
			     succ, labels, seens, cexit )
entity module;
int margin;
text r ;
control obj, succ, cexit ;
hash_table labels;
set seens ;
{
    string label ;

    if( succ == (control)NULL ) {
	return ;
    }
    label = local_name(control_slabel(module, succ, labels))+
	    strlen(LABEL_PREFIX);

    if (strcmp(label, RETURN_LABEL_NAME) == 0 ||
	seens == (set)NULL || 
	(get_bool_property("PRETTYPRINT_INTERNAL_RETURN") && succ == cexit) ||
	set_belong_p(seens, (char *)succ)) {
	ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, margin, label));
    }
}

/* TEXT_CONTROL prettyprints the control node OBJ in the MODULE with a
MARGIN. SEENS is a trail that keeps track of already printed nodes and
LABELS maps control nodes to label names. The previously printed control
is in PREVIOUS. */

text text_control(module, margin, obj, previous, seens, labels, cexit)
entity module;
int margin;
control obj, *previous, cexit ;
set seens ;
hash_table labels;
{
    text r = make_text(NIL);
    sentence s;
    unformatted u ;
    statement st = control_statement(obj) ;
    cons *succs, *preds ;
    cons *pc;
    string label;
    string label_name ;

    label = control_slabel(module, obj, labels);
    label_name = strdup(local_name(label)+strlen(LABEL_PREFIX)) ;

    switch(gen_length(preds=control_predecessors(obj))) {
    case 0:
	break ;
    case 1: 
	if (*previous == CONTROL(CAR(preds)) &&
	    (obj != cexit || 
	     !get_bool_property("PRETTYPRINT_INTERNAL_RETURN"))) {
	    break ;
	}
    default:
	if( empty_label_p( entity_name( statement_label( st )))) {
	    pc = CHAIN_SWORD(NIL,"CONTINUE") ;
	    s = make_sentence(is_sentence_unformatted,
			      make_unformatted(NULL, 0, margin, pc)) ;
	    unformatted_label(sentence_unformatted(s)) = label_name ;
	    ADD_SENTENCE_TO_TEXT(r, s);    
	}
    }
    switch(gen_length(succs=control_successors(obj))) {
    case 0:
	MERGE_TEXTS(r, text_statement(module, margin, st));
	add_control_goto(module, margin, r, obj, 
			 cexit, labels, seens, (control)NULL ) ;
	break ;
    case 1:
	MERGE_TEXTS(r, text_statement(module, margin, st));
	add_control_goto(module, margin, r, obj, 
			 CONTROL(CAR(succs)), labels, seens, cexit) ;
	break;
    case 2: {
	instruction i = statement_instruction(st);
	test t;

	assert(instruction_test_p(i));

	MERGE_TEXTS(r, init_text_statement(module, margin, st)) ;
	pc = CHAIN_SWORD(NIL, "IF (");
	t = instruction_test(i);
	pc = gen_nconc(pc, words_expression(test_condition(t)));
	pc = CHAIN_SWORD(pc, ") THEN");
	u = make_unformatted(NULL, statement_number(st), margin, pc) ;

	if( !empty_label_p( entity_name( statement_label( st )))) {
	    unformatted_label(u) = strdup(label_name) ;
	}
	s = make_sentence(is_sentence_unformatted,u) ;
	ADD_SENTENCE_TO_TEXT(r, s);
	add_control_goto(module, margin+INDENTATION, r, obj, 
			 CONTROL(CAR(succs)), labels, seens, cexit) ;
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	add_control_goto(module, margin+INDENTATION, r, obj, 
			 CONTROL(CAR(CDR(succs))), labels, (set)NULL, cexit) ;
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));
	break;
    }
    default:
	pips_error("text_graph", "incorrect number of successors\n");
    }
    return( r ) ;
}

/* TEXT_GRAPH prettyprints the control graph OBJ in the MODULE with a
MARGIN. SEENS is a trail that keeps track of already printed nodes and
LABELS maps control nodes to label names. The previously printed control
is in PREVIOUS. CEXIT is not printed, done latter. */

text text_graph(module, margin, obj, previous, seens, labels, cexit)
entity module;
int margin;
control obj, *previous, cexit ;
set seens ;
hash_table labels;
{
    text r ;

    if(set_belong_p(seens, (char *)obj) || obj == cexit ) {
	return( make_text( NIL )) ;
    }
    set_add_element(seens, seens, (char *)obj) ;
    r = text_control(module, margin, obj, previous, seens, labels, cexit);
     *previous = obj ;

    MAPL(ss, {
	MERGE_TEXTS(r, text_graph(module, margin, CONTROL(CAR(ss)), 
				  previous, seens, labels, cexit));
    }, control_successors(obj));

    return( r );
}

cons *words_parameters(e)
entity e;
{
    cons *pc = NIL;
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

list words_declaration(e)
entity e;
{
    extern bool declaration_delayed_p(entity);
    extern string bound_parameter_name(entity, string, int);

    list 
	pl = NIL;

    pl = CHAIN_SWORD(pl, entity_local_name(e));

    if (!type_variable_p(entity_type(e))) return(pl);

    if (variable_dimensions(type_variable(entity_type(e))) != NIL) 
    {
	list dims = variable_dimensions(type_variable(entity_type(e)));
	
	pl = CHAIN_SWORD(pl, "(");

	/*
	if (get_bool_property("PRETTYPRINT_HPFC") && declaration_delayed_p(e))
	{
	    int i;
	    list ls = NIL;

	    for(i=gen_length(dims); i>0; i--)
		ls = CONS(STRING, bound_parameter_name(e, "LO", i),
		     CONS(STRING, strdup(":"),
		     CONS(STRING, bound_parameter_name(e, "UP", i),
		     ENDP(ls) ? NIL : CONS(STRING, strdup(","), ls))));
	    
	    pl = gen_nconc(pl, ls);
	}
	else
	*/

	MAPL(pd, 
	 {
	     pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd))));
	     
	     if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, ",");
	 }, 
	     dims);
	
	pl = CHAIN_SWORD(pl, ")");
    }
    
    return(pl);
}

cons *words_basic(obj)
basic obj;
{
    cons *pc = NIL;

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

cons *words_value(obj)
value obj;
{
    cons *pc;

    if (value_symbolic_p(obj)) {
	pc = words_constant(symbolic_constant(value_symbolic(obj)));
    }
    else if (value_constant(obj)) {
	pc = words_constant(value_constant(obj));
    }
    else {
	pips_error("words_value", "unexpected tag");
	pc = NIL;
    }

    return(pc);
}

cons *words_constant(obj)
constant obj;
{
    cons *pc;

    pc=NIL;

    if (constant_int_p(obj)) {
	pc = CHAIN_IWORD(pc,constant_int(obj));
    }
    else {
	pips_error("words_constant", "unexpected tag");
    }

    return(pc);
}

cons *words_dimension(obj)
dimension obj;
{
    cons *pc;

    pc = words_expression(dimension_lower(obj));
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(dimension_upper(obj)));

    return(pc);
}

cons *words_expression(obj)
expression obj;
{
    cons *pc;

    pc = words_syntax(expression_syntax(obj));

    return(pc);
}

cons *words_subexpression(obj, precedence)
expression obj;
int precedence;
{
    cons *pc;

    if ( expression_call_p(obj) )
	pc = words_call(syntax_call(expression_syntax(obj)), precedence);
    else 
	pc = words_syntax(expression_syntax(obj));

     return(pc);
}

cons *words_loop_range(obj)
range obj;
{
    cons *pc;
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

cons *words_range(obj)
range obj;
{
    cons *pc = NIL ;

    pc = CHAIN_SWORD(pc,"(/I,I=");
    pc = gen_nconc(pc, words_expression(range_lower(obj)));
    pc = CHAIN_SWORD(pc,",");
    pc = gen_nconc(pc, words_expression(range_upper(obj)));
    pc = CHAIN_SWORD(pc,",");
    pc = gen_nconc(pc, words_expression(range_increment(obj)));
    pc = CHAIN_SWORD(pc,"/)") ;

    return(pc);
}

cons *words_reference(obj)
reference obj;
{
    cons *pc = NIL;
    entity e = reference_variable(obj);

    pc = CHAIN_SWORD(pc, entity_local_name(e));
/*    attach_something_to_last_word(pc, e); */
    
    if (reference_indices(obj) != NIL) {
	pc = CHAIN_SWORD(pc,"(");
	MAPL(pi, {
	    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(pi)), 0));
	    if (CDR(pi) != NIL)
		pc = CHAIN_SWORD(pc,",");
	}, reference_indices(obj));
	pc = CHAIN_SWORD(pc,")");
    }

    return(pc);
}

cons *words_label_name(s)
string s ;
{
    return(CHAIN_SWORD(NIL, local_name(s)+strlen(LABEL_PREFIX))) ;
}

cons *words_syntax(obj)
syntax obj;
{
    cons *pc;

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

cons *words_call(obj, precedence)
call obj;
int precedence;
{
    cons *pc;

    entity f = call_function(obj);
    value i = entity_initial(f);
    
    pc = (value_intrinsic_p(i)) ? words_intrinsic_call(obj, precedence) : 
	                          words_regular_call(obj);

    return(pc);
}

cons *words_regular_call(obj)
call obj;
{
    cons *pc = NIL;

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

cons *words_prefix_unary_op(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;
    expression e = EXPRESSION(CAR(call_arguments(obj)));
    int prec = words_intrinsic_precedence(obj);

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
    pc = gen_nconc(pc, words_subexpression(e, prec));

    return(pc);
}

cons *words_unary_minus(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;
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

/* 
 * If the infix operator is either "-" or "/", I perfer not to delete 
 * the parentheses of the second expression.
 * Ex: T = X - ( Y - Z ) and T = X / (Y*Z)
 *
 * Lei ZHOU       Nov. 4 , 1991
 */
cons *words_infix_binary_op(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;
    cons *args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);
    cons *we1 = words_subexpression(EXPRESSION(CAR(args)), prec);
    cons *we2;

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

cons *words_assign_op(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;
    cons *args = call_arguments(obj);
    int prec = words_intrinsic_precedence(obj);

    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(args)),  prec));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_subexpression(EXPRESSION(CAR(CDR(args))), prec));

    return(pc);
}

cons *words_nullary_op(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;

    pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));

    return(pc);
}

cons *words_io_control(iol, precedence)
cons **iol;
int precedence;
{
    cons *pc = NIL;
    cons *pio = *iol;

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

cons *words_implied_do(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;

    cons *pcc;
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

cons *words_unbounded_dimension(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

cons *words_list_directed(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;

    pc = CHAIN_SWORD(pc, "*");

    return(pc);
}

cons *words_io_inst(obj, precedence)
call obj;
int precedence;
{
    cons *pc = NIL;
    cons *pcio = call_arguments(obj);
    cons *pio_write;
    boolean good_fmt, good_unit, iolist_reached;

    /* AP: I try to convert WRITE to PRINT. Three conditions must be
       fullfilled. The first, and obvious, one, is that the function has
       to be WRITE. Secondly, "FMT" has to be equal to "*". Finally,
       "UNIT" has to be equal either to "*" or "6".  In such case,
       "WRITE(*,*)" is replaced by "PRINT *,". */
    good_fmt = FALSE;
    good_unit = FALSE;
    if (strcmp(entity_local_name(call_function(obj)), "WRITE") == 0) {
      pio_write = pcio;
      iolist_reached = FALSE;
      while ((pio_write != NIL) && (!iolist_reached)) {
	syntax s = expression_syntax(EXPRESSION(CAR(pio_write)));
	call c;
	expression arg = EXPRESSION(CAR(CDR(pio_write)));

	if (! syntax_call_p(s)) {
	    pips_error("words_io_inst", "call expected");
	}

	c = syntax_call(s);

	if (strcmp(entity_local_name(call_function(c)), "FMT=") == 0) {
	  if (strcmp(words_to_string(words_expression(arg)), "*") == 0)
	    good_fmt= TRUE;
	}

	if (strcmp(entity_local_name(call_function(c)), "UNIT=") == 0) {
	  if ((strcmp(words_to_string(words_expression(arg)), "*") == 0) ||
	      (strcmp(words_to_string(words_expression(arg)), "6") == 0))
	    good_unit = TRUE;
	}

	if (strcmp(entity_local_name(call_function(c)), "IOLIST=") == 0) {
	  iolist_reached = TRUE;
	  pio_write = CDR(pio_write);
	}
	else
	  pio_write = CDR(CDR(pio_write));
      }
    }

    if (good_fmt && good_unit) {
      /* AP: Allright for the substitution of WRITE by PRINT. For the
         IOLIST prettyprint, we skip everything but elements following the
         first "IOLIST=" keyword. */
      pc = CHAIN_SWORD(pc, "PRINT *,");
      pcio = pio_write;
    }
    else {
      pc = CHAIN_SWORD(pc, entity_local_name(call_function(obj)));
      pc = CHAIN_SWORD(pc, " (");
      /* FI: missing argument; I use "precedence" because I've no clue;
         see LZ */
      pc = gen_nconc(pc, words_io_control(&pcio, precedence));
      pc = CHAIN_SWORD(pc, ") ");
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
	    pc = CHAIN_SWORD(pc, ",");
	}
    }, pcio);
    return(pc) ;
}

cons *null(obj, precedence)
call obj;
int precedence;
{
    return(NIL);
}

/* precedence needed here
 * According to the Precedence of Operators 
 * Arithmetic > Character > Relational > Logical
 * Added by Lei ZHOU    Nov. 4,91
 */
struct intrinsic_handler {
    char * name;
    cons *(*f)();
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
    {"FORMAT", words_prefix_unary_op, 0},
    {UNBOUNDED_DIMENSION_NAME, words_unbounded_dimension, 0},
    {LIST_DIRECTED_FORMAT_NAME, words_list_directed, 0},

    {NULL, null, 0}
};

cons *words_intrinsic_call(obj, precedence)
call obj;
int precedence;
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

int words_intrinsic_precedence(obj)
call obj;
{
    char *n = entity_local_name(call_function(obj));

    return(intrinsic_precedence(n));
}

int intrinsic_precedence(n)
string n;
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
