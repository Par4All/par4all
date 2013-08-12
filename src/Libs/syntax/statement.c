/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "syntax.h"
#include "syn_yacc.h"



/* the purpose of the following data structure is to associate labels to
instructions. The data structure contains a string (the label's name)
and a statement (the statement which the label is attached to). */

#define INITIAL_STMTHEAP_BUFFER_SIZE 10

typedef struct {
    string l; /* the name of the label */
    statement s; /* the statement attached to l */
} stmt;

static stmt * StmtHeap_buffer;
static int StmtHeap_buffer_size;
static int CurrentStmt = 0;

static void
init_StmtHeap_buffer(void)
{
    if (StmtHeap_buffer_size!=0) return; /* if needed */
    pips_debug(9, "allocating StmtHeap buffer\n");
    StmtHeap_buffer_size = INITIAL_STMTHEAP_BUFFER_SIZE;
    StmtHeap_buffer = (stmt*) malloc(sizeof(stmt)*StmtHeap_buffer_size);
    pips_assert("malloc ok", StmtHeap_buffer);
}

static void
resize_StmtHeap_buffer(void)
{
    pips_debug(9, "resizing StmtHeap buffer\n");
    pips_assert("buffer initialized", StmtHeap_buffer_size>0);
    StmtHeap_buffer_size*=2;
    StmtHeap_buffer = (stmt*) realloc(StmtHeap_buffer, 
				      sizeof(stmt)*StmtHeap_buffer_size);
    pips_assert("realloc ok", StmtHeap_buffer);
}

void
parser_reset_StmtHeap_buffer(void)
{
    CurrentStmt = 0;
}

/* this functions looks up in table StmtHeap for the statement s whose
label is l. */

statement 
LabelToStmt(l)
string l;
{
  int i;

  for (i = 0; i < CurrentStmt; i++)
    if (strcmp(l, StmtHeap_buffer[i].l) == 0)
      return(StmtHeap_buffer[i].s);

  return(statement_undefined);
}



/* this function looks for undefined labels. a label is undefined if a
goto to that label has been encountered and if no statement with this
label has been parsed. */

void 
CheckAndInitializeStmt(void)
{
  int i;
  int MustStop = false;

  for (i = 0; i < CurrentStmt; i++) {
    statement s = StmtHeap_buffer[i].s;
    if (statement_instruction(s) == instruction_undefined) {
      MustStop = true;
      user_warning("CheckAndInitializeStmt", "Undefined label \"%s\"\n", 
		   label_local_name(statement_label(s)));
    }
  }

  if (MustStop) {
    ParserError("CheckAndInitializeStmt", "Undefined label(s)\n");
  }
  else {
    CurrentStmt = 0;
  }
}



/* this function stores a new association in table StmtHeap: the label
of statement s is e. */

void 
NewStmt(e, s)
entity e;
statement s;
{
  init_StmtHeap_buffer();

  pips_assert("The empty label is not associated to a statement",
	      !entity_empty_label_p(e));

  pips_assert("Label e is the label of statement s", e==statement_label(s));

  if (LabelToStmt(entity_name(e)) != statement_undefined) {
    user_log("NewStmt: duplicate label: %s\n", entity_name(e));
    ParserError("NewStmt", "duplicate label\n");
  }

  if (CurrentStmt >= StmtHeap_buffer_size)
    resize_StmtHeap_buffer();

  StmtHeap_buffer[CurrentStmt].l = entity_name(e);
  StmtHeap_buffer[CurrentStmt].s = s;
  CurrentStmt += 1;
}



/* The purpose of the following data structure is to build the control
structure of the procedure being analyzed. each time a control statement
(do loop, block if, ...) is analyzed, a new block is created and pushed
on the block stack. regular statement (assign, goto, return, ...) are
linked to the block that is on the top of the stack. blocks are removed
from the stack when the corresponding end statement is encountered
(endif, end of loop, ...). 

The block ending statements are ELSE, ENDIF,...

There does not seem to be any limit on the nesting level in Fortran standard.
MAXBLOCK is set to "large" value for our users. The IF/THEN/ELSEIF construct is replaced by nested IF/ELSE statements which increases the nesting level observed by the source reader.

Fabien Coelho suggests to refactor this part of the code with a Newgen stack automatically reallocated on overflow: 

stack s = stack_make(statement_domain, 0, 0);
stack_push(e, s);
e = stack_pop(s);
while (!stack_empty_p(s)) { ... }
stack_free(s);
*/

#define MAXBLOCK 200

typedef struct block {
    instruction i; /* the instruction that contains this block */
    string l;      /* the expected statement which will end this block */
    cons * c;      /* the list of statements contained in this block */
    int elsifs ;   /* ELSEIF statements are desugared as nested IFs. They
                      must be counted to generate the effect of the proper
                      number of ENDIF */
} block;
LOCAL block BlockStack[MAXBLOCK];
LOCAL int CurrentBlock = 0;

void 
ResetBlockStack()
{
  CurrentBlock = 0;
}

bool 
IsBlockStackEmpty()
{
  return(CurrentBlock == 0);
}

bool 
IsBlockStackFull()
{
  return(CurrentBlock == MAXBLOCK);
}

void 
PushBlock(i, l)
instruction i;
string l;
{
  if (IsBlockStackFull())
    ParserError("PushBlock", "top of stack reached\n");

  pips_assert("PushBlock", instruction_block_p(i));

  BlockStack[CurrentBlock].i = i;
  BlockStack[CurrentBlock].l = l;
  BlockStack[CurrentBlock].c = NULL;
  BlockStack[CurrentBlock].elsifs = 0;
  CurrentBlock += 1;
}

instruction 
PopBlock()
{
  if (IsBlockStackEmpty())
    ParserError("PopBlock", "bottom of stack reached\n");

  return(BlockStack[--CurrentBlock].i);
}



/* This functions creates a label. LABEL_PREFIX is added to its name, for
 * integer constants and labels not to have the same name space.
 *
 * If an empty string is passed, the empty label seems to be returned
 * since EMPTY_LABEL_NAME is defined as LABEL_PREFIX in ri-util-local.h
 * (FI, 5 March 1998)
 */

entity 
MakeLabel(s)
const char* s;
{
    entity l;
    static char *name = NULL ;

    if( name == NULL ) {
	name = (char *)malloc( LABEL_SIZE+sizeof(LABEL_PREFIX) ) ;
    }
    debug(5, "MakeLabel", "\"%s\"\n", s);

    strcpy(name, LABEL_PREFIX);
    strcat(name, s);	

    l = FindOrCreateEntity( (strcmp( name, LABEL_PREFIX )==0) ? 
			    TOP_LEVEL_MODULE_NAME :
			    CurrentPackage, name);

    if (entity_type(l) == type_undefined) {
	debug(5, "MakeLabel", "%s\n", name);
	entity_type(l) = MakeTypeStatement();
	entity_storage(l) = make_storage_rom();
	entity_initial(l) = make_value(is_value_constant,
				       make_constant_litteral());
    }
    else {
	debug(5, "MakeLabel", "%s already exists\n", name);
    }
    return(l);
}

statement 
MakeNewLabelledStatement(l, i)
entity l;
instruction i;
{
    statement s;

    debug(9, "MakeNewLabelledStatement", "begin for label \"%s\" and instruction %s\n",
	  label_local_name(l), instruction_identification(i));

    if(instruction_loop_p(i) && get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
	statement c = make_continue_statement(l);
	statement ls = instruction_to_statement(i);

	statement_number(ls) = get_statement_number();//get_next_statement_number();
	NewStmt(l, c);
	s = make_block_statement(CONS(STATEMENT,c,
				      CONS(STATEMENT, ls, NIL)));
    }
    else if(instruction_block_p(i)) {
	/* Associate label to the first statement in the block because
	 * block cannot be labelled.
	 */
	statement s1 = STATEMENT(CAR(instruction_block(i)));

	statement_label(s1) = l;
	NewStmt(l, s1);
	s = make_statement(entity_empty_label(),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments,
			   i, NIL, NULL, empty_extensions (), make_synchronization_none());
    }
    else {
	s = make_statement(l,
			   (instruction_goto_p(i))?
			   STATEMENT_NUMBER_UNDEFINED : get_statement_number(),//get_next_statement_number(),
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments,
			   i, NIL, NULL, empty_extensions (), make_synchronization_none());
	NewStmt(l, s);
    }

    debug(9, "MakeNewLabelledStatement", "end for label \"%s\"\n",
	  label_local_name(l));

    return s;
}

statement 
ReuseLabelledStatement(s, i)
statement s;
instruction i;
{
    statement new_s = statement_undefined;

    debug(9, "ReuseLabelledStatement", "begin for label \"%s\"\n",
	  label_local_name(statement_label(s)));

    pips_assert("Should have no number", 
		statement_number(s)==STATEMENT_NUMBER_UNDEFINED);
    pips_assert("The statement instruction should be undefined",
		instruction_undefined_p(statement_instruction(s)));

    if(instruction_loop_p(i) && get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
	/* Comments probably are lost... */
	instruction c = make_continue_instruction();
	statement ls = instruction_to_statement(i);

	statement_number(ls) = get_statement_number();//get_next_statement_number();
	statement_instruction(s) = c;

	new_s = instruction_to_statement(
	    make_instruction(is_instruction_sequence,
			     make_sequence(CONS(STATEMENT,s,
						CONS(STATEMENT, ls, NIL)))));
    }
    else if(instruction_block_p(i)) {
	/* Here, you are in trouble because the label cannot be carried 
	 * by the block. It should be carried by the first statement of
	 * the block... which has already been allocated.
	 * This only should occur with desugared constructs because they
	 * must bypass the MakeStatement() module to handle statement
	 * numbering properly.
	 *
	 * Reuse s1, the first statement of the block, to contain the
	 * whole block. Reuse s to contain the first instruction of the
	 * block.
	 */
	statement s1 = STATEMENT(CAR(instruction_block(i)));

	pips_assert("The first statement of the block is not a block",
		    !statement_block_p(s1));

	/* s only has got a label */
	statement_number(s) = statement_number(s1);
	statement_ordering(s) = statement_ordering(s1);
	statement_comments(s) = statement_comments(s1);
	statement_instruction(s) = statement_instruction(s1);

	statement_label(s1) = entity_empty_label();
	statement_number(s1) = STATEMENT_NUMBER_UNDEFINED;
	statement_ordering(s1) = STATEMENT_ORDERING_UNDEFINED;
	statement_comments(s1) = empty_comments;
	statement_instruction(s1) = i;

	instruction_block(i) = CONS(STATEMENT, s,
				    CDR(instruction_block(i)));

	pips_assert("The first statement of block s1 must be s\n",
		    STATEMENT(CAR(instruction_block(statement_instruction(s1))))
		    == s);

	new_s = s1;
    }
    else {
	statement_instruction(s) = i;
	/* 
	   statement_number(s) = (instruction_goto_p(i))?
	   STATEMENT_NUMBER_UNDEFINED : get_next_statement_number();
	*/
	/* Let's number labelled GOTO because a CONTINUE is derived later from them */
	statement_number(s) = get_statement_number(); //get_next_statement_number();
	new_s = s;
    }

    debug(9, "ReuseLabelledStatement", "end for label \"%s\"\n",
	  label_local_name(statement_label(s)));

    return new_s;
}

/* This function makes a statement. l is the label and i the
 * instruction. We make sure that the label is not declared twice.
 *
 * Comments are added by LinkInstToCurrentBlock() which calls MakeStatement()
 * because it links the instruction by linking its statement..
 *
 * GO TO statements are numbered like other statements although they
 * are destroyed by the controlizer. To be changed.
 */

statement 
MakeStatement(l, i)
entity l;
instruction i;
{
    statement s;

    debug(5, "MakeStatement", "Begin for label %s and instruction %s\n",
	  entity_name(l), instruction_identification(i));

    pips_assert("MakeStatement", type_statement_p(entity_type(l)));
    pips_assert("MakeStatement", storage_rom_p(entity_storage(l)));
    pips_assert("MakeStatement", value_constant_p(entity_initial(l)));
    pips_assert("MakeStatement", 
		constant_litteral_p(value_constant(entity_initial(l))));

    if (!entity_empty_label_p(l)) {
	/* There is an actual label */

	/* Well, there is no easy solution to handle labels when Fortran 
	 * constructs such as alternate returns, computed gotos and 
	 * assigned gotos are desugared because they may be part of a 
	 * logical IF, unknowingly.
	 */
	/*
	  if (instruction_block_p(i))
	  ParserError("makeStatement", "a block must have no label\n");
	*/

	/* FI, PJ: the "rice" phase does not handle labels on DO like 100 in:
	 *  100 DO 200 I = 1, N
	 *
	 * This should be trapped by "rice" when loops are checked to see
	 * if Allen/Kennedy's algorithm is applicable
	 */
	if (instruction_loop_p(i)) {
	    if(!get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
		user_warning("MakeStatement",
			     "DO loop reachable by GO TO via label %s "
			     "cannot be parallelized by PIPS\n",
			     entity_local_name(l));
	    }
	}

	if ((s = LabelToStmt(entity_name(l))) == statement_undefined) {
	    /* There is not forward reference to the this label. A new statement 
	     * can be safely allocated.
	     */
	    s = MakeNewLabelledStatement(l,i);
	}
	else {
	    /* A forward reference has been encountered and the corresponding
	     * statement has been allocated and has been referenced by at least
	     * one go to statement.
	     */

	    if(statement_instruction(s) != instruction_undefined) {
		    /* The CONTINUE slot can be re-used. It is likely to
		    be an artificial CONTINUE added to carry a
		    comment. Maybe it would be better to manage lab_I in a
		    more consistent way by resetting it as soon as it is
		    used. But I did not find the reset! */
		/*
		if(statement_continue_p(s)) {
		    free_instruction(statement_instruction(s));
		    statement_instruction(s) = instruction_undefined;
		    statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
		}
		else {
		    */
		    user_warning("MakeStatement", "Label %s may be used twice\n",
				 entity_local_name(l));
		    ParserError("MakeStatement", "Same label used twice\n");
		    /* } */
	    }
	    s = ReuseLabelledStatement(s, i);
	}
    }
    else {
	/* No actual label, no problem */
	s = make_statement(l,
			   (instruction_goto_p(i)||instruction_block_p(i))?
			   STATEMENT_NUMBER_UNDEFINED : get_statement_number(), //get_next_statement_number(),
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments,
			   i,NIL,NULL, empty_extensions (), make_synchronization_none());
    }

    return(s);
}



/* this function links the instruction i to the current block of
statements. if i is the last instruction of the block (i and the block
have the same label), the current block is popped from the stack. in
fortran, one instruction migth end more than one block. */

void 
LinkInstToCurrentBlock(i, number_it)
instruction i;
bool number_it;
{
    statement s;
    cons * pc;
    entity l = MakeLabel(lab_I);

    pips_debug(8, "Begin for instruction %s with label \"%s\"\n",
	       instruction_identification(i), &(lab_I[0]));

    /* A label cannot be used twice */
    reset_current_label_string();
 
    if (IsBlockStackEmpty())
	    ParserError("LinkInstToCurrentBlock", "no current block\n");

    if(instruction_block_p(i) && !entity_empty_label_p(l)) {
      /* a CONTINUE instruction must be added to carry the label,
         because blocks cannot be labelled */
	/*
      list ls = instruction_block(i);
      statement c = MakeStatement(l, make_continue_instruction());
      */
      /* The above continue is not a user statement an should not be numbered */
      /* OK, an argument could be added to MakeStatement()... */
	/* decrement_statement_number(); */

	/* instruction_block(i) = CONS (STATEMENT, c, ls); */
      if(number_it) {
	/* pips_assert("Why do you want to number a block?!?", false); */
	/* OK, let's be cool and ignore this request to save the caller a test */
	/* s = MakeStatement(entity_empty_label(), i); */
	  /* s = instruction_to_statement(i); */
	  ;
      }
      else{
	  /* s = instruction_to_statement(i); */
	  ;
      }
      s = MakeStatement(l, i);
    }
    else {
      s = MakeStatement(l, i);
    }

    if (iPrevComm != 0) {
	/* Because of labelled loop desugaring, new_i may be different from i */
	instruction new_i = statement_instruction(s);
	if(instruction_block_p(new_i)) {
	    statement fs = statement_undefined; // first statement of the block
	    statement ss = statement_undefined; // second statement, if it exist
	    statement cs = statement_undefined; // commented statement

	    /* Only desugared constructs such as labelled loop, computed go to or IO with
	     * error handling should produce blocks. Such blocks should be
	     * non-empty and not commented.
	     */
	    pips_assert("The block is non empty", !ENDP(instruction_block(new_i)));
	    /* Sometimes, we generate blocks with only one statement in it. E.g. alternate returns 
	    pips_assert("The block has at least two statements", !ENDP(CDR(instruction_block(new_i))));
	    */

	    fs = STATEMENT(CAR(instruction_block(new_i)));
	    /* For keeping pragma attached to a loop attached to it,
	       we have to find the loop instruction within the
	       block */
	    if(!ENDP(CDR(instruction_block(new_i)))) {
	      ss = STATEMENT(CAR(CDR(instruction_block(new_i))));

	      if(continue_statement_p(fs) && statement_loop_p(ss))
		cs = ss;
	      else
		cs = fs;
	    }
	    else {
		cs = fs;
	    }
	    /*
	    pips_assert("The first statement has no comments",
			statement_comments(cs) == empty_comments);
			*/
	    if(statement_comments(cs) != empty_comments) {
		user_log("Current comment of chosen statement: \"%s\"\n",
			 statement_comments(cs));
		user_log("Block comment to be carried by first statement: \"%s\"\n",
			 PrevComm);
		pips_internal_error("The first statement of the block should have no comments");
	    }

	    pips_assert("The chosen statement is not a block",
			!instruction_block_p(statement_instruction(cs)));

	    statement_comments(cs) = strdup(PrevComm);
	}
	else {
	    statement_comments(s) = strdup(PrevComm);
	}
	PrevComm[0] = '\0';
	iPrevComm = 0;
    }

    pc = CONS(STATEMENT, s, NULL);

    if (BlockStack[CurrentBlock-1].c == NULL) {
	instruction_block(BlockStack[CurrentBlock-1].i) = pc;
    }
    else {
	CDR(BlockStack[CurrentBlock-1].c) = pc;
    }
    BlockStack[CurrentBlock-1].c = pc;

    /* while i is the last instruction of the current block ... */
    while (BlockStack[CurrentBlock-1].l != NULL &&
	   strcmp(label_local_name(l), BlockStack[CurrentBlock-1].l) == 0)
		PopBlock();

    pips_debug(8, "End for instruction %s with label \"%s\"\n",
	       instruction_identification(i), label_local_name(l));
}		



/* this function creates an empty block */

instruction MakeEmptyInstructionBlock()
{
    return(make_instruction_block(NIL));
}



/* this function creates a simple Fortran statement such as RETURN,
CONTINUE, ... 

s is the name of the intrinsic function.

e is one optional argument (might be equal to expression_undefined). */

instruction 
MakeZeroOrOneArgCallInst(s, e)
char *s;
expression e;
{
    cons *l; /* la liste d'arguments */

    l = (e == expression_undefined) ? NIL : CONS(EXPRESSION, e, NIL);

    return(make_instruction(is_instruction_call, 
			    make_call(CreateIntrinsic(s), l)));
}



/* this function creates a goto instruction. n is the target label. */

instruction 
MakeGotoInst(n)
string n;
{
    entity l = entity_undefined;
    instruction i = instruction_undefined;

    l = MakeLabel(n);

    i = make_goto_instruction(l);

    return i;
}

/* In a "go to" instruction, the label does not appear explictly.
 * It is replaced by the statement to be jumped at.
 * If the statement carrying the label has been encountered before,
 * everything is fine. Else the target statement has to be synthesized
 * blindly ahead of time.
 */
instruction
make_goto_instruction(entity l)
{
  statement s = LabelToStmt(entity_name(l));
  instruction g = instruction_undefined;

  if (s == statement_undefined) {
    s = make_statement(l,
		       STATEMENT_NUMBER_UNDEFINED,
		       STATEMENT_ORDERING_UNDEFINED,
		       empty_comments,
		       instruction_undefined, NIL, NULL,
		       empty_extensions (), make_synchronization_none());
    NewStmt(l, s);
  }

  g = make_instruction(is_instruction_goto, s);

  return g;
}


instruction MakeComputedGotoInst(list ll, expression e)
{
    instruction inst = MakeAssignedOrComputedGotoInst(ll, e, false);

    return inst;
}

instruction MakeAssignedGotoInst(list ll, entity i)
{
    instruction inst;
    expression expr = entity_to_expression(i);

    DeclareVariable(i, type_undefined, NIL, storage_undefined, value_undefined);

    inst = MakeAssignedOrComputedGotoInst(ll, expr, true);

    return inst;
}

instruction 
MakeAssignedOrComputedGotoInst(list ll, expression ce, bool assigned)
{
  instruction ins = instruction_undefined;
  list cs = NIL;
  int l = 0;
  list cl = list_undefined;
  expression e = expression_undefined;
  syntax sce = expression_syntax(ce);
  statement s_init = statement_undefined;

  /* ce might have side effects */
  if(syntax_reference_p(sce)) {
    /* ce can be used several times without side effects */
    e = ce;
  }
  else if(syntax_call_p(sce)) {
    if(call_constant_p(syntax_call(sce))) {
      e = ce;
    }
    else {
      /* We cannot know yet if ce has side effects */
      /* expression_intrinsic_operation_p(ce): a user call may be hidden
	 at a lower level and some intrinsics may have side effects and
	 it might be more efficient not to recompute a complex
	 expression several times */
      /* Prefix starts with I to avoid an explicit declaration and a
         regeneration of declarations by the prettyprinter. */
      entity tmp = make_new_scalar_variable_with_prefix("ICG",
							get_current_module_entity(),
							make_basic(is_basic_int, (void*) 4));
      s_init = make_assign_statement(entity_to_expression(tmp), ce);

      e = entity_to_expression(tmp);
    }
  }
  else {
    pips_internal_error("No range expected", false);
  }
    

  for(l = gen_length(ll), cl = ll; !ENDP(cl); l--, POP(cl)) {
    string ln = STRING(CAR(cl));
    instruction g = MakeGotoInst(ln);
    expression cond = 
      MakeBinaryCall(
		     gen_find_tabulated(
					make_entity_fullname(TOP_LEVEL_MODULE_NAME,
							     EQUAL_OPERATOR_NAME), 
					entity_domain),
		     copy_expression(e),
		     int_to_expression(assigned? atoi(ln):l));
    /* Assigned GO TO: if the current label is not in the list, this is an error
     * in Fortran 90. ISO/IEC 1539 Section 8.2.4 page 108. Same in Fortran 77
     * standard, Section 11-2.
     */
    statement may_stop = (assigned && (cl==ll)) ?
      instruction_to_statement(MakeZeroOrOneArgCallInst(STOP_FUNCTION_NAME,
							expression_undefined))
      :
      make_empty_statement();
    instruction iif = 
      make_instruction(is_instruction_test,
		       make_test(cond,
				 instruction_to_statement(g),
				 may_stop));
    statement s = statement_undefined;

    s = instruction_to_statement(iif);

    /* Update the statement numbers of all possibly allocated statements */
    statement_number(s) = get_statement_number();
    if(stop_statement_p(may_stop))
      statement_number(may_stop) = get_statement_number();

    cs = CONS(STATEMENT, s, cs);
  }

  if(!statement_undefined_p(s_init))
    cs = CONS(STATEMENT, s_init, cs);

  /* MakeStatement won't increment the current statement number
   * because this is a block... so it has to be done here
   */
  //  (void) get_next_statement_number();
  ins = make_instruction_block(cs);

  (void) instruction_consistent_p(ins);

  /* FatalError("parser", "computed goto statement prohibited\n"); */

  return ins;
}

/* this function creates an affectation statement. 

   l is a reference (the left hand side).

   e is an expression (the right hand side).
*/

instruction 
MakeAssignInst(syntax l, expression e)
{
   expression lhs = expression_undefined;
   instruction i = instruction_undefined;

   if(syntax_reference_p(l)) {
     lhs = make_expression(l, normalized_undefined);
     i = make_assign_instruction(lhs, e);
   }
   else
   {
       if(syntax_call_p(l) &&
	  strcmp(entity_local_name(call_function(syntax_call(l))), 
		 SUBSTRING_FUNCTION_NAME) == 0) 
       {
	   list lexpr = CONS(EXPRESSION, e, NIL);
	   list asub = call_arguments(syntax_call(l));
	   
	   call_arguments(syntax_call(l)) = NIL;
	   free_syntax(l);
	   
	   i = make_instruction(is_instruction_call,
	      make_call(entity_intrinsic(ASSIGN_SUBSTRING_FUNCTION_NAME),
			gen_append(asub, lexpr)));
       }
       else
       {
	   if (syntax_call_p(l) && 
	     !value_symbolic_p(entity_initial(call_function(syntax_call(l)))))
	   {
	       if (get_bool_property("PARSER_EXPAND_STATEMENT_FUNCTIONS"))
	       {
		   /* Let us keep the statement function definition somewhere.
		    */
		   /* Preserve the current comments as well as the information
		      about the macro substitution */
		   statement stmt1 = make_continue_statement(entity_empty_label());
		   statement stmt2 = make_continue_statement(entity_empty_label());

		   pips_debug(5, "considering %s as a macro\n", 
			      entity_name(call_function(syntax_call(l))));

		   parser_add_a_macro(syntax_call(l), e);
		   statement_comments(stmt2) =
		       strdup(concatenate("C$PIPS STATEMENT FUNCTION ",
					  entity_local_name(call_function(syntax_call(l))),
					  " SUBSTITUTED\n", 0));
		   i = make_instruction_block(CONS(STATEMENT, stmt1,
						   CONS(STATEMENT, stmt2, NIL)));
	       }
	       else
	       {
		   /* FI: we stumble here when a Fortran macro is used. */
		   user_warning("MakeAssignInst", "%s() appears as lhs\n",
		       entity_local_name(call_function(syntax_call(l))));
		   ParserError("MakeAssignInst",
			       "bad lhs (function call or undeclared array)"
			       " or PIPS unsupported Fortran macro\n"
			       "you might consider switching the "
			       "PARSER_EXPAND_STATEMENT_FUNCTIONS property,\n"
			       "in the latter, at your own risk...\n");
	       }
	   }
	   else {
	       if(syntax_call_p(l)) {
		   /* FI: we stumble here when a Fortran PARAMETER is used as lhs. */
		   user_warning("MakeAssignInst", "PARAMETER %s appears as lhs\n",
		       entity_local_name(call_function(syntax_call(l))));
		   ParserError("MakeAssignInst",
			       "Illegal lhs\n");
	       }
	       else {
		   FatalError("MakeAssignInst", "Unexpected syntax tag\n");
	       }
	   }
       }
   }

   return i;
}

/* Update of the type returned by function f. nt must be a freshly
   allocated object. It is included in f's data structure */
void
update_functional_type_result(entity f, type nt)
{
  type ft = entity_type(f);
  type rt = type_undefined;

  //pips_assert("function type is functional", type_functional_p(ft));
  if(!type_functional_p(ft)) {
    /* The function is probably a formal parameter, its type is
       wrong. The return type is either void if it is called by CALL
       or its current implicit type. */
    if(storage_formal_p(entity_storage(f))) {
      pips_user_warning("Variable \"%s\" is a formal functional parameter\n",
			entity_user_name(f));
      ParserError(__FUNCTION__,
		  "Formal functional parameters are not yet supported\n");
    }
    else {
      pips_internal_error("Unexpected case");
    }
  }
  else {
    rt = functional_result(type_functional(ft));
  }

  pips_assert("result type is variable or unkown or void or undefined",
	      type_undefined_p(rt)
	      || type_unknown_p(rt)
	      || type_void_p(rt)
	      || type_variable_p(rt));

  pips_assert("new result type is variable or void",
	      type_void_p(nt)
	      ||type_variable_p(nt));

  free_type(rt);
  functional_result(type_functional(ft)) = nt;
}

void
update_functional_type_with_actual_arguments(entity e, list l)
{
  list pc = list_undefined;
  list pc2 = list_undefined;
  type t = type_undefined;
  functional ft = functional_undefined;

  pips_assert("update_functional_type_with_actual_arguments", !type_undefined_p(entity_type(e)));
  t = entity_type(e);
  pips_assert("update_functional_type_with_actual_arguments", type_functional_p(t));
  ft = type_functional(t);


  if( ENDP(functional_parameters(ft))) {
    /* OK, it is not safe: may be it's a 0-ary function */
    for (pc = l; pc != NULL; pc = CDR(pc)) {
      expression ae = EXPRESSION(CAR(pc));
      type t = type_undefined;
      parameter p = parameter_undefined;

      if(expression_reference_p(ae)) {
	reference r = expression_reference(ae);
	type tv = entity_type(reference_variable(r));

	if(type_functional_p(tv)) {
	  pips_user_warning("Functional actual argument %s found.\n"
			    "Functional arguments are not yet suported by PIPS\n",
			    entity_local_name(reference_variable(r)));
	}

	t = copy_type(tv);
      }
      else {
	basic b = basic_of_expression(ae);
	variable v = make_variable(b, NIL,NIL); 
	t = make_type(is_type_variable, v);
      }

      p = make_parameter(t,
			 MakeModeReference(),
			 make_dummy_unknown());
      functional_parameters(ft) = 
	gen_nconc(functional_parameters(ft),
		  CONS(PARAMETER, p, NIL));
    }
  }
  else if(get_bool_property("PARSER_TYPE_CHECK_CALL_SITES"))  {
    /* The pre-existing typing of e should match the new one */
    int i = 0;
    bool warning_p = false;

    for (pc = l, pc2 = functional_parameters(ft), i = 1;
	 !ENDP(pc) && !ENDP(pc2);
	 POP(pc), i++) {
      expression ae = EXPRESSION(CAR(pc));
      type at = type_undefined;
      type ft = parameter_type(PARAMETER(CAR(pc2)));
      type eft = type_varargs_p(ft)? type_varargs(ft) : ft;
      /* parameter p = parameter_undefined; */

      if(expression_reference_p(ae)) {
	reference r = expression_reference(ae);
	type tv = entity_type(reference_variable(r));

	if(type_functional_p(tv)) {
	  pips_user_warning("Functional actual argument %s found.\n"
			    "Functional arguments are not yet suported by PIPS\n",
			    entity_local_name(reference_variable(r)));
	}

	at = copy_type(tv);
      }
      else {
	basic b = basic_of_expression(ae);
	variable v = make_variable(b, NIL,NIL); 

	at = make_type(is_type_variable, v);
      }

      if((type_variable_p(eft)
	  && basic_overloaded_p(variable_basic(type_variable(eft))))
	 || type_equal_p(at, eft)) {
	/* OK */
	if(!type_varargs_p(ft)) 
	  POP(pc2);
      }
      else {
	user_warning("update_functional_type_with_actual_arguments",
		     "incompatible %d%s actual argument and type in call to %s "
		     "between lines %d and %d. Current type is not updated\n",
		     i, nth_suffix(i),
		     module_local_name(e), line_b_I, line_e_I);
	free_type(at);
	warning_p = true;
	break;
      }
      free_type(at);
    }

    if(!warning_p) {
      if(!(ENDP(pc) /* the actual parameter list must be exhausted */
	   && (ENDP(pc2) /* as well as the type parameter list */
	       || (ENDP(CDR(pc2)) /* unless the last type in the parameter list is a varargs */
		   && type_varargs_p(parameter_type(PARAMETER(CAR(pc2)))))))) {
	user_warning("update_functional_type_with_actual_arguments",
		     "inconsistent arg. list lengths for %s:\n"
		     " %d args according to type and %d actual arguments\n"
		     "between lines %d and %d. Current type is not updated\n",
		     module_local_name(e),
		     gen_length(functional_parameters(ft)), 
		     gen_length(l), line_b_I, line_e_I);
      }
    }
  }
}

/* this function creates a call statement. e is the called function. l
is the argument list, a list of expressions. */

instruction 
MakeCallInst(
    entity e, /* callee */
    cons * l  /* list of actual parameters */
)
{
    instruction i = instruction_undefined;
    list ar = get_alternate_returns();
    list ap = add_actual_return_code(l);
    storage s = entity_storage(e);
    bool ffp_p = false;
    entity fe = e;

    if(!storage_undefined_p(s)) {
	if(storage_formal_p(s)) {
	  ffp_p = true;
	    pips_user_warning("entity %s is a formal functional parameter\n",
			      entity_name(e));
	    /* ParserError("MakeCallInst",
			"Formal functional parameters are not supported "
			"by PIPS.\n"); */
	    /* FI: Before you can proceed to
	       update_functional_type_result(), you may have to fix
	       the type of e. Basically, if its type is not
	       functional, it should be made functional with result
	       void. I do not fix the problem in the parser because
	       tons of other problems are going to appear, at least
	       one for each PIPS analysis, starting with effects,
	       proper, cumulated, regions, transformers,
	       preconditions,... No quick fix, but a special effort
	       made after an explicit decision. */
	}
    }

    if(!ffp_p) {
      update_called_modules(e);

      /* The following assertion is no longer true when fucntions are
         passed as actual arguments. */
      /* pips_assert("e itself is returned",
	 MakeExternalFunction(e, MakeTypeVoid()) == e); */
      fe = MakeExternalFunction(e, MakeTypeVoid());
    }

    update_functional_type_result(fe, make_type(is_type_void,UU));
    update_functional_type_with_actual_arguments(fe, ap);

    if(!ENDP(ar)) {
	statement s = instruction_to_statement
	    (make_instruction(is_instruction_call, make_call(fe, ap)));

	statement_number(s) = get_statement_number();
	pips_assert("Alternate return substitution required\n", SubstituteAlternateReturnsP());
	i = generate_return_code_checks(ar);
	pips_assert("Must be a sequence", instruction_block_p(i));
	instruction_block(i) = CONS(STATEMENT,
				    s,
				    instruction_block(i));
    }
    else {
	i = make_instruction(is_instruction_call, make_call(fe, ap));
    }

    return i;
}



/* this function creates a do loop statement.

s is a reference to the do variable.

r is the range of the do loop.

l is the label of the last statement of the loop. */

void 
MakeDoInst(s, r, l)
syntax s;
range r;
string l;
{
    instruction ido, instblock_do;
    statement stmt_do;
    entity dovar, dolab;

    if (!syntax_reference_p(s))
	    FatalError("MakeDoInst", "function call as DO variable\n");

    if (reference_indices(syntax_reference(s)) != NULL)
	    FatalError("MakeDoInst", "variable reference as DO variable\n");

    dovar = reference_variable(syntax_reference(s));
    reference_variable(syntax_reference(s)) = entity_undefined;
    /* This free is not nice for the caller! Nor for the debugger. */
    free_syntax(s);

    dolab = MakeLabel((strcmp(l, "BLOCKDO") == 0) ? "" : l);

    instblock_do = MakeEmptyInstructionBlock();
    stmt_do = instruction_to_statement(instblock_do);

    if(get_bool_property("PARSER_LINEARIZE_LOOP_BOUNDS")) {
      normalized nl = NORMALIZE_EXPRESSION(range_lower(r));
      normalized nu = NORMALIZE_EXPRESSION(range_upper(r));
      normalized ni = NORMALIZE_EXPRESSION(range_increment(r));

      if(normalized_linear_p(nl) && normalized_linear_p(nu) &&  normalized_linear_p(ni)) {
	ido = make_instruction(is_instruction_loop,
			       make_loop(dovar, r, stmt_do, dolab,
					 make_execution(is_execution_sequential, 
							UU),
					 NIL));
      }
      else {
	/* Let's build a sequence with loop range assignments */
	instruction sido = make_instruction(is_instruction_loop,
					    make_loop(dovar, r, stmt_do, dolab,
						      make_execution(is_execution_sequential, 
								     UU),
						      NIL));
	list a = CONS(STATEMENT, instruction_to_statement(sido), NIL);
 
	if(!normalized_linear_p(ni)) {
	  entity nv = make_new_scalar_variable_with_prefix("INC_",
							   get_current_module_entity(),
							   make_basic(is_basic_int, (void*) 4));
	  instruction na = make_assign_instruction(entity_to_expression(nv),range_increment(r));
	  range_increment(r) = entity_to_expression(nv);
	  a = CONS(STATEMENT, instruction_to_statement(na), a);
	}
 
	if(!normalized_linear_p(nu)) {
	  entity nv = make_new_scalar_variable_with_prefix("U_",
							   get_current_module_entity(),
							   make_basic(is_basic_int, (void*) 4));
	  instruction na = make_assign_instruction(entity_to_expression(nv),range_upper(r));
	  range_upper(r) = entity_to_expression(nv);
	  a = CONS(STATEMENT, instruction_to_statement(na), a);
	}

	if(!normalized_linear_p(nl)) {
	  entity nv = make_new_scalar_variable_with_prefix("L_",
							   get_current_module_entity(),
							   make_basic(is_basic_int, (void*) 4));
	  instruction na = make_assign_instruction(entity_to_expression(nv),range_lower(r));
	  range_lower(r) = entity_to_expression(nv);
	  a = CONS(STATEMENT, instruction_to_statement(na), a);
	}
	ido = make_instruction_block(a);
      }
    }
    else {
      ido = make_instruction(is_instruction_loop,
			     make_loop(dovar, r, stmt_do, dolab,
				       make_execution(is_execution_sequential, 
						      UU),
				       NIL));
    }

    LinkInstToCurrentBlock(ido, true);
   
    PushBlock(instblock_do, l);
}

/* This function creates a while do loop statement.
 *
 * c is the loop condition
 * l is the label of the last statement of the loop.
 */

void 
MakeWhileDoInst(expression c, string l)
{
    instruction iwdo, instblock_do;
    statement stmt_do;
    entity dolab;
    expression cond = expression_undefined;

    if(!logical_expression_p(c)) {
      /* with the f77 compiler, this is equivalent to c.NE.0*/
      cond = MakeBinaryCall(entity_intrinsic(NON_EQUAL_OPERATOR_NAME),
			    c, int_to_expression(0));
      pips_user_warning("WHILE condition between lines %d and %d is not a logical expression.\n",
			line_b_I,line_e_I);
    }
    else {
      cond = c;
    }

    dolab = MakeLabel((strcmp(l, "BLOCKDO") == 0) ? "" : l);

    instblock_do = MakeEmptyInstructionBlock();
    stmt_do = instruction_to_statement(instblock_do);

    iwdo = make_instruction(is_instruction_whileloop,
			   make_whileloop(cond, stmt_do, dolab,make_evaluation_before()));

    LinkInstToCurrentBlock(iwdo, true);
   
    PushBlock(instblock_do, l);
}

expression fix_if_condition(expression e)
{
  expression cond = expression_undefined;

  if(!logical_expression_p(e)) {
    /* with the f77 compiler, this is equivalent to e.NE.0 if e is an
       integer expression. */
    if(integer_expression_p(e)) {
      cond = MakeBinaryCall(entity_intrinsic(NON_EQUAL_OPERATOR_NAME),
			    e, int_to_expression(0));
      pips_user_warning("IF condition between lines %d and %d is not a logical expression.\n",
			line_b_I,line_e_I);
    }
    else {
      ParserError("MakeBlockIfInst", "IF condition is neither logical nor integer.\n");
    }
  }
  else {
    cond = e;
  }
  return cond;
}

/* this function creates a logical if statement. the true part of the
 * test is a block with only one instruction (i), and the false part is an
 * empty block.  
 *
 * Modifications:
 *  - there is no need for a block in the true branch, any statement can do
 *  - there is no need for a CONTINUE statement in the false branch, an empty block
 *    is plenty
 *  - MakeStatement() cannot be used for the true and false branches because it
 *    disturbs the statement numering
 */

instruction 
MakeLogicalIfInst(e, i)
expression e;
instruction i;
{
  /* It is not easy to number bt because Yacc reduction order does not help... */
    statement bt = instruction_to_statement(i);
    statement bf = make_empty_block_statement();
    expression cond = fix_if_condition(e);
    instruction ti = make_instruction(is_instruction_test, 
				      make_test(cond, bt, bf));
				      
    if (i == instruction_undefined)
	    FatalError("MakeLogicalIfInst", "bad instruction\n");

    /* Instruction i should not be a block, unless:
     * - an alternate return 
     * - a computed GO TO
     * - an assigned GO TO
     * has been desugared.
     *
     * If the logical IF is labelled, the label has been stolen by the
     * first statement in the block. This shows that label should only
     * be affected by MakeStatement and not by desugaring routines.
     */
    if(instruction_block_p(i)) {
	list l = instruction_block(i);
	/* statement first = STATEMENT(CAR(l)); */
	/* Only the alternate return case assert:
	pips_assert("Block of two instructions or call with return code checks",
		    (gen_length(l)==2 && assignment_statement_p(first))
		    ||
		    (statement_call_p(first))
	    );
	    */
	MAP(STATEMENT, s, {
	    statement_number(s) = get_statement_number ();
	}, l);
    }
    else {
	statement_number(bt) = get_statement_number();
    }

    return ti;
}

/* this function transforms an arithmetic if statement into a set of
regular tests. long but easy to understand without comments.

e is the test expression. e is inserted in the instruction returned
(beware of sharing)

l1, l2, l3 are the three labels of the original if statement.

      IF (E) 10, 20, 30

becomes

      IF (E .LT. 0) THEN
         GOTO 10
      ELSE
         IF (E .EQ. 0) THEN
	    GOTO 20
	 ELSE
	    GOTO 30
	 ENDIF
      ENDIF

*/

instruction 
MakeArithmIfInst(e, l1, l2, l3)
expression e;
string l1, l2, l3;
{
    expression e1, e2;
    statement s1, s2, s3, s;
    instruction ifarith = instruction_undefined;

    /* FI: Should be improved by testing equality between l1, l2 and l3
     * Cases observed:
     *  l1 == l2
     *  l2 == l3
     *  l1 == l3
     * Plus, just in case, l1==l2==l3
     */

    if(strcmp(l1,l2)==0) {
	if(strcmp(l2,l3)==0) {
	    /* This must be quite unusual, but the variables in e have to be dereferenced
	     * to respect the use-def chains, e may have side effects,...
	     *
	     * If the optimizer is very good, the absolute value of e 
	     * should be checked positive?
	     */
	    e1 = MakeUnaryCall(CreateIntrinsic("ABS"), e);
	    e2 = MakeBinaryCall(CreateIntrinsic(".GE."), 
				e1, int_to_expression(0));

	    s1 = instruction_to_statement(MakeGotoInst(l1));
	    s2 = make_empty_block_statement();

	    ifarith = make_instruction(is_instruction_test, 
				       make_test(e2,s1,s2));
	}
	else {
	    e1 = MakeBinaryCall(CreateIntrinsic(".LE."), 
				e, int_to_expression(0));

	    s1 = instruction_to_statement(MakeGotoInst(l1));
	    s3 = instruction_to_statement(MakeGotoInst(l3));

	    ifarith = make_instruction(is_instruction_test, 
				       make_test(e1,s1,s3));
	}
    }
    else if(strcmp(l1,l3)==0) {
	e1 = MakeBinaryCall(CreateIntrinsic(".EQ."), 
			    e, int_to_expression(0));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));

	ifarith = make_instruction(is_instruction_test, 
						      make_test(e1,s2,s1));
    }
    else if(strcmp(l2,l3)==0) {
	e1 = MakeBinaryCall(CreateIntrinsic(".LT."), 
			    e, int_to_expression(0));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));

	ifarith = make_instruction(is_instruction_test, 
				   make_test(e1,s1,s2));
    }
    else {
	/* General case */
	e1 = MakeBinaryCall(CreateIntrinsic(".LT."), 
			    e, int_to_expression(0));
	e2 = MakeBinaryCall(CreateIntrinsic(".EQ."), 
			    copy_expression(e), int_to_expression(0));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));
	s3 = instruction_to_statement(MakeGotoInst(l3));

	s = instruction_to_statement(make_instruction(is_instruction_test, 
						      make_test(e2,s2,s3)));
	statement_number(s) = get_statement_number();

	ifarith = make_instruction(is_instruction_test, make_test(e1,s1,s));
    }

    return ifarith;
}

/* this function and the two next ones create a block if statement. the
true and the else part of the test are two empty blocks. e is the test
expression.

the true block is pushed on the stack. it will contain the next
statements, and will end with a else statement or an endif statement.

if a else statement is reached, the true block is popped and the false
block is pushed to gather the false part statements. if no else
statement is found, the true block will be popped with the endif
statement and the false block will remain empty. */

void 
MakeBlockIfInst(e,elsif)
expression e;
int elsif;
{  
    instruction bt, bf, i;
    expression cond = fix_if_condition(e);

    bt = MakeEmptyInstructionBlock();
    bf = MakeEmptyInstructionBlock();

    i = make_instruction(is_instruction_test,
			 make_test(cond,
				   MakeStatement(MakeLabel(""), bt),
				   MakeStatement(MakeLabel(""), bf)));

    LinkInstToCurrentBlock(i, true);
   
    PushBlock(bt, "ELSE");
    BlockStack[CurrentBlock-1].elsifs = elsif ;
}

/* This function is used to handle either an ELSE or an ELSEIF construct */

int 
MakeElseInst(bool is_else_p)
{
    statement if_stmt;
    test if_test;
    int elsifs;
    bool has_comments_p = (iPrevComm != 0);

    if(CurrentBlock==0) {
	/* No open block can be closed by this ELSE */
	ParserError("MakeElseInst", "unexpected ELSE statement\n");
    }

    elsifs = BlockStack[CurrentBlock-1].elsifs ;

    if (strcmp("ELSE", BlockStack[CurrentBlock-1].l))
	ParserError("MakeElseInst", "block if statement badly nested\n");

    if (has_comments_p) {
	/* Generate a CONTINUE to carry the comments but not the label
           because the ELSE is not represented in the IR and cannot carry
           comments. The ELSEIF is transformed into an IF which can carry
           comments and label but the prettyprint of structured code is
           nicer if the comments are carried by a CONTINUE in the previous
           block. Of course, this is not good for unstructured code since
           comments end up far from their intended target or attached to
	   a dead CONTINUE if the previous block ends up with a GO TO.

	   The current label is temporarily hidden. */
	string ln = strdup(get_current_label_string());
	reset_current_label_string();
	LinkInstToCurrentBlock(make_continue_instruction(), false);
	set_current_label_string(ln);
	free(ln);
    }

    (void) PopBlock();

    if_stmt = STATEMENT(CAR(BlockStack[CurrentBlock-1].c));

    if (! instruction_test_p(statement_instruction(if_stmt)))
	FatalError("MakeElseInst", "no block if statement\n");

    if_test = instruction_test(statement_instruction(if_stmt));

    PushBlock(statement_instruction(test_false(if_test)), "ENDIF");

    if (is_else_p && !empty_current_label_string_p()) {
	/* generate a CONTINUE to carry the label because the ELSE is not
           represented in the IR */
	LinkInstToCurrentBlock(make_continue_instruction(), false);
    }

    return( BlockStack[CurrentBlock-1].elsifs = elsifs ) ;
}

void 
MakeEndifInst()
{
    int elsifs = -1;

    if(CurrentBlock==0) {
	ParserError("MakeEndifInst", "unexpected ENDIF statement\n");
    }

    if (iPrevComm != 0) {
	/* generate a CONTINUE to carry the comments */
	LinkInstToCurrentBlock(make_continue_instruction(), false);
    }

    if (BlockStack[CurrentBlock-1].l != NULL &&
	strcmp("ELSE", BlockStack[CurrentBlock-1].l) == 0) {
	elsifs = MakeElseInst(true);
	LinkInstToCurrentBlock(make_continue_instruction(), false);
    }
    if (BlockStack[CurrentBlock-1].l == NULL ||
	strcmp("ENDIF", BlockStack[CurrentBlock-1].l)) {
	ParserError("MakeEndifInst", "block if statement badly nested\n");
    }
    else {
	elsifs = BlockStack[CurrentBlock-1].elsifs ;
    }
    pips_assert( "MakeEndifInst", elsifs >= 0 ) ;

    do {
	(void) PopBlock();
    } while( elsifs-- != 0 ) ;
}

void 
MakeEnddoInst()
{
    if(CurrentBlock<=1) {
	ParserError("MakeEnddoInst", "Unexpected ENDDO statement\n");
    }

    if (strcmp("BLOCKDO", BlockStack[CurrentBlock-1].l)
	&&strcmp(lab_I, BlockStack[CurrentBlock-1].l))
	    ParserError("MakeEnddoInst", "block do statement badly nested\n");

    /*LinkInstToCurrentBlock(MakeZeroOrOneArgCallInst("ENDDO", 
						    expression_undefined));*/
    /* Although it is not really an instruction, the ENDDO statement may 
     * carry comments and be labelled when closing a DO label structure.
     */
    LinkInstToCurrentBlock(make_continue_instruction(), false);

    /* An unlabelled ENDDO can only close one loop. This cannot be
     * performed by LinkInstToCurrentBlock().
     */
    if (strcmp("BLOCKDO", BlockStack[CurrentBlock-1].l)==0)
      (void) PopBlock();
}

string 
NameOfToken(token)
int token;
{
    string name;

    switch (token) {
      case TK_BUFFERIN: 
	name = "BUFFERIN"; 
	break;
      case TK_BUFFEROUT: 
	name = "BUFFEROUT"; 
	break;
      case TK_INQUIRE: 
	name = "INQUIRE"; 
	break;
      case TK_OPEN: 
	name = "OPEN"; 
	break;
      case TK_CLOSE: 
	name = "CLOSE"; 
	break;
      case TK_PRINT: 
	name = "PRINT"; 
	break;
      case TK_READ: 
	name = "READ"; 
	break;
      case TK_REWIND: 
	name = "REWIND"; 
	break;
      case TK_WRITE: 
	name = "WRITE"; 
	break;
      case TK_ENDFILE: 
	name = "ENDFILE"; 
	break;
      case TK_BACKSPACE: 
	name = "BACKSPACE"; 
	break;
      default:
	FatalError("NameOfToken", "unknown token\n");
	name = string_undefined; /* just to avoid a gcc warning */
	break;
    }

    return(name);
}

/* Generate a test to jump to l if flag f is TRUE
 * Used to implement control effects of IO's due to ERR= and END=.
 *
 * Should not use MakeStatement() directly or indirectly to avoid
 * counting these pseudo-instructions
 */
statement 
make_check_io_statement(string n, expression u, entity l)
{
    entity a = FindEntity(IO_EFFECTS_PACKAGE_NAME, n);
    reference r = make_reference(a, CONS(EXPRESSION, u, NIL));
    expression c = reference_to_expression(r);
    statement b = instruction_to_statement(make_goto_instruction(l));
    instruction t = make_instruction(is_instruction_test,
				     make_test(c, b, make_empty_block_statement()));
    statement check = instruction_to_statement(t);

    statement_consistent_p(check);

    return check;
}

/* this function creates an IO statement. keyword indicates which io
statement is to be built (READ, WRITE, ...).

lci is a list of 'control specifications'. its has the following format:

        ("UNIT=", 6, "FMT=", "*", "RECL=", 80, "ERR=", 20)

lio is the list of expressions to write or references to read. */

instruction 
MakeIoInstA(keyword, lci, lio)
int keyword;
cons *lci;
cons *lio;
{
  cons *l;
  /* The composite IO with potential branches for ERR and END */
  instruction io = instruction_undefined;
  /* The pure io itself */
  instruction io_call = instruction_undefined;
  /* virtual tests to implement ERR= and END= clauses */
  statement io_err = statement_undefined;
  statement io_end = statement_undefined;
  expression unit = expression_undefined;

  for (l = lci; l != NULL; l = CDR(CDR(l))) {
    syntax s1;
    entity e1;

    s1 = expression_syntax(EXPRESSION(CAR(l)));

    e1 = call_function(syntax_call(s1));

    if (strcmp(entity_local_name(e1), "UNIT=") == 0) {
      if( ! expression_undefined_p(unit) )
	free_expression(unit);
      unit = EXPRESSION(CAR(CDR(l)));
    }
  }

  /* we scan the list of specifications to detect labels (such as in
     ERR=20, END=30, FMT=50, etc.), that were stored as integer constants
     (20, 30, 50) and that must be replaced by labels (_20, _30, _50). */
  for (l = lci; l != NULL; l = CDR(CDR(l))) {
    syntax s1, s2;
    entity e1, e2;

    s1 = expression_syntax(EXPRESSION(CAR(l)));
    s2 = expression_syntax(EXPRESSION(CAR(CDR(l))));

    pips_assert("syntax is a call", syntax_call_p(s1));
    e1 = call_function(syntax_call(s1));
    pips_assert("value is constant", value_constant_p(entity_initial(e1)));
    pips_assert("constant is not int (thus litteral or call)", 
		!constant_int_p(value_constant(entity_initial(e1))));

    if (strcmp(entity_local_name(e1), "ERR=") == 0 || 
	strcmp(entity_local_name(e1), "END=") == 0 ||
	strcmp(entity_local_name(e1), "FMT=") == 0) {
      if (syntax_call_p(s2)) {
	e2 = call_function(syntax_call(s2));
	if (value_constant_p(entity_initial(e2))) {
	  if (constant_int_p(value_constant(entity_initial(e2)))) {
	    /* here is a label */
	    call_function(syntax_call(s2)) = 
	      MakeLabel(entity_local_name(e2));
	  }
	}
	e2 = call_function(syntax_call(s2));
	if (strcmp(entity_local_name(e1), "FMT=") != 0
	    && expression_undefined_p(unit)) {
	  /* UNIT is not defined for INQUIRE (et least)
	   * Let's use LUN 0 by default for END et ERR.
	   */
	  unit = int_to_expression(0);
	}
	if (strcmp(entity_local_name(e1), "ERR=") == 0) {
	  io_err = make_check_io_statement(IO_ERROR_ARRAY_NAME, unit, e2);
	}
	else if (strcmp(entity_local_name(e1), "END=") == 0) {
	  io_end = make_check_io_statement(IO_EOF_ARRAY_NAME, unit, e2);
	}
	else {
	  //free_expression(unit);
	  ;
    }
      }
    }
  }

  /*
    for (l = lci; CDR(l) != NULL; l = CDR(l)) ;

    CDR(l) = lio;
    l = lci;
  */

  lci = gen_nconc(lci, lio);

  io_call = make_instruction(is_instruction_call,
			     make_call(CreateIntrinsic(NameOfToken(keyword)),
				       lci));

  if(statement_undefined_p(io_err) && statement_undefined_p(io_end)) {
    io = io_call;
  }
  else {
    list ls = NIL;
    if(!statement_undefined_p(io_err)) {
      ls = CONS(STATEMENT, io_err, ls);
    }
    if(!statement_undefined_p(io_end)) {
      ls = CONS(STATEMENT, io_end, ls);
    }
    ls = CONS(STATEMENT, MakeStatement(entity_empty_label(), io_call), ls);
    io = make_instruction(is_instruction_sequence, make_sequence(ls));
    instruction_consistent_p(io);
  }
    
  return io;
}



/* this function creates a BUFFER IN or BUFFER OUT io statement. this is
not ansi fortran.

e1 is the logical unit.

nobody known the exact meaning of e2

e3 et e4 are references that indicate which variable elements are to be
buffered in or out. */

instruction 
MakeIoInstB(keyword, e1, e2, e3, e4)
int keyword;
expression e1, e2, e3, e4;
{
    cons * l;

    l = CONS(EXPRESSION, e1, 
	     CONS(EXPRESSION, e2,
		  CONS(EXPRESSION, e3,
		       CONS(EXPRESSION, e4, NULL))));

    return(make_instruction(is_instruction_call,
			    make_call(CreateIntrinsic(NameOfToken(keyword)),
				      l)));
}

instruction 
MakeSimpleIoInst1(int keyword, expression unit)
{
    instruction inst = instruction_undefined;
    expression std, format, unite;
    cons * lci;

    switch(keyword) {
    case TK_READ:
    case TK_PRINT:
	std = MakeNullaryCall(CreateIntrinsic
			      (LIST_DIRECTED_FORMAT_NAME));
	unite = MakeCharacterConstantExpression("UNIT=");
	format = MakeCharacterConstantExpression("FMT=");

	lci = CONS(EXPRESSION, unite,
		   CONS(EXPRESSION, std,
			CONS(EXPRESSION, format,
			     CONS(EXPRESSION, unit, NULL))));
	/* Functionally PRINT is a special case of WRITE */
	inst = MakeIoInstA((keyword==TK_PRINT)?TK_WRITE:TK_READ,
			 lci, NIL);
	break;
    case TK_WRITE:
    case TK_OPEN:
    case TK_CLOSE:
    case TK_INQUIRE:
	ParserError("Syntax",
		    "Illegal syntax in IO statement, "
		    "Parentheses and arguments required");
    case TK_BACKSPACE:
    case TK_REWIND:
    case TK_ENDFILE:
	unite = MakeCharacterConstantExpression("UNIT=");
	lci = CONS(EXPRESSION, unite,
		   CONS(EXPRESSION, unit, NULL));
	inst = MakeIoInstA(keyword, lci, NIL);
	break;
    default:
	ParserError("Syntax","Unexpected token in IO statement");
    }
    return inst;
}

instruction 
MakeSimpleIoInst2(int keyword, expression f, list io_list)
{
    instruction inst = instruction_undefined;
    expression std, format, unite;
    list cil;

    switch(keyword) {
    case TK_READ:
    case TK_PRINT:
	std = MakeNullaryCall(CreateIntrinsic
			      (LIST_DIRECTED_FORMAT_NAME));
	unite = MakeCharacterConstantExpression("UNIT=");
	format = MakeCharacterConstantExpression("FMT=");

	cil = CONS(EXPRESSION, unite,
		   CONS(EXPRESSION, std,
			CONS(EXPRESSION, format,
			     CONS(EXPRESSION, f, NULL))));
	inst = MakeIoInstA((keyword==TK_PRINT)?TK_WRITE:TK_READ,
			   cil, io_list);
	break;
    case TK_WRITE:
    case TK_OPEN:
    case TK_CLOSE:
    case TK_INQUIRE:
    case TK_BACKSPACE:
    case TK_REWIND:
    case TK_ENDFILE:
	ParserError("Syntax",
		    "Illegal syntax in IO statement, Parentheses are required");
    default:
	ParserError("Syntax","Unexpected token in IO statement");
    }
    return inst;
}


/* Are we in the declaration or in the executable part?
 * Have we seen a FORMAT statement before an executable statement?
 * For more explanation, see check_first_statement() below.
 */

static int seen = false;
static int format_seen = false;
static int declaration_lines = -1;

/* Well, some constant defined in reader.c
 * and not deserving a promotion in syntax-local.h
 */
#define UNDEF (-2)

void 
reset_first_statement()
{
    seen = false;
    format_seen = false;
    declaration_lines = -1;
}

void
set_first_format_statement()
{
    if(!format_seen && !seen) {
	format_seen = true;
	/* declaration_lines = line_b_I-1; */
	    debug(8, "set_first_format_statement", "line_b_C=%d, line_b_I=%d\n",
		  line_b_C, line_b_I);
	declaration_lines = (line_b_C!=UNDEF)?line_b_C-1:line_b_I-1;
    }
}

bool
first_executable_statement_seen()
{
    return seen;
}

bool
first_format_statement_seen()
{
    return format_seen;
}

void
check_in_declarations()
{
    if(seen) {
	ParserError("Syntax", 
		    "Declaration appears after executable statement");
    }
    else if(format_seen && !seen) {
	/* A FORMAT statement has been found in the middle of the declarations */
	if(!get_bool_property("PRETTYPRINT_ALL_DECLARATIONS")) {
	    pips_user_warning("FORMAT statement within declarations. In order to "
			      "analyze this code, "
			      "please set property PRETTYPRINT_ALL_DECLARATIONS "
			      "or move this FORMAT down in executable code.\n");
	    ParserError("Syntax", "Source cannot be parsed with current properties");
	}
    }
}

/* This function is called each time an executable statement is encountered
 * but is effective the first time only.
 *
 * It mainly copies the declaration text in the symbol table because it is
 * impossible (very difficult) to reproduce it in a user-friendly manner.
 *
 * The declaration text stops at the first executable statement or at the first
 * FORMAT statement.
 */
void 
check_first_statement()
{
    int line_start = true;
    int in_comment = false;
    int out_of_constant_string = true;
    int in_constant_string = false;
    int end_of_constant_string = false;
    char string_sep = '\000';

    if (! seen) 
    {
	FILE *fd;
	int cpt = 0, ibuffer = 0, c;

	/* dynamic local buffer
	 */
	int buffer_size = 1000;
	char * buffer = (char*) malloc(buffer_size);
	pips_assert("malloc ok", buffer);

	seen = true;
	
	/* we must read the input file from the begining and up to the 
	   line_b_I-1 th line, and the texte read must be stored in buffer */

	if(!format_seen) {
	    /* declaration_lines = line_b_I-1; */
	    debug(8, "check_first_statement", "line_b_C=%d, line_b_I=%d\n",
		  line_b_C, line_b_I);
	    declaration_lines = (line_b_C!=UNDEF)?line_b_C-1:line_b_I-1;
	}

	fd = safe_fopen(CurrentFN, "r");
	while ((c = getc(fd)) != EOF) {
	    if(line_start == true)
		in_comment = strchr(START_COMMENT_LINE,c) != NULL;
	    /* buffer[ibuffer++] = in_comment? c : toupper(c); */
	    if(in_comment) {
	      buffer[ibuffer++] =  c;
	    }
	    else {
	      /* Constant strings must be taken care of */
	      if(out_of_constant_string) {
		if(c=='\'' || c == '"') {
		  string_sep = c;
		  out_of_constant_string = false;
		  in_constant_string = true;
		  buffer[ibuffer++] =  c;
		} 
		else {
		  buffer[ibuffer++] =  toupper(c);
		}
	      }
	      else
		if(in_constant_string) {
		  if(c==string_sep) {
		    in_constant_string = false;
		    end_of_constant_string = true;
		  }
		  buffer[ibuffer++] =  c;
		}
		else 
		  if(end_of_constant_string) {
		    if(c==string_sep) {
		      in_constant_string = true;
		      end_of_constant_string = false;
		      buffer[ibuffer++] =  c;
		    }
		    else {
		      out_of_constant_string = true;
		      end_of_constant_string = false;
		      buffer[ibuffer++] =  toupper(c);
		    }
		  }
	    }

	    if (ibuffer >= buffer_size-10)
	    {
		pips_assert("buffer initialized", buffer_size>0);
		buffer_size*=2;
		buffer = (char*) realloc(buffer, buffer_size);
		pips_assert("realloc ok", buffer);
	    }

	    if (c == '\n') {
		cpt++;
		line_start = true;
		in_comment = false;
	    }
	    else {
		line_start = false;
	    }

	    if (cpt == declaration_lines)
		break;
	}
	safe_fclose(fd, CurrentFN);
	buffer[ibuffer++] = '\0';
	/* Standard version */
	code_decls_text(EntityCode(get_current_module_entity())) = buffer;
	buffer = NULL;
	/* For Cathare-2, get rid of 100 to 200 MB of declaration text: */
	/*
	   code_decls_text(EntityCode(get_current_module_entity())) = strdup("");
	   free(buffer);
	*/
	/* strdup(buffer); */
	/* free(buffer), buffer=NULL; */

	/* kill the first statement's comment because it's already
	   included in the declaration text */
	/* FI: I'd rather keep them together! */
	/*
	PrevComm[0] = '\0';
	iPrevComm = 0;
	*/
	/*
	Comm[0] = '\0';
	iComm = 0;
	*/

	/* clean up the declarations */
	/* Common sizes are not yet known because ComputeAddresses() has not been called yet */
	/* update_common_sizes(); */

	/* It might seem logical to perform these calls from EndOfProcedure()
	 * here. But at least ComputeAddresses() is useful for implictly 
	 * declared variables. 
	 * These calls are better located in EndOfProcedure().
	 */
	/*
	 * UpdateFunctionalType(FormalParameters);
	 *
	 * ComputeEquivalences();
	 * ComputeAddresses();
	 *
	 * check_common_layouts(get_current_module_entity());
	 *
	 * SaveChains();
	 */
    }
}
