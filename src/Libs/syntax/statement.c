/*
 * $Id$
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "parser_private.h"
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

/* to produce statement numbers */
static int stat_num = 1;
static bool skip_num = FALSE ;

void 
reset_statement_number()
{
    stat_num = 1;
    skip_num = FALSE;
}

int
get_next_statement_number()
{
  int next = stat_num;

  stat_num = skip_num? stat_num+2 : stat_num+1;
  skip_num = FALSE;

  return next;
}

int
look_at_next_statement_number()
{
  return stat_num;
}

int
get_future_statement_number()
{
  int next = stat_num+1;
  pips_assert("skip_num must be false",!skip_num);
  skip_num = TRUE;
  return next;
}

void
decrement_statement_number()
{
  stat_num--;
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
    int MustStop = FALSE;

    for (i = 0; i < CurrentStmt; i++) {
	statement s = StmtHeap_buffer[i].s;
	if (statement_instruction(s) == instruction_undefined) {
	    MustStop = TRUE;
	    user_log("CheckAndInitializeStmt %s\n", 
		     entity_name(statement_label(s)));
	}
    }

    if (MustStop) {
	ParserError("CheckAndInitializeStmt", "Undefined labels\n");
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

There does not seem to be any limit on the nesting level in Fortran standard.
MAXBLOCK is set to "large" value for our users.
*/

#define MAXBLOCK 100

typedef struct block {
	instruction i; /* the instruction that contains this block */
	string l;      /* the label that will end this block */
	cons * c;      /* the list of statements contained in this block */
	int elsifs ;
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



/* this functions creates a label. LABEL_PREFIX is added to its name, for
integer constants and labels not to have the name space. */

entity 
MakeLabel(s)
string s;
{
	entity l;
	static char *name = NULL ;

	if( name == NULL ) {
	    name = (char *)malloc( LABEL_SIZE+strlen(LABEL_PREFIX)+1 ) ;
	}
	debug(5, "MakeLabel", "%s\n", s);

	strcpy(name, LABEL_PREFIX);
	strcat(name, s);	

	l = FindOrCreateEntity( (strcmp( name, LABEL_PREFIX )==0) ? 
		        TOP_LEVEL_MODULE_NAME :
		        CurrentPackage, name);

	if (entity_type(l) == type_undefined) {
	    debug(5, "MakeLabel", "%s\n", name);
	    entity_type(l) = MakeTypeStatement();
	    entity_storage(l) = MakeStorageRom();
	    entity_initial(l) = make_value(is_value_constant,
					   MakeConstantLitteral());
	}
	return(l);
}



/* this function makes a statement. l is the label and i the
 * instruction. we make sure that the label is not declared twice.
 *
 * GO TO statement are numbered like other statements although they
 * are destroyed by the controlizer. To be changed.
 */

statement 
MakeStatement(l, i)
entity l;
instruction i;
{
  statement s;

  debug(5, "MakeStatement", "%s\n", entity_name(l));

  pips_assert("MakeStatement", type_statement_p(entity_type(l)));
  pips_assert("MakeStatement", storage_rom_p(entity_storage(l)));
  pips_assert("MakeStatement", value_constant_p(entity_initial(l)));
  pips_assert("MakeStatement", 
	      constant_litteral_p(value_constant(entity_initial(l))));

  if (!entity_empty_label_p(l)) {
    if (instruction_block_p(i))
      ParserError("makeStatement", "a block must have no label\n");

    /* FI, PJ: the "rice" phase does not handle labels on DO like 100 in:
     *  100 DO 200 I = 1, N
     *
     * This should be trapped by "rice" when loops are checked to see
     * if Allen/Kennedy's algorithm is applicable
     */
    if (instruction_loop_p(i)) {
      if(!get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
	user_warning("MakeStatement",
		     "DO loop reachable by GO TO via label %s cannot be parallelized by PIPS\n",
		     entity_local_name(l));
      }
    }

    if ((s = LabelToStmt(entity_name(l))) == statement_undefined) {
      if(instruction_loop_p(i) && get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
	statement c = make_continue_statement(l);
	statement ls = instruction_to_statement(i);

	statement_number(ls) = get_next_statement_number();
	NewStmt(l, c);
	s = make_block_statement(CONS(STATEMENT,c,
				      CONS(STATEMENT, ls, NIL)));
      }
      else {
	s = make_statement(l, 
			   (instruction_goto_p(i)||instruction_block_p(i))?
			   STATEMENT_NUMBER_UNDEFINED : get_next_statement_number(),
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments,
			   i);
	NewStmt(l, s);
      }
    }
    else {
      if(statement_instruction(s) != instruction_undefined) {
	user_warning("MakeStatement", "Label %s may be used twice\n",
		     entity_local_name(l));
	/* FI: commented out to avoid useless user triggered core dumps */
	/*
	  pips_assert("MakeStatement", 
	  statement_instruction(s) == instruction_undefined);
	  */
	if(statement_instruction(s) != instruction_undefined) {
	  ParserError("MakeStatement", "Same label used twice\n");
	}
      }
      pips_assert("Should have no number", 
		  statement_number(s)==STATEMENT_NUMBER_UNDEFINED);

      if(instruction_loop_p(i) && get_bool_property("PARSER_SIMPLIFY_LABELLED_LOOPS")) {
	statement c = make_continue_statement(l);
	statement ls = instruction_to_statement(i);

	statement_number(ls) = get_next_statement_number();
	statement_instruction(s) = make_instruction(is_instruction_sequence,
						    make_sequence(CONS(STATEMENT,c,
								       CONS(STATEMENT, ls, NIL))));
      }
      else {
	statement_instruction(s) = i;
	/* 
	statement_number(s) = (instruction_goto_p(i)||instruction_block_p(i))?
	  STATEMENT_NUMBER_UNDEFINED : get_next_statement_number();
	  */
	/* Let's number labelled GOTO because a CONTINUE is derived later from them */
	statement_number(s) = (instruction_block_p(i))?
	  STATEMENT_NUMBER_UNDEFINED : get_next_statement_number();
      }
    }
  }
  else {
    s = make_statement(l, 
		       (instruction_goto_p(i)||instruction_block_p(i))?
		         STATEMENT_NUMBER_UNDEFINED : get_next_statement_number(),
		       STATEMENT_ORDERING_UNDEFINED,
		       empty_comments, 
		       i);
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
    entity l = MakeLabel(strdup(lab_I));
 
    if (IsBlockStackEmpty())
	    ParserError("LinkInstToCurrentBlock", "no current block\n");

    if(instruction_block_p(i) && !entity_empty_label_p(l)) {
      /* a CONTINUE instruction must be added to carry the label,
         because blocks cannot be labelled */
      list ls = instruction_block(i);
      statement c = MakeStatement(l, make_continue_instruction());

      /* The above continue is not a user statement an should not be numbered */
      /* OK, an argument could be added to MakeStatement()... */
      decrement_statement_number();

      instruction_block(i) = CONS (STATEMENT, c, ls);
      if(number_it) {
	/* pips_assert("Why do you want to number a block?!?", FALSE); */
	/* OK, let's be cool and ignore this request to save the caller a test */
	/* s = MakeStatement(entity_empty_label(), i); */
	s = instruction_to_statement(i);
      }
      else
	s = instruction_to_statement(i);
    }
    else {
      s = MakeStatement(MakeLabel(strdup(lab_I)), i);
      if(!number_it) {
	decrement_statement_number();
      }
    }

    if (iPrevComm != 0 && !instruction_block_p(i)) {
	statement_comments(s) = strdup(PrevComm);
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
	   strcmp(lab_I, BlockStack[CurrentBlock-1].l) == 0)
		PopBlock();
}		



/* this function creates an empty block */

instruction MakeEmptyInstructionBlock()
{
    return(make_instruction_block(NIL));
}



/* this function creates a simple fortran statement such as RETURN,
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
		       instruction_undefined);
    NewStmt(l, s);
  }

  g = make_instruction(is_instruction_goto, s);

  return g;
}


instruction 
MakeComputedGotoInst(ll, i)
list ll;
entity i;
{
    instruction ins = instruction_undefined;
    list cs = NIL;
    int l = 0;
    list cl = list_undefined;

    /* Warning: the desugaring of a computed goto generates a
     * block... wich cannot be labelled when a stament is made later
     *
     * But a CONTINUE is added by MakeStatement a first block statements...
     * although the first statement of the block could carry the label.
     */

    DeclareVariable(i, type_undefined, NIL, storage_undefined, value_undefined);

    for(l = gen_length(ll), cl = ll; !ENDP(cl); l--, POP(cl)) {
	string ln = STRING(CAR(cl));
	instruction g = MakeGotoInst(ln);
	expression cond = 
	    MakeBinaryCall(
			   gen_find_tabulated(
					      make_entity_fullname(TOP_LEVEL_MODULE_NAME,
								   EQUAL_OPERATOR_NAME), 
					      entity_domain),
			   entity_to_expression(i),
			   int_to_expression(l));
	instruction iif = make_instruction(is_instruction_test,
					  make_test(cond,
						    instruction_to_statement(g),
						    make_empty_statement()));
	statement s = statement_undefined;

	if(ENDP(CDR(cl)) && strcmp(lab_I,"")) {
	    /* If this is the last generated test (i.e. the first one since they are
	     * generated backwards) and if the computed GO TO has a label, do something
	     * about it
	     */
	    entity lab = MakeLabel(strdup(lab_I));

	    if ((s = LabelToStmt(entity_name(lab))) == statement_undefined) {
		s = instruction_to_statement(iif);
		NewStmt(lab, s);
	    }
	    else {
		pips_assert("Instruction field must be undefined", 
			    instruction_undefined_p(statement_instruction(s)));
		statement_instruction(s) = iif;
	    }
	    lab_I[0] = '\0';
	}
	else {
	    s = instruction_to_statement(iif);
	}

	statement_number(s) = look_at_next_statement_number();
	cs = CONS(STATEMENT, s, cs);
    }
    /* Add a label on first statement */
    /* This is not possible because the corresponding statement may already exist. */
    /* Hence (at least one reason for) the spurious CONTINUE added in MakeStatement... */
    /*
    statement_label(STATEMENT(CAR(cs))) = MakeLabel(strdup(lab_I));
    */

    /* MakeStatement won't increment the current statement number
     * because this is a block... so it has to be done here
     */
    (void) get_next_statement_number();
    ins = make_instruction_block(cs);

    (void) gen_consistent_p(ins);

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
	   if(syntax_call_p(l)) 
	   {
	       if (get_bool_property("PARSER_EXPAND_STATEMENT_FUNCTIONS"))
	       {
		   /* Let us keep it somewhere. */
		   pips_debug(5, "considering %s as a macro\n", 
			      entity_name(call_function(syntax_call(l))));
		   parser_add_a_macro(syntax_call(l), e);
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
	       FatalError("MakeAssignInst", "Unexpected syntax tag\n");
	   }
       }
   }
   
   return i;
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
	    basic b = basic_of_expression(EXPRESSION(CAR(pc)));
	    variable v = make_variable(b, NIL); 
	    type t = make_type(is_type_variable, v);
	    parameter p = make_parameter(t,
					 MakeModeReference());

	    functional_parameters(ft) = 
		gen_nconc(functional_parameters(ft),
			  CONS(PARAMETER, p, NIL));
	}
    }
    else if(get_bool_property("PARSER_TYPE_CHECK_CALL_SITES"))  {
	/* The pre-existing typing of e should match the new one */
	int i = 0;
	bool warning_p = FALSE;

	for (pc = l, pc2 = functional_parameters(ft), i = 1;
	     !ENDP(pc) && !ENDP(pc2);
	     POP(pc), i++) {
	    basic b = basic_of_expression(EXPRESSION(CAR(pc)));
	    variable v = make_variable(b, NIL); 
	    type at = make_type(is_type_variable, v);
	    type ft = parameter_type(PARAMETER(CAR(pc2)));
	    type eft = type_varargs_p(ft)? type_varargs(ft) : ft;

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
		warning_p = TRUE;
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

    update_called_modules(e);

    pips_assert("MakeCallInst", MakeExternalFunction(e, MakeTypeVoid()) == e); 

    update_functional_type_with_actual_arguments(e, ap);

    if(!ENDP(ar)) {
	statement s = instruction_to_statement
	    (make_instruction(is_instruction_call, make_call(e, ap)));

	statement_number(s) = look_at_next_statement_number();
	pips_assert("Alternate return substitution required\n", SubstituteAlternateReturnsP());
	i = generate_return_code_checks(ar);
	pips_assert("Must be a sequence", instruction_block_p(i));
	instruction_block(i) = CONS(STATEMENT,
				    s,
				    instruction_block(i));
    }
    else {
	i = make_instruction(is_instruction_call, make_call(e, ap));
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
    gen_free(s);

    dolab = MakeLabel((strcmp(l, "BLOCKDO") == 0) ? "" : l);

    instblock_do = MakeEmptyInstructionBlock();
    stmt_do = instruction_to_statement(instblock_do);

    ido = make_instruction(is_instruction_loop,
			   make_loop(dovar, r, stmt_do, dolab,
				     make_execution(is_execution_sequential, 
						    UU),
				     NIL));

    LinkInstToCurrentBlock(ido, TRUE);
   
    PushBlock(instblock_do, l);
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
    instruction ti = make_instruction(is_instruction_test, 
				      make_test(e, bt, bf));
				      
    if (i == instruction_undefined)
	    FatalError("MakeLogicalIfInst", "bad instruction\n");

    /* instruction i should not be ablock, unless an alternate return is being processed */
    if(instruction_block_p(i)) {
	/* There should be two statements in the sequence.
	 * They have to be re-numbered.
	 */
	list l = instruction_block(i);
	int sn = get_future_statement_number();
	pips_assert("Block of two instructions", gen_length(l)==2);
	MAP(STATEMENT, s, {
	    statement_number(s) = sn;
	}, l);
    }
    else {
	statement_number(bt) = get_future_statement_number();
    }

    return ti;
}

/* this function transforms an arithmetic if statement into a set of
regular tests. long but easy to understand without comments.

e is the test expression.

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
				e1, MakeIntegerConstantExpression("0"));

	    s1 = instruction_to_statement(MakeGotoInst(l1));
	    s2 = make_empty_block_statement();

	    ifarith = make_instruction(is_instruction_test, 
				       make_test(e2,s1,s2));
	}
	else {
	    e1 = MakeBinaryCall(CreateIntrinsic(".LE."), 
				e, MakeIntegerConstantExpression("0"));

	    s1 = instruction_to_statement(MakeGotoInst(l1));
	    s3 = instruction_to_statement(MakeGotoInst(l3));

	    ifarith = make_instruction(is_instruction_test, 
				       make_test(e1,s1,s3));
	}
    }
    else if(strcmp(l1,l3)==0) {
	e1 = MakeBinaryCall(CreateIntrinsic(".EQ."), 
			    e, MakeIntegerConstantExpression("0"));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));

	ifarith = make_instruction(is_instruction_test, 
						      make_test(e1,s2,s1));
    }
    else if(strcmp(l2,l3)==0) {
	e1 = MakeBinaryCall(CreateIntrinsic(".LT."), 
			    e, MakeIntegerConstantExpression("0"));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));

	ifarith = make_instruction(is_instruction_test, 
				   make_test(e1,s1,s2));
    }
    else {
	/* General case */
	e1 = MakeBinaryCall(CreateIntrinsic(".LT."), 
			    e, MakeIntegerConstantExpression("0"));
	e2 = MakeBinaryCall(CreateIntrinsic(".EQ."), 
			    e, MakeIntegerConstantExpression("0"));

	s1 = instruction_to_statement(MakeGotoInst(l1));
	s2 = instruction_to_statement(MakeGotoInst(l2));
	s3 = instruction_to_statement(MakeGotoInst(l3));

	s = instruction_to_statement(make_instruction(is_instruction_test, 
						      make_test(e2,s2,s3)));
	statement_number(s) = look_at_next_statement_number();

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

    bt = MakeEmptyInstructionBlock();
    bf = MakeEmptyInstructionBlock();

    i = make_instruction(is_instruction_test,
			 make_test(e,
				   MakeStatement(MakeLabel(""), bt),
				   MakeStatement(MakeLabel(""), bf)));

    LinkInstToCurrentBlock(i, TRUE);
   
    PushBlock(bt, "ELSE");
    BlockStack[CurrentBlock-1].elsifs = elsif ;
}

int 
MakeElseInst()
{
    statement if_stmt;
    test if_test;
    int elsifs;

    if(CurrentBlock==0) {
	ParserError("MakeElseInst", "unexpected ELSE statement\n");
    }

    elsifs = BlockStack[CurrentBlock-1].elsifs ;

    if (strcmp("ELSE", BlockStack[CurrentBlock-1].l))
	    FatalError("MakeElseInst", "block if statement badly nested\n");

    if (iPrevComm != 0) {
      /* generate a CONTINUE to carry the comments */
      LinkInstToCurrentBlock(make_continue_instruction(), FALSE);
    }

    (void) PopBlock();

    if_stmt = STATEMENT(CAR(BlockStack[CurrentBlock-1].c));

    if (! instruction_test_p(statement_instruction(if_stmt)))
	    FatalError("MakeElseInst", "no block if statement\n");

    if_test = instruction_test(statement_instruction(if_stmt));

    PushBlock(statement_instruction(test_false(if_test)), "ENDIF");
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
      LinkInstToCurrentBlock(make_continue_instruction(), FALSE);
    }

    if (BlockStack[CurrentBlock-1].l != NULL &&
	strcmp("ELSE", BlockStack[CurrentBlock-1].l) == 0) {
	elsifs = MakeElseInst();
	LinkInstToCurrentBlock(make_continue_instruction(), FALSE);
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

    if (strcmp("BLOCKDO", BlockStack[CurrentBlock-1].l))
	    ParserError("MakeEnddoInst", "block do statement badly nested\n");

    /*LinkInstToCurrentBlock(MakeZeroOrOneArgCallInst("ENDDO", 
						    expression_undefined));*/
    /* Although it is not really an instruction, the ENDDO statement may 
     * carry comments
     */
    LinkInstToCurrentBlock(make_continue_instruction(), FALSE);

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
  entity a = global_name_to_entity(IO_EFFECTS_PACKAGE_NAME, n);
  reference r = make_reference(a, CONS(EXPRESSION, u, NIL));
  expression c = reference_to_expression(r);
  statement b = instruction_to_statement(make_goto_instruction(l));
  instruction t = make_instruction(is_instruction_test,
				   make_test(c, b, make_empty_block_statement()));
  statement check = instruction_to_statement(t);

  gen_consistent_p(check);

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
	  unit = copy_expression(EXPRESSION(CAR(CDR(l))));
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

	pips_assert("MakeIoInstA", syntax_call_p(s1));
	e1 = call_function(syntax_call(s1));
	pips_assert("MakeIoInstA", value_constant_p(entity_initial(e1)));
	pips_assert("MakeIoInstA", 
	       constant_litteral_p(value_constant(entity_initial(e1))));

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
		if (strcmp(entity_local_name(e1), "ERR=") == 0) {
		  io_err = make_check_io_statement(IO_ERROR_ARRAY_NAME, unit, e2);
		}
		else if (strcmp(entity_local_name(e1), "END=") == 0) {
		  io_end = make_check_io_statement(IO_EOF_ARRAY_NAME, unit, e2);
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
      gen_consistent_p(io);
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

/* Are we in the declaration or in the executable part? */

static int seen = FALSE;

void 
reset_first_statement()
{
    seen = FALSE;
}

bool
first_executable_statement_seen()
{
    return seen;
}

void
check_in_declarations()
{
    if(first_executable_statement_seen()) {
	ParserError("Syntax", 
		    "Declaration appears after executable statement");
    }
}

/* This function is called when the first executable statement is encountered */
void 
check_first_statement()
{
    int line_start = TRUE;
    int in_comment = FALSE;
    int out_of_constant_string = TRUE;
    int in_constant_string = FALSE;
    int end_of_constant_string = FALSE;
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

	seen = TRUE;
	
	/* we must read the input file from the begining and up to the 
	   line_b_I-1 th line, and the texte read must be stored in buffer */

	fd = safe_fopen(CurrentFN, "r");
	while ((c = getc(fd)) != EOF) {
	    if(line_start == TRUE)
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
		  out_of_constant_string = FALSE;
		  in_constant_string = TRUE;
		  buffer[ibuffer++] =  c;
		} 
		else {
		  buffer[ibuffer++] =  toupper(c);
		}
	      }
	      else
		if(in_constant_string) {
		  if(c==string_sep) {
		    in_constant_string = FALSE;
		    end_of_constant_string = TRUE;
		  }
		  buffer[ibuffer++] =  c;
		}
		else 
		  if(end_of_constant_string) {
		    if(c==string_sep) {
		      in_constant_string = TRUE;
		      end_of_constant_string = FALSE;
		      buffer[ibuffer++] =  c;
		    }
		    else {
		      out_of_constant_string = TRUE;
		      end_of_constant_string = FALSE;
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
		line_start = TRUE;
		in_comment = FALSE;
	    }
	    else {
		line_start = FALSE;
	    }

	    if (cpt == line_b_I-1)
		break;
	}
	safe_fclose(fd, CurrentFN);
	buffer[ibuffer++] = '\0';
	code_decls_text(EntityCode(get_current_module_entity())) = 
	    strdup(buffer);

	free(buffer), buffer=NULL;

	/* kill the first statement's comment because it's already
	   included in the declaration text */
	PrevComm[0] = '\0';
	iPrevComm = 0;
	/*
	Comm[0] = '\0';
	iComm = 0;
	*/

	/* clean up the declarations */
	update_common_sizes();

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

	reset_statement_number();
    }
}
