#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"

#include "syntax.h"
#include "syn_yacc.h"



/* the purpose of the following data structure is to associate labels to
instructions. The data structure contains a string (the label's name)
and a statement (the statement which the label is attached to). */
#define MAXSTMT 200
typedef struct stmt {
    string l; /* the name of the label */
    statement s; /* the statement attached to l */
} stmt;
LOCAL stmt StmtHeap[MAXSTMT];
LOCAL int CurrentStmt = 0;



/* to produce statement numbers */
static int stat_num = -1;

void reset_statement_number()
{
    stat_num = 1;
}


/* this functions looks up in table StmtHeap for the statement s whose
label is l. */

statement LabelToStmt(l)
string l;
{
    int i;

    for (i = 0; i < CurrentStmt; i++)
	    if (strcmp(l, StmtHeap[i].l) == 0)
		    return(StmtHeap[i].s);

    return(statement_undefined);
}



/* this function looks for undefined labels. a label is undefined if a
goto to that label has been encountered and if no statement with this
label has been parsed. */

void CheckAndInitializeStmt()
{
    int i;
    int MustStop = FALSE;

    for (i = 0; i < CurrentStmt; i++) {
	statement s = StmtHeap[i].s;
	if (statement_instruction(s) == instruction_undefined) {
	    MustStop = TRUE;
	    user_log("CheckAndInitializeStmt %s\n", entity_name(statement_label(s)));
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

void NewStmt(e, s)
entity e;
statement s;
{
  if (LabelToStmt(entity_name(e)) != statement_undefined) {
    user_log("NewStmt: duplicate label: %s\n", entity_name(e));
    ParserError("NewStmt", "duplicate label\n");
  }

    if (CurrentStmt == MAXSTMT)
	    ParserError("NewStmt", "statement heap full\n");

    StmtHeap[CurrentStmt].l = entity_name(e);
    StmtHeap[CurrentStmt].s = s;
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

void ResetBlockStack()
{
    CurrentBlock = 0;
}

bool IsBlockStackEmpty()
{
	return(CurrentBlock == 0);
}

bool IsBlockStackFull()
{
	return(CurrentBlock == MAXBLOCK);
}

void PushBlock(i, l)
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

instruction PopBlock()
{
	if (IsBlockStackEmpty())
		ParserError("PopBlock", "bottom of stack reached\n");

	return(BlockStack[--CurrentBlock].i);
}



/* this functions creates a label. LABEL_PREFIX is added to its name, for
integer constants and labels not to have the name space. */

entity MakeLabel(s)
string s;
{
	entity l;
	static char *name = NULL ;

	if( name == NULL ) {
	    name = (char *)malloc( LABEL_SIZE+strlen(LABEL_PREFIX)+1 ) ;
	}
	debug(5, "", "[MakeLabel] %s\n", s);

	strcpy(name, LABEL_PREFIX);
	strcat(name, s);	

	l = FindOrCreateEntity( (strcmp( name, LABEL_PREFIX )==0) ? 
		        TOP_LEVEL_MODULE_NAME :
		        CurrentPackage, name);

	if (entity_type(l) == type_undefined) {
	    debug(5, "", "[MakeLabel] %s\n", name);
	    entity_type(l) = MakeTypeStatement();
	    entity_storage(l) = MakeStorageRom();
	    entity_initial(l) = make_value(is_value_constant,
					   MakeConstantLitteral());
	}
	return(l);
}



/* this function makes a statement. l is the label and i the
instruction. we make sure that the label is not declared twice. */


statement MakeStatement(l, i)
entity l;
instruction i;
{
    statement s;

    debug(5, "", "[MakeStatement] %s\n", entity_name(l));

    pips_assert("MakeStatement", type_statement_p(entity_type(l)));
    pips_assert("MakeStatement", storage_rom_p(entity_storage(l)));
    pips_assert("MakeStatement", value_constant_p(entity_initial(l)));
    pips_assert("MakeStatement", 
	   constant_litteral_p(value_constant(entity_initial(l))));

    if (strcmp(entity_local_name(l), EMPTY_LABEL_NAME) != 0) {
	if (instruction_block_p(i))
		ParserError("makeStatement", "a block must have no label\n");

	/* FI, PJ: the "rice" phase does not handle labels on DO like 100 in:
	 *  100 DO 200 I = 1, N
	 *
	 * This should be trapped by "rice" when loops are checked to see
	 * if Allen/Kennedy's algorithm is applicable
	 */
	if (instruction_loop_p(i)) {
	    user_warning("MakeStatement",
			 "DO loop reachable by GO TO via label %s cannot be parallelized by PIPS\n",
			 entity_local_name(l));
	}

	if ((s = LabelToStmt(entity_name(l))) == statement_undefined) {
	    s = make_statement(l, 
			       stat_num++, 
			       STATEMENT_ORDERING_UNDEFINED,
			       string_undefined, 
			       i);
	    NewStmt(l, s);
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
	    statement_instruction(s) = i;
	}
    }
    else {
	s = make_statement(l, 
			   stat_num++, 
			   STATEMENT_ORDERING_UNDEFINED,
			   string_undefined, 
			   i);
    }
	
    return(s);
}



/* this function links the instruction i to the current block of
statements. if i is the last instruction of the block (i and the block
have the same label), the current block is popped from the stack. in
fortran, one instruction migth end more than one block. */

void LinkInstToCurrentBlock(i)
instruction i;
{
    statement s;
    cons * pc;
 
    if (IsBlockStackEmpty())
	    ParserError("LinkInstToCurrentBlock", "no current block\n");

    if(instruction_block_p(i)) {
      /* a CONTINUE instruction must be added to carry the label,
         because blocks cannot be labelled */
      list l = instruction_block(i);
      statement c = MakeStatement(MakeLabel(strdup(lab_I)), 
				  make_continue_instruction());

      instruction_block(i) = CONS (STATEMENT, c, l);
      s = MakeStatement(entity_empty_label(), i);
    }
    else {
      s = MakeStatement(MakeLabel(strdup(lab_I)), i);
    }

    if (iPrevComm != 0) {
	statement_comments(s) = strdup(PrevComm);
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



/* this function creatyes an empty block */

instruction MakeEmptyInstructionBlock()
{
    return(make_instruction_block(NIL));
}



/* this function creates a simple fortran statement such as RETURN,
CONTINUE, ... 

s is the name of the intrinsic function.

e is one optional argument (might be equal to expression_undefined). */

instruction MakeZeroOrOneArgCallInst(s, e)
char *s;
expression e;
{
    cons *l; /* la liste d'arguments */

    l = (e == expression_undefined) ? NIL : CONS(EXPRESSION, e, NIL);

    return(make_instruction(is_instruction_call, 
			    make_call(CreateIntrinsic(s), l)));
}



/* this function creates a goto instruction. n is the target label. */

instruction MakeGotoInst(n)
string n;
{
    statement s;
    entity l;

    l = MakeLabel(n);
    s = LabelToStmt(entity_name(l));

    if (s == statement_undefined) {
	s = make_statement(l, 
			   stat_num++,
			   STATEMENT_ORDERING_UNDEFINED,
			   string_undefined, 
			   instruction_undefined);
	NewStmt(l, s);
    }

    return(make_instruction(is_instruction_goto, s));
}


instruction MakeComputedGotoInst(ll, i)
list ll;
entity i;
{
    instruction ins = instruction_undefined;
    list cs = NIL;
    int l = 0;
    list cl = list_undefined;

    /* Warning: the desugaring of a computed goto generates a
     * block... wich cannot be labelled when a stament is made later
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
						    MakeStatement(MakeLabel(""), g),
						    make_empty_statement()));
	statement s =  make_stmt_of_instr(iif);

	cs = CONS(STATEMENT, s, cs);
    }
    ins = make_instruction_block(cs);

    (void) gen_consistent_p(ins);

    /* FatalError("parser", "computed goto statement prohibited\n"); */

    return ins;
}

/* this function creates an affectation statement. 

l is a reference (the left hand side).

e is an expression (the right hand side). */

instruction MakeAssignInst(l, e)
syntax l;
expression e;
{
   expression lhs = expression_undefined;
   instruction i = instruction_undefined;

   if(syntax_reference_p(l)) {
     lhs = make_expression(l, normalized_undefined);
     i = make_assign_instruction(lhs, e);
   }
   else {
     if(syntax_call_p(l) &&
	strcmp(entity_local_name(call_function(syntax_call(l))), 
	       SUBSTRING_FUNCTION_NAME) == 0) {
       list lexpr = CONS(EXPRESSION, e, NIL);
       list asub = call_arguments(syntax_call(l));

       call_arguments(syntax_call(l)) = NIL;
       free_syntax(l);

       i = make_instruction(is_instruction_call,
			    make_call(entity_intrinsic(ASSIGN_SUBSTRING_FUNCTION_NAME),
				      gen_append(asub, lexpr)));
     }
     else
       /* FI: we stumble here when a Fortran macro is used */
       ParserError("MakeAssignInst",
		   "bad lhs (function call or undeclared array) or "
		   "unsupported Fortran macro\n");
   }

   return i;
}



/* this function creates a call statement. e is the called function. l
is the argument list. */

instruction MakeCallInst(e, l)
entity e;
cons * l;
{
    update_called_modules(e);

    pips_assert("MakeCallInst", MakeExternalFunction(e, MakeTypeVoid()) == e);

    return(make_instruction(is_instruction_call, make_call(e, l)));
}



/* this function creates a do loop statement.

s is a reference to the do variable.

r is the range of the do loop.

l is the label of the last statement of the loop. */

void MakeDoInst(s, r, l)
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
    stmt_do = MakeStatement(MakeLabel(""), instblock_do);

    ido = make_instruction(is_instruction_loop,
			   make_loop(dovar, r, stmt_do, dolab,
				     make_execution(is_execution_sequential, 
						    UU),
				     NIL));

    LinkInstToCurrentBlock(ido);
   
    PushBlock(instblock_do, l);
}

/* this function creates a logical if statement. the true part of the
test is a block with only one instruction (i), and the false part is an
empty block.  */

instruction MakeLogicalIfInst(e, i)
expression e;
instruction i;
{
    instruction bt, bf;

    if (i == instruction_undefined)
	    FatalError("MakeLogicalIfInst", "bad instruction\n");

    bt = MakeEmptyInstructionBlock();
    instruction_block(bt) = CONS(INSTRUCTION, MakeStatement(MakeLabel(""), i),
				 instruction_block(bt));

    bf = MakeEmptyInstructionBlock();
    instruction_block(bf) =
	    CONS(INSTRUCTION, 
		 MakeStatement(MakeLabel(""), 
			       MakeZeroOrOneArgCallInst("CONTINUE",
							expression_undefined)),
		 instruction_block(bf));

    return(make_instruction(is_instruction_test, 
			    make_test(e, 
				      MakeStatement(MakeLabel(""), bt), 
				      MakeStatement(MakeLabel(""), bf))));
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

instruction MakeArithmIfInst(e, l1, l2, l3)
expression e;
string l1, l2, l3;
{
    expression e1, e2;
    statement s1, s2, s3, s;
    instruction b1, b2, b3, b;

    e1 = MakeBinaryCall(CreateIntrinsic(".LT."), 
			e, MakeIntegerConstantExpression("0"));
    e2 = MakeBinaryCall(CreateIntrinsic(".EQ."), 
			e, MakeIntegerConstantExpression("0"));

    s1 = MakeStatement(MakeLabel(""), MakeGotoInst(l1));
    s2 = MakeStatement(MakeLabel(""), MakeGotoInst(l2));
    s3 = MakeStatement(MakeLabel(""), MakeGotoInst(l3));

    b1 = MakeEmptyInstructionBlock();
    instruction_block(b1) = CONS(INSTRUCTION, s1, instruction_block(b1));
    b2 = MakeEmptyInstructionBlock();
    instruction_block(b2) = CONS(INSTRUCTION, s2, instruction_block(b2));
    b3 = MakeEmptyInstructionBlock();
    instruction_block(b3) = CONS(INSTRUCTION, s3, instruction_block(b3));

    s = MakeStatement(MakeLabel(""),
		      make_instruction(is_instruction_test, 
				       make_test(e2,
						 MakeStatement(MakeLabel(""), 
							       b2), 
						 MakeStatement(MakeLabel(""),
							       b3))));
    b = MakeEmptyInstructionBlock();
    instruction_block(b) = CONS(INSTRUCTION, s, instruction_block(b));

    return(make_instruction(is_instruction_test, 
		     make_test(e1,
			       MakeStatement(MakeLabel(""), b1),
			       MakeStatement(MakeLabel(""), b))));
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

void MakeBlockIfInst(e,elsif)
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

    LinkInstToCurrentBlock(i);
   
    PushBlock(bt, "ELSE");
    BlockStack[CurrentBlock-1].elsifs = elsif ;
}

int MakeElseInst()
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

    (void) PopBlock();

    if_stmt = STATEMENT(CAR(BlockStack[CurrentBlock-1].c));

    if (! instruction_test_p(statement_instruction(if_stmt)))
	    FatalError("MakeElseInst", "no block if statement\n");

    if_test = instruction_test(statement_instruction(if_stmt));

    PushBlock(statement_instruction(test_false(if_test)), "ENDIF");
    return( BlockStack[CurrentBlock-1].elsifs = elsifs ) ;
}

void MakeEndifInst()
{
    int elsifs = -1;

    if(CurrentBlock==0) {
	ParserError("MakeEndifInst", "unexpected ENDIF statement\n");
    }

    if (BlockStack[CurrentBlock-1].l != NULL &&
	strcmp("ELSE", BlockStack[CurrentBlock-1].l) == 0) {
	elsifs = MakeElseInst();
	LinkInstToCurrentBlock(MakeZeroOrOneArgCallInst("CONTINUE",
							expression_undefined));
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

void MakeEnddoInst()
{
    if(CurrentBlock<=1) {
	ParserError("MakeEnddoInst", "Unexpected ENDDO statement\n");
    }

    if (strcmp("BLOCKDO", BlockStack[CurrentBlock-1].l))
	    ParserError("MakeEnddoInst", "block do statement badly nested\n");

    /*LinkInstToCurrentBlock(MakeZeroOrOneArgCallInst("ENDDO", 
						    expression_undefined));*/
    (void) PopBlock();
}

string NameOfToken(token)
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

/* this function creates a io statement. keyword indicates which io
statement is to be built (READ, WRITE, ...).

lci is a list of 'control specifications'. its has the following format:

        ("UNIT=", 6, "FMT=", "*", "RECL=", 80, "ERR=", 20)

lio is the list of expressions to write or references to read. */

instruction MakeIoInstA(keyword, lci, lio)
int keyword;
cons *lci;
cons *lio;
{
    cons *l;

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
	    }
	}
    }

    for (l = lci; CDR(l) != NULL; l = CDR(l)) ;

    CDR(l) = lio;
    l = lci;

    return(make_instruction(is_instruction_call,
			    make_call(CreateIntrinsic(NameOfToken(keyword)),
				      l)));
}



/* this function creates a BUFFER IN or BUFFER OUT io statement. this is
not ansi fortran.

e1 is the logical unit.

nobody known the exact meaning of e2

e3 et e4 are references that indicate which variable elements are to be
buffered in or out. */

instruction MakeIoInstB(keyword, e1, e2, e3, e4)
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

static int seen = FALSE;

void reset_first_statement()
{
    seen = FALSE;
}

#define SIZE 32384
void check_first_statement()
{
    int line_start = TRUE;
    int in_comment = FALSE;
    int out_of_constant_string = TRUE;
    int in_constant_string = FALSE;
    int end_of_constant_string = FALSE;
    char string_sep = '\000';

    if (! seen) {
	FILE *fd;
	int cpt = 0, ibuffer = 0, c;
	static char buffer[SIZE];

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

	    if (ibuffer >= SIZE) {
		ParserError("check_first_statement", "Static buffer too small, resize!\n");
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
	code_decls_text(EntityCode(get_current_module_entity())) = strdup(buffer);

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

	reset_statement_number();
    }
}
