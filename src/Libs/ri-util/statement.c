 /* 
    Function for statement, and its subtypes:
     - instruction

    Lei ZHOU         12 Sep. 1991
    Francois IRIGOIN
  */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"

#include "text.h"
#include "text-util.h"
#include "ri.h"

#include "ri-util.h"

bool
empty_comments_p(string s)
{
  /* Could be replaced by a macro. See macro empty_comment */
  pips_assert("comments cannot be NULL", s!=NULL);
  return (s == NULL || string_undefined_p(s));
}

/* PREDICATES ON STATEMENTS */

bool empty_statement_p(st)
statement st;
{
    instruction i;

    return(entity_empty_label_p(statement_label(st)) &&
	    instruction_block_p(i=statement_instruction(st)) &&
	    ENDP(instruction_block(i)));
}

bool assignment_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    return (fortran_instruction_p(i, ASSIGN_OPERATOR_NAME));
}

bool return_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);
    bool return_p = fortran_instruction_p(i, RETURN_FUNCTION_NAME);

    return return_p;
}

bool continue_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    return (fortran_instruction_p(i, CONTINUE_FUNCTION_NAME));
}

bool stop_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    return (fortran_instruction_p(i, STOP_FUNCTION_NAME));
}

bool format_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    return (fortran_instruction_p(i, FORMAT_FUNCTION_NAME));
}

bool statement_less_p(st1, st2)
statement st1, st2;
{
    int o1 = statement_ordering( st1 ) ;
    int o2 = statement_ordering( st2 ) ;

    if (ORDERING_NUMBER( o1 ) != ORDERING_NUMBER( o2 )) {
	fprintf(stderr, "cannot compare %d (%d,%d) and %d (%d,%d)\n",
		statement_number(st1), 
		ORDERING_NUMBER( o1 ), ORDERING_STATEMENT( o1 ),
		statement_number(st2), 
		ORDERING_NUMBER( o2 ), ORDERING_STATEMENT( o2 ));

	abort();
    }

    return( ORDERING_STATEMENT(o1) < ORDERING_STATEMENT(o2)) ;
}

bool statement_possible_less_p(st1, st2)
statement st1, st2;
{
    int o1 = statement_ordering( st1 ) ;
    int o2 = statement_ordering( st2 ) ;

    if (ORDERING_NUMBER( o1 ) != ORDERING_NUMBER( o2 )) {
	return(TRUE);
    }
    else
	return( ORDERING_STATEMENT(o1) < ORDERING_STATEMENT(o2));
}

bool statement_loop_p(s)
statement s;
{
    return(instruction_loop_p(statement_instruction(s)));
}

bool statement_test_p(statement s)
{
    return(instruction_test_p(statement_instruction(s)));
}

/* This function should not be used ! */
bool
statement_continue_p(s)
statement s;
{
    return continue_statement_p(s);
}

bool
unlabelled_statement_p(st)
statement st;
{
    return(entity_empty_label_p(statement_label(st)));
}

bool
nop_statement_p(statement s)
{
    /* NOP statements are useful to fill in empty test branches.
     * The definition of NOP in PIPS has changed over the years.
     * A NOP statement is an empty block. Like blocks, it cannot be
     * labelled nor commented. It has no statement number because it
     * is invisible to users.
     *
     * Dangling comments, like labels, are attached to CONTINUE statements.
     *
     * Note 1: blocks are now called "sequences"
     * Note 2: see also empty_statement_p()
     */
    bool nop = FALSE;
    instruction i = statement_instruction(s);

    if(instruction_block_p(i) && ENDP(instruction_block(i))) {
	pips_assert("No label!", entity_empty_label_p(statement_label(s)));
	pips_assert("No comments", empty_comments_p(statement_comments(s)));
	pips_assert("No statement number", statement_number(s) == STATEMENT_NUMBER_UNDEFINED);
	nop = TRUE;
    }

    return nop;
}

/* Return true if the statement is an empty instruction block without
   label or a continue without label or a recursive combination of
   above. */
bool
empty_statement_or_labelless_continue_p(statement st)
{
   instruction i;

   if (!entity_empty_label_p(statement_label(st)))
      return FALSE;
   if (continue_statement_p(st))
      return TRUE;
   i = statement_instruction(st);
   if (instruction_block_p(i)) {
       MAP(STATEMENT, s,
           {
	       if (!empty_statement_or_labelless_continue_p(s))
		   /* Well there is at least one possibly usefull thing... */
		   return FALSE;
           },
	   instruction_block(i));
       return TRUE;
   }
   return FALSE;
}


/* Return true if the statement is an empty instruction block or a
   continue or a recursive combination of above. */
bool
empty_statement_or_continue_p(statement st)
{
   instruction i;

   if (continue_statement_p(st))
      return TRUE;
   i = statement_instruction(st);
   if (instruction_block_p(i)) {
       MAP(STATEMENT, s,
           {
	       if (!empty_statement_or_continue_p(s))
		   /* Well there is at least one possibly usefull thing... */
		   return FALSE;
           },
              instruction_block(i));
       return TRUE;
   }
   return FALSE;
}


/* Return true if the statement is an empty instruction block or a
   continue without comments or without LABEL or a recursive
   combination of above. */
bool
empty_statement_or_continue_without_comment_p(statement st)
{
   instruction i;
   string the_comments = statement_comments(st);

   /* The very last condition should be sufficient */
   if (!empty_comments_p(the_comments))
       return FALSE;

   if (!entity_empty_label_p(statement_label(st)))
      return FALSE;
   if (continue_statement_p(st))
      return TRUE;
   
   i = statement_instruction(st);
   if (instruction_block_p(i)) {
       MAP(STATEMENT, s,
	   {
	       if (!empty_statement_or_continue_without_comment_p(s))
		   return FALSE;
	   },
	   instruction_block(i));
       /* Everything in the block are commentless continue or empty
          statements: */
       return TRUE;
   }
   /* Everything else useful: */
   return FALSE;
}


bool statement_call_p(s)
statement s;
{
    return(instruction_call_p(statement_instruction(s)));
}


bool check_io_statement_p(statement s)
{
  bool check_io = FALSE;
  instruction i = statement_instruction(s);

  if(instruction_test_p(i)) {
    syntax c = expression_syntax(test_condition(instruction_test(i)));

    if(syntax_reference_p(c)) {
      check_io = io_entity_p(reference_variable(syntax_reference(c)));
    }
  }

  return check_io;
}

bool perfectly_nested_loop_p(stat)
statement stat;
{
    instruction ins = statement_instruction(stat);
    tag t = instruction_tag(ins);

    switch( t ) {
    case is_instruction_block: {
	list lb = instruction_block(ins);

	if ( lb != NIL && (lb->cdr) != NIL && (lb->cdr)->cdr == NIL 
	    && ( continue_statement_p(STATEMENT(CAR(lb->cdr))) ) ) {
	    if ( assignment_statement_p(STATEMENT(CAR(lb))) )
		return TRUE;
	    else
		return(perfectly_nested_loop_p(STATEMENT(CAR(lb))));
	}
	else if ( lb != NIL && (lb->cdr) == NIL )
	    return(perfectly_nested_loop_p(STATEMENT(CAR(lb))));
	else if ( lb != NIL ) {
	    /* biased for WP65 */
	    return assignment_block_p(ins);
	}
	else
	    /* extreme case: empty loop nest */
	    return TRUE;
	break;
    }
    case is_instruction_loop: {
	loop lo = instruction_loop(ins);
	statement sbody = loop_body(lo);
	    
	if ( assignment_statement_p(sbody) ) 
	    return TRUE;
	else
	    return(perfectly_nested_loop_p(sbody));
	break;
    }
    default:
	break;
    }

    return FALSE;
}

/* checks that a block is a list of assignments, possibly followed by
   a continue */
bool assignment_block_p(i)
instruction i;
{
    MAPL(cs,
     {
	 statement s = STATEMENT(CAR(cs));

	 if(!assignment_statement_p(s))
	     if(!(continue_statement_p(s) && ENDP(CDR(cs)) ))
		 return FALSE;
     },
	 instruction_block(i));
    return TRUE;
}

/* functions to generate statements */

statement make_empty_statement()
{
    return(make_statement(entity_empty_label(), 
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED, 
			  empty_comments,
			  make_instruction_block(NIL)));
}

/* to be compared with instruction_to_statement() which is a macro (thanks to FC?) ! */

statement make_stmt_of_instr(instr)
instruction instr;
{
    return(make_statement(entity_empty_label(), 
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED, 
			  empty_comments,
			  instr));
}

instruction 
make_assign_instruction(expression l,
                        expression r)
{
   call c = call_undefined;
   instruction i = instruction_undefined;

   pips_assert("make_assign_statement",
               syntax_reference_p(expression_syntax(l)));
   c = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME),
                 CONS(EXPRESSION, l, CONS(EXPRESSION, r, NIL)));
   i = make_instruction(is_instruction_call, c);

   return i;
}


statement
make_assign_statement(expression l,
                      expression r)
{
    return make_stmt_of_instr(make_assign_instruction(l, r));
}

/* FI: make_block_statement_with_stop is obsolete, do not use */

statement make_block_statement_with_stop()
{
    statement b;
    statement stop;
    entity stop_function;

    stop_function = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME, 
						   MODULE_SEP_STRING,
						   "STOP",
						   NULL),
				       entity_domain);

    pips_assert("make_block_statement_with_stop", 
		stop_function != entity_undefined);

    stop = make_statement(entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  make_instruction(is_instruction_call,
					   make_call(stop_function,NIL)));

    b = make_statement(entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  make_instruction_block(CONS(STATEMENT, stop, NIL)));

    ifdebug(8) {
	fputs("make_block_statement_with_stop",stderr);
	print_text(stderr, text_statement(entity_undefined,0,b));
    }

    return b;
}

statement
make_nop_statement()
{
    /* Attention: this function and the next one produce the same result,
     * but they are not semantically equivalent. An empty block may be
     * defined to be filled in, not to be used as a NOP statement.
     */
    statement s = make_empty_block_statement();

    return s;
}

statement make_empty_block_statement()
{
    statement b;

    b = make_block_statement(NIL);

    return b;
}

statement make_block_statement(body)
list body;
{
    statement b;

    b = make_statement(entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  make_instruction_block(body));

    return b;
}


instruction make_instruction_block(list statements)
{
#ifdef is_instruction_sequence
    return make_instruction(is_instruction_sequence,
			    make_sequence(statements));
#else
    return make_instruction(is_instruction_block, statements);
#endif
}

statement make_return_statement()
{
  /* The special RETURN label should be used, shouldn't it? */
    return make_call_statement(RETURN_FUNCTION_NAME, NIL, 
			       entity_undefined, empty_comments);
}


instruction make_continue_instruction()
{
    entity called_function;

    called_function = entity_intrinsic(CONTINUE_FUNCTION_NAME);
    return make_instruction(is_instruction_call,
			    make_call(called_function,NIL));
}


statement make_continue_statement(l)
entity l;
{
    return make_call_statement(CONTINUE_FUNCTION_NAME, NIL, l, 
			       empty_comments);
}



instruction MakeUnaryCallInst(f,e)
entity f;
expression e;
{
    return(make_instruction(is_instruction_call,
			    make_call(f, CONS(EXPRESSION, e, NIL))));
}

/* this function creates a call to a function with zero arguments.  */

expression MakeNullaryCall(f)
entity f;
{
    return(make_expression(make_syntax(is_syntax_call, make_call(f, NIL)),
			   normalized_undefined));
}


/* this function creates a call to a function with one argument. */

expression MakeUnaryCall(f, a)
entity f;
expression a;
{
    call c =  make_call(f, CONS(EXPRESSION, a, NIL));

    return(make_expression(make_syntax(is_syntax_call, c),
			   normalized_undefined));
}


statement make_call_statement(function_name, args, l, c)
string function_name;
list args;
entity l; /* label, default entity_undefined */
string c; /* comments, default empty_comments (was: "" (was: string_undefined)) */
{
    entity called_function;
    statement cs;

    called_function = entity_intrinsic(function_name);

    l = (l==entity_undefined)? entity_empty_label() : l;
    cs = make_statement(l,
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  c,
			  make_instruction(is_instruction_call,
					   make_call(called_function,NIL)));

    ifdebug(8) {
	pips_debug(8, "cs is\n");
	print_statement(cs);
    }

    return cs;
}

/* */

statement perfectly_nested_loop_to_body(loop_nest)
statement loop_nest;
{
    instruction ins = statement_instruction(loop_nest);

    switch(instruction_tag(ins)) {
    case is_instruction_call:
        return loop_nest;
        
    case is_instruction_block: {
	list lb = instruction_block(ins);
	statement first_s = STATEMENT(CAR(lb));
	instruction first_i = statement_instruction(first_s);

	if(instruction_call_p(first_i))
	    return loop_nest;
	else {
	    if(instruction_block_p(first_i)) 
		return perfectly_nested_loop_to_body(STATEMENT(CAR(instruction_block(first_i))));
	    else {
		pips_assert("perfectly_nested_loop_to_body",
			    instruction_loop_p(first_i));
		return perfectly_nested_loop_to_body( first_s);
	    }
	}
	break;
    } 
    case is_instruction_loop: {
	statement sbody = loop_body(instruction_loop(ins));
	return (perfectly_nested_loop_to_body(sbody));
	break;
    }
    default:
	pips_error("perfectly_nested_loop_to_body","illegal tag\n");
	break;
    }
    return(statement_undefined); /* just to avoid a warning */
}

/* Direct accesses to second level fields */

loop statement_loop(s)
statement s;
{
    pips_assert("statement_loop", statement_loop_p(s));

    return(instruction_loop(statement_instruction(s)));
}

call statement_call(s)
statement s;
{
    pips_assert("statement_call", statement_call_p(s));

    return(instruction_call(statement_instruction(s)));
}

list statement_block(s)
statement s;
{
    pips_assert("statement_block", statement_block_p(s));

    return(instruction_block(statement_instruction(s)));
}

/* predicates on instructions */

bool instruction_assign_p(i)
instruction i;
{
    return fortran_instruction_p(i, ASSIGN_OPERATOR_NAME);
}

bool instruction_continue_p(i)
instruction i;
{
    return fortran_instruction_p(i, CONTINUE_FUNCTION_NAME);
}

bool instruction_return_p(i)
instruction i;
{
    return fortran_instruction_p(i, RETURN_FUNCTION_NAME);
}

bool instruction_stop_p(i)
instruction i;
{
    return fortran_instruction_p(i, STOP_FUNCTION_NAME);
}

bool instruction_format_p(i)
instruction i;
{
    return fortran_instruction_p(i, FORMAT_FUNCTION_NAME);
}

bool fortran_instruction_p(i, s)
instruction i;
string s;
{
  bool call_s_p = FALSE;

    if (instruction_call_p(i)) {
	call c = instruction_call(i);
	entity f = call_function(c);

	if (strcmp(entity_local_name(f), s) == 0)
	  call_s_p = TRUE;
    }

    return call_s_p;
}

/*
  returns the numerical value of loop l increment expression.
  aborts if this expression is not an integral constant expression.
  modification : returns the zero value when it isn't constant
  Y.Q. 19/05/92
*/

int loop_increment_value(l)
loop l;
{
    range r = loop_range(l);
    expression ic = range_increment(r);
    normalized ni;
    int inc;

    ni = NORMALIZE_EXPRESSION(ic);

    if (! EvalNormalized(ni, &inc)){
	/*user_error("loop_increment_value", "increment is not constant");*/
	debug(8,"loop_increment_value", "increment is not constant");
	return(0);
    }
    return(inc);
}

bool assignment_block_or_statement_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    switch(instruction_tag(i)) {
    case is_instruction_block:
	return assignment_block_p(statement_instruction(s));
	break;
    case is_instruction_test:
	break;
    case is_instruction_loop:
	break;
    case is_instruction_goto:
	pips_error("assignment_block_p", "unexpected GO TO\n");
    case is_instruction_call:
	return assignment_statement_p(s);
    case is_instruction_unstructured:
	break;
    default: pips_error("assignment_block_or_statement_p",
			"ill. instruction tag %d\n", instruction_tag(i));
    }
    return FALSE;
}

void print_statement_set(fd, r)
FILE *fd;
set r;
{
    fprintf(fd, "Set contains statements");

    SET_MAP(s, {
	fprintf(fd, " %02d", statement_number((statement) s));
    }, r);

    fprintf(fd, "\n");
}

/* (the text is not freed, massive memory leak:-) 
 */ 
void print_statement(statement s)
{
    debug_on("TEXT_DEBUG_LEVEL");
    print_text(stderr, text_statement(entity_undefined,0,s));
    debug_off();
}

static hash_table number_to_statement = hash_table_undefined;

static void update_number_to_statement(s)
statement s;
{
    if(statement_number(s)!=STATEMENT_NUMBER_UNDEFINED) {
	hash_put(number_to_statement, (char *) statement_number(s), (char *) s);
    }
}

hash_table build_number_to_statement(nts, s)
hash_table nts;
statement s;
{
    pips_assert("build_number_to_statement", nts!=hash_table_undefined
		&& !statement_undefined_p(s));

    number_to_statement = nts;

    gen_recurse(s, statement_domain, gen_true, update_number_to_statement);

    /* nts is updated by side effect on number_to_statement */
    number_to_statement = hash_table_undefined;

    return nts;
}

void print_number_to_statement(nts)
hash_table nts;
{
    HASH_MAP(number, stmt, {
	fprintf(stderr,"%d\t", (int) number);
	print_statement((statement) stmt);
    }, nts);
}

hash_table allocate_number_to_statement()
{
    hash_table nts = hash_table_undefined;

    /* let's assume that 50 statements is a good approximation of a module
     * size 
     */
    nts = hash_table_make(hash_int, 50);

    return nts;
}

/* get rid of all labels in controlized code before duplication: all
 * labels have become useless and they cannot be freely duplicated.
 *
 * One caveat: FORMAT statements!
 */
statement clear_labels(s)
statement s;
{
    gen_recurse (s, statement_domain, gen_true, clear_label);
    return s;
}

void clear_label(s)
statement s;
{
    statement_label(s) = entity_empty_label();


    if(format_statement_p(s)) {
	user_error("clear_label", "Cannot clear FORMAT label!\n");
    }

    if(instruction_loop_p(statement_instruction(s))) 
	loop_label(instruction_loop(statement_instruction(s))) = 
	    entity_empty_label();
}

/*
 *   moved from HPFC by FC, 15 May 94
 *
 */

statement list_to_statement(l)
list l;
{
    switch (gen_length(l))
    {
    case 0:
	return(statement_undefined);
    case 1:
    {
	statement stat=STATEMENT(CAR(l));
	gen_free_list(l);
	return(stat);
    }
    default:
	return(make_block_statement(l));
    }

    return(statement_undefined);
}

statement st_make_nice_test(condition, ltrue, lfalse)
expression condition;
list ltrue,lfalse;
{
    statement
	stattrue = list_to_statement(ltrue),
	statfalse = list_to_statement(lfalse);
    bool
	notrue=(stattrue==statement_undefined),
	nofalse=(statfalse==statement_undefined);
    
    if ((notrue) && (nofalse)) 
	return(make_continue_statement(entity_undefined)); 

    if (nofalse) 
    {
	return
	    (make_stmt_of_instr
	     (make_instruction
	      (is_instruction_test,
	       make_test(condition,
			 stattrue,
			 make_continue_statement(entity_undefined)))));
    }

    if (notrue) 
    {
	expression
	    newcond = MakeUnaryCall(entity_intrinsic(NOT_OPERATOR_NAME),
				    condition);

	return
	    (make_stmt_of_instr
	     (make_instruction
	      (is_instruction_test,
	       make_test(newcond,
			 statfalse,
			 make_continue_statement(entity_undefined)))));
    }

    return(make_stmt_of_instr(make_instruction(is_instruction_test,
					       make_test(condition,
							 stattrue,
							 statfalse))));
}


/* statement makeloopbody(l) 
 * make statement of a  loop body
 *
 * FI: the name of this function is not very good; the function should be put
 * in ri-util/statement.c, if it is really useful. The include list is
 * a joke!
 *
 * move here from generation... FC 16/05/94
 */
statement makeloopbody(l,s_old)
loop l;
statement s_old;
{
    statement state_l;
    instruction instr_l;
    statement l_body;

    instr_l = make_instruction(is_instruction_loop,l);
    state_l = make_statement(statement_label(s_old),
			     statement_number(s_old),
			     statement_ordering(s_old),
			     statement_comments(s_old),
			     instr_l);
    l_body = make_statement(entity_empty_label(),
			    STATEMENT_NUMBER_UNDEFINED,
			    STATEMENT_ORDERING_UNDEFINED,
			    empty_comments,
			make_instruction_block(CONS(STATEMENT,state_l,NIL)));

    return(l_body);
}


/*============================================================================*/
/* statement make_block_with_stmt(statement stmt): makes sure that the given
 * statement "stmt" is a block of instructions. If it is not the case, this
 * function returns a new statement with a block of one statement. This
 * statement is "stmt".
 * If "stmt" is already a block, it is returned unmodified.
 */
statement make_block_with_stmt(stmt)
statement stmt;
{
if (instruction_tag(statement_instruction(stmt)) != is_instruction_block)
  {
  /* We create a new statement with an empty block of instructions.
   * "make_empty_statement()" is defined in Lib/ri-util, it creates a statement
   * with a NIL instruction block. */
  statement block_stmt = make_empty_statement();

  /* Then, we put the statement "stmt" in the block. */
  instruction_block(statement_instruction(block_stmt)) = CONS(STATEMENT,
                                                              stmt, NIL);
  return (block_stmt);
  }
return (stmt);
}

string statement_identification(statement s)
{
    static char buffer[50];
    char *instrstring = NULL;
    int so = statement_ordering(s);

    switch (instruction_tag(statement_instruction(s)))
    {
    case is_instruction_loop:
	instrstring="LOOP";
	break;
    case is_instruction_test:
	instrstring="TEST";
	break;
    case is_instruction_goto:
	instrstring="GOTO";
	break;
    case is_instruction_call:
      {if(continue_statement_p(s))
	instrstring="CONTINUE";
      else if(return_statement_p(s))
	instrstring="RETURN";
      else if(stop_statement_p(s))
	instrstring="STOP";
      else if(format_statement_p(s))
	instrstring="FORMAT";
      else if(assignment_statement_p(s))
	instrstring="ASSIGN";
      else
	instrstring="CALL";
      break;
      }
    case is_instruction_block:
	instrstring="BLOCK";
	break;
    case is_instruction_unstructured:
	instrstring="UNSTRUCTURED";
	break;
    default: pips_error("assignment_block_or_statement_p",
			"ill. instruction tag %d\n", 
			instruction_tag(statement_instruction(s)));
    }

    sprintf(buffer, "%d (%d, %d) at 0x%x: %s\n",
	    statement_number(s),
	    ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so),
	    (unsigned int) s,
	    instrstring);

    return buffer;
}


/* Return true if the statement has an empty statement: */
bool
statement_with_empty_comment_p(statement s)
{
    string the_comments = statement_comments(s);
    return empty_comments_p(the_comments);
}


static string gather_all_comments_of_a_statement_string;


static bool 
gather_all_comments_of_a_statement_filter(statement s)
{
  string the_comments = statement_comments(s);
  if (!empty_comments_p(the_comments)) {
    string old = gather_all_comments_of_a_statement_string;
    gather_all_comments_of_a_statement_string =
      strdup(concatenate(old, the_comments, NULL));
    free(old);
  }
  return TRUE;
}


/* Gather all the comments recursively found in the given statement
   and return them in a string.

   Do not forget to free the string returned later when no longer
   used. */
string
gather_all_comments_of_a_statement(statement s)
{
    gather_all_comments_of_a_statement_string = strdup("");
    gen_recurse(s, statement_domain,
		gather_all_comments_of_a_statement_filter, gen_null);
    return gather_all_comments_of_a_statement_string;
}


/* Append a comment string to the comments of a statement. */
void
append_comments_to_statement(statement s,
			     string the_comments)
{
    string old = statement_comments(s);
    
    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;
    
    if (empty_comments_p(old))
	/* There is no comment yet: */
	statement_comments(s) = strdup(the_comments);
    else {
	statement_comments(s) = strdup(concatenate(old, the_comments, NULL));
	free(old);
    }
}


/* Insert a comment string at the beginning of the comments of a statement. */
void
insert_comments_to_statement(statement s,
			     string the_comments)
{
    string old = statement_comments(s);

    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;
    
    if (empty_comments_p(old))
	/* There are no comments yet: */
	statement_comments(s) = strdup(the_comments);
    else {
	statement_comments(s) = strdup(concatenate(the_comments, old, NULL));
	free(old);
    }
}

/* See if statement s is labelled and can be reached by a GO TO */
entity
statement_to_label(statement s)
{
  entity l = statement_label(s);

  if(format_statement_p(s)) {
    /* You cannot jump onto a non-executable statement */
    l = entity_empty_label();
  }
  else if(entity_empty_label_p(l)) {
    instruction i = statement_instruction(s);

    switch(instruction_tag(i)) {

    case is_instruction_sequence:
      /* The initial statement may be an unlabelled CONTINUE... */
      MAPL(cs, {
	statement stmt = STATEMENT(CAR(cs));
	l = statement_to_label(stmt);
	if(!entity_empty_label_p(l)||!continue_statement_p(stmt))
	  break;
      }, sequence_statements(instruction_sequence(i)));
      break;    
    case is_instruction_unstructured:
      l = statement_to_label(unstructured_control(instruction_unstructured(i)));
      break;
    case is_instruction_call:
    case is_instruction_test:
    case is_instruction_loop:
    case is_instruction_goto:
      break;
    default:
      pips_error("statement_to_label", "Ill. tag %d for instruction",
		  instruction_tag(i));
    }
  }

  debug(8, "statement_to_label", "stmt %s, pointed by = %s\n",
	statement_identification(s), entity_local_name(l));

  return l;
}

/* Returns FALSE is no syntactic control path exits s (i.e. even if TRUE is returned
 * there might be no control path). Subroutines and
 * functions are assumed to always return to keep the analysis intraprocedural.
 * See the continuation library for more advanced precondition-based analyses.
 *
 * TRUE is a safe default value.
 *
 * The function name is misleading: a RETURN statement does not return...
 * It should be called "statement_does_continue()"
 */
bool
statement_does_return(statement s)
{
  bool returns = TRUE;
  instruction i = statement_instruction(s);
  test t = test_undefined;

  switch(instruction_tag(i)) {
  case is_instruction_sequence:
      MAPL(sts,
           {
	     statement st = STATEMENT(CAR(sts)) ;
	     if (!statement_does_return(st)) {
	       returns = FALSE;
	       break;
	     }
	   },
	     sequence_statements(instruction_sequence(i)));

      break;    
    case is_instruction_unstructured:
      returns = unstructured_does_return(instruction_unstructured(i));
      break;
    case is_instruction_call:
      /* the third condition is due to a bug/feature of unspaghettify */
      returns = !stop_statement_p(s) && !return_statement_p(s) &&
	!(continue_statement_p(s) && entity_return_label_p(statement_label(s)));
      break;
    case is_instruction_test:
      t = instruction_test(i);
      returns = statement_does_return(test_true(t)) ||
	statement_does_return(test_false(t));
      break;
    case is_instruction_loop:
      /* No precise answer, unless you can prove the loop executes at
       * least one iteration.
       */
      returns = TRUE;
      break;
    case is_instruction_goto:
      /* returns = statement_does_return(instruction_goto(i)); */
      returns = FALSE;
      break;
    default:
      pips_error("statement_does_return", "Ill. tag %d for instruction",
		  instruction_tag(i));
  }

  debug(8, "statement_does_return", "stmt %s, does return= %d\n",
	statement_identification(s), returns);

  return returns;
}

bool
unstructured_does_return(unstructured u)
{
  bool returns = TRUE;
  return returns;
}
