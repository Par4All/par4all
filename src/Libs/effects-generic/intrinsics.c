/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: intrinsics.c
 * ~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of 
 * all types of proper effects and proper references of intrinsics.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "makefile.h"

#include "properties.h"
#include "pipsmake.h"

#include "transformer.h"
#include "semantics.h"
#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"


/********************************************************* LOCAL FUNCTIONS */

static list no_write_effects(entity e,list args);
static list affect_effects(entity e,list args);
static list assign_substring_effects(entity e,list args);
static list substring_effect(entity e,list args);
static list io_effects(entity e, list args);
static list effects_of_ioelem(expression exp, tag act);
static list effects_of_iolist(list exprs, tag act);
static list effects_of_implied_do(expression exp, tag act);


/* the following data structure indicates wether an io element generates
a read effects or a write effect. the kind of effect depends on the
instruction type: for instance, access generates a read effect if used
within an open statement, and a write effect when used inside an inquire
statement */

typedef struct IoElementDescriptor {
    string StmtName;
    string IoElementName;
    tag ReadOrWrite, MayOrMust;
} IoElementDescriptor;

static IoElementDescriptor IoElementDescriptorUndefined;

static IoElementDescriptor IoElementDescriptorTable[] = {
    {"OPEN",      "UNIT=",        is_action_read, is_approximation_must},
    {"OPEN",      "ERR=",         is_action_read, is_approximation_may},
    {"OPEN",      "FILE=",        is_action_read, is_approximation_must},
    {"OPEN",      "STATUS=",      is_action_read, is_approximation_may},
    {"OPEN",      "ACCESS=",      is_action_read, is_approximation_must},
    {"OPEN",      "FORM=",        is_action_read, is_approximation_must},
    {"OPEN",      "RECL=",        is_action_read, is_approximation_must},
    {"OPEN",      "BLANK=",       is_action_read, is_approximation_may},
    {"OPEN",      "IOSTAT=",      is_action_write, is_approximation_may},

    {"CLOSE",     "UNIT=",        is_action_read, is_approximation_must},
    {"CLOSE",     "ERR=",         is_action_read, is_approximation_may},
    {"CLOSE",     "STATUS=",      is_action_read, is_approximation_may},
    {"CLOSE",     "IOSTAT=",      is_action_write, is_approximation_may},

    {"INQUIRE",   "UNIT=",        is_action_read, is_approximation_must},
    {"INQUIRE",   "ERR=",         is_action_read, is_approximation_may},
    {"INQUIRE",   "FILE=",        is_action_read, is_approximation_must},
    {"INQUIRE",   "IOSTAT=",      is_action_write, is_approximation_must},
    {"INQUIRE",   "EXIST=",       is_action_write, is_approximation_must},
    {"INQUIRE",   "OPENED=",      is_action_write, is_approximation_must},
    {"INQUIRE",   "NUMBER=",      is_action_write, is_approximation_must},
    {"INQUIRE",   "NAMED=",       is_action_write, is_approximation_must},
    {"INQUIRE",   "NAME=",        is_action_write, is_approximation_must},
    {"INQUIRE",   "ACCESS=",      is_action_write, is_approximation_must},
    {"INQUIRE",   "SEQUENTIAL=",  is_action_write, is_approximation_must},
    {"INQUIRE",   "DIRECT=",      is_action_write, is_approximation_must},
    {"INQUIRE",   "FORM=",        is_action_write, is_approximation_must},
    {"INQUIRE",   "FORMATTED=",   is_action_write, is_approximation_must},
    {"INQUIRE",   "UNFORMATTED=", is_action_write, is_approximation_must},
    {"INQUIRE",   "RECL=",        is_action_write, is_approximation_must},
    {"INQUIRE",   "NEXTREC=",     is_action_write, is_approximation_must},
    {"INQUIRE",   "BLANK=",       is_action_write, is_approximation_must},

    {"BACKSPACE", "UNIT=",        is_action_read, is_approximation_must},
    {"BACKSPACE", "ERR=",         is_action_read, is_approximation_may},
    {"BACKSPACE", "IOSTAT=",      is_action_write, is_approximation_may},

    {"ENDFILE",   "UNIT=",        is_action_read, is_approximation_must},
    {"ENDFILE",   "ERR=",         is_action_read, is_approximation_may},
    {"ENDFILE",   "IOSTAT=",      is_action_write, is_approximation_may},

    {"REWIND",    "UNIT=",        is_action_read, is_approximation_must},
    {"REWIND",    "ERR=",         is_action_read, is_approximation_may},
    {"REWIND",    "IOSTAT=",      is_action_write, is_approximation_may},

    {"READ",      "FMT=",         is_action_read, is_approximation_must},
    {"READ",      "UNIT=",        is_action_read, is_approximation_must},
    {"READ",      "REC=",         is_action_read, is_approximation_must},
    {"READ",      "ERR=",         is_action_read, is_approximation_may},
    {"READ",      "END=",         is_action_read, is_approximation_must},
    {"READ",      "IOSTAT=",      is_action_write, is_approximation_may},
    {"READ",      "IOLIST=",      is_action_write, is_approximation_must},

    {"WRITE",     "FMT=",         is_action_read, is_approximation_must},
    {"WRITE",     "UNIT=",        is_action_read, is_approximation_must},
    {"WRITE",     "REC=",         is_action_read, is_approximation_must},
    {"WRITE",     "ERR=",         is_action_read, is_approximation_may},
    {"WRITE",     "END=",         is_action_read, is_approximation_must},
    {"WRITE",     "IOSTAT=",      is_action_write, is_approximation_may},
    {"WRITE",     "IOLIST=",      is_action_read, is_approximation_must},
    {0,           0,              0}
};


/* the following data structure describes an intrinsic function: its
name and the function to apply on a call to this intrinsic to get the
effects of the call */

typedef struct IntrinsicDescriptor
{
    string name;
    list (*f)();
} IntrinsicDescriptor;

static IntrinsicDescriptor IntrinsicDescriptorTable[] = {
    {"+",                        no_write_effects},
    {"-",                        no_write_effects},
    {"/",                        no_write_effects},
    {"*",                        no_write_effects},
    {"INV",                      no_write_effects},
    {"--",                       no_write_effects},
    {"**",                       no_write_effects},
    {".EQV.",                    no_write_effects},
    {".NEQV.",                   no_write_effects},
    {".OR.",                     no_write_effects},
    {".AND.",                    no_write_effects},
    {".LT.",                     no_write_effects},
    {".GT.",                     no_write_effects},
    {".LE.",                     no_write_effects},
    {".GE.",                     no_write_effects},
    {".EQ.",                     no_write_effects},
    {".NE.",                     no_write_effects},
    {"//",                       no_write_effects},
    {".NOT.",                    no_write_effects},

    {"CONTINUE",                 no_write_effects},
    {"ENDDO",                    no_write_effects},
    {"PAUSE",                    no_write_effects},
    {"RETURN",                   no_write_effects},
    {"STOP",                     no_write_effects},
    {"END",                      no_write_effects},
    {"FORMAT",                   no_write_effects},

    { IMPLIED_COMPLEX_NAME,      no_write_effects},
    { IMPLIED_DCOMPLEX_NAME,     no_write_effects},

    {"INT",                      no_write_effects},
    {"IFIX",                     no_write_effects},
    {"IDINT",                    no_write_effects},
    {"REAL",                     no_write_effects},
    {"FLOAT",                    no_write_effects},
    {"DFLOAT",                   no_write_effects},
    {"SNGL",                     no_write_effects},
    {"DBLE",                     no_write_effects},
    {"DREAL",                    no_write_effects}, /* Added for Arnauld Leservot */
    {"CMPLX",                    no_write_effects},
    {"DCMPLX",                   no_write_effects},
    {"ICHAR",                    no_write_effects},
    {"CHAR",                     no_write_effects},
    {"AINT",                     no_write_effects},
    {"DINT",                     no_write_effects},
    {"ANINT",                    no_write_effects},
    {"DNINT",                    no_write_effects},
    {"NINT",                     no_write_effects},
    {"IDNINT",                   no_write_effects},
    {"IABS",                     no_write_effects},
    {"ABS",                      no_write_effects},
    {"DABS",                     no_write_effects},
    {"CABS",                     no_write_effects},
    {"CDABS",                    no_write_effects},

    {"MOD",                      no_write_effects},
    {"AMOD",                     no_write_effects},
    {"DMOD",                     no_write_effects},
    {"ISIGN",                    no_write_effects},
    {"SIGN",                     no_write_effects},
    {"DSIGN",                    no_write_effects},
    {"IDIM",                     no_write_effects},
    {"DIM",                      no_write_effects},
    {"DDIM",                     no_write_effects},
    {"DPROD",                    no_write_effects},
    {"MAX",                      no_write_effects},
    {"MAX0",                     no_write_effects},
    {"AMAX1",                    no_write_effects},
    {"DMAX1",                    no_write_effects},
    {"AMAX0",                    no_write_effects},
    {"MAX1",                     no_write_effects},
    {"MIN",                      no_write_effects},
    {"MIN0",                     no_write_effects},
    {"AMIN1",                    no_write_effects},
    {"DMIN1",                    no_write_effects},
    {"AMIN0",                    no_write_effects},
    {"MIN1",                     no_write_effects},
    {"LEN",                      no_write_effects},
    {"INDEX",                    no_write_effects},
    {"AIMAG",                    no_write_effects},
    {"DIMAG",                    no_write_effects},
    {"CONJG",                    no_write_effects},
    {"DCONJG",                   no_write_effects},
    {"SQRT",                     no_write_effects},
    {"DSQRT",                    no_write_effects},
    {"CSQRT",                    no_write_effects},

    {"EXP",                      no_write_effects},
    {"DEXP",                     no_write_effects},
    {"CEXP",                     no_write_effects},
    {"LOG",                      no_write_effects},
    {"ALOG",                     no_write_effects},
    {"DLOG",                     no_write_effects},
    {"CLOG",                     no_write_effects},
    {"LOG10",                    no_write_effects},
    {"ALOG10",                   no_write_effects},
    {"DLOG10",                   no_write_effects},
    {"SIN",                      no_write_effects},
    {"DSIN",                     no_write_effects},
    {"CSIN",                     no_write_effects},
    {"COS",                      no_write_effects},
    {"DCOS",                     no_write_effects},
    {"CCOS",                     no_write_effects},
    {"TAN",                      no_write_effects},
    {"DTAN",                     no_write_effects},
    {"ASIN",                     no_write_effects},
    {"DASIN",                    no_write_effects},
    {"ACOS",                     no_write_effects},
    {"DACOS",                    no_write_effects},
    {"ATAN",                     no_write_effects},
    {"DATAN",                    no_write_effects},
    {"ATAN2",                    no_write_effects},
    {"DATAN2",                   no_write_effects},
    {"SINH",                     no_write_effects},
    {"DSINH",                    no_write_effects},
    {"COSH",                     no_write_effects},
    {"DCOSH",                    no_write_effects},
    {"TANH",                     no_write_effects},
    {"DTANH",                    no_write_effects},

    {"LGE",                      no_write_effects},
    {"LGT",                      no_write_effects},
    {"LLE",                      no_write_effects},
    {"LLT",                      no_write_effects},

    {LIST_DIRECTED_FORMAT_NAME,  no_write_effects},
    {UNBOUNDED_DIMENSION_NAME,   no_write_effects},

    {"=",                        affect_effects},

    {"WRITE",                    io_effects},
    {"REWIND",                   io_effects},
    {"BACKSPACE",                io_effects},
    {"OPEN",                     io_effects},
    {"CLOSE",                    io_effects},
    {"INQUIRE",                  io_effects},
    {"READ",                     io_effects},
    {"BUFFERIN",                 io_effects},
    {"BUFFEROUT",                io_effects},
    {"ENDFILE",                  io_effects},
    {IMPLIED_DO_NAME,            effects_of_implied_do},

    {SUBSTRING_FUNCTION_NAME,    substring_effect},
    {ASSIGN_SUBSTRING_FUNCTION_NAME, assign_substring_effects},

    /* These operators are used within the OPTIMIZE transformation in
       order to manipulate operators such as n-ary add and multiply or
       multiply-add operators ( JZ - sept 98) */
    {EOLE_SUM_OPERATOR_NAME,     no_write_effects },
    {EOLE_PROD_OPERATOR_NAME,    no_write_effects },
    {EOLE_FMA_OPERATOR_NAME,     no_write_effects },

    {NULL, 0}
};



/* list generic_proper_effects_of_intrinsic(entity e, list args)
 * input    : a intrinsic function name, and the list or arguments. 
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list
generic_proper_effects_of_intrinsic(entity e, list args)
{
    string s = entity_local_name(e);
    IntrinsicDescriptor *pid = IntrinsicDescriptorTable;
    list lr;

    pips_debug(3, "begin\n");

    while (pid->name != NULL) {
        if (strcmp(pid->name, s) == 0) {
	        lr = (*(pid->f))(e, args);
		pips_debug(3, "end\n");
                return(lr);
	    }

        pid += 1;
    }

    pips_error("generic_proper_effects_of_intrinsic", "unknown intrinsic %s\n", s);

    return(NIL);
}



static list 
no_write_effects(entity e,list args)
{
    list lr;

    debug(5, "no_write_effects", "begin\n");
    lr = generic_proper_effects_of_expressions(args);
    debug(5, "no_write_effects", "end\n");
    return(lr);
}

static list 
affect_effects(entity e,list args)
{
    list le = NIL;

    expression lhs = EXPRESSION(CAR(args));
    syntax s = expression_syntax(lhs);

    expression rhs = EXPRESSION(CAR(CDR(args)));

    pips_debug(5, "begin\n");

    if (! syntax_reference_p(s))
            pips_error("affect_effects", "not a reference\n");

    le = generic_proper_effects_of_lhs(syntax_reference(s));

    le = gen_nconc(le, generic_proper_effects_of_expression(rhs));

    pips_debug(5, "end\n");

    return(le);
}

static list
assign_substring_effects(entity e, list args)
{
    list le = NIL;

    expression lhs = EXPRESSION(CAR(args));
    syntax s = expression_syntax(lhs);
    expression l = EXPRESSION(CAR(CDR(args)));
    expression u = EXPRESSION(CAR(CDR(CDR(args))));
    expression rhs = EXPRESSION(CAR(CDR(CDR(CDR(args)))));

    pips_debug(5, "begin\n");

    if (! syntax_reference_p(s))
            pips_error("assign_substring_effects", "not a reference\n");


    le = generic_proper_effects_of_lhs(syntax_reference(s));
    le = gen_nconc(le, generic_proper_effects_of_expression(l));
    le = gen_nconc(le, generic_proper_effects_of_expression(u));

    le = gen_nconc(le, generic_proper_effects_of_expression(rhs));

    pips_debug(5, "end\n");
    return(le);
}

static list
substring_effect(entity e, list args)
{
    list le = NIL;
    expression expr = EXPRESSION(CAR(args));
    expression l = EXPRESSION(CAR(CDR(args)));
    expression u = EXPRESSION(CAR(CDR(CDR(args))));

    pips_debug(5, "begin\n");

    if (! syntax_reference_p(expression_syntax(expr)))
            pips_error("substring_effect", "not a reference\n");

    le = generic_proper_effects_of_expression(expr);
    le = gen_nconc(le, generic_proper_effects_of_expression(l));
    le = gen_nconc(le, generic_proper_effects_of_expression(u));

    pips_debug(5, "end\n");

    return(le);
}

static IoElementDescriptor*
SearchIoElement(char *s, char *i)
{
    IoElementDescriptor *p = IoElementDescriptorTable;

    while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0 && strcmp(p->IoElementName, i) == 0)
                return(p);
        p += 1;
    }

    pips_error("SearchIoElement", "unknown io element %s %s\n", s, i);
    /* Never reaches this point. Only to avoid a warning at compile time. BC. */
    return(&IoElementDescriptorUndefined);
}

static list
io_effects(entity e, list args)
{
    list le = NIL, pc, lep;

    pips_debug(5, "begin\n");

    for (pc = args; pc != NIL; pc = CDR(pc)) {
	IoElementDescriptor *p;
	entity ci;
        syntax s = expression_syntax(EXPRESSION(CAR(pc)));

        pips_assert("io_effects", syntax_call_p(s));

	ci = call_function(syntax_call(s));
	p = SearchIoElement(entity_local_name(e), entity_local_name(ci));

	pc = CDR(pc);

	if (strcmp(p->IoElementName, "IOLIST=") == 0) {
	    lep = effects_of_iolist(pc, p->ReadOrWrite);
	}
	else {
	    lep = effects_of_ioelem(EXPRESSION(CAR(pc)), 
				    p->ReadOrWrite);
	}

	if (p->MayOrMust == is_approximation_may)        
	    effects_to_may_effects(lep);

	le = gen_nconc(le, lep);

	/* effects effects on logical units - taken from effects/io.c */
	if ((get_bool_property ("PRETTYPRINT_IO_EFFECTS")) &&
	    (pc != NIL) &&
	    (strcmp(p->IoElementName, "UNIT=") == 0))
	{
	    /* We simulate actions on files by read/write actions
	       to a static integer array 
	       GO:
	       It is necessary to do a read and and write action to
	       the array, because it updates the file-pointer so
	       it reads it and then writes it ...*/
	    entity private_io_entity;
	    reference ref;
	    list indices = NIL;

	    indices = gen_nconc(indices,
				CONS(EXPRESSION,
				     EXPRESSION(CAR(pc)),NIL));

	    private_io_entity = global_name_to_entity
		(IO_EFFECTS_PACKAGE_NAME,
		 IO_EFFECTS_ARRAY_NAME);

	    pips_assert("io_effects", private_io_entity != entity_undefined);

	    ref = make_reference(private_io_entity,indices);
	    le = gen_nconc(le, generic_proper_effects_of_reference(ref));
	    le = gen_nconc(le, generic_proper_effects_of_lhs(ref));
	}	
    }

    pips_debug(5, "end\n");

    return(le);
}    

static list
effects_of_ioelem(expression exp, tag act)
{   
    list lr;

    pips_debug(5, "begin\n");
    if (act == is_action_write)
    {
	syntax s = expression_syntax(exp);

	pips_debug(6, "is_action_write\n");
	pips_assert("effects_of_ioelem", syntax_reference_p(s));

	lr = generic_proper_effects_of_lhs(syntax_reference(s));
    }
    else
    {  
	debug(6, "effects_of_io_elem", "is_action_read\n");  
	lr = generic_proper_effects_of_expression(exp);
    }   
 
    pips_debug(5, "end\n");
    return(lr);
}

static list
effects_of_iolist(list exprs, tag act)
{
    list le = NIL;
    list lep = NIL;
    expression exp = EXPRESSION(CAR(exprs));

    pips_debug(5, "begin\n");

    if (expression_implied_do_p(exp)) 
	lep = effects_of_implied_do(exp, act);
    else
    {
	if (act == is_action_write)
	{
	    syntax s = expression_syntax(exp);

	    debug(6, "effects_of_io_list", "is_action_write");
	    /* pips_assert("effects_of_iolist", syntax_reference_p(s)); */
	    if(syntax_reference_p(s))
	      lep = generic_proper_effects_of_lhs(syntax_reference(s));
	    else
	    {
	      /* write action on a substring */
	      if(syntax_call_p(s) &&
		 strcmp(entity_local_name(call_function(syntax_call(s))),
			SUBSTRING_FUNCTION_NAME) == 0 )
	      {
		expression e = EXPRESSION(CAR(call_arguments(syntax_call(s))));
		expression l = EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		expression u = EXPRESSION(CAR(CDR(CDR(call_arguments(syntax_call(s))))));

		lep = generic_proper_effects_of_lhs
		    (syntax_reference(expression_syntax(e)));
		lep = gen_nconc(lep, generic_proper_effects_of_expression(l));
		lep = gen_nconc(lep, generic_proper_effects_of_expression(u));
	      }
	      else {
		pips_error("effects_of_io_list", "Impossible memory write effect!");
	      }
	    }
	}
	else {	
	    debug(6, "effects_of_io_elem", "is_action_read");
	    lep = generic_proper_effects_of_expression(exp);
	}
    }

    le = gen_nconc(le, lep);

    pips_debug(5, "end\n");

    return(le);
}

/* an implied do is a call to an intrinsic function named IMPLIED-DO;
 * its first argument is the loop index, the second one is a range, and the
 * remaining ones are expressions to be written or references to be read,
 * or another implied_do (BA).
 */

static list
effects_of_implied_do(expression exp, tag act)
{
    list le, lep, lr, args;
    expression arg1, arg2; 
    entity index;
    range r;
    reference ref;
    transformer context = effects_private_current_context_head();
    transformer local_context = transformer_undefined;

    pips_assert("effects_of_implied_do", expression_implied_do_p(exp));

    pips_debug(5, "begin\n");

    args = call_arguments(syntax_call(expression_syntax(exp)));
    arg1 = EXPRESSION(CAR(args));       /* loop index */
    arg2 = EXPRESSION(CAR(CDR(args)));  /* range */
    
    pips_assert("effects_of_implied_do", 
		syntax_reference_p(expression_syntax(arg1)));

    pips_assert("effects_of_implied_do", 
		syntax_range_p(expression_syntax(arg2)));

    index = reference_variable(syntax_reference(expression_syntax(arg1)));
    ref = make_reference(index, NIL);

    r = syntax_range(expression_syntax(arg2));

    /* effects of implied do index 
     * it is must_written but may read because the implied loop 
     * might execute no iteration. 
     */

    le = generic_proper_effects_of_lhs(ref); /* the loop index is must-written */
    /* Read effects are masked by the first write to the implied-do loop variable */
	
    /* effects of implied-loop bounds and increment */
    le = gen_nconc(le, generic_proper_effects_of_expression(arg2));

    /* Do we use context information */
    if (! transformer_undefined_p(context))
    {    
	transformer tmp_trans;
	Psysteme context_sc;

    /* the preconditions of the current statement don't include those
     * induced by the implied_do, because they are local to the statement.
     * But we need them to properly calculate the regions.
     * the solution is to add to the current context the preconditions 
     * due to the current implied_do (think of nested implied_do).
     * Beware: the implied-do index variable may already appear 
     * in the preconditions. So we have to eliminate it first.
     * the regions are calculated, and projected along the index.
     * BA, September 27, 1993.
     */

	local_context = transformer_dup(context);
	/* we first eliminate the implied-do index variable */
	context_sc = predicate_system(transformer_relation(local_context));
	if(base_contains_variable_p(context_sc->base, (Variable) index))
	{
	    sc_and_base_projection_along_variable_ofl_ctrl(&context_sc,
							   (Variable) index,
							   NO_OFL_CTRL);
	    predicate_system_(transformer_relation(local_context)) =
		newgen_Psysteme(context_sc);
	}
	/* tmp_trans simulates the transformer of the implied-do loop body */
	tmp_trans = transformer_identity();
	local_context = add_index_range_conditions(local_context, index, r, 
						   tmp_trans);
	free_transformer(tmp_trans);
	transformer_arguments(local_context) = 
	    arguments_add_entity(transformer_arguments(local_context), 
				 entity_to_new_value(index)); 


	ifdebug(7) {
	    pips_debug(7, "local context : \n%s\n", 
		       precondition_to_string(local_context));
	}	
    }
    else
	local_context = transformer_undefined;

    effects_private_current_context_push(local_context);
    
    MAP(EXPRESSION, expr, 
	{ 
	    syntax s = expression_syntax(expr);
	    
	    if (syntax_reference_p(s))
		if (act == is_action_write) 
		    lep = generic_proper_effects_of_lhs(syntax_reference(s));
		else
		    lep = generic_proper_effects_of_expression(expr);
	    else
		if (syntax_range_p(s))
		    lep = generic_proper_effects_of_range(syntax_range(s));
		else
		    /* syntax_call_p(s) is true here */
		    if (expression_implied_do_p(expr))
			lep = effects_of_implied_do(expr, act);
		    else
			lep = generic_r_proper_effects_of_call(syntax_call(s));
	    	    
	    /* indices are removed from effects because this is a loop */
	    lr = NIL;
	    MAP(EFFECT, eff,
		{	     
		    if (effect_entity(eff) != index)
			lr =  effects_add_effect(lr, eff);
		    else
		    {
			debug(5, "effects_of_implied_do", "index removed");
			free_effect(eff);
		    }
		}, lep);
	    gen_free_list(lep);
	    le = gen_nconc(le, lr);	    
	}, CDR(CDR(args)));
    

    (*effects_union_over_range_op)(le, 
				   index, 
				   r, descriptor_undefined);

    ifdebug(6) {
	pips_debug(6, "effects:\n");
	(*effects_prettyprint_func)(le);
	fprintf(stderr, "\n");
    }
    
    effects_private_current_context_pop();
    pips_debug(5, "end\n");

    return(le);
}
