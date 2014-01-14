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
/*{{{  banner*/
/* package regions :  Alexis Platonoff, 22 Aout 1990
 *
 * This File contains the main functions that compute the regions of
 * a call to an intrinsic function : "+", "COS", etc...
 */
/*}}}*/
#include "all.h"
/*{{{  data structures for handling descriptors*/
/* the following data structure indicates whether an io element generates
a read regions or a write region. the kind of region depends on the
instruction type: for instance, access generates a read region if used
within an open statement, and a write region when used inside an inquire
statement */

typedef struct IoElementDescriptor {
    string StmtName;
    string IoElementName;
    tag ReadOrWrite, MayOrMust;
} IoElementDescriptor;

static IoElementDescriptor IoElementDescriptorUndefined;

static IoElementDescriptor IoElementDescriptorTable[] = {
    {"OPEN",      "UNIT=",        is_action_read, is_approximation_exact},
    {"OPEN",      "ERR=",         is_action_read, is_approximation_may},
    {"OPEN",      "FILE=",        is_action_read, is_approximation_exact},
    {"OPEN",      "STATUS=",      is_action_read, is_approximation_may},
    {"OPEN",      "ACCESS=",      is_action_read, is_approximation_exact},
    {"OPEN",      "FORM=",        is_action_read, is_approximation_exact},
    {"OPEN",      "RECL=",        is_action_read, is_approximation_exact},
    {"OPEN",      "BLANK=",       is_action_read, is_approximation_may},
    {"OPEN",      "IOSTAT=",      is_action_write, is_approximation_may},

    {"CLOSE",     "UNIT=",        is_action_read, is_approximation_exact},
    {"CLOSE",     "ERR=",         is_action_read, is_approximation_may},
    {"CLOSE",     "STATUS=",      is_action_read, is_approximation_may},
    {"CLOSE",     "IOSTAT=",      is_action_write, is_approximation_may},

    {"INQUIRE",   "UNIT=",        is_action_read, is_approximation_exact},
    {"INQUIRE",   "ERR=",         is_action_read, is_approximation_may},
    {"INQUIRE",   "FILE=",        is_action_read, is_approximation_exact},
    {"INQUIRE",   "IOSTAT=",      is_action_write, is_approximation_exact},
    {"INQUIRE",   "EXIST=",       is_action_write, is_approximation_exact},
    {"INQUIRE",   "OPENED=",      is_action_write, is_approximation_exact},
    {"INQUIRE",   "NUMBER=",      is_action_write, is_approximation_exact},
    {"INQUIRE",   "NAMED=",       is_action_write, is_approximation_exact},
    {"INQUIRE",   "NAME=",        is_action_write, is_approximation_exact},
    {"INQUIRE",   "ACCESS=",      is_action_write, is_approximation_exact},
    {"INQUIRE",   "SEQUENTIAL=",  is_action_write, is_approximation_exact},
    {"INQUIRE",   "DIRECT=",      is_action_write, is_approximation_exact},
    {"INQUIRE",   "FORM=",        is_action_write, is_approximation_exact},
    {"INQUIRE",   "FORMATTED=",   is_action_write, is_approximation_exact},
    {"INQUIRE",   "UNFORMATTED=", is_action_write, is_approximation_exact},
    {"INQUIRE",   "RECL=",        is_action_write, is_approximation_exact},
    {"INQUIRE",   "NEXTREC=",     is_action_write, is_approximation_exact},
    {"INQUIRE",   "BLANK=",       is_action_write, is_approximation_exact},

    {"BACKSPACE", "UNIT=",        is_action_read, is_approximation_exact},
    {"BACKSPACE", "ERR=",         is_action_read, is_approximation_may},
    {"BACKSPACE", "IOSTAT=",      is_action_write, is_approximation_may},

    {"ENDFILE",   "UNIT=",        is_action_read, is_approximation_exact},
    {"ENDFILE",   "ERR=",         is_action_read, is_approximation_may},
    {"ENDFILE",   "IOSTAT=",      is_action_write, is_approximation_may},

    {"REWIND",    "UNIT=",        is_action_read, is_approximation_exact},
    {"REWIND",    "ERR=",         is_action_read, is_approximation_may},
    {"REWIND",    "IOSTAT=",      is_action_write, is_approximation_may},

    {"READ",      "FMT=",         is_action_read, is_approximation_exact},
    {"READ",      "UNIT=",        is_action_read, is_approximation_exact},
    {"READ",      "REC=",         is_action_read, is_approximation_exact},
    {"READ",      "ERR=",         is_action_read, is_approximation_may},
    {"READ",      "END=",         is_action_read, is_approximation_exact},
    {"READ",      "IOSTAT=",      is_action_write, is_approximation_may},
    {"READ",      "IOLIST=",      is_action_write, is_approximation_exact},

    {"WRITE",     "FMT=",         is_action_read, is_approximation_exact},
    {"WRITE",     "UNIT=",        is_action_read, is_approximation_exact},
    {"WRITE",     "REC=",         is_action_read, is_approximation_exact},
    {"WRITE",     "ERR=",         is_action_read, is_approximation_may},
    {"WRITE",     "END=",         is_action_read, is_approximation_exact},
    {"WRITE",     "IOSTAT=",      is_action_write, is_approximation_may},
    {"WRITE",     "IOLIST=",      is_action_read, is_approximation_exact},
    {0,           0,              0,              0}
};


/* the following data structure describes an intrinsic function: its
name and the function to apply on a call to this intrinsic to get the
effects of the call */

typedef struct {
    string name;
    list (*f)();
} IntrinsicEffectDescriptor;

static IntrinsicEffectDescriptor IntrinsicDescriptorTable[] = {
    {"+",                        no_write_comp_regions},
    {"-",                        no_write_comp_regions},
    {"/",                        no_write_comp_regions},
    {"*",                        no_write_comp_regions},
    {"--",                       no_write_comp_regions},
    {"**",                       no_write_comp_regions},
    {".EQV.",                    no_write_comp_regions},
    {".NEQV.",                   no_write_comp_regions},
    {".OR.",                     no_write_comp_regions},
    {".AND.",                    no_write_comp_regions},
    {".LT.",                     no_write_comp_regions},
    {".GT.",                     no_write_comp_regions},
    {".LE.",                     no_write_comp_regions},
    {".GE.",                     no_write_comp_regions},
    {".EQ.",                     no_write_comp_regions},
    {".NE.",                     no_write_comp_regions},
    {"//",                       no_write_comp_regions},
    {".NOT.",                    no_write_comp_regions},

    {"CONTINUE",                 no_write_comp_regions},
    {"ENDDO",                    no_write_comp_regions},
    {"PAUSE",                    no_write_comp_regions},
    {"RETURN",                   no_write_comp_regions},
    {"STOP",                     no_write_comp_regions},
    {"END",                      no_write_comp_regions},
    {"FORMAT",                   no_write_comp_regions},

    {"INT",                      no_write_comp_regions},
    {"IFIX",                     no_write_comp_regions},
    {"IDINT",                    no_write_comp_regions},
    {"REAL",                     no_write_comp_regions},
    {"FLOAT",                    no_write_comp_regions},
    {"SNGL",                     no_write_comp_regions},
    {"DBLE",                     no_write_comp_regions},
    {"CMPLX",                    no_write_comp_regions},
    {"ICHAR",                    no_write_comp_regions},
    {"CHAR",                     no_write_comp_regions},
    {"AINT",                     no_write_comp_regions},
    {"DINT",                     no_write_comp_regions},
    {"ANINT",                    no_write_comp_regions},
    {"DNINT",                    no_write_comp_regions},
    {"NINT",                     no_write_comp_regions},
    {"IDNINT",                   no_write_comp_regions},
    {"IABS",                     no_write_comp_regions},
    {"ABS",                      no_write_comp_regions},
    {"DABS",                     no_write_comp_regions},
    {"CABS",                     no_write_comp_regions},

    {"MOD",                      no_write_comp_regions},
    {"AMOD",                     no_write_comp_regions},
    {"DMOD",                     no_write_comp_regions},
    {"ISIGN",                    no_write_comp_regions},
    {"SIGN",                     no_write_comp_regions},
    {"DSIGN",                    no_write_comp_regions},
    {"IDIM",                     no_write_comp_regions},
    {"DIM",                      no_write_comp_regions},
    {"DDIM",                     no_write_comp_regions},
    {"DPROD",                    no_write_comp_regions},
    {"MAX",                      no_write_comp_regions},
    {"MAX0",                     no_write_comp_regions},
    {"AMAX1",                    no_write_comp_regions},
    {"DMAX1",                    no_write_comp_regions},
    {"AMAX0",                    no_write_comp_regions},
    {"MAX1",                     no_write_comp_regions},
    {"MIN",                      no_write_comp_regions},
    {"MIN0",                     no_write_comp_regions},
    {"AMIN1",                    no_write_comp_regions},
    {"DMIN1",                    no_write_comp_regions},
    {"AMIN0",                    no_write_comp_regions},
    {"MIN1",                     no_write_comp_regions},
    {"LEN",                      no_write_comp_regions},
    {"INDEX",                    no_write_comp_regions},
    {"AIMAG",                    no_write_comp_regions},
    {"CONJG",                    no_write_comp_regions},
    {"SQRT",                     no_write_comp_regions},
    {"DSQRT",                    no_write_comp_regions},
    {"CSQRT",                    no_write_comp_regions},

    {"EXP",                      no_write_comp_regions},
    {"DEXP",                     no_write_comp_regions},
    {"CEXP",                     no_write_comp_regions},
    {"LOG",                      no_write_comp_regions},
    {"ALOG",                     no_write_comp_regions},
    {"DLOG",                     no_write_comp_regions},
    {"CLOG",                     no_write_comp_regions},
    {"LOG10",                    no_write_comp_regions},
    {"ALOG10",                   no_write_comp_regions},
    {"DLOG10",                   no_write_comp_regions},
    {"SIN",                      no_write_comp_regions},
    {"DSIN",                     no_write_comp_regions},
    {"CSIN",                     no_write_comp_regions},
    {"COS",                      no_write_comp_regions},
    {"DCOS",                     no_write_comp_regions},
    {"CCOS",                     no_write_comp_regions},
    {"TAN",                      no_write_comp_regions},
    {"DTAN",                     no_write_comp_regions},
    {"ASIN",                     no_write_comp_regions},
    {"DASIN",                    no_write_comp_regions},
    {"ACOS",                     no_write_comp_regions},
    {"DACOS",                    no_write_comp_regions},
    {"ATAN",                     no_write_comp_regions},
    {"DATAN",                    no_write_comp_regions},
    {"ATAN2",                    no_write_comp_regions},
    {"DATAN2",                   no_write_comp_regions},
    {"SINH",                     no_write_comp_regions},
    {"DSINH",                    no_write_comp_regions},
    {"COSH",                     no_write_comp_regions},
    {"DCOSH",                    no_write_comp_regions},
    {"TANH",                     no_write_comp_regions},
    {"DTANH",                    no_write_comp_regions},

    {"LGE",                      no_write_comp_regions},
    {"LGT",                      no_write_comp_regions},
    {"LLE",                      no_write_comp_regions},
    {"LLT",                      no_write_comp_regions},

    {LIST_DIRECTED_FORMAT_NAME,  no_write_comp_regions},
    {UNBOUNDED_DIMENSION_NAME,   no_write_comp_regions},

    {"=",                        affect_comp_regions},

    {"WRITE",                    io_comp_regions},
    {"REWIND",                   io_comp_regions},
    {"OPEN",                     io_comp_regions},
    {"CLOSE",                    io_comp_regions},
    {"INQUIRE",                  io_comp_regions},
    {"READ",                     io_comp_regions},
    {"BUFFERIN",                 io_comp_regions},
    {"BUFFEROUT",                io_comp_regions},
    {"ENDFILE",                  io_comp_regions},
    {IMPLIED_DO_NAME,            comp_regions_of_implied_do},
    {NULL, 0}
};
/*}}}*/


/*{{{  comp_regions_of_intrinsic*/
/* list comp_regions_of_intrinsic(entity e, list args, transformer context)
 * input    : a intrinsic function name, the list or arguments, and
 *            the calling context. 
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_intrinsic(e, args, context)
entity e;
list args;
transformer context;
{
    return (proper_comp_regions_of_intrinsic(e, args, context));
}
/*}}}*/
/*{{{  proper_comp_regions_of_intrinsic*/
/* list proper_comp_regions_of_intrinsic(entity e, list args, transformer context)
 * input    : a intrinsic function name, the list or arguments, and
 *            the calling context. 
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list proper_comp_regions_of_intrinsic(e, args, context)
entity e;
list args;
transformer context;
{
    const char* s = entity_local_name(e);
    IntrinsicEffectDescriptor *pid = IntrinsicDescriptorTable;
    list lr;

    debug(3, "proper_comp_regions_of_intrinsic", "begin\n");

    while (pid->name != NULL) {
        if (strcmp(pid->name, s) == 0) {
	        lr = (*(pid->f))(e, args, context);
		      debug(3, "proper_comp_regions_of_intrinsic", "end\n");
                return(lr);
	    }

        pid += 1;
    }

    pips_internal_error("unknown intrinsic %s", s);

    return(NIL);
}
/*}}}*/
/*{{{  no_write_comp_regions*/
/*===============================================================================*/
list
no_write_comp_regions(entity __attribute__ ((unused)) e,
		      list args,
		      transformer context)
{
    list lr;

    debug(5, "no_write_comp_regions", "begin\n");
    lr = comp_regions_of_expressions(args,context);
    debug(5, "no_write_comp_regions", "end\n");
    return(lr);
}


/*}}}*/
/*{{{  affect_comp_regions*/
/*===============================================================================*/
list
affect_comp_regions(entity __attribute__ ((unused)) e,
		    list args,
		    transformer context)
{
    list le = NIL;

    expression lhs = EXPRESSION(CAR(args));
    syntax s = expression_syntax(lhs);

    expression rhs = EXPRESSION(CAR(CDR(args)));

    debug(5, "affect_comp_regions", "begin\n");

    if (! syntax_reference_p(s))
            pips_internal_error("not a reference");


    le = comp_regions_of_write(syntax_reference(s), context);

    le = CompRegionsExactUnion(le, comp_regions_of_expression(rhs, context), 
			  effects_same_action_p);

    debug(5, "affect_comp_regions", "end\n");

    return(le);
}
/*}}}*/

/*{{{  SearchIoElements*/
static IoElementDescriptor *SearchIoElement(s, i)
const char *s, *i;
{
    IoElementDescriptor *p = IoElementDescriptorTable;

    while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0 && strcmp(p->IoElementName, i) == 0)
                return(p);
        p += 1;
    }

    pips_internal_error("unknown io element %s %s", s, i);
    /* Never reaches this point. Only to avoid a warning at compile time. BC. */
    return(&IoElementDescriptorUndefined);
}
/*}}}*/
/*{{{  io_comp_regions*/
list io_comp_regions(e, args, context)
entity e;
list args;
transformer context;
{
    list le = NIL, pc, lep;

    debug(5, "io_comp_regions", "begin\n");

    for (pc = args; pc != NIL; pc = CDR(pc)) {
	IoElementDescriptor *p;
	entity ci;
        syntax s = expression_syntax(EXPRESSION(CAR(pc)));

        pips_assert("io_comp_regions", syntax_call_p(s));

	ci = call_function(syntax_call(s));
	p = SearchIoElement(entity_local_name(e), entity_local_name(ci));

	pc = CDR(pc);

	if (strcmp(p->IoElementName, "IOLIST=") == 0) {
	    lep = comp_regions_of_iolist(pc, p->ReadOrWrite, context);
	}
	else {
	    lep = comp_regions_of_ioelem(EXPRESSION(CAR(pc)), 
				    p->ReadOrWrite, context);
	}

	if (p->MayOrMust == is_approximation_may) 
	{
	    MAP(REGION, reg, 
	    {
		approximation_tag(effect_approximation(reg)) = 
		    is_approximation_may;
	    }, lep);
	}

	le = CompRegionsExactUnion(le, lep, effects_same_action_p);

	/* regions effects on logical units - taken from effects/io.c */
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

	    private_io_entity = FindEntity(TOP_LEVEL_MODULE_NAME,
		 IO_EFFECTS_ARRAY_NAME);

	    pips_assert("regions_effects", private_io_entity != entity_undefined);

	    ref = make_reference(private_io_entity,indices);
	    le = CompRegionsExactUnion(le, 
				 comp_regions_of_read(ref, context), 
				 effects_same_action_p);
	    le = CompRegionsExactUnion(le, 
				 comp_regions_of_write(ref, context), 
				 effects_same_action_p);
	    
	}
	
    }

    debug(5, "io_comp_regions", "end\n");

    return(le);
}    
/*}}}*/
/*{{{  comp_regions_of_io_element*/
/*===============================================================================*/
list comp_regions_of_ioelem(exp, act, context)
expression exp;
tag act;
transformer context;
{   
    list lr;

    debug(5, "comp_regions_of_io_elem", "begin\n");
    if (act == is_action_write) {
	syntax s = expression_syntax(exp);

	debug(6, "comp_regions_of_io_elem", "is_action_write\n");
	pips_assert("comp_regions_of_ioelem", syntax_reference_p(s));

	lr = comp_regions_of_write(syntax_reference(s), context);
    }
    else {  
	debug(6, "comp_regions_of_io_elem", "is_action_read\n");  
	lr = comp_regions_of_expression(exp, context);
    }   
 
    debug(5, "comp_regions_of_elem", "end\n");

    return(lr);
}
/*}}}*/
/*{{{  comp_regions_of_io_list*/
/*===============================================================================*/
list comp_regions_of_iolist(exprs, act, context)
list exprs;
tag act;
transformer context;
{
  list le = NIL;
  debug(5, "comp_regions_of_io_list", "begin\n");
  while (!ENDP(exprs))
    {
      expression exp = EXPRESSION(CAR(exprs));
      list lep = NIL;
	
      /* There is a bug with effects of io list  
       
	 READ *,N,(T(I),I=1,N)

	 there is write effect on N but for the io list, 
	 we don't have this effect !
       
	 Cause : there is no loop for the list exprs !!! /NN:2000/ */

      if (expression_implied_do_p(exp)) {
	lep = comp_regions_of_implied_do(exp, act, context);
      }
      else {
	if (act == is_action_write) {
	  syntax s = expression_syntax(exp);

	  debug(6, "comp_regions_of_io_list", "is_action_write");
	  pips_assert("comp_regions_of_iolist", syntax_reference_p(s));
	  lep = comp_regions_of_write(syntax_reference(s), context);
	}
	else {	
	  debug(6, "comp_regions_of_io_elem", "is_action_read");
	  lep = comp_regions_of_expression(exp, context);
	}
      }
      
      le = CompRegionsExactUnion(le, lep, effects_same_action_p);
      
      exprs = CDR(exprs);
    }

  debug(5, "comp_regions_of_io_list", "end\n");

  return(le);
}
/*}}}*/
/*{{{  comp_regions_of_implied_do*/
/* an implied do is a call to an intrinsic function named IMPLIED-DO;
 * its first argument is the loop index, the second one is a range, and the
 * remaining ones are expressions to be written or references to be read,
 * or another implied_do (BA).
 */

list comp_regions_of_implied_do(exp, act, context)
expression exp;
tag act;
transformer context;
{
    list le, lep, lr, args;
    transformer local_context;
    expression arg1, arg2; 
    entity index;
    range r;
    reference ref;

    pips_assert("comp_regions_of_implied_do", expression_implied_do_p(exp));

    debug(5, "comp_regions_of_implied_do", "begin\n");

    args = call_arguments(syntax_call(expression_syntax(exp)));
    arg1 = EXPRESSION(CAR(args));       /* loop index */
    arg2 = EXPRESSION(CAR(CDR(args)));  /* range */
    
    pips_assert("comp_regions_of_implied_do", 
		syntax_reference_p(expression_syntax(arg1)));

    pips_assert("comp_regions_of_implied_do", 
		syntax_range_p(expression_syntax(arg2)));


    index = reference_variable(syntax_reference(expression_syntax(arg1)));
    ref = make_reference(index, NIL);

    r = syntax_range(expression_syntax(arg2));

    /* regions of implied do index 
     * it is must_written but may read because the implied loop 
     * might execute no iteration. 
     */

    le = comp_regions_of_write(ref, context); /* the loop index is must-written */
    /* Read effects are masked by the first write to the implied-do loop variable */
	
    /* regions of implied-loop bounds and increment */
    le = CompRegionsExactUnion(comp_regions_of_expression(arg2,context), le, 
			  effects_same_action_p);


    /* the preconditions of the current statement don't include those
     * induced by the implied_do, because they are local to the statement.
     * But we need them to properly calculate the regions.
     * the solution is to add to the current context the preconditions 
     * due to the current implied_do (think of nested implied_do).
     * the regions are calculated, and projected along the index.
     * BA, September 27, 1993.
     */

    local_context = transformer_dup(context);
    local_context = add_index_range_conditions(local_context, index, r, 
					       transformer_undefined);
    transformer_arguments(local_context) = 
	arguments_add_entity(transformer_arguments(local_context), 
			     entity_to_new_value(index)); 

    ifdebug(7) {
	debug(7, "comp_regions_of_implied_do", "local context : \n%s\n", 
	      precondition_to_string(local_context));
    }

    MAP(EXPRESSION, expr, 
	{ 
	    syntax s = expression_syntax(expr);
	    
	    if (syntax_reference_p(s))
		if (act == is_action_write) 
		    lep = comp_regions_of_write(syntax_reference(s),local_context);
		else
		    lep = comp_regions_of_expression(expr, local_context);
	    else
		/* on a un autre implied_do imbrique' */
		lep = comp_regions_of_implied_do(expr, act, local_context);
	    
	    
	    /* indices are removed from regions because this is a loop */
	    lr = NIL;
	    MAP(REGION, reg,
		{	     
		    if (region_entity(reg) != index)
			lr =  region_add_to_regions(reg,lr);
		    else
		    {
			debug(5, "comp_regions_of_implied_do", "index removed");
			region_free(reg);
		    }
		}, lep);
	    gen_free_list(lep);
	    le = CompRegionsExactUnion(le, lr, effects_same_action_p);
	    
	}, CDR(CDR(args)));
    
    ifdebug(7) {
	debug(7, "comp_regions_of_implied_do", "regions before projection :\n");
	print_regions(le);
	fprintf(stderr, "\n");
    }
    
    project_regions_along_loop_index(le, 
				     entity_to_new_value(index), 
				     r);

    ifdebug(6) {
	debug(6, "comp_regions_of_implied_do", "regions after projection :\n");
	print_regions(le);
	fprintf(stderr, "\n");
    }


    transformer_free(local_context);
    local_context = transformer_undefined;
    
    debug(5, "comp_regions_of_implied_do", "end\n");

    return(le);
}
/*}}}*/







