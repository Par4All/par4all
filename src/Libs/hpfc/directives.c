/*
 * HPFC module by Fabien COELHO
 *
 * $RCSfile: directives.c,v $ ($Date: 1995/03/23 16:54:39 $, )
 * version $Revision$,
 */

#include <stdio.h>
#include <string.h> 
extern fprintf();
extern system();

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"

#include "ri.h" 
#include "database.h"
#include "hpf.h" 

#include "ri-util.h" 
#include "misc.h" 
#include "properties.h"
#include "pipsdbm.h"
#include "resources.h"
#include "bootstrap.h"
#include "control.h"

#include "hpfc.h"
#include "defines-local.h"

/*   in Syntax
 */
extern void MakeExternalFunction();
extern type MakeTypeVoid();

#define HPF_PREFIX "HPFC"
#define BLOCK_SUFFIX "K"
#define CYCLIC_SUFFIX "C"
#define STAR_SUFFIX "S"

/*-----------------------------------------------------------------
 *
 *   UTILITIES
 *
 */
DEFINE_LOCAL_STACK(current_stmt, statement);

/* recognize an hpf directive special entity.
 * (the prefix of which is HPF_PREFIX, as a convention)
 */
bool hpf_directive_string_p(s)
string s;
{
    int len = strlen(HPF_PREFIX);
    return(strncmp(HPF_PREFIX, s, len)==0);
}

bool hpf_directive_entity_p(e)
entity e;
{
    return(top_level_entity_p(e) && 
	   hpf_directive_string_p(entity_local_name(e)));
}

/* look for hpf directives thru the AST and handle them.
 */

/*-----------------------------------------------------------------
 *
 *  PROCESSORS and TEMPLATE directives.
 *
 * just change the basic type to overloaded and 
 * store the entity as a processor or a template.
 */
static reference expression_to_reference(e)
expression e;
{
    syntax s = expression_syntax(e);
    message_assert("reference expected", syntax_reference_p(s));
    return(syntax_reference(s));
}

static entity expression_to_entity(e)
expression e;
{
    return(reference_variable(expression_to_reference(e)));
}

static list expression_list_to_entity_list(l)
list /* of expressions */ l;
{
    list /* of entities */ n = NIL;

    MAPL(ce,
     {
	 n = CONS(ENTITY, 
		  expression_to_entity(EXPRESSION(CAR(ce))), 
		  n);
     },
	 l);
    
    return(n);		 
}

static void switch_basic_type_to_overloaded(e)
entity e;
{
    basic b = entity_basic(e);
    assert(b!=basic_undefined);
    basic_tag(b)=is_basic_overloaded;
}

static void new_processor(e)
expression e;
{
    entity p = expression_to_entity(e);
    switch_basic_type_to_overloaded(p);
    set_processor(p);

    debug(3, "new_processor", "entity is %s\n", entity_name(p));
}

static void new_template(e)
expression e;
{
    entity t = expression_to_entity(e);
    switch_basic_type_to_overloaded(t);
    set_template(t);

    debug(3, "new_template", "entity is %s\n", entity_name(t));
}

/*-----------------------------------------------------------------
 * one simple ALIGN directive is handled.
 * retrieve the alignment from references array and template
 */
/*  TRUE if the template dimension subscript is an alignment.
 *  FALSE if the dimension is replicated. 
 */
static bool alignment_p(align_src, subscript, padim, prate, pshift)
list /* of expressions */ align_src;
expression subscript;
int *padim;
Value *prate, *pshift;
{
    normalized n = expression_normalized(subscript);
    Pvecteur v, v_src;
    int size, array_dim;

    if (normalized_complex_p(n))
	return(FALSE);

    /*  else the subscript is linear
     */

    v = normalized_linear(n);
    size = vect_size(v);
    *pshift = vect_coeff(TCST, v);

    /*  the alignment should be an affine expression
     */
    message_assert("align subscript is not affine",
		   *pshift==0 ? size<=1 : size<=2)

    /*   constant alignment case
     */
    if (size==0 || (*pshift!=0 && size==1))
    {
	*padim = 0, *prate = 0;
	return(TRUE);
    }

    for(array_dim = 1;
	!ENDP(align_src);
	align_src=CDR(align_src), array_dim++)
    {
	n = expression_normalized(EXPRESSION(CAR(align_src)));
	if (normalized_linear_p(n))
	{
	    v_src = normalized_linear(n);

	    message_assert("simple index expected",
			   vect_size(v_src)==1 && var_of(v_src)!=TCST);
	    
	    *prate = vect_coeff(var_of(v_src), v);

	    if (*prate!=0) 
	    {
		*padim = array_dim;
		return(TRUE);   /* alignment ok */
	    }
	}
    }

    *padim = 0, *prate = 0;
    return(TRUE);
}

static void one_align_directive(alignee, temp)
reference alignee, temp;
{
    list
	aligns    = NIL,
	align_src = reference_indices(alignee),
	align_sub = reference_indices(temp);
    entity 
	template = reference_variable(temp),
	array    = reference_variable(alignee);
    int
	array_dim,
	template_dim = 1;
    Value
	rate, shift;

    assert(entity_template_p(template));

    /*  each array dimension is looked for a possible alignment
     */
    for(template_dim=1; 
	!ENDP(align_sub); 
	align_sub=CDR(align_sub), template_dim++)
    {
	if (alignment_p(align_src, EXPRESSION(CAR(align_sub)),
			&array_dim, &rate, &shift))
	{
	    aligns = CONS(ALIGNMENT, 
			  make_alignment(array_dim,
					 template_dim,
					 Value_to_expression(rate),
					 Value_to_expression(shift)),
			  aligns);
	}
    }
    
    set_array_as_distributed(array);
    store_entity_align(array, make_align(aligns, template));

    debug(3, "one_align_directive", "%s aligned with %s\n",
	  entity_name(array), entity_name(template));
}



/*-----------------------------------------------------------------
 * one DISTRIBUTE directive management
 */
/* returns the expected style tag for the given distribution format,
 * plus a pointer to the list of arguments.
 */
static tag distribution_format(e, pl)
expression e;
list *pl;
{
    syntax s = expression_syntax(e);
    call c;
    entity function;
    string name;

    message_assert("invalid distribution format", syntax_call_p(s));

    c = syntax_call(s);
    function = call_function(c);
    *pl = call_arguments(c);

    message_assert("invalid distribution format", 
		   hpf_directive_entity_p(function));

    name = entity_local_name(function);
    
    if (same_string_p(name, HPF_PREFIX BLOCK_SUFFIX))
	return(is_style_block);
    else 
    if (same_string_p(name, HPF_PREFIX CYCLIC_SUFFIX))
	return(is_style_cyclic);
    else
    if (same_string_p(name, HPF_PREFIX STAR_SUFFIX))
	return(is_style_none);
    else
	user_error("distribution_format", "invalid");

    return(-1); /* just to avoid a gcc warning */
}

static void one_distribute_directive(distributee, proc)
reference distributee, proc;
{
    expression 
	parameter = expression_undefined;
    entity
	processor = reference_variable(proc),
	template  = reference_variable(distributee);
    list
	lformat = reference_indices(distributee),
	largs,
	ldist = NIL;
    tag format;

    assert(ENDP(reference_indices(proc)));

    for(;
	!ENDP(lformat);
	lformat=CDR(lformat))
    {
	format = distribution_format(EXPRESSION(CAR(lformat)), &largs);

	switch (format)
	{
	case is_style_block:
	case is_style_cyclic:
	    message_assert("invalid distribution", gen_length(largs)<=1);

	    if (ENDP(largs))
		parameter = expression_undefined;
	    else
		parameter = EXPRESSION(CAR(largs)),
		EXPRESSION(CAR(largs)) = expression_undefined;
	    break;
	case is_style_none:
	    parameter = expression_undefined;
	    break;
	default:
	    pips_error("one_distribute_directive", "unexpected style tag\n");
	}

	ldist = CONS(DISTRIBUTION, 
		     make_distribution(make_style(format, UU), parameter),
		     ldist);
    }
    
    store_entity_distribute(template, 
			    make_distribute(gen_nreverse(ldist), processor));
}

/*-----------------------------------------------------------------
 *
 *    DIRECTIVE HANDLERS
 *
 * each directive is handled by a function here.
 * these handler may use the statement stack to proceed.
 * signature: void HANDLER (instruction i, list args)
 */

static void handle_unexpected_directive(i, args)
instruction i;
list args;
{
    user_error("handle_hpf_directives", "unexpected hpf directive\n");
}

/*-----------------------------------------------------------------
 *
 * HPF OBJECTS DECLARATIONS
 *
 * namely TEMPLATE and PROCESSORS directives.
 * 
 */
static void handle_processors_directive(i, args)
instruction i;
list args;
{
    gen_map(new_processor, args); /* see new_processor */
}

static void handle_template_directive(i, args)
instruction i;
list args;
{
    gen_map(new_template, args); /* see new_template */
}

/*-----------------------------------------------------------------
 *
 * HPF STATIC MAPPING
 *
 * namely ALIGN and DISTRIBUTE directives.
 *
 */
static void handle_align_directive(i, args)
instruction i;
list args;
{
    list last=gen_last(args);
    reference template;

    /* last points to the last item of args, which should be the template
     */
    assert(gen_length(args)>=2);
    template = expression_to_reference(EXPRESSION(CAR(last)));

    normalize_all_expressions_of(i);

    for(; args!=last; args=CDR(args))
	one_align_directive(expression_to_reference(EXPRESSION(CAR(args))), 
			    template);
    
}

static void handle_distribute_directive(i, args)
instruction i;
list args;
{
    list last=gen_last(args);
    reference proc;

    /* last points to the last item of args, which should be the processors
     */
    assert(gen_length(args)>=2);
    proc = expression_to_reference(EXPRESSION(CAR(last)));

    normalize_all_expressions_of(i);

    for(; args!=last; args=CDR(args))
       one_distribute_directive(expression_to_reference(EXPRESSION(CAR(args))), 
				proc);
}

/*-----------------------------------------------------------------
 *
 * HPF PARALLELISM DIRECTIVES
 *
 * namely INDEPENDENT and NEW directives.
 */
/* ??? I wait for the next statements in a particular order, what
 * should not be necessary. Means I should deal with independent 
 * directives on the PARSED_CODE rather than after the controlized.
 */
static void handle_independent_directive(i0, args)
instruction i0;
list /* of expressions */ args;
{
    list /* of entities */ l = expression_list_to_entity_list(args);
    statement s;
    instruction i;
    loop o;
    entity index;

    debug(2, "handle_independent_directive", "%d index(es)\n", gen_length(l));

    init_ctrl_graph_travel(current_stmt_head(), gen_true);
    
    while(next_ctrl_graph_travel(&s))
    {
	i = statement_instruction(s);
	
	if (instruction_loop_p(i))  /* what we're looking for */
	{
	    o = instruction_loop(i);
	    index = loop_index(o);

	    if (ENDP(l)) /* simple independent case */
	    {
		debug(3, "handle_independent_directive", "parallel loop\n");

		execution_tag(loop_execution(o)) = is_execution_parallel;
		close_ctrl_graph_travel();
		return;
	    }
	    /*  else general independent case
	     */
	    if (gen_find_eq(index, l)==index)
	    {
		debug(3, "handle_independent_directive", 
		      "parallel loop (%s)\n", entity_name(index));

		execution_tag(loop_execution(o)) = is_execution_parallel;
		gen_remove(&l, index);

		if (ENDP(l)) /* the end */
		{
		    close_ctrl_graph_travel();
		    return;
		}
	    }
	    /*  else I should not be here ???
	     */
	}
    }
    
    pips_error("handle_independent_directive", "no loop!\n");
    close_ctrl_graph_travel();
}

static void handle_new_directive(i, args)
instruction i;
list args;
{
    return; /* (that's indeed a first implementation:-) */
}

/*-----------------------------------------------------------------
 *
 * DYNAMIC HPF DIRECTIVES.
 *
 * these functions are not yet implemented in hpfc.
 */
static void handle_dynamic_directive(i, args)
instruction i;
list args;
{
    handle_unexpected_directive(i, args);
}

static void handle_realign_directive(i, args)
instruction i;
list args;
{
    handle_unexpected_directive(i, args);
}

static void handle_redistribute_directive(i, args)
instruction i;
list args;
{
    handle_unexpected_directive(i, args);
}

/*-----------------------------------------------------------------
 *
 * DIRECTIVE HANDLING
 *
 * find the handler for a given entity.
 */
struct DirectiveHandler 
{
  string name;      /* all names should start with the same prefix */
  void (*handler)();
};

static struct DirectiveHandler handlers[] =
{ 
  {HPF_PREFIX "A", handle_align_directive },
  {HPF_PREFIX "B", handle_realign_directive },
  {HPF_PREFIX "D", handle_distribute_directive },
  {HPF_PREFIX "E", handle_redistribute_directive },
  {HPF_PREFIX "I", handle_independent_directive },
  {HPF_PREFIX "N", handle_new_directive },
  {HPF_PREFIX "P", handle_processors_directive },
  {HPF_PREFIX "T", handle_template_directive },
  {HPF_PREFIX "Y", handle_dynamic_directive },
  {HPF_PREFIX BLOCK_SUFFIX,  handle_unexpected_directive },
  {HPF_PREFIX CYCLIC_SUFFIX, handle_unexpected_directive },
  {HPF_PREFIX STAR_SUFFIX,   handle_unexpected_directive },
  { (string) 0, handle_unexpected_directive }
};

static void (*directive_handler(name))()
string name;
{
    struct DirectiveHandler *x=handlers;

    while (x->name!=(string) NULL && strcmp(name,x->name)!=0) 
	x++;

    return(x->handler);
}

/* newgen recursion thru the IR.
 */
static bool directive_filter(i)
instruction i;
{
    if (instruction_call_p(i))
    {
	call c = instruction_call(i);
	entity e = call_function(c);

	if (hpf_directive_entity_p(e))
	{
	    debug(8, "directive_filter", "hpfc entity is %s\n", entity_name(e));
	    /*
	     * call the appropriate handler for the directive
	     */
	    (directive_handler(entity_local_name(e)))
		(i, call_arguments(c));

	    free_call(c);
	    instruction_call(i) = 
		make_call(entity_intrinsic(CONTINUE_FUNCTION_NAME), NIL);
	}

	return(FALSE); /* no instructions within a call! */
    }

    return(TRUE);
}

static bool stmt_filter(s)
statement s;
{
    current_stmt_push(s);
    return(TRUE);
}

static void stmt_rewrite(s)
statement s;
{
    (void) current_stmt_pop();
}

void handle_hpf_directives(s)
statement s;
{
    make_current_stmt_stack();

    gen_multi_recurse(s,
		      /*
		       *  STATEMENT
		       */
		      statement_domain,
		      stmt_filter,
		      stmt_rewrite,
		      /*
		       * INSTRUCTION (I don't want the calls in expressions)
		       */
		      instruction_domain,
		      directive_filter,
		      gen_null,
		      NULL);

    assert(current_stmt_empty_p());
    free_current_stmt_stack();

    DebugPrintCode(5, get_current_module_entity(), s);
}

/* that is all
 */
