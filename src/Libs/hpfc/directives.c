/* HPFC module by Fabien COELHO
 *
 * these functions deal with HPF directives.
 *
 * $RCSfile: directives.c,v $ ($Date: 1995/04/03 17:00:36 $, )
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
    MAPL(ce, n = CONS(ENTITY, expression_to_entity(EXPRESSION(CAR(ce))), n), l);
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

static void new_dynamic(e)
expression e;
{
    entity a = expression_to_entity(e);
    set_entity_as_dynamic(a);

    debug(3, "new_dynamic", "entity is %s\n", entity_name(a));
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

    /*  else the subscript is affine
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

    /*   affine alignment case
     */
    for(array_dim = 1; !ENDP(align_src); POP(align_src), array_dim++)
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

    /*   matching array dimension not found, replicated!
     */
    *padim = 0, *prate = 0;
    return(FALSE);
}

/*  builds an align from the alignee and template references.
 *  used by both align and realign management.
 */
static align extract_the_align(alignee, temp)
reference alignee, temp;
{
    list
	/* of alignments  */ aligns    = NIL,
	/* of expressions */ align_src = reference_indices(alignee),
	                     align_sub = reference_indices(temp);
    entity template = reference_variable(temp);
    int array_dim, template_dim = 1;
    Value rate, shift;

    assert(entity_template_p(template));
    
    /*  each array dimension is looked for a possible alignment
     */
    for(template_dim=1; !ENDP(align_sub); POP(align_sub), template_dim++)
    {
	if (alignment_p(align_src, EXPRESSION(CAR(align_sub)),
			&array_dim, &rate, &shift))
	    aligns = CONS(ALIGNMENT, 
			  make_alignment(array_dim,
					 template_dim,
					 Value_to_expression(rate),
					 Value_to_expression(shift)),
			  aligns);
    }

    return(make_align(aligns, template)); /* built align is returned */
}

static void one_align_directive(alignee, temp, dynamic)
reference alignee, temp;
bool dynamic;
{
    entity 
	template = reference_variable(temp),
	array    = reference_variable(alignee);
    align
	a = extract_the_align(alignee, temp);
    
    debug(3, "one_align_directive", "%s %saligned with %s\n",
	  entity_name(array), dynamic ? "re" : "", entity_name(template));

    if (dynamic)
    {
	assert(array_distributed_p(array) && dynamic_entity_p(array));

	/*  existing array? propagation? and so on...
	 */
	pips_error("one_align_directive", "dynamic not implemented yet");
    }
    else
    {
	set_array_as_distributed(array);
	store_entity_align(array, a);
    }       
}

static void handle_align_and_realign_directive(f, args, dynamic)
entity f;
list /* of expressions */ args;
bool dynamic;
{
    list last=gen_last(args);
    reference template;

    /* last points to the last item of args, which should be the template
     */
    assert(gen_length(args)>=2);
    template = expression_to_reference(EXPRESSION(CAR(last)));

    gen_map(normalize_all_expressions_of, args);

    for(; args!=last; POP(args))
	one_align_directive(expression_to_reference(EXPRESSION(CAR(args))), 
			    template, dynamic);
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
    entity function;
    string name;
    call c;

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

/*  builds the distribute from the distributee and processor references.
 */
static distribute extract_the_distribute(distributee, proc)
reference distributee, proc;
{
    expression 
	parameter = expression_undefined;
    entity
	processor = reference_variable(proc);
    list
	lformat = reference_indices(distributee),
	largs,
	ldist = NIL;
    tag format;

    /* the template arguments are scanned to build the distribution
     */
    for(; !ENDP(lformat); POP(lformat))
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
    
    return(make_distribute(gen_nreverse(ldist), processor));
}

static void one_distribute_directive(distributee, proc, dynamic)
reference distributee, proc;
bool dynamic;
{
    entity
	processor = reference_variable(proc),
	template  = reference_variable(distributee);
    distribute
	d = extract_the_distribute(distributee, proc);

    assert(ENDP(reference_indices(proc))); /* no ... ONTO P(something) */

    debug(3, "one_distribute_directive", "%s %sdistributed onto %s\n",
	  entity_name(template), dynamic ? "re" : "", entity_name(processor));

    if (dynamic)
    {
	/* existing template, and related arrays, renamming, propagation...
	 */
	pips_error("one_redistribute_directive", "not implemented yet");
    }
    else
	store_entity_distribute(template, d);
}

static void handle_distribute_and_redistribute_directive(f, args, dynamic)
entity f;
list /* of expressions */ args;
bool dynamic;
{
    list /* of expression */ last = gen_last(args);
    reference proc;

    /* last points to the last item of args, which should be the processors
     */
    assert(gen_length(args)>=2);
    proc = expression_to_reference(EXPRESSION(CAR(last)));

    gen_map(normalize_all_expressions_of, args);

    for(; args!=last; POP(args))
       one_distribute_directive(expression_to_reference(EXPRESSION(CAR(args))), 
				proc, dynamic);
}

/*-----------------------------------------------------------------
 *
 *    DIRECTIVE HANDLERS
 *
 * each directive is handled by a function here.
 * these handler may use the statement stack to proceed.
 * signature: void HANDLER (entity f, list args)
 */

static void handle_unexpected_directive(f, args)
entity f;
list /* of expressions */ args;
{
    user_error("handle_hpf_directives", "unexpected hpf directive\n");
}

/*-----------------------------------------------------------------
 *
 * HPF OBJECTS DECLARATIONS
 *
 *   namely TEMPLATE and PROCESSORS directives.
 */
static void handle_processors_directive(f, args)
entity f;
list /* of expressions */ args;
{
    gen_map(new_processor, args); /* see new_processor */
}

static void handle_template_directive(f, args)
entity f;
list /* of expressions */ args;
{
    gen_map(new_template, args); /* see new_template */
}

/*-----------------------------------------------------------------
 *
 * HPF STATIC MAPPING
 *
 *   namely ALIGN and DISTRIBUTE directives.
 */
static void handle_align_directive(f, args)
entity f;
list /* of expressions */ args;
{
    handle_align_and_realign_directive(f, args, FALSE);
}

static void handle_distribute_directive(f, args)
entity f;
list /* of expressions */ args;
{
    handle_distribute_and_redistribute_directive(f, args, FALSE);
}

/*-----------------------------------------------------------------
 *
 * HPF PARALLELISM DIRECTIVES
 *
 *   namely INDEPENDENT and NEW directives.
 */
/* ??? I wait for the next statements in a particular order, what
 * should not be necessary. Means I should deal with independent 
 * directives on the PARSED_CODE rather than after the CONTROLIZED.
 */
static void handle_independent_directive(f, args)
entity f;
list /* of expressions */ args;
{
    list /* of entities */ l = expression_list_to_entity_list(args);
    instruction i;
    entity index;
    statement s;
    loop o;

    debug(2, "handle_independent_directive", "%d index(es)\n", gen_length(l));

    /*  travels thru the full control graph to find the loops 
     *  and tag them as parallel.
     */
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
	    if (gen_in_list_p(index, l))
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
	}
    }
    
    close_ctrl_graph_travel();
    user_error("handle_independent_directive", "no loop found!\n");
}

static void handle_new_directive(f, args)
entity f;
list /* of expressions */ args;
{
    return; /* (that's indeed a first implementation:-) */
}

/*-----------------------------------------------------------------
 *
 * DYNAMIC HPF DIRECTIVES.
 *
 */
static void handle_dynamic_directive(f, args)
entity f;
list /* of expressions */ args;
{
    gen_map(new_dynamic, args); /* see new_dynamic */
}

static void handle_realign_directive(f, args)
entity f;
list /* of expressions */ args;
{
    handle_align_and_realign_directive(f, args, TRUE);
}

static void handle_redistribute_directive(f, args)
entity f;
list /* of expressions */ args;
{
    handle_distribute_and_redistribute_directive(f, args, TRUE);
}

/*-----------------------------------------------------------------
 *
 * DIRECTIVE HANDLING
 *
 *   find the handler for a given entity.
 */
struct DirectiveHandler 
{
  string name;       /* all names should start with the same prefix */
  void (*handler)(); /* handler for directive "name" */
};

static struct DirectiveHandler handlers[] =
{ 
  {HPF_PREFIX BLOCK_SUFFIX,  handle_unexpected_directive },
  {HPF_PREFIX CYCLIC_SUFFIX, handle_unexpected_directive },
  {HPF_PREFIX STAR_SUFFIX,   handle_unexpected_directive },
  {HPF_PREFIX "A", handle_align_directive },
  {HPF_PREFIX "B", handle_realign_directive },
  {HPF_PREFIX "D", handle_distribute_directive },
  {HPF_PREFIX "E", handle_redistribute_directive },
  {HPF_PREFIX "I", handle_independent_directive },
  {HPF_PREFIX "N", handle_new_directive },
  {HPF_PREFIX "P", handle_processors_directive },
  {HPF_PREFIX "T", handle_template_directive },
  {HPF_PREFIX "Y", handle_dynamic_directive },
  { (string) NULL, handle_unexpected_directive }
};

/* returns the handler for directive name.
 */
static void (*directive_handler(name))()
string name;
{
    struct DirectiveHandler *x=handlers;
    while (x->name!=(string) NULL && strcmp(name,x->name)!=0) x++;
    return(x->handler);
}

/* newgen recursion thru the IR.
 */
static bool directive_filter(c)
call c;
{
    entity f = call_function(c);
    
    if (hpf_directive_entity_p(f))
    {
	debug(8, "directive_filter", "hpfc entity is %s\n", entity_name(f));

	/* call the appropriate handler for the directive
	 */
	(directive_handler(entity_local_name(f)))(f, call_arguments(c));
	
	/*  the directive is switched to a CONTINUE call.
	 */
	free_call(c);
	instruction_call(statement_instruction(current_stmt_head())) = 
	    make_call(entity_intrinsic(CONTINUE_FUNCTION_NAME), NIL);
    }
    
    return(FALSE); /* no instructions within a call! */
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
        statement_domain, stmt_filter, stmt_rewrite, /* STATEMENT */
	expression_domain, gen_false, gen_null,      /* EXPRESSION */
	call_domain, directive_filter, gen_null,     /* CALL */
		      NULL);

    assert(current_stmt_empty_p());
    free_current_stmt_stack();
}

/* that is all
 */
