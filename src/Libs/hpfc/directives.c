/* HPFC module by Fabien COELHO
 *
 * these functions deal with HPF directives.
 *
 * $RCSfile: directives.c,v $ version $Revision$,
 * ($Date: 1995/07/21 16:32:47 $, )
 */

#include "defines-local.h"

#include "pipsdbm.h"
#include "resources.h"
#include "bootstrap.h"
#include "control.h"

/*  directive names encoding: HPF_PREFIX + one character
 */
#define HPF_PREFIX     "HPFC"

#define BLOCK_SUFFIX   "K"
#define CYCLIC_SUFFIX  "C"
#define STAR_SUFFIX    "S"

#define ALIGN_SUFFIX   "A"
#define REALIGN_SUFFIX "B"
#define DIST_SUFFIX    "D"
#define REDIST_SUFFIX  "E"
#define INDEP_SUFFIX   "I"
#define NEW_SUFFIX     "N"
#define PROC_SUFFIX    "P"
#define TEMPL_SUFFIX   "T"
#define PURE_SUFFIX    "U"
#define DYNA_SUFFIX    "Y"

/*-----------------------------------------------------------------
 *
 *   UTILITIES
 *
 */
/* local primary dynamics
 */
GENERIC_STATIC_STATUS(/**/, the_dynamics, list, NIL, gen_free_list)

void add_a_dynamic(c)
entity c;
{
    the_dynamics = gen_once(c, the_dynamics);
}

/*  the local stack is used to retrieve the current statement while 
 *  scanning the AST with gen_recurse.
 */
DEFINE_LOCAL_STACK(current_stmt, statement);

/* recognize an hpf directive special entity.
 * (the prefix of which is HPF_PREFIX, as a convention)
 * both functions are available, based on the name and on the entity.
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

bool realign_directive_p(f)
entity f;
{
    return(top_level_entity_p(f) && 
	   same_string_p(HPF_PREFIX REALIGN_SUFFIX, entity_local_name(f)));
}

bool redistribute_directive_p(f)
entity f;
{
    return(top_level_entity_p(f) && 
	   same_string_p(HPF_PREFIX REDIST_SUFFIX, entity_local_name(f)));
}

/*-----------------------------------------------------------------
 *
 *  PROCESSORS and TEMPLATE directives.
 *
 * just change the basic type to overloaded and 
 * store the entity as a processor or a template.
 */
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
    add_a_dynamic(a);

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

    /*  the alignment should be a simple affine expression
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

    /* built align is returned. should be normalized?
     */
    return(make_align(aligns, template));
}

/* handle s as the initial alignment...
 * to be called after the dynamics arrays...
 */
static void initial_alignment(s)
statement s;
{
    MAP(ENTITY, array,
    {
	if (array_distributed_p(array))
        {
	    propagate_synonym(s, array, array);
	    update_renamings(s, CONS(RENAMING, make_renaming(array, array),
				     load_renamings(s)));
	}
    },
	get_the_dynamics());
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

    normalize_align(array, a);
    
    debug(3, "one_align_directive", "%s %saligned with %s\n",
	  entity_name(array), dynamic ? "re" : "", entity_name(template));

    if (dynamic)
    {
	statement current = current_stmt_head();
	entity new_array;

	message_assert("realigning non dynamic array",
		       array_distributed_p(array) && dynamic_entity_p(array));

	new_array = array_synonym_aligned_as(array, a);
	propagate_synonym(current, array, new_array);
	update_renamings(current, 
			 CONS(RENAMING, make_renaming(array, new_array),
			      load_renamings(current)));
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
    list last = gen_last(args);
    reference template;

    /* last points to the last item of args, which should be the template
     */
    assert(gen_length(args)>=2);
    template = expression_to_reference(EXPRESSION(CAR(last)));

    gen_map(normalize_all_expressions_of, args);

    if (dynamic) store_renamings(current_stmt_head(), NIL);

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
list /* of expressions */ *pl;
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
    
    if (same_string_p(name, HPF_PREFIX BLOCK_SUFFIX))  /* BLOCK() */
	return(is_style_block);
    else 
    if (same_string_p(name, HPF_PREFIX CYCLIC_SUFFIX)) /* CYCLIC() */
	return(is_style_cyclic);
    else
    if (same_string_p(name, HPF_PREFIX STAR_SUFFIX))   /* * [star] */
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
    expression parameter = expression_undefined;
    entity processor = reference_variable(proc);
    list
	/* of expressions */   lformat = reference_indices(distributee),
	                       largs,
	/* of distributions */ ldist = NIL;
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

	    parameter = ENDP(largs) ? 
		expression_undefined :                   /* implicit size */
		copy_expression(EXPRESSION(CAR(largs))); /* explicit size */

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

/*  handles a simple (one template) distribute or redistribute directive.
 */
static void one_distribute_directive(distributee, proc, dynamic)
reference distributee, proc;
bool dynamic;
{
    entity processor = reference_variable(proc),
           template  = reference_variable(distributee);
    distribute
	d = extract_the_distribute(distributee, proc);

    assert(ENDP(reference_indices(proc))); /* no ... ONTO P(something) */
    
    normalize_distribute(template, d);

    debug(3, "one_distribute_directive", "%s %sdistributed onto %s\n",
	  entity_name(template), dynamic ? "re" : "", entity_name(processor));

    if (dynamic)
    {
	statement current = current_stmt_head();
	entity new_t;

	message_assert("redistributing non dynamic template",
		  entity_template_p(template) && dynamic_entity_p(template));

	new_t = template_synonym_distributed_as(template, d);
	propagate_synonym(current, template, new_t);

	/*  all arrays aligned to template are propagated in turn.
	 */
	MAP(ENTITY, array,
	 {
	     align a = new_align_with_template(load_entity_align(array), new_t);
	     entity new_array = array_synonym_aligned_as(array, a);

	     propagate_synonym(current, array, new_array);
	     update_renamings(current, 
			      CONS(RENAMING, make_renaming(array, new_array),
				   load_renamings(current)));
	 },
	    alive_arrays(current, template));
    }
    else
	store_entity_distribute(template, d);
}

/*  handles a full distribute or redistribute directive.
 */
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

    /*  calls the simple case handler.
     */
    for(; args!=last; POP(args))
       one_distribute_directive(expression_to_reference(EXPRESSION(CAR(args))), 
				proc, dynamic);
}

/*-----------------------------------------------------------------
 *
 *    DIRECTIVE HANDLERS
 *
 * each directive is handled by a function here.
 * these handlers may use the statement stack to proceed.
 * signature: void HANDLER (entity f, list args)
 */

/*  default case issues an error.
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
 *
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

	    if (ENDP(l)) /* simple independent case, first loop is tagged // */
	    {
		debug(3, "handle_independent_directive", "parallel loop\n");

		execution_tag(loop_execution(o)) = is_execution_parallel;
		close_ctrl_graph_travel();
		return;
	    }
	    /*  else general independent case (with a list of indexes)
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
    user_error("handle_independent_directive", "some loop not found!\n");
}

/* ??? not implemented and not used. The independent directive is trusted
 * by the compiler to apply its optimizations...
 */
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

/*   may be used to declare functions as pure. 
 *   ??? it is not a directive in HPF, but I put it this way in F77.
 *   ??? pure declarations are not yet used by HPFC.
 */
static void handle_pure_directive(f, args)
entity f;
list /* of expressions */ args;
{
    entity module = get_current_module_entity();
    assert(ENDP(args));
    add_a_pure(module);

    debug(3, "handle_pure_directive", "entity is %s\n", entity_name(module));
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
  {HPF_PREFIX BLOCK_SUFFIX,   handle_unexpected_directive },
  {HPF_PREFIX CYCLIC_SUFFIX,  handle_unexpected_directive },
  {HPF_PREFIX STAR_SUFFIX,    handle_unexpected_directive },
  {HPF_PREFIX ALIGN_SUFFIX,   handle_align_directive },
  {HPF_PREFIX REALIGN_SUFFIX, handle_realign_directive },
  {HPF_PREFIX DIST_SUFFIX,    handle_distribute_directive },
  {HPF_PREFIX REDIST_SUFFIX,  handle_redistribute_directive },
  {HPF_PREFIX INDEP_SUFFIX,   handle_independent_directive },
  {HPF_PREFIX NEW_SUFFIX,     handle_new_directive },
  {HPF_PREFIX PROC_SUFFIX,    handle_processors_directive },
  {HPF_PREFIX TEMPL_SUFFIX,   handle_template_directive },
  {HPF_PREFIX DYNA_SUFFIX,    handle_dynamic_directive },
  {HPF_PREFIX PURE_SUFFIX,    handle_pure_directive },
  { (string) NULL,            handle_unexpected_directive }
};

/* returns the handler for directive name.
 * assumes the name should point to a directive.
 */
static void (*directive_handler(name))()
string name;
{
    struct DirectiveHandler *x=handlers;
    while (x->name!=(string) NULL && strcmp(name,x->name)!=0) x++;
    return(x->handler);
}

/* list of statements to be cleaned. the operation is delayed because
 * the directives are needed in place to stop the dynamic updates.
 */
static list /* of statements */ to_be_cleaned = NIL;

/* the directive is freed and replaced by a continue call or
 * a copy loop nest, depending on the renamings.
 */
static void clean_statement(s)
statement s;
{
    instruction i = statement_instruction(s);

    assert(instruction_call_p(i));

    free_call(instruction_call(i));
    instruction_call(i) = call_undefined;

    if (bound_renamings_p(s))
    {
	list /* of renamings */  lr = load_renamings(s),
	     /* of statements */ block = NIL;
	
	debug(4, "clean_statement",
	      "remapping statement 0x%x\n", (unsigned int) s);

	MAP(RENAMING, r,
	{
	    entity o = renaming_old(r);
	    entity n = renaming_new(r);
	    
	    debug(5, "clean_statement", 
		  "%s -> %s\n", entity_name(o), entity_name(n));
	    block = CONS(STATEMENT, generate_copy_loop_nest(o, n), block);
	},
	    lr);

	free_instruction(i);
	statement_instruction(s) =
	    make_instruction(is_instruction_block, block);
    }
    else
	instruction_call(i) =
	    make_call(entity_intrinsic(CONTINUE_FUNCTION_NAME), NIL);
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

	/* call the appropriate handler for the directive.
	 */
	(directive_handler(entity_local_name(f)))(f, call_arguments(c));
	
	/* the current statement will have to be cleaned.
	 */
	to_be_cleaned = CONS(STATEMENT, current_stmt_head(), to_be_cleaned);
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

/* void handle_hpf_directives(s)
 * statement s;
 *
 * what: handles the HPF directives in statement s.
 * how: recurses thru the AST, looking for special "directive calls".
 *      when found, a special handler is called for the given directive.
 * input: the code statement s
 * output: none
 * side effects: (many)
 *  - the hpfc data structures are set/updated to store the hpf mapping.
 *  - parallel loops are tagged parallel.
 *  - a static stack is used to retrieve the current statement.
 *  - the ctrl_graph travelling is used, so should be initialized.
 *  - the special calls are freed and replaced by continues.
 * bugs or features:
 *  - the "new" directive is not used to tag private variables.
 */
void handle_hpf_directives(s)
statement s;
{
    make_current_stmt_stack();
    init_dynamic_locals();
    init_the_dynamics();

    to_be_cleaned = NIL;
    store_renamings(s, NIL);

    gen_multi_recurse(s,
        statement_domain,  stmt_filter,      stmt_rewrite, /* STATEMENT */
	expression_domain, gen_false,        gen_null,     /* EXPRESSION */
	call_domain,       directive_filter, gen_null,     /* CALL */
		      NULL);

    initial_alignment(s);

    DEBUG_STAT(7, "intermediate code", s);

    simplify_remapping_graph();
    gen_map(clean_statement, to_be_cleaned);
    assert(current_stmt_empty_p());

    gen_free_list(to_be_cleaned), to_be_cleaned=NIL;
    free_current_stmt_stack();
    close_dynamic_locals();
    close_the_dynamics();

    DEBUG_CODE(5, "resulting code", get_current_module_entity(), s);
}

/* that is all
 */
