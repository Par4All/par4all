 /* 
  *
  */
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

static bool module_coherent_p=TRUE;
static entity checked_module = entity_undefined;

bool variable_in_module_p2(v,m)
entity v;
entity m;
{
    bool in_module_1 = 
	same_string_p(module_local_name(m), entity_module_name(v));
    list decl =  code_declarations (value_code(entity_initial (m)));

    bool in_module_2 = entity_is_argument_p(v,decl);

    ifdebug(8) {
	if (!in_module_1 || !in_module_2) {
	    pips_debug(8, "variable %s not in declarations of %s\n", 
		       entity_name(v),entity_name(m));
	    dump_arguments(decl);
	}
    }
    return (in_module_1 && in_module_2);
}


void variable_declaration_verify(reference ref)
{

    if (!variable_in_module_p2(reference_variable(ref),checked_module))
    { 	module_coherent_p = FALSE;
	fprintf(stderr,
			   "variable non declaree %s\n",
			   entity_name(reference_variable(ref)));
    }
}

void symbolic_constant_declaration_verify(call c)
{
    
    if (value_symbolic_p(entity_initial(call_function(c)))) {
	if (!variable_in_module_p2(call_function(c),checked_module))
	{    module_coherent_p = FALSE;
	     ifdebug(4) fprintf(stderr,
				"variable non declaree %s\n",
				entity_name(call_function(c)));
	 }
    }
}

void add_non_declared_reference_to_declaration(reference ref)
{
    entity var = reference_variable(ref);
    /* type t = entity_type(var); */

    if (!variable_in_module_p2(var,checked_module))
    { 
	value val = entity_initial(checked_module);
	code ce = value_code(val);
	list decl =  code_declarations (ce);
	entity ent= entity_undefined;
	string name=strdup(concatenate(
	    module_local_name(checked_module), 
	    MODULE_SEP_STRING,
	    entity_local_name(var), 
	    NULL));
	if ((ent = gen_find_tabulated(name,entity_domain)) ==entity_undefined)
	{
	    ifdebug(8) 
		(void) fprintf(stderr,
			       "ajout de la variable symbolique %s au module %s\n",
			       entity_name(var), 
			       entity_name(checked_module)); 

	    ent =  make_entity(name,
			       gen_copy_tree(entity_type(var)),
			       gen_copy_tree(entity_storage(var)),
			       gen_copy_tree(entity_initial(var)));

	    code_declarations(ce) = gen_nconc(decl,
					      CONS(ENTITY, ent, NIL));
	}
    }
}

void add_symbolic_constant_to_declaration(call c)
{    
    if (value_symbolic_p(entity_initial(call_function(c)))) {
	value val = entity_initial(checked_module);
	code ce = value_code(val);
	list decl =  code_declarations (ce);
	entity ent= entity_undefined;
	string name=strdup(concatenate(module_local_name(checked_module), 
				       MODULE_SEP_STRING,
				       entity_local_name(call_function(c)), NULL));
	if ((ent = gen_find_tabulated(name,entity_domain)) == entity_undefined) {
	    ifdebug(8) 
		(void) fprintf(stderr,"ajout de la variable symbolique %s au module %s\n",
			       entity_name(call_function(c)), 
			       entity_name(checked_module)); 

	    ent =  make_entity( name,
			       gen_copy_tree(entity_type(call_function(c))),
			       gen_copy_tree(entity_storage(call_function(c))),
			       gen_copy_tree(entity_initial(call_function(c))));

	    code_declarations(ce) = gen_nconc(decl,
					      CONS(ENTITY, ent, NIL));
	}
    }
}

bool variable_declaration_coherency_p(entity module, statement st)
{
    module_coherent_p = TRUE;
    checked_module = module;
    gen_recurse(st,
		reference_domain,
		gen_true,
		add_non_declared_reference_to_declaration);
    module_coherent_p = TRUE;
    gen_recurse(st,
		call_domain,
		gen_true,
		add_symbolic_constant_to_declaration);
    checked_module = entity_undefined;
    return(module_coherent_p);
}

/* ------------------------------------------------------------------
 *
 * new and as efficient as possible version
 * My programs may be rather long with many references and variables...
 *
 * FC 08/06/94
 */

/* should be in Newgen? */
#ifndef bool_undefined
#define bool_undefined ((bool) (-1))
#define bool_undefined_p(b) ((b)==bool_undefined)
#endif

/* ??? used as ok or undefined 
 */
GENERIC_GLOBAL_FUNCTION(referenced_variables, entity_int)
GENERIC_LOCAL_FUNCTION(declared_variables, entity_int)

/* the list is needed to insure a deterministic scanning, what would be
 * rather random over installations thru hash_table_map for instance.
 */
static list /* of entity */
    referenced_variables_list = NIL;

static void store_this_variable(var)
entity var;
{
    assert(!entity_undefined_p(var));

    /* I put this out because of shared entities (HPFC-PACKAGE)
     * that must be considered again. ??? 
     */
    referenced_variables_list = 
	CONS(ENTITY, var, referenced_variables_list);

    if (!bound_referenced_variables_p(var))
    {
	pips_debug(9, "%s\n", entity_name(var));
	store_referenced_variables(var, TRUE);
    }
}

static void store_a_referenced_variable(ref)
reference ref;
{
    /* assert(!reference_undefined_p(ref)); */
    store_this_variable(reference_variable(ref));
}

static void store_the_loop_index(l)
loop l;
{
    /* assert(!loop_undefined_p(l)); */
    store_this_variable(loop_index(l));
}

/*  to be called if the referenced map need not be shared between modules
 *  otherwise the next function is okay.
 */
void insure_declaration_coherency_of_module(
    entity module,
    statement stat)
{
    init_referenced_variables();
    insure_declaration_coherency(module, stat, NIL);
    close_referenced_variables();
}

/*  the referenced_variable_map is global and must be made/freed
 *  before/after the call
 */
void insure_declaration_coherency(
    entity module,
    statement stat,
    list /* of entity */ le) /* added entities, for includes... */
{
    list decl = entity_declarations(module), new_decl = NIL, added = NIL;

    debug_on("RI_UTIL_DEBUG_LEVEL");

    assert(!referenced_variables_undefined_p());

    pips_debug(5, "Processing module %s\n", entity_name(module));

    init_declared_variables();
    referenced_variables_list = NIL;
    
    MAP(ENTITY, e, store_this_variable(e), le);
    /* MAP(ENTITY, e, store_declared_variables(e, TRUE), decl); */

    gen_multi_recurse(stat,
		      /*   Direct References   
		       */
		      reference_domain, gen_true, store_a_referenced_variable,
		      /*   References in Loops   
		       */
		      loop_domain, gen_true, store_the_loop_index,
		      NULL);

    /*    checks each declared variable for a reference
     */
    MAPL(ce,
     {
	 entity var = ENTITY(CAR(ce));

	 if (!entity_variable_p(var) ||
	     !local_entity_of_module_p(var, module) ||
	     storage_formal_p(entity_storage(var)) ||
	     value_symbolic_p(entity_initial(var)) ||
	     (bound_referenced_variables_p(var) &&
	      !bound_declared_variables_p(var)))
	 {
	     pips_debug(7, "declared variable %s is referenced, kept\n",
			entity_name(var));
	     new_decl = CONS(ENTITY, var, new_decl);
	     store_declared_variables(var, TRUE);
	 }
	 else
	 {
	     pips_debug(7, "declared variable %s not referenced, removed\n",
			entity_name(var));
	 }
     },
	 decl);

    /*    checks each referenced variable for a declaration 
     */
    MAPL(ce,
     {
	 entity var = ENTITY(CAR(ce));

	 if (!bound_declared_variables_p(var) &&
	     !basic_overloaded_p(entity_basic(var)))
	 {
	     pips_debug(7, "referenced var %s added\n", entity_name(var));
	     added = CONS(ENTITY, var, added);
	     store_declared_variables(var, TRUE);
	 }
	 else
	 {
	     pips_debug(7, "referenced var %s declared\n", entity_name(var));
	 }
     },
	 referenced_variables_list);

    new_decl = gen_nreverse(new_decl); /* back to initial order */
    new_decl = gen_nconc(new_decl, added);

    /*  More determinism: the declarations are sorted
     * ??? bug: parameters cannot be sorted...
     */
    /* gen_sort_list(new_decl, compare_entities); */
    entity_declarations(module) = new_decl; /* rough! */

    /* the temporaries are cleaned
     */
    close_declared_variables();
    gen_free_list(decl);
    gen_free_list(referenced_variables_list),
    referenced_variables_list = NIL;

    debug_off();
}


/*  Look for the end of header comments: */
static string
get_end_of_header_comments(string the_comments)
{
    string end_of_comment = the_comments;
    
    for(;;) {
	string next_line;
	/* If we are at the end of string or the line is not a comment
           (it must be a PROGRAM, FUNCTION,... line), it is the end of
           the header comment: */
	if (!comment_string_p(end_of_comment))
	    break;

	if ((next_line = strchr(end_of_comment, '\n')) != NULL
	    || *end_of_comment !=  '\0')
	    /* Look for next line */
	    end_of_comment = next_line + 1;
	else
	    /* No return char: it is the last line and it is a
               comment... Should never happend in a real program
               without any useful header! */
	    pips_assert("get_end_of_header_comments found only comments!",
			FALSE);
    }
    return end_of_comment;
}


/* Get the header comments (before PROGRAM, FUNCTION,...) from the
   text declaration: */
sentence
get_header_comments(entity module)
{
    int length;
    string end_of_comment;
    string extracted_comment;
    /* Get the textual header: */
    string the_comments = code_decls_text(entity_code(module));

    end_of_comment = get_end_of_header_comments(the_comments);
    /* Keep room for the trailing '\0': */
    length = end_of_comment - the_comments;
    extracted_comment = (string) malloc(length + 1);
    (void) strncpy(extracted_comment, the_comments, length);
    extracted_comment[length] = '\0';
    return make_sentence(is_sentence_formatted, extracted_comment);  
}


/* Get all the declaration comments, that are comments from the
   PROGRAM, FUNCTION,... stuff up to the end of the declarations: */
sentence
get_declaration_comments(entity module)
{
    string comment;
    string old_extracted_comments;
    string extracted_comments = strdup("");
    /* Get the textual header: */
    string the_comments = code_decls_text(entity_code(module));
    
    comment = get_end_of_header_comments(the_comments);
    /* Now, gather all the comments: */
    for(;;) {
	string next_line;
	if (*comment == '\0')
	    break;
	next_line = strchr(comment, '\n');
	if (comment_string_p(comment)) {
	    string the_comment;
	    /* Add it to the extracted_comments: */
	    if (next_line == NULL)
		/* There is no newline, so this is the last
		   comment line: */
		the_comment = strdup(comment);
	    else {
		/* Build a string that holds the comment: */
		the_comment = (string) malloc(next_line - comment + 2);
		(void) strncpy(the_comment, comment, next_line - comment + 1);
		the_comment[next_line - comment + 1] = '\0';
	    }
	    old_extracted_comments = extracted_comments;
	    extracted_comments = strdup(concatenate(old_extracted_comments,
						    the_comment,
						    NULL));
	    free(old_extracted_comments);
	    free(the_comment);
	}
	/* Have a look to next line if any... */
	if (next_line == NULL)
	    break;
	comment = next_line + 1;
    }
    
    return make_sentence(is_sentence_formatted, extracted_comments);
}

/*
 *  that is all
 */
