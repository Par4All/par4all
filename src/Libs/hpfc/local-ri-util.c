
/*
 * What could be in ri-util, or even in ri.h!
 *
 * Fabien Coelho May 1993
 */


#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"

#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "hpfc.h"
#include "defines-local.h"

extern char *flint_print_basic();

/*
#define expression_undefined_p(e) ((e) == expression_undefined)
#define statement_undefined_p(e) ((e) == statement_undefined)
*/



/*
 * entity make_scalar_entity(name, module_name, base)
 */
entity make_scalar_entity(name, module_name, base)
string name;
string module_name;
basic base;
{
    string 
	full_name;
    entity 
	e, f, a;
    basic 
	b = base;

    full_name =
	strdup(concatenate(module_name, MODULE_SEP_STRING, name, NULL));

    debug(8,"make_scalar_entity","name %s\n",full_name);

    e = make_entity(full_name,
		    type_undefined, 
		    storage_undefined, 
		    value_undefined);

    entity_type(e) = (type) MakeTypeVariable(b, NIL);

    f = local_name_to_top_level_entity(module_name);

    /* ??? suddenly this function has started to give back 
     * a strange pointer (0xfffffff0) for newly creatd modules (NODE, HOST), 
     * so I get it out of the job.
     * ??? storage is not initialized from now on. FC 15/09/93.
     * then it seems to work again. FC, the same day.
     */
    a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME); 

    entity_storage(e) = make_storage(is_storage_ram,
				     (make_ram(f, a,
					       (basic_tag(base)!=is_basic_overloaded)?
					       (add_variable_to_area(a, e)):(0),
					       NIL)));

    entity_initial(e) = make_value(is_value_constant,
				   MakeConstantLitteral());

    return(e);
}

/*
 * looks for an entity which should be a scalar of the specified
 * basic. If found, returns it, else one is created.
 */
entity find_or_create_scalar_entity(name, module_name, base)
string name;
string module_name;
tag base;
{
    entity 
	e = entity_undefined;
    string 
	nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

    if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined) 
    {
	pips_assert("find_or_create_scalar_entity",
		    (entity_scalar_p(e) && 
		     entity_basic_p(e, base)));

	return(e);
    }

    return(make_scalar_entity(name, module_name, MakeBasic(base)));
}

/*
 *
 */
basic expression_basic(expr)
expression expr;
{
    syntax the_syntax=expression_syntax(expr);

    switch(syntax_tag(the_syntax))
    {
    case is_syntax_reference:
	return(entity_basic(reference_variable(syntax_reference(the_syntax))));
	break;
    case is_syntax_range:
	/* should be int */
	return(expression_basic(range_lower(syntax_range(the_syntax))));
	break;
    case is_syntax_call:
	/*
	 * here is a little problem with pips...
	 * every intrinsics are overloaded, what is not 
	 * exactly what is desired...
	 */
    {
	return(entity_basic(call_function(syntax_call(the_syntax))));
	break;
    }
    default:
	pips_error("expression_basic","unexpected syntax tag\n");
	break;
    }

    return(basic_undefined);
}

/*
 *
 */
basic MakeBasic(the_tag)
int the_tag;
{
    switch(the_tag)
    {
    case is_basic_int: 
	return(make_basic(is_basic_int,4));
	break;
    case is_basic_float: 
	return(make_basic(is_basic_float,4));
	break;
    case is_basic_logical: 
	return(make_basic(is_basic_logical,4));
	break;
    case is_basic_complex: 
	return(make_basic(is_basic_complex,8));
	break;
    case is_basic_overloaded: 
	return(make_basic(is_basic_overloaded,NULL));
	break;
    case is_basic_string: 
	return(make_basic(is_basic_string,string_undefined));
	break;
    default:
	pips_error("MakeBasic","unexpected basic tag\n");
	break;
    }
    
    return(basic_undefined);
}

/*
 * print_entity_variable(e)
 * 
 * if it is just a variable, the type is printed,
 * otherwise just the entity name is printed
 */
void print_entity_variable(e)
entity e;
{
    variable v;

    (void) fprintf(stderr,"name: %s\n",entity_name(e));
    
    if (!type_variable_p(entity_type(e)))
	return;

    v = type_variable(entity_type(e));

    fprintf(stderr,"basic %s\n",flint_print_basic(variable_basic(v)));
    MAPL(cd,{print_dimension(DIMENSION(CAR(cd)));},variable_dimensions(v));
}

void print_dimension(d)
dimension d;
{
    fprintf(stderr,"dimension :\n");
    print_expression(dimension_lower(d));
    print_expression(dimension_upper(d));
}

list entity_declarations(e)
entity e;
{
    return(code_declarations(entity_code(e)));
}

/*
 * MakeTernaryCall
 */
expression MakeTernaryCall(f,e1,e2,e3)
entity f;
expression e1,e2,e3;
{
/*     pips_assert("MakeTernaryCall",entity_function_p(f)); */

    return(make_expression(make_syntax(is_syntax_call,
				       make_call(f,
						 CONS(EXPRESSION,
						      e1,
						      CONS(EXPRESSION,
							   e2,
							   CONS(EXPRESSION,
								e3,
								NULL))))),
			   normalized_undefined));
						      
}

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


/*
 * FindIthDimension
 */
dimension FindIthDimension(e, i)
entity e;
int i;
{
    cons * pc;

    if (!type_variable_p(entity_type(e))) 
	pips_error("FindIthDimension","not a variable\n");

    if (i <= 0)
	pips_error("FindIthDimension","invalid dimension\n");

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	pc = CDR(pc);

    if (pc == NULL) 
	pips_error("FindIthDimension","not enough dimensions\n");

    return(DIMENSION(CAR(pc)));
}

/*
 * type_variable_dup
 */
type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable
	    v = type_variable(t);

	return(MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v))));
    }
    else
	return(t); /* !!! means sharing */
}

/*
 * ldimensions_dup
 */
list ldimensions_dup(l)
list l;
{
    return((ENDP(l))?
	   (NULL):
	   (CONS(DIMENSION,
		 dimension_dup(DIMENSION(CAR(l))),
		 ldimensions_dup(CDR(l)))));
}

/*
 * dimension_dup
 */
dimension dimension_dup(d)
dimension d;
{
    return(make_dimension(expression_dup(dimension_lower(d)),
			  expression_dup(dimension_upper(d))));
}

/*
 * MakeIntegerResult
 *
 * made after MakeOverloadedResult in ri-util/type.c
 */
type MakeIntegerResult()
{
    return(MakeTypeArray(make_basic(is_basic_int, 4), NIL));
}

/*
 * perfectly_nested_parallel_loop_to_body
 *
 * FC 930722: modified from perfectly_nested_loop_to_body, ri-util,
 */
statement perfectly_nested_parallel_loop_to_body(loop_nest, plloop)
statement loop_nest;
list *plloop;
{
    instruction 
	ins = statement_instruction(loop_nest);

    switch(instruction_tag(ins)) 
    {
    case is_instruction_block: 
    {
	list 
	    lb = instruction_block(ins);
	statement 
	    first_s = STATEMENT(CAR(lb));
	instruction 
	    first_i = statement_instruction(first_s);
	
	if (instruction_loop_p(first_i) && (gen_length(lb)==1)) /* perfect ??? */
	{
	    if (execution_parallel_p(loop_execution(instruction_loop(first_i))))
	    {
		*plloop = gen_nconc(*plloop, CONS(LOOP, instruction_loop(first_i), NULL));
		return(perfectly_nested_parallel_loop_to_body(first_s, plloop));
	    }
	    else 
		return(loop_nest);
	}
	else
	    return(loop_nest);

	break;
    }
    case is_instruction_loop: 
    {
	statement 
	    sbody = loop_body(instruction_loop(ins));

	if (execution_parallel_p(loop_execution(instruction_loop(ins))))
	{
	    *plloop = gen_nconc(*plloop, CONS(LOOP, instruction_loop(ins), NULL));
	    return(perfectly_nested_parallel_loop_to_body(sbody, plloop));
	}
	else
	    return(loop_nest);
	break;
    }
    case is_instruction_test:
    case is_instruction_call:
    case is_instruction_unstructured:
    case is_instruction_goto:
	return(loop_nest);
    default:
	pips_error("perfectly_nested_parallel_loop_to_body","illegal tag\n");
	break;
    }
    
    return(loop_nest); /* just to avoid a gcc warning */
}

/*
 * expression call_to_expression(c)
 */
expression call_to_expression(c)
call c;
{
    return(make_call(make_syntax(is_syntax_call, c),
		     make_normalized(is_normalized_complex, UU)));
}

/*
 * expression make_call_expression(e, l)
 */
expression make_call_expression(e, l)
entity e;
list l;
{
    return(call_to_expression(make_call(e, l)));
}


