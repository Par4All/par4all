/* 
 *
 * 
 */
#include <stdio.h>
extern int fprintf();
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "ri.h"

#include "ri-util.h"

#include "misc.h"

extern value EvalExpression();

/* this function computes the total size of a variable, ie. the product of
the number of elements and the size of each element. */

int SizeOfArray(e)
entity e;
{
	variable a;

	assert(type_variable_p(entity_type(e)));
	a = type_variable(entity_type(e));

	return(SizeOfElements(variable_basic(a))*
	       NumberOfElements(variable_dimensions(a)));
}




/* this function returns the length in bytes of the fortran type
represented by a basic. */

int SizeOfElements(b)
basic b;
{
	int e;

	switch (basic_tag(b)) {
	    case is_basic_int:
		e = basic_int(b);
		break;
	    case is_basic_float:
		e = basic_float(b);
		break;
	    case is_basic_logical:
		e = basic_logical(b);
		break;
	    case is_basic_complex:
		e = basic_complex(b);
		break;
	    case is_basic_string:
		assert(value_constant_p(basic_string(b)));
		assert(constant_int_p(value_constant(basic_string(b))));
		e = constant_int(value_constant(basic_string(b)));
		break;
	    default:
		fprintf(stderr, "[SizeOfElements] case default\n");
		abort();
	}

	return(e);
}




/* this function computes the number of elements of a variable. ld is the
list of dimensions of the variable */

int NumberOfElements(ld)
cons * ld;
{
    cons *pc;
    int ne = 1;

    for (pc = ld; pc != NULL; pc = CDR(pc)) {
	ne *= SizeOfDimension(DIMENSION(CAR(pc)));
    }

    return(ne);
}



/* this function returns the size of the ith dimension of a variable e. if
called for the 0th dimension, it returns the variable element size. */

int SizeOfIthDimension(e, i)
entity e;
int i;
{
    cons * pc;

    if (!type_variable_p(entity_type(e))) {
	fprintf(stderr, "[SizeOfIthDimension] not a variable\n");
	abort();
    }

    if (i == 0)
	return(SizeOfElements((variable_basic(type_variable(entity_type(e))))));

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	pc = CDR(pc);

    if (pc == NULL) {
	fprintf(stderr, "[SizeOfIthDimension] not enough dimensions\n");
	abort();
    }

    return(SizeOfDimension(DIMENSION(CAR(pc))));
}



/* this function computes the size of a dimension. */

int SizeOfDimension(d)
dimension d;
{
    return(ExpressionToInt(dimension_upper(d))-
	   ExpressionToInt(dimension_lower(d))+1);
}




/* this function computes the value of an integer constant expression
and returns it to the calling function. it aborts if the expression is
not constant. */

int ExpressionToInt(e)
expression e;
{
    value v;
    constant c;
    int i;

    v = EvalExpression(e);

    if (value_constant_p(v)) {
	c = value_constant(v);
	if (constant_int_p(c)) {
	    i = constant_int(c);
	}
	else {
	    fprintf(stderr, "[ExpressionToInt] integer constant expected\n");
	    abort();
	}
    }
    else {
	fprintf(stderr, "[ExpressionToInt] constant expected\n");
	abort();
    }

    gen_free(v);
    return(i);
}

int NumberOfDimension(e)
entity e;
{
    type t = entity_type(e);
    int nd;
    cons *pc;

    assert(type_variable_p(t));

    pc = variable_dimensions(type_variable(t));
    nd = 0;

    while (! ENDP(pc)) {
	nd += 1;
	pc = CDR(pc);
    }

    return(nd);
}



#define INTERVAL_INTERSECTION(a,b,c,d) (!((b) <= (c) || (d) <= (a)))

/* 
this function returns TRUE if e1 and e2 have some memory locations in common
*/

bool entity_conflict_p(e1, e2)
entity e1, e2;
{
    /* a hash table to map entities to their number of elements */
    static hash_table entity_to_size = (hash_table) NULL;

    storage s1, s2;
    ram r1 = ram_undefined, r2 = ram_undefined;
    int o1, o2, l1, l2;
    entity f1, f2, a1, a2;

    if(same_entity_p(e1, e2)) return(TRUE);

    s1 = entity_storage(e1);
    s2 = entity_storage(e2);

    if (! (storage_ram_p(s1) && storage_ram_p(s2)))
	return(FALSE);

    if (entity_to_size == (hash_table) NULL)
	entity_to_size = hash_table_make(hash_pointer, 0);

    if ((l1 = (int) hash_get(entity_to_size, (char *) e1)) == (int) HASH_UNDEFINED_VALUE) {
	l1 = SizeOfArray(e1);
	hash_put(entity_to_size, (char *) e1, (char *) l1);
    }

    r1 = storage_ram(s1);
    f1 = ram_function(r1);
    a1 = ram_section(r1);
    o1 = ram_offset(r1);
    l1 = l1+o1-1;
	    
    if ((l2 = (int) hash_get(entity_to_size, (char *) e2)) == (int) HASH_UNDEFINED_VALUE) {
	l2 = SizeOfArray(e2);
	hash_put(entity_to_size, (char *) e2, (char *) l2);
    }

    r2 = storage_ram(s2);
    f2 = ram_function(r2);
    a2 = ram_section(r2);
    o2 = ram_offset(r2);
    l2 = l2+o2-1;
	    
    /* return(r1 != ram_undefined && r2 != ram_undefined && 
	   f1 == f2 && a1 == a2 &&
	   INTERVAL_INTERSECTION(o1, l1, o2, l2)); */

    /* FI: it's too late to check if r1 and r2 are defined: you already have core dumped!
     * also, f1 and f2 are not relevant since location are governed by area a1 and a2
     */
    return( a1 == a2 && INTERVAL_INTERSECTION(o1, l1, o2, l2));
}
