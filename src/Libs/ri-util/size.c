/* 
 * $Id$
 *
 * $Log: size.c,v $
 * Revision 1.15  1999/01/12 20:39:02  irigoin
 * Performance improvements for entity_conflict_p(). Instead of gathering all
 * information and making a final test, partial tests are performed as soon
 * as possible. Function storage_space_of_variable() is expensive and should
 * not be called unless necessary. It might be possible to speed up
 * entity_conflict_p() some more by checking scalar variables and by avoiding
 * the call to storage_space_of_variable().
 *
 *
 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "ri-util.h"
#include "misc.h"

extern value EvalExpression();

/* this function computes the total size of a variable, ie. the product of
the number of elements and the size of each element. */

bool
SizeOfArray(entity e, int * s)
{
	variable a;
	bool ok = TRUE;
	int se = 0;
	int ne = 0;

	assert(type_variable_p(entity_type(e)));
	a = type_variable(entity_type(e));

	se = SizeOfElements(variable_basic(a));
	ok = NumberOfElements(variable_dimensions(a), &ne);
	* s = ne * se;

	return ok;
}

int
array_size(entity a)
{
    int s = 0;

    if(!SizeOfArray(a, &s)) {
	pips_error("array_size",
		   "Array \"%s\" with illegal varying array size\n",
		   entity_name(a));
    }
    return s;
}

Value 
ValueSizeOfArray(entity e)
{
	variable a;
	Value longueur, taille_elt;

	assert(type_variable_p(entity_type(e)));
	a = type_variable(entity_type(e));

	taille_elt = (Value) SizeOfElements(variable_basic(a));
	longueur = ValueNumberOfElements(variable_dimensions(a));
	
	return(value_mult(taille_elt,longueur));
	
}
    

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/* this function returns the length in bytes of the fortran type
represented by a basic. */

int 
SizeOfElements(b)
basic b;
{
  int e = -1;

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
  case is_basic_string: {
    constant c = constant_undefined;

    /* pips_assert("SizeOfElements", gen_consistent_p(b)); */

    if(value_constant_p(basic_string(b)))
      c = value_constant(basic_string(b));
    else if(value_symbolic_p(basic_string(b)))
      c = symbolic_constant(value_symbolic(basic_string(b)));
    else
      user_error("SizeOfElements",
		 "Sizing of character variable by illegal value (tag=%d)",
		 basic_tag(b));

    if(constant_int_p(c))
      e = constant_int(c);
    else
      user_error("SizeOfElements",
		 "Sizing of character variable by non-integer constant");
    break;
  }
  default:
    pips_error("SizeOfElements", "Ill. tag %d for basic", basic_tag(b));
  }

  return e;
}

/* END_EOLE */


/* this function computes the number of elements of a variable. ld is the
list of dimensions of the variable */

int
element_number(list ld)
{
    int en = 0;

    if(!NumberOfElements(ld, &en)) {
	pips_error("element_number", "Probably varying size array\n");
    }

    return en;
}

bool
NumberOfElements(list ld, int * n)
{
    list pc;
    int ne = 1;
    bool ok = TRUE;

    for (pc = ld; pc != NULL && ok; pc = CDR(pc)) {
	int s;
	ok = SizeOfDimension(DIMENSION(CAR(pc)), &s);
	ne *= s;
    }

    *n = ne;
    return ok;
}

Value 
ValueNumberOfElements(list ld)
{
    list pc;
    Value ne = VALUE_ONE;

    for (pc = ld; pc != NULL; pc = CDR(pc)) {
	ne = value_mult(ne, ValueSizeOfDimension(DIMENSION(CAR(pc))));
    }

    return(ne);
}



/* this function returns the size of the ith dimension of a variable e. if
called for the 0th dimension, it returns the variable element size. */

int 
SizeOfIthDimension(entity e, int i)
{
    list pc = NIL;
    int s = 0;

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

    if(!(SizeOfDimension(DIMENSION(CAR(pc)), &s))) {
	fprintf(stderr, "[SizeOfIthDimension] Non constant %dth dimension\n", i);
	abort();
    }

    return s;
}



/* this function computes the size of a dimension. */

int
dimension_size(dimension d)
{
    int s = 0;

    if(!SizeOfDimension(d, &s)) {
	pips_error("dimension_size", "Probably varying size array\n");
    }

    return s;
}

bool
SizeOfDimension(dimension d, int * s)
{
    int l = 0;
    int u = 0;
    bool ok = TRUE;
	
    ok = expression_integer_value(dimension_upper(d), &u) &&
	   expression_integer_value(dimension_lower(d), &l);
    *s = u - l + 1;
    return ok;
}


/* FI: I do not understand the "Value" cast */

Value 
ValueSizeOfDimension(dimension d)
{
    Value dl, du;
    int l = 0 ;
    int u = 0;
    bool ok;

    ok = expression_integer_value(dimension_upper(d), &u);
    du = (Value) u;
    ok = ok && expression_integer_value(dimension_lower(d), &l);
    dl = (Value) l;

    if(!ok) {
	fprintf(stderr, "[ValueSizeOfIthDimension] Non constant dimension\n");
	abort();
    }
    
    return(value_plus(value_minus(du,dl), VALUE_ONE));
}




/* this function computes the value of an integer constant expression
 * and returns it to the calling function. it aborts if the expression is
 * not constant.
 *
 * See expression_integer_value() to check before aborting
 */

int 
ExpressionToInt(expression e)
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

    free_value(v);
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


/* a hash table to map entities to their numbers of elements
 *
 * This table is critical to compute use-def chains at an acceptable
 * speed because the computation of a variable allocated space
 * is very slow. Declaration are preserved in PIPS and constant
 * expressions must be evaluated.
 *
 * Note: the current implementation is not safe. The hash table
 * may be kept from module to module (which should be OK) and 
 * from workspace to workspace, which is not.
 *
 * This is an unusual object because it's life time is the
 * workspace life time and not the module analysis life time.
 * This hash table acts as a cache of the symbol table.
 */
static hash_table entity_to_size = hash_table_undefined;

void set_entity_to_size()
{
    if (entity_to_size != hash_table_undefined) {
	pips_error("set_entity_to_size",
		   "hash table should have been deallocated\n");
	/* hash_table_clear(entity_to_size); */
    }

    entity_to_size = hash_table_make(hash_pointer, 0);
}

void reset_entity_to_size()
{
    if (entity_to_size == hash_table_undefined) {
	pips_error("reset_entity_to_size",
		   "hash table should have been allocated\n");
    }
    else {
	hash_table_free(entity_to_size);
	entity_to_size = hash_table_undefined;
    }
}

int 
storage_space_of_variable(entity v)
{
    /* Storage size is expressed in bytes */
    int l;

    if (entity_to_size == hash_table_undefined) {
	user_warning("storage_space_of_variable",
		     "hash table should have been allocated\n");
	entity_to_size = hash_table_make(hash_pointer, 0);
    }

    if ((l = (int) hash_get(entity_to_size, (char *) v))
	== (int) HASH_UNDEFINED_VALUE) {
	if(!SizeOfArray(v, &l)) {
	    fprintf(stderr, "[storage_space_of_variable] Non constant array size\n");
	    abort();
	}
	hash_put(entity_to_size, (char *) v, (char *) l);
    }

    return l;
}

#define INTERVAL_INTERSECTION(a,b,c,d) (!((b) <= (c) || (d) <= (a)))

/* 
this function returns TRUE if e1 and e2 have some memory locations in common
*/

bool 
entity_conflict_p(e1, e2)
entity e1, e2;
{
    bool intersect_p = FALSE;
    storage s1, s2;
    ram r1 = ram_undefined, r2 = ram_undefined;
    int o1, o2, l1, l2;
    entity f1, f2, a1, a2;

    if(same_entity_p(e1, e2)) return TRUE;

    s1 = entity_storage(e1);
    s2 = entity_storage(e2);

    if (! (storage_ram_p(s1) && storage_ram_p(s2)))
	return FALSE;

    r1 = storage_ram(s1);
    r2 = storage_ram(s2);

    a1 = ram_section(r1);
    a2 = ram_section(r2);

    if(a1!=a2) return FALSE;

    o1 = ram_offset(r1);
    o2 = ram_offset(r2);

    if(o1==o2) return TRUE;

    f1 = ram_function(r1);
    f2 = ram_function(r2);

    if(f1==f2 && (ENDP(ram_shared(r1)) || ENDP(ram_shared(r2))))
       return FALSE;

    l1 = storage_space_of_variable(e1);
    l1 = l1+o1-1;


    l2 = storage_space_of_variable(e2);
    l2 = l2+o2-1;
	    
    /* return(r1 != ram_undefined && r2 != ram_undefined && 
	   f1 == f2 && a1 == a2 &&
	   INTERVAL_INTERSECTION(o1, l1, o2, l2)); */

    /* FI: it's too late to check if r1 and r2 are defined:
     * you already have core dumped!
     * also, f1 and f2 are not relevant since location are governed
     * by area a1 and a2
     */

    intersect_p = ( a1 == a2 && INTERVAL_INTERSECTION(o1, l1, o2, l2));

    return intersect_p;
}
