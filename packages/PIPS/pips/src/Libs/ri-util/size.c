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
/* Functions to compute the numer of bytes required to store a
   variable or an object of a given type in memory; used by memory
   allocation functions in parsers, when allocation is possible
   (STATIC and DYNAMIC). If the size is unknown, the variable is
   allocated in the STACK area. See ri.pdf, section about "area". */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "ri-util.h"
#include "misc.h"

int number_of_initial_values(list args)
{
  int niv = 0;
  list carg = list_undefined;

  for(carg=args; !ENDP(carg); POP(carg)) {
    expression e = EXPRESSION(CAR(carg));

    if(expression_call_p(e)) {
      call c = syntax_call(expression_syntax(e));
      entity f = call_function(c);
      list sargs = call_arguments(c);

      if(ENTITY_BRACE_INTRINSIC_P(f)) {
	niv += number_of_initial_values(sargs);
      }
      else
	niv++;
    }
    else
      niv++;
  }

  return niv;
}

/* This function computes the total size of a variable in bytes,
ie. the product of the number of elements and the size of each element
in byte, when this size is a constant.

Arrays cannot be sized by a variable according to Fortran 77 standard,
unless they are formal parameters (see SafeSizeOfArray). This
restriction is lifted for arrays allocated in the area *STACK*.

Arrays can be sized by expressions according to C latest standard (see
Validation/C_syntax/array_declarators.c). Then their sizes are not
available at compile time and their addresses in stack cannot be
computed at compiler time (see CSafeSizeOfArray).

Since the size is returned in a signed int, there is a maximum PIPS
size for an array.
*/

bool SizeOfArray(entity e, int * s)
{
  type et = entity_type(e);
  type uet = ultimate_type(et);
  variable a, ua;
  bool ok = true;
  int se = -1;
  int ne = -1;
  int mne = -1;

  assert(type_variable_p(et) && type_variable_p(uet));
  a = type_variable(et);
  ua = type_variable(uet);

  se = SizeOfElements(variable_basic(ua));

  ok = NumberOfElements(variable_basic(a), variable_dimensions(a), &ne);

  if(!ok) {
    /* Let's try to use the initial value */
    value ev = entity_initial(e);
    if(!value_undefined_p(ev)) {
      if(value_expression_p(ev)) {
	expression eve = value_expression(ev);
	//type evet = expression_to_type(eve);
	basic eveb = basic_of_expression(eve); /* FI: should eveb be freed? */

	/* Is it an array of characters initialized with a string expression? */
	if(char_type_p(et) && !basic_undefined_p(eveb) && basic_string_p(eveb)) {
	  ne = string_type_size(eveb);
	  ok = true;
	}
	else if(expression_call_p(eve)) {
	  call evec = syntax_call(expression_syntax(eve));
	  entity f = call_function(evec);
	  list args = call_arguments(evec);

	  /* Is it a call to the BRACE_INTRINSIC operator? */
	  if(ENTITY_BRACE_INTRINSIC_P(f)) {
	    /* This is too simple unfortunately: OK, but why? which
	       test case? */
	    /* ne = gen_length(args); */
	    int ni = number_of_initial_values(args);
	    //int nf = number_of_fields(et);
	    int nf;

	    if(type_variable_p(uet)) {
	      variable ev = type_variable(uet);
	      basic eb = variable_basic(ev);
	      if(basic_derived_p(eb)) {
		entity de = basic_derived(eb);
		nf = number_of_items(entity_type(de));
	      }
	      else
		nf = 1;
	    }
	    ne = ni/nf;
	    if(nf*ne!=ni) {
	      /* Should be a call to CParserError()... */
	      //pips_user_error("Number of initialization values (%d) incompatible"
	      //	      " with number of type fields (%d)\n", ni, nf);
	      // let's assume the source code is correct...
	      ne = gen_length(args);
	    }
	    ok = true;
	  }
	  /* Check for other dimensions which must be all declared: the
	     first dimension only can be implicit */
	  /* Already taken care of by "ni" */
	  /*
	    if(ok && gen_length(variable_dimensions(a))>1) {
	    bool sok = false;
	    int sne = -1;
	    sok = NumberOfElements(variable_basic(a), CDR(variable_dimensions(a)), &sne);
	    if(sok) {
	    ne *= sne;
	    }
	    else {
	    ok = false;
	    }
	    }
	  */
	}
      }
      else if(value_constant_p(ev)) {
	pips_internal_error("Not implemented yet");
      }
    }
  }

  /* Check for 32 bit signed overflows */
  mne = se>0 ? INT_MAX/se : INT_MAX;

  if(ok) {
    if(mne>=ne)
      *s = ne*se;
    else {
      pips_user_warning("Array size incompatible with 32 bit signed integers\n"
			"Maximum number of elements: %d, number of elements declared: %d\n",
			mne, ne);
      ok = false;
    }
  }
  else {
    *s = se;
  }

  return ok;
}

int
array_size(entity a)
{
    int s = 0;

    if(!SizeOfArray(a, &s)) {
	pips_internal_error("Array \"%s\" with illegal varying array size",
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

int CSafeSizeOfArray(entity a)
{
  int s;

  if(!SizeOfArray(a, &s)) {
      pips_user_warning("Varying size for array \"%s\"\n", entity_name(a));
      /* should be a pips_user_error() to avoid useless and dangerous
	 results */
      pips_user_warning("Not yet supported properly by PIPS\n");
  }

  return s;
}

int entity_memory_size(entity dt)
{
  /* dt is assumed to be a derived type: struct, union, and maybe enum */
  type t = ultimate_type(entity_type(dt));
  int s = type_memory_size(t);

  return s;
}

int type_memory_size(type t)
{
  int s = 0;

  switch(type_tag(t)) {
  case is_type_statement:
  case is_type_area:
  case is_type_functional:
  case is_type_varargs:
  case is_type_unknown:
  case is_type_void:
    pips_internal_error("arg. with ill. tag %d", type_tag(t));
    break;
  case is_type_variable:
    {
        /* Seems to be the case for defined types; FI: why are they
           allocated in RAM while they only exist at compile time ? */
        s = SizeOfElements(variable_basic(type_variable(t)));
        FOREACH(DIMENSION,dim, variable_dimensions(type_variable(t)))
            s*=dimension_size(dim);

    } break;
  case is_type_struct:
    MAP(ENTITY, v, {s+=CSafeSizeOfArray(v);}, type_struct(t));
    break;
  case is_type_union:
    MAP(ENTITY, v, {s = s>CSafeSizeOfArray(v)? s : CSafeSizeOfArray(v);}, type_union(t));
    break;
  case is_type_enum:
    s = 4; /* How is it implemented? 32 bit integer? */
    break;
  default:
    pips_internal_error("arg. with unknown tag %d", type_tag(t));
    break;
  }
  /* A struct (or a union) may be hidden, with its declaration
     located in the source code of a library. For instance,
     "_IO_FILE_plus" in stdio.h. Since it contains at least one byte,
     let set s to 1? Or let the caller deal with the problem? */
  // s = s>0? s : 1;
  return s;
}

/* This function returns the length in bytes of the Fortran or C type
   represented by a basic, except for a varying size string (formal
   parameter).

   What is the semantics for bitfields? Return its size rounded to a byte
   number large enough to fit, and not the size in bit. */
_int SizeOfElements(basic b)
{
  int e = -1;

  switch (basic_tag(b)) {
  case is_basic_int:
    {
      /* Some of these values are target architecture dependent:
	 e == 1 character
	 e == 2 short int
	 e == 4 int
	 e == 6 long int
	 e == 8 long long int
	 They are defined in ri-util-local.h
	 To be consistent with the machine compiling and executing
	 PIPS, we could use a switch(e) and the corresponding sizeof().
       */

    e = basic_int(b);
    e = e % 10;
    if(e==DEFAULT_LONG_INTEGER_TYPE_SIZE)
      e = DEFAULT_INTEGER_TYPE_SIZE;
    break;
    }
  case is_basic_float:
    e = basic_float(b);
    break;
  case is_basic_logical:
    e = basic_logical(b);
    break;
  case is_basic_complex:
    e = basic_complex(b);
    /* As for int, e encodes some fine typing information: remove it */
    e = (e/8)*8;
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
  case is_basic_bit: {
    constant c = symbolic_constant(basic_bit(b));
    if(constant_int_p(c))
    // Get the size in bits:
      e = constant_int(c);
    else
      user_error("SizeOfElements",
		 "Sizing of bit-field is non-integer constant");
    // Round the size to full byte number:
    e = (e + 7)/8;
  }
    break;
  case is_basic_pointer:
    e = DEFAULT_POINTER_TYPE_SIZE;
    break;
  case is_basic_derived:
    e = entity_memory_size(basic_derived(b));
    break;
  case is_basic_typedef:
    e = CSafeSizeOfArray(basic_typedef(b));
    break;
  default:
    pips_internal_error("Ill. tag %d for basic", basic_tag(b));
  }

  /* Size can be zero, i.e. unknown, for an external variable */
  //pips_assert("e is not zero", e!=0);

  return e;
}

/* END_EOLE */


/* this function computes the number of elements of a variable. ld is the
list of dimensions of the variable */

int
element_number(basic b, list ld)
{
    int en = 0;

    if(!NumberOfElements(b, ld, &en)) {
	pips_internal_error("Probably varying size array");
    }

    return en;
}

bool
NumberOfElements(basic b, list ld, int * n)
{
  list pc;
  int ne = 1;
  bool ok = true;
  int sne = 1;

  /* do we have many elements at the lower typedef levels? */
  if(basic_typedef_p(b)) {
    entity e = basic_typedef(b);
    // Lots of asserts skipped here
    variable ev = type_variable(entity_type(e));

    ok = NumberOfElements(variable_basic(ev), variable_dimensions(ev), &sne);
  }

  /* let's take care of the current level */
  if(ok) {
    for (pc = ld; pc != NULL && ok; pc = CDR(pc)) {
        expression sod = SizeOfDimension(DIMENSION(CAR(pc)));
        intptr_t s;
        ok=expression_integer_value(sod,&s);
        free_expression(sod);
        ne *= s;
    }
  }

  *n = ne*sne;
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
    intptr_t s = 0;

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

    expression sod = SizeOfDimension(DIMENSION(CAR(pc)));
    if(!(expression_integer_value(sod, &s))) {
	fprintf(stderr, "[SizeOfIthDimension] Non constant %dth dimension\n", i);
	abort();
    }
    free_expression(sod);

    return s;
}



/* this function computes the size of a dimension. */

int
dimension_size(dimension d)
{
    expression sod= SizeOfDimension(d);
    intptr_t i;
    if(expression_integer_value(sod,&i))
        free_expression(sod);
    else
        pips_internal_error("dimension is not constant, use SizeOfDimension instead");
    return i;
}

expression
SizeOfDimension(dimension d)
{
    return 
        make_op_exp(PLUS_OPERATOR_NAME,
                make_op_exp(MINUS_OPERATOR_NAME,copy_expression(dimension_upper(d)),copy_expression(dimension_lower(d))),
                int_to_expression(1)
                )
                ;
}

static void *do_sizeofdimension_reduction(void *v, const list l)
{
    return make_op_exp(MULTIPLY_OPERATOR_NAME,
            (expression)v,
            SizeOfDimension(DIMENSION(CAR(l))));
}

/** computes the product of all dimensions in @p dims*/
expression
SizeOfDimensions(list dims)
{
    return (expression)gen_reduce(int_to_expression(1),do_sizeofdimension_reduction,dims);
}


/* FI: I do not understand the "Value" cast */

Value 
ValueSizeOfDimension(dimension d)
{
    Value dl, du;
    intptr_t l = 0 ;
    intptr_t u = 0;
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

// FI: should be at least NumberOfDimensions, with an s
int NumberOfDimension(entity e)
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
	pips_internal_error("hash table should have been deallocated");
	/* hash_table_clear(entity_to_size); */
    }

    entity_to_size = hash_table_make(hash_pointer, 0);
}

void reset_entity_to_size()
{
    if (entity_to_size == hash_table_undefined) {
	pips_internal_error("hash table should have been allocated");
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
    char * s;

    if (entity_to_size == hash_table_undefined) {
	user_warning("storage_space_of_variable",
		     "hash table should have been allocated\n");
	entity_to_size = hash_table_make(hash_pointer, 0);
    }
    s = hash_get(entity_to_size, (char *) v);
    l = (_int) s;
    if (s == HASH_UNDEFINED_VALUE) {
      if(!SizeOfArray(v, &l)) {
	fprintf(stderr, "[storage_space_of_variable] Non constant array size\n");
	abort();
      }
      hash_put(entity_to_size, (char *) v, (char *) (_int) l);
    }

    return l;
}

#define INTERVAL_INTERSECTION(a,b,c,d) (!((b) <= (c) || (d) <= (a)))

/*
  this function returns true if e1 and e2 MAY have some memory locations
  in common

  This function used to be called entities_may_conflict_p() but abstract
  locations are new entities which require a generalization.
*/
bool variable_entity_may_conflict_p(entity e1, entity e2)
{
    bool intersect_p = false;
    storage s1, s2;
    ram r1 = ram_undefined, r2 = ram_undefined;
    int o1, o2, l1, l2;
    entity f1, f2, a1, a2;

    if(same_entity_p(e1, e2)) return true;

    s1 = entity_storage(e1);
    s2 = entity_storage(e2);

    if (! (storage_ram_p(s1) && storage_ram_p(s2)))
	return false;

    r1 = storage_ram(s1);
    r2 = storage_ram(s2);

    a1 = ram_section(r1);
    a2 = ram_section(r2);

    if(a1!=a2) return false;

    /* Can we have and check static aliasing in a1? */
    if(stack_area_p(a1))
      return false;

    if(heap_area_p(a1))
      return false;

    if (c_module_p(get_current_module_entity()))
      return false;

    /* Let's assume we are dealing with Fortran code, but another test
       should be added about the current module language. No test on
       dynamic aliasing since we are dealing here with direct read and
       write effects. */
    o1 = ram_offset(r1);
    o2 = ram_offset(r2);

    if(o1==o2) return true;

    f1 = ram_function(r1);
    f2 = ram_function(r2);

    if(f1==f2 && (ENDP(ram_shared(r1)) || ENDP(ram_shared(r2))))
       return false;

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

