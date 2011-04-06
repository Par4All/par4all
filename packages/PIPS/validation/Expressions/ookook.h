//Implementation for binary operators
#define __DEFOP2(name, type, lval) \
	type name(type lval a, type b);

#define __IMPOP2(name, op, type, lval) \
	type name(type lval a, type b) { return lval a op b; }

#define SUFFIX(type) SUF_ ## type

#define OPERATOR(op) OP_ ## op

#define CONC(a,b) a ## b
#define LCONC(a,b) CONC(a,b)
#define FUNC_NAME(prefix, name, type) LCONC(prefix, LCONC(name,SUFFIX(type)))

#define _DEFOP2(prefix, name, type, lval) \
	__DEFOP2(FUNC_NAME(prefix, name, type), \
		type, lval)

#define _IMPOP2(prefix, name, type, lval) \
	__IMPOP2(FUNC_NAME(prefix, name, type), \
		OPERATOR(name), type, lval)

#ifdef OP_IMPL
#define _DOOP2(prefix, name, type, lval) _IMPOP2(prefix, name, type, lval)
#else
#define _DOOP2(prefix, name, type, lval) _DEFOP2(prefix, name, type, lval)
#endif

#define DEFOP2(prefix, name, type) _DEFOP2(prefix, name, type, )
#define LDEFOP2(prefix, name, type) _DEFOP2(prefix, name, type, *)

#define IMPOP2(prefix, name, type) _IMPOP2(prefix, name, type, )
#define LIMPOP2(prefix, name, type) _IMPOP2(prefix, name, type, *)

#define DOOP2(prefix, name, type) _DOOP2(prefix, name, type, )
#define LDOOP2(prefix, name, type) _DOOP2(prefix, name, type, *)

//Register suffixes here
#define SUF_char c
#define SUF_short s
#define SUF_int i
#define SUF_long l
#define SUF_float f
#define SUF_double d
#define SUF__Bool b
#define SUF__Complex C
#define SUF__Imaginary I

//Register operators here
#define OP_plus +
#define OP_minus -
#define OP_mul *
#define OP_div /
#define OP_mod %
#define OP_assign =
#define OP_mul_up *=
#define OP_div_up /=
#define OP_mod_up %=
#define OP_plus_up +=
#define OP_minus_up -=
#define OP_leq <=
#define OP_lt <
#define OP_geq >=
#define OP_gt >
#define OP_eq ==
#define OP_neq !=

//Register functions here
DOOP2(op_,plus,int)
DOOP2(op_,minus,int)
DOOP2(op_,plus_eq,int)
DOOP2(op_,leq,int)
LDOOP2(op_,plus_up,int)
DOOP2(op_,assign,int)

DOOP2(op_,plus,float)
DOOP2(op_,minus,float)
DOOP2(op_,mul,float)
DOOP2(op_,div,float)
LDOOP2(op_,plus_up,float)
DOOP2(op_,assign,float)

#include <complex.h>
DOOP2(op_,plus,complex)
DOOP2(op_,minus,complex)
DOOP2(op_,mul,complex)
DOOP2(op_,div,complex)
LDOOP2(op_,plus_up,complex)
DOOP2(op_,assign,complex)
