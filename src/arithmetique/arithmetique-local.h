/* package arithmetique
 *
 * Francois Irigoin, mai 1989
 *
 * Modifications
 *  - reprise de DIVIDE qui etait faux (Remi Triolet, Francois Irigoin, 
 *    april 90)
 *  - simplification de POSITIVE_DIVIDE par suppression d'un modulo
 */

/* We would like linear to be generic about the "integer" type used
 * to represent integer values. Thus Value is defined here. It should
 * be changed to "int" "long" or "long long". In an ideal world,
 * any source modification should be limited to this package.
 *
 * Indeed, we cannot switch easily to bignums that need constructors 
 * dans destructors... That would lead to too many modifications...
 * C++ would make things easier and cleaner...
 *
 * Fabien COELHO
 */

/* to be included for _MIN and _MAX: #include <limits.h>
 */

/* gnu and sun do not have the same conventions for "long long"
 */
#ifndef LONG_LONG_MAX
#define LONG_LONG_MAX LLONG_MAX
#define LONG_LONG_MIN LLONG_MIN
#endif

#if defined(LINEAR_VALUE_IS_LONGLONG)
typedef long long Value;
#define VALUE_FMT "%lld"
#define VALUE_CONST(val) (val##LL)
#define VALUE_MIN LONG_LONG_MIN
#define VALUE_MAX LONG_LONG_MAX
#define VALUE_ZERO 0LL
#define VALUE_ONE  1LL
#define VALUE_MONE -1LL
/* I cannot trust gcc for this...
 * some faster checks with 0x7ffffff000 sg and so ? 
 */
#define VALUE_TO_LONG(val) \
    ((long)(val>=(Value)LONG_MIN&&val<=(Value)LONG_MAX)?val:abort())
#define VALUE_TO_INT(val) \
    ((int)(val>=(Value)INT_MIN&&val<=(Value)INT_MAX)?val:abort())
#define VALUE_TO_DOUBLE(val) ((double)val)
/* end LINEAR_VALUE_IS_LONGLONG
 */
#elif defined(LINEAR_VALUE_IS_LONG)
typedef long Value;
#define VALUE_FMT "%ld"
#define VALUE_CONST(val) (val##L)
#define VALUE_MIN LONG_MIN
#define VALUE_MAX LONG_MAX
#define VALUE_ZERO 0L
#define VALUE_ONE  1L
#define VALUE_MONE -1L
#define VALUE_TO_LONG(val) (val)
#define VALUE_TO_INT(val) ((int)val)
#define VALUE_TO_DOUBLE(val) ((double)val)
/* end LINEAR_VALUE_IS_LONG
 */
#elif defined(LINEAR_VALUE_IS_FLOAT)
typedef float Value;
#define VALUE_FMT "%f"
#define VALUE_CONST(val) (val)
#define VALUE_MIN FLOAT_MIN
#define VALUE_MAX FLOAT_MAX
#define VALUE_ZERO 0
#define VALUE_ONE  1
#define VALUE_MONE -1
#define VALUE_TO_LONG(val) ((long)val)
#define VALUE_TO_INT(val) ((int)val)
#define VALUE_TO_DOUBLE(val) ((double)val)
/* end LINEAR_VALUE_IS_FLOAT
 */
/* the purpose of the chars version is to detect invalid assignments
 */
#elif defined(LINEAR_VALUE_IS_CHARS)
typedef char * Value;
#define VALUE_FMT "%s"
#define VALUE_CONST(val) ((char*)val)
#define VALUE_MIN (char*)INT_MIN
#define VALUE_MAX (char*)INT_MAX
#define VALUE_ZERO (char*)0
#define VALUE_ONE  (char*)1
#define VALUE_MONE (char*)-1
#define VALUE_TO_LONG(val) ((long)val)
#define VALUE_TO_INT(val) ((int)val)
#define VALUE_TO_DOUBLE(val) ((double)(int)val)
/* end LINEAR_VALUE_IS_CHARS
 */
#else /* default: LINEAR_VALUE_IS_INT */
typedef int Value;
#define VALUE_FMT "%d"
#define VALUE_CONST(val) (val)
#define VALUE_MIN INT_MIN
#define VALUE_MAX INT_MAX
#define VALUE_ZERO 0
#define VALUE_ONE  1
#define VALUE_MONE -1
#define VALUE_TO_LONG(val) ((long)val)
#define VALUE_TO_INT(val) ((int)val)
#define VALUE_TO_DOUBLE(val) ((double)val)
/* end LINEAR_VALUE_IS_INT
 */
#endif 

#define VALUE_NAN VALUE_MIN

#define int_to_value(i) ((Value)i)
#define long_to_value(l) ((Value)l)
#define float_to_value(f) ((Value)f)
#define double_to_value(d) ((Value)d)

/* boolean operators on values
 */
#define value_eq(v1,v2) ((v1)==(v2))
#define value_ne(v1,v2) ((v1)!=(v2))
#define value_gt(v1,v2) ((v1)>(v2))
#define value_ge(v1,v2) ((v1)>=(v2))
#define value_lt(v1,v2) ((v1)<(v2))
#define value_le(v1,v2) ((v1)<=(v2))

/* trian operators on values
 */
#define value_sign(v) (value_eq(v,VALUE_ZERO)?0:value_lt(v,VALUE_ZERO)?-1:1)
#define value_compare(v1,v2) (value_eq(v1,v2)?0:value_lt(v1,v2)?-1:1)

/* binary operators on values
 */
#define value_plus(v1,v2)  ((v1)+(v2))
#define value_div(v1,v2)   ((v1)/(v2))
#define value_mod(v1,v2)   ((v1)%(v2))
#define value_mult(v1,v2)  ((v1)*(v2))
#define value_minus(v1,v2) ((v1)-(v2))
#define value_pdiv(v1,v2)  (divide(v1,v2))
#define value_pmod(v1,v2)  (modulo(v1,v2))
#define value_min(v1,v2)   (value_le(v1,v2)? v1: v2)
#define value_max(v1,v2)   (value_ge(v1,v2)? v1: v2)

/* assigments
 */
#define value_assign(ref,val) ref=(val)
#define value_addto(ref,val) ref+=(val)
#define value_product(ref,val) ref*=(val)
#define value_substract(ref,val) ref-=(val)
#define value_division(ref,val) ref/=(val)
#define value_modulus(ref,val) ref%=(val)
#define value_pdivision(ref,val) value_assign(ref,value_pdiv(ref,val))
#define value_oppose(ref) value_assign(ref,value_uminus(ref))
#define value_absolute(ref) value_assign(ref,value_abs(ref))

/* unary operators on values
 */
#define value_uminus(val)  (-(val))
#define value_abs(val)     (value_ge(val,VALUE_ZERO)? (val): value_uminus(val))

#define value_pos_p(val)      value_gt(val,VALUE_ZERO)
#define value_neg_p(val)      value_lt(val,VALUE_ZERO)
#define value_posz_p(val)     value_ge(val,VALUE_ZERO)
#define value_negz_p(val)     value_le(val,VALUE_ZERO)
#define value_zero_p(val)     value_eq(val,VALUE_ZERO)
#define value_notzero_p(val)  value_ne(val,VALUE_ZERO)
#define value_one_p(val)      value_eq(val,VALUE_ONE)
#define value_notone_p(val)   value_ne(val,VALUE_ONE)
#define value_mone_p(val)     value_eq(val,VALUE_MONE)

/* LINEAR_VALUE_IS_CHARS is used for type checking.
 * some operations are not allowed on (char*), thus
 * they are switched to some other operation here...
 */
#if defined(LINEAR_VALUE_IS_CHARS)
#undef float_to_value
#define float_to_value(f) ((Value)(int)f)
#undef double_to_value
#define double_to_value(f) ((Value)(int)f)
#define value_fake_binary(v1,v2) ((char*)((int)(v1)^(int)(v2)))
#define value_bool_binary(v1,v2) (((int)(v1)^(int)(v2)))
#undef value_uminus
#define value_uminus(v) (v)
#undef value_mult
#define value_mult(v1,v2) value_fake_binary(v1,v2)
#undef value_mod
#define value_mod(v1,v2) value_fake_binary(v1,v2)
#undef value_ge
#define value_ge(v1,v2) value_bool_binary(v1,v2)
#undef value_gt
#define value_gt(v1,v2) value_bool_binary(v1,v2)
#undef value_le
#define value_le(v1,v2) value_bool_binary(v1,v2)
#undef value_lt
#define value_lt(v1,v2) value_bool_binary(v1,v2)
#undef value_plus
#define value_plus(v1,v2) value_fake_binary(v1,v2)
#undef value_minus
#define value_minus(v1,v2) value_fake_binary(v1,v2)
#undef value_pdiv
#define value_pdiv(v1,v2) value_fake_binary(v1,v2)
#undef value_div
#define value_div(v1,v2) value_fake_binary(v1,v2)
#undef value_mod
#define value_mod(v1,v2) value_fake_binary(v1,v2)
#undef value_addto
#define value_addto(v1,v2) value_assign(v1,value_plus(v1,v2))
#undef value_substract
#define value_substract(v1,v2) value_addto(v1,v2)
#undef value_product
#define value_product(v1,v2) value_addto(v1,v2)
#undef value_division
#define value_division(v1,v2) value_addto(v1,v2)
#undef value_negz_p
#define value_negz_p(v) ((int)v)
#endif

/* valeur absolue
 */
#ifndef ABS
#define ABS(x) ((x)>=0 ? (x) : -(x))
#endif

/* minimum et maximum 
 * if they are defined somewhere else, they are very likely 
 * to be defined the same way. Thus the previous def is not overwritten.
 */
#ifndef MIN
#define MIN(x,y) ((x)>=(y)?(y):(x))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>=(y)?(x):(y))
#endif

/* signe d'un entier: -1, 0 ou 1 */
#define SIGN(x) ((x)>0? 1 : ((x)==0? 0 : -1))

/* division avec reste toujours positif
 * basee sur les equations:
 * a/(-b) = - (a/b)
 * (-a)/b = - ((a+b-1)/b)
 * ou a et b sont des entiers positifs
 */
#define DIVIDE(x,y) ((y)>0? POSITIVE_DIVIDE(x,y) : \
		     -POSITIVE_DIVIDE((x),(-(y))))

/* division avec reste toujours positif quand y est positif: assert(y>=0) */
#define POSITIVE_DIVIDE(x,y) ((x)>0 ? (x)/(y) : - (-(x)+(y)-1)/(y))

/* modulo a resultat toujours positif */
#define MODULO(x,y) ((y)>0 ? POSITIVE_MODULO(x,y) : POSITIVE_MODULO(-x,-y))

/* modulo par rapport a un nombre positif: assert(y>=0)
 *
 * Ce n'est pas la macro la plus efficace que j'aie jamais ecrite: il faut
 * faire, dans le pire des cas, deux appels a la routine .rem, qui n'est
 * surement pas plus cablee que la division ou la multiplication
 */
#define POSITIVE_MODULO(x,y) ((x) > 0 ? (x)%(y) : \
			      ((x)%(y) == 0 ? 0 : ((y)-(-(x))%(y))))
			      
/* Pour la recherche de performance, selection d'une implementation
 * particuliere des fonctions
 */

#define pgcd(a,b) pgcd_slow(a,b)

#define divide(a,b) DIVIDE(a,b)

#define modulo(a,b) MODULO(a,b)

typedef struct {Value num, den; int numero ; } frac ;
typedef struct col{int taille, existe ; frac *colonne ;} tableau ;

/* end of $RCSfile: arithmetique-local.h,v $
 */
