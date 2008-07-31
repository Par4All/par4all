/* 
 * $Id$
 *
 * FI: these values were taken from syntax/tokyacc.h but I do not 
 * think they matter. Wouldn't an enum be better?
 *
 * MOD was an initial exception. So are MINIMUM and MAXIMUM
 */

#define AND     55
#define EQ     56
#define EQV     57
#define GE     58
#define GT     59
#define LE     60
#define LT     61
#define NE     62
#define NEQV     63
#define NOT     64
#define OR     65
#define MINUS     73
#define PLUS     74
#define SLASH     75
#define STAR     76
#define POWER     77
#define MOD       78  /* not evaluated, but later added in IsBinaryOperator*/
#define CONCAT     84
#define MINIMUM 85
#define MAXIMUM 86
#define CAST_OP 87
#define BITWISE_AND 88
#define BITWISE_OR 89
#define BITWISE_XOR 90
#define RIGHT_SHIFT 91
#define LEFT_SHIFT 92
