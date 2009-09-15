#ifndef TYPE
#define _OP(a,b) a##b
#define OP(a,b) _OP(a,b)
#define TYPE float
#define SUFF f
#include "include/terasm.c"
#define TYPE int
#define SUFF i
#include "include/terasm.c"
#else
TYPE OP(add,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs+rhs;
}
TYPE OP(sub,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs-rhs;
}
TYPE OP(mul,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs*rhs;
}
TYPE OP(div,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs/rhs;
}
TYPE OP(set,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=rhs;
}
TYPE OP(padd,SUFF)(TYPE *lhs, int rhs)
{
    return lhs=lhs[rhs];
}
#endif
