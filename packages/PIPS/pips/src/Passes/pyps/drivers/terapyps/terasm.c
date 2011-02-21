#define _OP(a,b) a##b
#define OP(a,b) _OP(a,b)
#define TYPE int
#define SUFF i

TYPE OP(ref,SUFF)(TYPE *ptr, int index)
{
	return *(ptr+index);
}

TYPE OP(add,SUFF)(TYPE lhs, TYPE rhs0, TYPE rhs1)
{
    return lhs=rhs0+rhs1;
}
TYPE OP(addr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs+*rhs;
}
TYPE OP(sub,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs-rhs;
}
TYPE OP(subr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs-*rhs;
}
TYPE OP(lshift,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs<<rhs;
}
TYPE OP(rshift,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs>>rhs;
}
TYPE OP(prshift,SUFF)(TYPE *lhs, TYPE rhs)
{
    return *lhs=*lhs>>rhs;
}
TYPE OP(mul,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs*rhs;
}
TYPE OP(div,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs/rhs;
}
TYPE OP(mulr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs**rhs;
}
TYPE OP(set,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=rhs;
}
TYPE OP(pset,SUFF)(TYPE *lhs, TYPE rhs)
{
    return *lhs=rhs;
}
TYPE OP(setp,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=*rhs;
}
TYPE OP(psetp,SUFF)(TYPE *lhs, TYPE *rhs)
{
    return *lhs=*rhs;
}
TYPE* OP(padd,SUFF)(TYPE *lhs, TYPE *rhs, int val)
{
    return lhs=rhs+val;
}
TYPE* OP(psub,SUFF)(TYPE *lhs, int rhs)
{
    return lhs=lhs-rhs;
}

