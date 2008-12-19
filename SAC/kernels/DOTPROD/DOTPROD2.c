void inner_loop(int _i, int *_a, int*_b, int *_c)
{
    _a[_i] = _b[_i] + _c[_i];
}

void dotprod(int* a, int *b, int *c)
{
    int i;
    for(i=0;i<1000000;++i)
    {
        inner_loop(i,a,b,c);
    }


    return;
}


