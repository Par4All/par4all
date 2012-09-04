void call17(int *x)
{
    /* check aliases */
    int *y = x;
    y[0]=1;
    return;
} 
