void use_def_elim01(int *x)
{
    /* check aliases */
    int *y = x;
    y[0]=1;
}
