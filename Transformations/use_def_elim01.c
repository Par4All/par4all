void use_def_elim(int *x)
{
    /* check aliases */
    int *y = x;
    y[0]=1;
}
