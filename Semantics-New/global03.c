int nelt;
void do_refine(int *irefine)
{
    int k, ne[4] ;
    int num_refine; 
    nelt+=1;
    for (k = 0; k < 1; k++) {
        ne[k] = nelt;
    }
    *irefine +=  num_refine; // changing the += into a + removes the error
}
