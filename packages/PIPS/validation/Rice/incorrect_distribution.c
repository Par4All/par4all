typedef double  REAL;

/*
** Extract from linpack benchmark and simplified, the loop should be distributed
*/
static REAL ddot_r(int n,REAL *dx,int incx,REAL *dy,int incy) {
    REAL dtemp;
    int i,ix,iy;

    dtemp = ZERO;

    ix = 0;
    iy = 0;
    for (i = 0;i < n; i++) {
      dtemp = dtemp + dx[ix]*dy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
    
    return(dtemp);
}


