void foo_l(int a[5],int b[5],int c[5])
{
    a[0]=b[0]+c[0];
    a[1]=b[1]+c[1];
    a[2]=b[2]+c[2];

    a[0]=b[0]*c[0];
    a[1]=b[1]*c[1];
    a[2]=b[2]*c[2];
    a[3]=b[3]*c[3];
    a[4]=b[4]*c[4];
}

void april()
{
    int A[5],B[5],C[5];
    foo_l(A,B,C);
}
