int foo(int c[10])
{
    int i,j,k;
    for(i=0;i<10;i++)
    {
        j=i;
        for(j;j<10;j++)
        {
            k=k+1;
            c[j]=k;
        }
    }
    return c[10];
}

int main(int argc, char *argv[])
{
    int c[10];
    int i,k=0;
    for(i=0;i<10;i++)
    {
        c[i]=i;
    }
    k=foo(c);
    printf("%d",k);
    return 0;
}
