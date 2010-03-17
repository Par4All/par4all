int foo(int c[10])
{
    int i,k;
    for(i=0;i<10;i++)
    {
        k=c[i];
        k=k+1;
        c[i]=k;
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
