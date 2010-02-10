int bar(int a, int b)
{
    int n;
    int j=0;
    for(n=0;n<a;n++)
    {
        int c= b;
        j+=2*c;
    }
    return j;
}

int foo(int argc, char **argv)
{
    int c ;
    c = bar(argc, argc);
    return 0;
}

