main()
{
    struct { int a; int b;} c = { 2,4};
    c.b=5;
    printf("%d",c.a+c.b);
}
