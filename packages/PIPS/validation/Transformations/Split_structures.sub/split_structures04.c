typedef struct { int a; int b;} my;
main()
{
    my c = { 2,4};
    c.b=5;
    printf("%d",c.a+c.b);
}
