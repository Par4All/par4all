typedef struct { int a; int b;} my;
void foo(my* m)
{
    m->a=3;
}
main()
{
    my c = { 2,4};
    c.b=5;
    foo(&c);
    printf("%d",c.a+c.b);
}
