struct a { int b; };
void foo(struct a *_)
{
    _->b=1;
};
void bar()
{
    struct a A = { 0 };
    foo(&A);
    printf("%d\n",A.b);
}

main()
{
    bar();
    return 0;
}
