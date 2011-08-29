int foo(int i)
{
    int i__=3;
    {
        int i_=i__;
        i+=i_;
    }
    return i;
}
int bar(int i_)
{
    int i__=0,i=0;
    int j=0;
    j+=foo(i_);
    return j;
}
int main()
{
    printf("%d\n",bar(0));
    return 0;
}

