void fp(int x)
{
    void (*f)(int)=fp;
    void (*g)(int);
    g(x);
} 
