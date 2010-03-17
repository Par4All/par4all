struct a { struct b { int jedi; } the; };
int foo(int c)
{
    int force;
    struct a of = { {c} };
pas_trop:for(force=1;force<18;force+=2)
    {
        of.the.jedi+=force;
    }
    return of.the.jedi;
}
