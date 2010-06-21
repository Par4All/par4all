#include <stdio.h>
int main()
{
    struct s { int t0;int t1;};
    struct s s0,*s1;
    s1=&s0;
    s0.t0=1;
    s0.t1=1;
    (*(&s1))->t0+=s1->t1;
    printf("%d",s0.t0);
    return 0;
}
