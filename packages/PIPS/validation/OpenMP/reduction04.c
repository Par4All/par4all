#include <stdio.h>
int main()
{
    int ua2[3][1] = { {1},{2},{3}};
    int uc0=0;
    for(int lv1 = 0; lv1 <= 2; lv1 += 1)
        uc0 |= ua2[lv1][0]!=(int) 0;
    printf("%d\n",uc0);
    return 0;
}
