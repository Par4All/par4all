void foo()
{
    int i,tab[4];
    if(i<0)
        i=0;
    for(i=0;i<sizeof(tab)/sizeof(tab[0]);i++)
        if(tab[i]>i)tab [i]=i;
}
