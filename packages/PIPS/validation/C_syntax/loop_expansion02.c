/* Bug parser: op is replaced by pop... */

pop(int op)
{
    int i=0;
    for(;i<op;++i)
    {
        printf("%d",i);
    }
}
