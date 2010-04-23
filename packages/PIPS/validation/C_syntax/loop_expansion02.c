/* Bug parser when returned value type is implicit: pop:op is replaced
   by op:pop... */

pop(int op)
{
  int i=0;
  for(;i<op;++i)
    {
      printf("%d",i);
    }
}
