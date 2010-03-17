int foo(int predicate )
{
  int x = 0;
  if (predicate == 0)
    {
      goto switch_0_0;
    }
  else
    {
      if (predicate == 1)
        {
          goto switch_0_1;
        }
      else
        {
          if (predicate == 2)
            {
              goto switch_0_2;
            }
          else
            {
              if (predicate == 3)
                {
                  goto switch_0_3;
                }
              else
                {
                  if (1)
                    {
                      goto switch_0_default;
                    }
                  else
                    {
                      if (0)
                        {
                        switch_0_0: return (111);
                        switch_0_1: x = x + 1;
                        switch_0_2: return (x + 3);
                        switch_0_3: goto switch_0_break;
                        switch_0_default: return (222);
                        }
                      else
                        {
                        switch_0_break: ;
                        }
                    }
                }
            }
        }
    }
  return (333);
}
