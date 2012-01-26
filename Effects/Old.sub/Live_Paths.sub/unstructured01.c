int unstructured02()
{
  int begin, end, step, current;
  int i;

  step = 1;
  begin = 0;
  end = 10;
  current = begin;

  while((current != end ))
    {
      for(i = 0; i< 20; i++)
	if( current == end) goto SKIP;

      if( step == 0)
	{
	  if (current == end ) goto SKIP;
	  if (current == begin ) goto SKIP;
	}
      else
	{
	  if (current < end ) goto SKIP;
	  current -= step;
	}
    SKIP:
      if( current == end ) break;
      current = end;
    }

  return 1;
}
