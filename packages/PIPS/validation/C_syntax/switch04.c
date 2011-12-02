int switch04(int predicate)
{
  int x = 0;
  switch (predicate) {
  case 0: x = x;
    break;
  case 1|2: x = x + 1;
  }
  return x;
}

