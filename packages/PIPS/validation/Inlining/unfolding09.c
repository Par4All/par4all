int foo(int c)
{
  int err = 0;

  // BEFORE SWITCH
  switch (c)
  {
  case 4:
    err |= deep(4);
    // BREAK 4
    break;
  case 8:
    err |= deep(8);
    // BREAK 8
    break;
  default:
    // memory leak & RETURN
    return 1;
  }

  // AFTER SWITCH
  return err;
}

int deep(int ret)
{
  return ret;
}

int main(int argc, char * argv[])
{
  foo(8);
  return 0;
}
