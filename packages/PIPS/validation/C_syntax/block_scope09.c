int x = 1;
int y = 2;

// The formal parameter hides the global x
int block_scope09(int x)
{
  int k;
  {
    // This decaration hides the formal parameter
    int x;

    x = 2;
  }

  k = 1;
  y++;
}
