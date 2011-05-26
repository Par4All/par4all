/* The pragma carried by a block is lost by the controlizer */

int main(void)
{
  int i;

#pragma toto
  {
    int j;
    i=0;
  }
  return i;
}
