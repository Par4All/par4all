/* The pragma carried by a block is lost by the controlizer */

int main(void)
{
  int i;

#pragma toto
  {
    i=0;
  }
  return i;
}
