// Impact of anywhere effect
//
// The precondition i==1 just before the return in the main is lost

int * foo()
{
  return 0;
}

void print(char * s)
{
}

void effects_anywhere01(int i, int *p)
{
  if(i!=1) {
    exit(1);
  }
  else{
    int k;

    k = i;

    // If this statement is commented out, the information i==1 is
    // found after the call in the main programm
    int l = *p;

    if(k!=1)
      print("error\n");
  }
  return;
}

int main()
{
  int *p = foo();
  int i;
  effects_anywhere01(i, p);
  return 0;
}
