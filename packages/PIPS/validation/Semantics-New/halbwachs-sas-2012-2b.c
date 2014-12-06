// Halbwachs & Henry, SAS 2012, p. 200

// Requires transformer lists

void foo()
{
  int n = 0;
  float x;
  while(1) {
    if(x) {
      if(n<60)
	n++;
      else
	n = 0;
    }
  }
  return;
}
