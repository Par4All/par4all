/* Check a simple proof step for Ramzi Darmoul: k==0 because the
   non-linear condition i*i>1 is impossible */


int main(int j, char **unused) {
  int i;
  int k;

  i = (j>1)? 1 : (j<-1)? -1 : j;

  if(i*i>1)
    k = 1;
  else
    k = 0;

  return k;
}
