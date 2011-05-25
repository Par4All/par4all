


int main() {
  int a[10];

// All of these doesn't pass clan and shouldn't be detected as scop by PIPS
// Only * / + - are authorized in indices.

  for(int i=0; i<10;i++) 
    a[i%2] = 0;

  for(int i=0; i<10;i++) 
    a[i&2] = 0;

  for(int i=0; i<10;i++) 
    a[(int)(i/(2.f))] = 0;

// Strangely, this one is ok, we might transform the previous one to this form 
  int j;
  for(int i=0; i<10;i++) {
    j = i&2;
    a[j] = 0;
  }




}
