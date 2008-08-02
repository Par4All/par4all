// Declarations within statement: accepted by gcc

extern void DESC_init_1D_SF();

void trigger_20080717110542703()
{
  DESC_init_1D_SF();

  float tr[10];
  int i;

  for(i=0;i<10;i++){
    tr[i]+= 1.;
  }
}
