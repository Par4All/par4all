#define NP 1000000
void indirect00(int data[NP], int histo[NP])
{
  int i;
  for (i = 0; i < NP; i++)
    histo[i] = 0;
  for (i = 0; i < NP; i++)
    ++histo[data[i]];
}

