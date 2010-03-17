#define TAILLE 1000
int main() {
  int a[TAILLE];
  int j;

  // A do loop ! Should be parallel.
  for(j = 0; j < TAILLE; j++)
    a[j] = j;
  for(j = 0; j < TAILLE; j++)
    a[j] = j;

  return 0;
}
