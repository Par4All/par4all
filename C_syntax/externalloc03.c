//Special case of extern variable
int i = 1;

int main(int argc, char *argv[]) {
  int i = 2;
  {
    extern int i;
    printf("%d\n", i);
  }
  return 0;
}
