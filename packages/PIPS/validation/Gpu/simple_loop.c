enum { N = 10000};

double a[N], b[N], c[N];

int main() {
  int i;

  for(i = 0; i < N; i++)
    a[i] = b[i] + c[i];

  return 0;
}
