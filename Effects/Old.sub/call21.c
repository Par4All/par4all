
typedef struct {int data[10];} mystruct;

void foo(int t[10])
{
  for (int i=0; i<10; i++)
    t[i] = i;
}

int main()
{
  mystruct t;
  foo(t.data);
  return 0;
}
