
int titi = 4;

typedef struct {
  int var1;
  int var2;
  int var3;
} SomeStruct;

struct {
  int var1;
  SomeStruct var2;
  int var3;
} toto;

int main () {
  int result = 0;
  result = toto.var1 + toto.var2.var2 + toto.var3 + titi;
  return result;
}

