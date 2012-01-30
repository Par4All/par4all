
typedef struct {
  int var1;
  int var2;
  int var3;
} SomeStruct;

int titi = 4;
SomeStruct toto = {1, 2, 3};

int main () {
  int result = 0;
  result = toto.var1 + toto.var2 + toto.var3 + titi;
  return result;
}

