typedef struct {int champ1; int champ2;} mystruct;


void decl04(mystruct s)
{
  mystruct s_loc = s;

  s_loc.champ1 = s_loc.champ1 -1;
}

int main()
{
  mystruct s;

  s.champ1 = 3;
  s.champ2 = 4;
  decl04(s);
  return 0;
}
