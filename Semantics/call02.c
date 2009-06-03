/* Chekc that unvisible side effects are not shown in the summary transformer */

int j = 0;

void call02(int i, float x)
{
  i++;
  j++;
  x++;
}
