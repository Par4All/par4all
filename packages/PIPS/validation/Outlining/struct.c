typedef unsigned char uint8;

struct my_struct {
        uint8 v[1];
};



void outline_struct(struct my_struct *a_struct)
{
  uint8 *v_struct = a_struct->v;

outline: v_struct[0]++;
}

