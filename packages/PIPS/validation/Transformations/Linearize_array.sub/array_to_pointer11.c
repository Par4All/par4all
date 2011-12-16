typedef double t_precision;

typedef unsigned int size_t;

size_t size;

void p4a_kernel_1(size_t size, size_t i, size_t j, t_precision ptr[size][size], t_precision val)
{
   // Loop nest P4A end
   if (i<=-1+size&&j<=-1+size)
      ptr[i][j] = val;
}


void p4a_kernel_wrapper_1(size_t size, size_t i, size_t j, t_precision ptr[size][size], t_precision val)
{
   // Index has been replaced by P4A_vp_0:
   i = 0;
   // Index has been replaced by P4A_vp_1:
   j = 1;
   // Loop nest P4A end
   p4a_kernel_1(size, i, j, ptr, val);
}


void p4a_kernel_launcher_1(size_t size, t_precision ptr[size][size], t_precision val)
{
   //PIPS generated variable
   size_t i, j;
   p4a_kernel_wrapper_1(size, i, j, ptr, val);
}


void init(t_precision val)
{
   //PIPS generated variable
   t_precision (*P_3)[size][size] = (t_precision (*)[size][size]) 0;
   p4a_kernel_launcher_1(size, *P_3, val);
}
