/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010
*
*/

extern void scilab_rt_colorbar_i0i0_(int scalarin0, 
    int scalarin1);

extern void scilab_rt_colorbar_i0d0_(int scalarin0, 
    double scalarin1);

extern void scilab_rt_colorbar_d0i0_(double scalarin0, 
    int scalarin1);

extern void scilab_rt_colorbar_d0d0_(double scalarin0, 
    double scalarin1);

extern void scilab_rt_colorbar_i0i0i2_(int scalarin0, 
    int scalarin1, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_colorbar_i0d0i2_(int scalarin0, 
    double scalarin1, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_colorbar_d0i0i2_(double scalarin0, 
    int scalarin1, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_colorbar_d0d0i2_(double scalarin0, 
    double scalarin1, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_colorbar_i0i0i2s0_(int scalarin0, 
    int scalarin1, 
    int in00, int in01, int matrixin0[in00][in01], 
    char* scalarin2);

extern void scilab_rt_colorbar_i0d0i2s0_(int scalarin0, 
    double scalarin1, 
    int in00, int in01, int matrixin0[in00][in01], 
    char* scalarin2);

extern void scilab_rt_colorbar_d0i0i2s0_(double scalarin0, 
    int scalarin1, 
    int in00, int in01, int matrixin0[in00][in01], 
    char* scalarin2);

extern void scilab_rt_colorbar_d0d0i2s0_(double scalarin0, 
    double scalarin1, 
    int in00, int in01, int matrixin0[in00][in01], 
    char* scalarin2);

