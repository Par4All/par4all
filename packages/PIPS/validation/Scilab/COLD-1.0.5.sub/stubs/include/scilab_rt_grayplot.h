/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010
*
*/

extern void scilab_rt_grayplot__();

extern void scilab_rt_grayplot_i2i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21]);

extern void scilab_rt_grayplot_i2i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21]);

extern void scilab_rt_grayplot_i2d2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21]);

extern void scilab_rt_grayplot_i2d2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21]);

extern void scilab_rt_grayplot_d2i2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21]);

extern void scilab_rt_grayplot_d2i2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21]);

extern void scilab_rt_grayplot_d2d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21]);

extern void scilab_rt_grayplot_d2d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21]);

extern void scilab_rt_grayplot_i2i2i2s0_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_i2i2d2s0_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_i2d2i2s0_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_i2d2d2s0_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_d2i2i2s0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_d2i2d2s0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_d2d2i2s0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_d2d2d2s0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0);

extern void scilab_rt_grayplot_i2i2i2s0i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2i2i2s0d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2i2d2s0i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2i2d2s0d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2d2i2s0i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2d2i2s0d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2d2d2s0i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2d2d2s0d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2i2i2s0i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2i2i2s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2i2d2s0i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2i2d2s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2d2i2s0i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2d2i2s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2d2d2s0i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31]);

extern void scilab_rt_grayplot_d2d2d2s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31]);

extern void scilab_rt_grayplot_i2i2i2s0i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2i2s0i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2i2s0d2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2i2s0d2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2d2s0i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2d2s0i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2d2s0d2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2i2d2s0d2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2i2s0i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2i2s0i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2i2s0d2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2i2s0d2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2d2s0i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2d2s0i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2d2s0d2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_i2d2d2s0d2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2i2s0i2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2i2s0i2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2i2s0d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2i2s0d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2d2s0i2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2d2s0i2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2d2s0d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2i2d2s0d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2i2s0i2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2i2s0i2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2i2s0d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2i2s0d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, int matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2d2s0i2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2d2s0i2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2d2s0d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41]);

extern void scilab_rt_grayplot_d2d2d2s0d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    char* scalarin0, 
    int in30, int in31, double matrixin3[in30][in31], 
    int in40, int in41, double matrixin4[in40][in41]);

