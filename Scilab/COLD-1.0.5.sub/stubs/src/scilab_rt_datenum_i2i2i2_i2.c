
void scilab_rt_datenum_i2i2i2_i2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sout00, int sout01, int out0[sout00][sout01])
{

  int i, j;
  int val0=0, val1=0, val2=0;

  for(i = 0; i < sin00; ++i){
    for(j = 0; j < sin01; ++j){
      val0 += in0[i][j];
    }
  }

  for(i = 0; i < sin10; ++i){
    for(j = 0; j < sin11; ++j){
      val1 += in1[i][j];
    }
  }

  for(i = 0; i < sin20; ++i){
    for(j = 0; j < sin21; ++j){
      val2 += in2[i][j];
    }
  }
  for (i = 0; i < sout00; ++i ){
    for (j = 0; j < sout01; ++j ){
      out0[i][j] = val0 + val1 + val2; 
    }
  }



}
