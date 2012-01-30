
void scilab_rt_datenum_i2i2i2i2i2i2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sin30, int sin31, int in3[sin30][sin31],
    int sin40, int sin41, int in4[sin40][sin41],
    int sin50, int sin51, int in5[sin50][sin51],
    int sout00, int sout01, double out0[sout00][sout01])
{
  int i, j;
  int val0=0, val1=0, val2=0, val3=0, val4=0, val5=0;

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
  for(i = 0; i < sin30; ++i){
    for(j = 0; j < sin31; ++j){
      val3 += in3[i][j];
    }
  }
  for(i = 0; i < sin40; ++i){
    for(j = 0; j < sin41; ++j){
      val4 += in4[i][j];
    }
  }
  for(i = 0; i < sin50; ++i){
    for(j = 0; j < sin51; ++j){
      val5 += in5[i][j];
    }
  }

  for (i = 0; i < sout00; ++i ){
    for (j = 0; j < sout01; ++j ){
      out0[i][j] = val0 + val1 + val2 + val3 + val4 + val5;
    }
  }



}
