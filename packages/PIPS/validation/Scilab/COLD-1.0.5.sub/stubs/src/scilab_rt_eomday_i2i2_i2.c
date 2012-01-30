void scilab_rt_eomday_i2i2_i2(int sin00, int sin01, int year[sin00][sin01], int sin10, int sin11, int month[sin10][sin11], int sout00, int sout01, int last_day[sout00][sout01]){


  int common_year[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
  int leap_year[12]   = {31,29,31,30,31,30,31,31,30,31,30,31};
  int i,j;


  for (i = 0 ; i < sin00 ; ++i){
    for (j = 0 ; j < sin01 ; ++j){
        last_day[i][j] = leap_year[month[i][j]] + common_year[month[i][j]] + year[i][j];
    }
  }


} 
