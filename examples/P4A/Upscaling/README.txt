contact : Stephanie.Even@telecom-bretagne.eu

make cuda :


Initialization calls :
      p4a_launcher_main(*P4A_var_buffer0);
      P4A_copy_from_accel_1d(sizeof(unsigned char), 90400, 90400, 0, &buffer[0], *P4A_var_buffer0);

      What about sizeof(unsigned char) as the first argument to P4A_copy_from_accel_1d ?

      In the buffer_copy subroutine :
      P4A_copy_to_accel_1d(sizeof(unsigned char), 90400, 90400, 0, &buffer[0], *P4A_var_buffer0);
      P4A_copy_to_accel_1d(sizeof(unsigned char), 92920, 92920, 0, &y[0], *P4A_var_y0);
      p4a_launcher_buffer_copy(*P4A_var_buffer0, *P4A_var_y0);

   The same, even when supressing unfolding in the p4a_process.
   
