      program continuation01

      do i = 1, 2
         write(6, 263) i
263      FORMAT('0INITIAL ESTIMATE OF THETA(',I2,') IS NOT GIVEN'/' BUT 
     &THE CORRESPONDING LOWER AND/OR UPPER BOUND IS ','INAPPROPRIATE')
      enddo
      end
