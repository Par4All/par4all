C     Make sure than parallel loops are preserved

      program loop_interchange08
      real a(10,11)

      do 100 i = 1, 10
         do j = 1, 11
            a(i,j) = 0.
         enddo
 100  continue

      end
