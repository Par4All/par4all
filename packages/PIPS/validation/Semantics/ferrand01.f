      program ferrand01
      integer i
      integer*1 A( 200 )
      integer*1 B( 200 )
      integer*1 C( 200 )
      integer*1 BI
      integer*1 CI

      do 20 i = 0, 199
         BI = B(i)
         CI = C(i)
         A(i) = BI + CI
 20   enddo

      end
