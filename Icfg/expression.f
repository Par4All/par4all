
      program GO
      integer A, S2
      external S2

      A= 5
      call S1(A)

      do A= 6,S2(7)
         call S1(A)
      enddo

      end

      subroutine S1(B)
      integer B


      print *, B
      end

      integer function S2(B)
      integer B

      s2 = b + 2
      end

