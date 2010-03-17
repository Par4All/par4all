
      program GO
      integer A

      A= 5
      call S1(A)

      do A= 6,10
         call S1(A)
      enddo

      end

      subroutine S1(B)
      integer B


      print *, B
      end
