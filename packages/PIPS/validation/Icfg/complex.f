
      program GO
      integer A

      j = 7

      if  (is2(is2(7)) .eq. 11) then
         call is1(j)
       else
         print *,j
      endif
      call is1(j)
      end

      subroutine IS1(B)
      integer B


      print *, B
      end

      function IS2(B)
      integer B

      is2 = b + 2
      end

