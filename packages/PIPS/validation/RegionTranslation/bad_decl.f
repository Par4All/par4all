      program bad_decl
      integer A(10,10)

      call bad(A)

      end
      
      subroutine bad(B)
      integer B(1)

      do i = 1,50
         B(i) = 0
      enddo
      end
