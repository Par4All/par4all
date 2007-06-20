      program IF1
      integer A(10,10), i, j 
      
      do i = 1, 10
         do j = 1, 10
            call TEST(A, i, j)
         enddo
      enddo
      
      end

      subroutine TEST(A,i,j)
      integer A(10,10), i, j
      
      if (i.lt.j) then
         A(i,j) = A(j,i)
      endif
      end
