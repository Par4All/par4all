      program proper
      integer i,A(30)

      do i = 11,20
         A(i) = A(i-10)+A(i+10)
      enddo
      do i = 11,20
         call dum1(A(i), A(i-10), A(i+10))
      enddo
      do i = 11,20
        call dum2(A(i), A(i-10)+A(i+10))
      enddo
      end


      subroutine dum1(i,j,k)
      integer i,j,k
      i = j + k
      end

      subroutine dum2(i,j)
      integer i,j
      i = j
      end
      


