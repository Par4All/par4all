      subroutine MATMUL(a, b, c)

      integer i, j, k
      integer mat1(1:1000, 1:1000), mat2(1:1000, 1:1000), 
     & mat3(1:1000, 1:1000)

      do 10 i = 1, 1000
         do 20 j = 1, 1000
c BEGIN_FPGA_MATMULHRE
            mat3(i, j) = 0
            do 30 k = 1, 1000
               mat3(i, j) = mat3(i, j) + mat1(i, k) * mat2(k, j)
 30   enddo
c END_FPGA_MATMULHRE
 20   enddo
 10   enddo

      return
      end
