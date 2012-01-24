      program private
c	privatisation malgre les DO implicites	
c	dans les I/O
        real a(10,10)
        integer i,j,k

        do j = 1, 32
          do i = 1, 10
            a(i,j) = 0.
          enddo
	  print *, ((a(i,k), i=1,10), k=1,i)
        enddo
        print *, ((a(i,k), i=1,10), k=1,j)
        end
