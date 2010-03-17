      program applu

      common/cgcon/ nx, ny, nz
      common/cvar/ u(5,33,33,33)
      read (5,*) nx, ny, nz

      if ( ( nx .lt. 5 ) .or.
     $     ( ny .lt. 5 ) .or.
     $     ( nz .lt. 5 ) ) then
c
         write (6,2001)
 2001    format (5x,'PROBLEM SIZE IS TOO SMALL - ',
     $        /5x,'SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5')
         stop
      end if
      if ( ( nx .gt. 33 ) .or.
     $     ( ny .gt. 33 ) .or.
     $     ( nz .gt. 33 ) ) then

         write (6,2002)
 2002    format (5x,'PROBLEM SIZE IS TOO LARGE - ',
     $        /5x,'NX, NY AND NZ SHOULD BE LESS THAN OR EQUAL TO ',
     $        /5x,'ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY')
         stop
      end if
c
      call setbv
      call error
      stop
      end

      subroutine setbv
      common/cgcon/ nx, ny, nz
      common/cvar/ u(5,33,33,33)

      do j = 1, ny
         do i = 1, nx
            call exact ( i, j, 1, u( 1, i, j, 1 ) )
            call exact ( i, j, nz, u( 1, i, j, nz ) )
         end do
      end do
      do k = 1, nz
         do i = 1, nx
            call exact ( i, 1, k, u( 1, i, 1, k ) )
            call exact ( i, ny, k, u( 1, i, ny, k ) )
         end do
      end do
      do k = 1, nz
         do j = 1, ny
            call exact ( 1, j, k, u( 1, 1, j, k ) )
            call exact ( nx, j, k, u( 1, nx, j, k ) )
         end do
      end do
      return
      end

      subroutine exact ( i, j, k, u000ijk )
      common/cgcon/ nx, ny, nz
      dimension u000ijk(*)
   
      do m = 1, 5
         u000ijk(m) =  m
      end do
      return
      end
c
      subroutine error
      common/cgcon/ nx, ny, nz
      common/cvar/ u(5,33,33,33)
      dimension u000ijk(5)
      do k = 2, nz-1
         do j = 2, ny-1
            do i = 2, nx-1
c     
               call exact ( i, j, k, u000ijk )
               
            end do
         end do
      end do
      return
      end







