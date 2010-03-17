      program loop
      integer i
      real r
      double precision d
      complex c
      double complex dc
      logical l
      character h
! type of index
      do i=1, 10, 2
         print *, 'hello'
      enddo
      do r=1, 10, 2
         print *, 'hello'
      enddo
      do d=1, 10, 2
         print *, 'hello'
      enddo
      do c=1, 10, 2
         print *, 'hello'
      enddo
      do dc=1, 10, 2
         print *, 'hello'
      enddo
      do l=1, 10, 2
         print *, 'hello'
      enddo
      do h=1, 10, 2
         print *, 'hello'
      enddo
! int index
      do i=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do i=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do i=1.0E0, 10.0E0, 2.0E0
         print *, 'hello'
      enddo
      do i=1.0D0, 10.0D0, 2.0D0
         print *, 'hello'
      enddo
      do i=l, c, dc
         print *, 'hello'
      enddo
! real index
      do r=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do r=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do r=1.0E0, 10.0E0, 2.0E0
         print *, 'hello'
      enddo
      do r=1.0D0, 10.0D0, 2.0D0
         print *, 'hello'
      enddo
! double index
      do d=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do d=1.0, 10.0, 2.0
         print *, 'hello'
      enddo
      do d=1.0E0, 10.0E0, 2.0E0
         print *, 'hello'
      enddo
      do d=1.0D0, 10.0D0, 2.0D0
         print *, 'hello'
      enddo
      end
