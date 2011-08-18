      subroutine saxpy(sx,sy)

      INCLUDE 'SAXPY_INC.f'

c
c     constant times a vector plus a vector.
c     uses unrolled loop for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c

      real sx(SIZE),sy(SIZE),sa
      integer i,incx,incy,ix,iy,m,mp1

      sa = 3
      incx = 1
      incy = 1
c
      if (sa .eq. 0.0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      do 10 i = 1,SIZE
        sy(iy) = sy(iy) + sa*sx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c

  

c 20   PRINT *, 'bef'
 20   do 50 i = 1,SIZE
        sy(i) = sy(i) + sa*sx(i)
   50 continue
c      PRINT *, 'aft'
      return
      return
      end

      subroutine saxpyref(sx,sy)

      INCLUDE 'SAXPY_INC.f'

c
c     constant times a vector plus a vector.
c     uses unrolled loop for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      real sx(SIZE),sy(SIZE),sa
      integer i,incx,incy,ix,iy,m,mp1

      sa = 3
      incx = 1
      incy = 1
c
      if (sa .eq. 0.0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      do 10 i = 1,SIZE
        sy(iy) = sy(iy) + sa*sx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
  

 20   i=1
      do 50 i = 1,SIZE
        sy(i) = sy(i) + sa*sx(i)
   50 continue
c      PRINT *, 'aft2'
      return
      return
      end

      INTEGER FUNCTION CLOCK()
      
      CLOCK = 0

      RETURN
      END

      PROGRAM MAIN

      INCLUDE 'SAXPY_INC.f'

      INTEGER i, j, incx, incy
      INTEGER bef, af, diff
      INTEGER befRef, afRef, diffRef
      INTEGER CLOCK
      REAL  sa, sx(1:SIZE), sy(1:SIZE)
      REAL  sxRef(1:SIZE), syRef(1:SIZE)

      LOGICAL success

      incx = 1
      incy = 1
      sa = 3
      DO 10 i = 1, SIZE
          sx(i) = i
          sy(i) = i
          sxRef(i) = i
          syRef(i) = i
10    ENDDO

      bef = CLOCK()
      DO 20 i = 1, 10000
          CALL saxpy(sx,sy)
20    ENDDO
      af = CLOCK()

      befRef = CLOCK()
      DO 21 i = 1, 10000
          CALL saxpyRef(sxRef,syRef)
21    ENDDO
      afRef = CLOCK()

      success = .TRUE.


      DO 22 i = 1, SIZE
         IF((ABS(sy(i) - syRef(i)) / syRef(i)) .GT. 0.000001) THEN
             success = .FALSE.
         ENDIF
22    ENDDO
      PRINT *, sy(100), syRef(100)
      IF(success .EQV. .TRUE.) THEN
         PRINT *, 'SUCCESS'
      ELSE
         PRINT *, 'ERROR'
      ENDIF

      diff = af - bef
      diffRef = afRef - befRef
      PRINT *, 'time: ', diff
      PRINT *, 'reference time: ', diffRef

      RETURN
      END
