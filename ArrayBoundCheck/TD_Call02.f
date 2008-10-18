c
c     Call with many array references
c
c
c
      subroutine jacu
c
c***compute the upper triangular part of the jacobian matrix
c
c Author: Sisira Weeratunga
c         NASA Ames Research Center
c         (10/25/90)
c
c #include 'applu.incl'  

      implicit real*8 (a-h,o-z)
c
      parameter ( c1 = 1.40d+00, c2 = 0.40d+00,
     $            c3 = 1.00d-01, c4 = 1.00d+00,
     $            c5 = 1.40d+00 )
c
c***grid
c
      common/cgcon/ nx, ny, nz,
     $              ii1, ii2, ji1, ji2, ki1, ki2, itwj,
     $              dxi, deta, dzeta,
     $              tx1, tx2, tx3,
     $              ty1, ty2, ty3,
     $              tz1, tz2, tz3
c
c***dissipation
c
      common/disp/ dx1,dx2,dx3,dx4,dx5,
     $             dy1,dy2,dy3,dy4,dy5,
     $             dz1,dz2,dz3,dz4,dz5,
     $             dssp
c
c***field variables and residuals
c
      common/cvar/ u(5,33,33,33),
     $             rsd(5,33,33,33),
     $             frct(5,33,33,33)
c
c***output control parameters
c
      common/cprcon/ ipr, iout, inorm
c
c***newton-raphson iteration control parameters
c
      common/ctscon/ itmax, invert,
     $               dt, omega, tolrsd(5),
     $               rsdnm(5), errnm(5), frc, ttotal
c
      common/cjac/ a(5,5,33,33,33),
     $             b(5,5,33,33,33),
     $             c(5,5,33,33,33),
     $             d(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      common/cexact/ ce(5,13)
c



c
      r43 = ( 4.0d+00 / 3.0d+00 )
      c1345 = c1 * c3 * c4 * c5
      c34 = c3 * c4
c      
      do k = 2, nz - 1
c
         do j = 2, ny - 1
c
            do i = 2, nx - 1
c               
c***form the first block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i+1,j,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               a(1,1,i,j,k) = - dt * tx1 * dx1
               a(1,2,i,j,k) =   dt * tx2
               a(1,3,i,j,k) =   0.0d+00
               a(1,4,i,j,k) =   0.0d+00
               a(1,5,i,j,k) =   0.0d+00
c
               a(2,1,i,j,k) =  dt * tx2
     $          * ( - ( u(2,i+1,j,k) * tmp1 ) ** 2
     $     + c2 * 0.50d+00 * (  u(2,i+1,j,k) * u(2,i+1,j,k)
     $                        + u(3,i+1,j,k) * u(3,i+1,j,k)
     $                        + u(4,i+1,j,k) * u(4,i+1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - r43 * c34 * tmp2 * u(2,i+1,j,k) )
               a(2,2,i,j,k) =  dt * tx2
     $          * ( ( 2.0d+00 - c2 ) * ( u(2,i+1,j,k) * tmp1 ) )
     $          - dt * tx1 * ( r43 * c34 * tmp1 )
     $          - dt * tx1 * dx2
               a(2,3,i,j,k) =  dt * tx2
     $              * ( - c2 * ( u(3,i+1,j,k) * tmp1 ) )
               a(2,4,i,j,k) =  dt * tx2
     $              * ( - c2 * ( u(4,i+1,j,k) * tmp1 ) )
               a(2,5,i,j,k) =  dt * tx2 * c2 
c
               a(3,1,i,j,k) =  dt * tx2
     $              * ( - ( u(2,i+1,j,k) * u(3,i+1,j,k) ) * tmp2 )
     $         - dt * tx1 * ( - c34 * tmp2 * u(3,i+1,j,k) )
               a(3,2,i,j,k) =  dt * tx2 * ( u(3,i+1,j,k) * tmp1 )
               a(3,3,i,j,k) =  dt * tx2 * ( u(2,i+1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx3
               a(3,4,i,j,k) = 0.0d+00
               a(3,5,i,j,k) = 0.0d+00
c
               a(4,1,i,j,k) = dt * tx2
     $          * ( - ( u(2,i+1,j,k)*u(4,i+1,j,k) ) * tmp2 )
     $          - dt * tx1 * ( - c34 * tmp2 * u(4,i+1,j,k) )
               a(4,2,i,j,k) = dt * tx2 * ( u(4,i+1,j,k) * tmp1 )
               a(4,3,i,j,k) = 0.0d+00
               a(4,4,i,j,k) = dt * tx2 * ( u(2,i+1,j,k) * tmp1 )
     $          - dt * tx1 * ( c34 * tmp1 )
     $          - dt * tx1 * dx4
               a(4,5,i,j,k) = 0.0d+00
c
               a(5,1,i,j,k) = dt * tx2
     $          * ( ( c2 * (  u(2,i+1,j,k) * u(2,i+1,j,k)
     $                      + u(3,i+1,j,k) * u(3,i+1,j,k)
     $                      + u(4,i+1,j,k) * u(4,i+1,j,k) ) * tmp2
     $              - c1 * ( u(5,i+1,j,k) * tmp1 ) )
     $          * ( u(2,i+1,j,k) * tmp1 ) )
     $          - dt * tx1
     $          * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i+1,j,k)**2 ))

C     $              - (     c34 - c1345 ) * tmp3 * ( u(3,i+1,j,k)**2 )

C     $              - (     c34 - c1345 ) * tmp3 * ( u(4,i+1,j,k)**2 )
C     $              - c1345 * tmp2 * u(5,i+1,j,k) )
               a(5,2,i,j,k) = dt * tx2
     $          * ( c1 * ( u(5,i+1,j,k) * tmp1 )
     $             - 0.50d+00 * c2
     $             * ( (  3.0d+00*u(2,i+1,j,k)*u(2,i+1,j,k)
     $                  + u(3,i+1,j,k)*u(3,i+1,j,k)
     $                  + u(4,i+1,j,k)*u(4,i+1,j,k) ) * tmp2 ) )
     $           - dt * tx1
     $           * ( r43*c34 - c1345 ) * tmp2 * u(2,i+1,j,k)
               a(5,3,i,j,k) = dt * tx2
     $           * ( - c2 * ( u(3,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(3,i+1,j,k)
               a(5,4,i,j,k) = dt * tx2
     $           * ( - c2 * ( u(4,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 )
     $           - dt * tx1
     $           * (  c34 - c1345 ) * tmp2 * u(4,i+1,j,k)
               a(5,5,i,j,k) = dt * tx2
     $           * ( c1 * ( u(2,i+1,j,k) * tmp1 ) )
     $           - dt * tx1 * c1345 * tmp1
     $           - dt * tx1 * dx5
c
c***form the second block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j+1,k)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               b(1,1,i,j,k) = - dt * ty1 * dy1
               b(1,2,i,j,k) =   0.0d+00
               b(1,3,i,j,k) =  dt * ty2
               b(1,4,i,j,k) =   0.0d+00
               b(1,5,i,j,k) =   0.0d+00
c
               b(2,1,i,j,k) =  dt * ty2
     $           * ( - ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 )
     $           - dt * ty1 * ( - c34 * tmp2 * u(2,i,j+1,k) )
               b(2,2,i,j,k) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 )
     $          - dt * ty1 * ( c34 * tmp1 )
     $          - dt * ty1 * dy2
               b(2,3,i,j,k) =  dt * ty2 * ( u(2,i,j+1,k) * tmp1 )
               b(2,4,i,j,k) = 0.0d+00
               b(2,5,i,j,k) = 0.0d+00
c
               b(3,1,i,j,k) =  dt * ty2
     $           * ( - ( u(3,i,j+1,k) * tmp1 ) ** 2
     $      + 0.50d+00 * c2 * ( (  u(2,i,j+1,k) * u(2,i,j+1,k)
     $                           + u(3,i,j+1,k) * u(3,i,j+1,k)
     $                           + u(4,i,j+1,k) * u(4,i,j+1,k) )
     $                          * tmp2 ) )
     $       - dt * ty1 * ( - r43 * c34 * tmp2 * u(3,i,j+1,k) )
               b(3,2,i,j,k) =  dt * ty2
     $                   * ( - c2 * ( u(2,i,j+1,k) * tmp1 ) )
               b(3,3,i,j,k) =  dt * ty2 * ( ( 2.0d+00 - c2 )
     $                   * ( u(3,i,j+1,k) * tmp1 ) )
     $       - dt * ty1 * ( r43 * c34 * tmp1 )
     $       - dt * ty1 * dy3
               b(3,4,i,j,k) =  dt * ty2
     $                   * ( - c2 * ( u(4,i,j+1,k) * tmp1 ) )
               b(3,5,i,j,k) =  dt * ty2 * c2
c
               b(4,1,i,j,k) =  dt * ty2
     $              * ( - ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 )
     $       - dt * ty1 * ( - c34 * tmp2 * u(4,i,j+1,k) )
               b(4,2,i,j,k) = 0.0d+00
               b(4,3,i,j,k) =  dt * ty2 * ( u(4,i,j+1,k) * tmp1 )
               b(4,4,i,j,k) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 )
     $                        - dt * ty1 * ( c34 * tmp1 )
     $                        - dt * ty1 * dy4
               b(4,5,i,j,k) = 0.0d+00
c
               b(5,1,i,j,k) =  dt * ty2
     $          * ( ( c2 * (  u(2,i,j+1,k) * u(2,i,j+1,k)
     $                      + u(3,i,j+1,k) * u(3,i,j+1,k)
     $                      + u(4,i,j+1,k) * u(4,i,j+1,k) ) * tmp2
     $               - c1 * ( u(5,i,j+1,k) * tmp1 ) )
     $          * ( u(3,i,j+1,k) * tmp1 ) )
     $          - dt * ty1
     $          * ( - (     c34 - c1345 )*tmp3*(u(2,i,j+1,k)**2))

C     $              - ( r43*c34 - c1345 )*tmp3*(u(3,i,j+1,k)**2)


C     $              - (     c34 - c1345 )*tmp3*(u(4,i,j+1,k)**2)
C     $              - c1345*tmp2*u(5,i,j+1,k) )
               b(5,2,i,j,k) =  dt * ty2
     $          * ( - c2 * ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 )
     $          - dt * ty1
     $          * ( c34 - c1345 ) * tmp2 * u(2,i,j+1,k)
               b(5,3,i,j,k) =  dt * ty2
     $          * ( c1 * ( u(5,i,j+1,k) * tmp1 )
     $          - 0.50d+00 * c2 
     $          * ( (  u(2,i,j+1,k)*u(2,i,j+1,k)
     $               + 3.0d+00 * u(3,i,j+1,k)*u(3,i,j+1,k)
     $               + u(4,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 ) )
     $          - dt * ty1
     $          * ( r43*c34 - c1345 ) * tmp2 * u(3,i,j+1,k)
               b(5,4,i,j,k) =  dt * ty2
     $          * ( - c2 * ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 )
     $          - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j+1,k)
               b(5,5,i,j,k) =  dt * ty2
     $          * ( c1 * ( u(3,i,j+1,k) * tmp1 ) )
     $          - dt * ty1 * c1345 * tmp1
     $          - dt * ty1 * dy5
c
c***form the third block sub-diagonal
c
               tmp1 = 1.0d+00 / u(1,i,j,k+1)
               tmp2 = tmp1 * tmp1
               tmp3 = tmp1 * tmp2
c
               c(1,1,i,j,k) = - dt * tz1 * dz1
               c(1,2,i,j,k) =   0.0d+00
               c(1,3,i,j,k) =   0.0d+00
               c(1,4,i,j,k) = dt * tz2
               c(1,5,i,j,k) =   0.0d+00
c
               c(2,1,i,j,k) = dt * tz2
     $           * ( - ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(2,i,j,k+1) )
               c(2,2,i,j,k) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 )
     $           - dt * tz1 * c34 * tmp1
     $           - dt * tz1 * dz2 
               c(2,3,i,j,k) = 0.0d+00
               c(2,4,i,j,k) = dt * tz2 * ( u(2,i,j,k+1) * tmp1 )
               c(2,5,i,j,k) = 0.0d+00
c
               c(3,1,i,j,k) = dt * tz2
     $           * ( - ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $           - dt * tz1 * ( - c34 * tmp2 * u(3,i,j,k+1) )
               c(3,2,i,j,k) = 0.0d+00
               c(3,3,i,j,k) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 )
     $           - dt * tz1 * ( c34 * tmp1 )
     $           - dt * tz1 * dz3
               c(3,4,i,j,k) = dt * tz2 * ( u(3,i,j,k+1) * tmp1 )
               c(3,5,i,j,k) = 0.0d+00
c
               c(4,1,i,j,k) = dt * tz2
     $        * ( - ( u(4,i,j,k+1) * tmp1 ) ** 2
     $            + 0.50d+00 * c2
     $            * ( ( u(2,i,j,k+1) * u(2,i,j,k+1)
     $                + u(3,i,j,k+1) * u(3,i,j,k+1)
     $                + u(4,i,j,k+1) * u(4,i,j,k+1) ) * tmp2 ) )
     $        - dt * tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k+1) )
               c(4,2,i,j,k) = dt * tz2
     $             * ( - c2 * ( u(2,i,j,k+1) * tmp1 ) )
               c(4,3,i,j,k) = dt * tz2
     $             * ( - c2 * ( u(3,i,j,k+1) * tmp1 ) )
               c(4,4,i,j,k) = dt * tz2 * ( 2.0d+00 - c2 )
     $             * ( u(4,i,j,k+1) * tmp1 )
     $             - dt * tz1 * ( r43 * c34 * tmp1 )
     $             - dt * tz1 * dz4
               c(4,5,i,j,k) = dt * tz2 * c2
c
               c(5,1,i,j,k) = dt * tz2
     $     * ( ( c2 * (  u(2,i,j,k+1) * u(2,i,j,k+1)
     $                 + u(3,i,j,k+1) * u(3,i,j,k+1)
     $                 + u(4,i,j,k+1) * u(4,i,j,k+1) ) * tmp2
     $       - c1 * ( u(5,i,j,k+1) * tmp1 ) )
     $            * ( u(4,i,j,k+1) * tmp1 ) )
     $       - dt * tz1
     $       * ( - ( c34 - c1345 ) * tmp3 * (u(2,i,j,k+1)**2))

C     $           - ( c34 - c1345 ) * tmp3 * (u(3,i,j,k+1)**2)

C     $           - ( r43*c34 - c1345 )* tmp3 * (u(4,i,j,k+1)**2)
C     $          - c1345 * tmp2 * u(5,i,j,k+1) )
               c(5,2,i,j,k) = dt * tz2
     $       * ( - c2 * ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(2,i,j,k+1)
               c(5,3,i,j,k) = dt * tz2
     $       * ( - c2 * ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 )
     $       - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k+1)
               c(5,4,i,j,k) = dt * tz2
     $       * ( c1 * ( u(5,i,j,k+1) * tmp1 )
     $       - 0.50d+00 * c2
     $       * ( (  u(2,i,j,k+1)*u(2,i,j,k+1)
     $            + u(3,i,j,k+1)*u(3,i,j,k+1)
     $            + 3.0d+00*u(4,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) )
     $       - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k+1)
               c(5,5,i,j,k) = dt * tz2
     $       * ( c1 * ( u(4,i,j,k+1) * tmp1 ) )
     $       - dt * tz1 * c1345 * tmp1
     $       - dt * tz1 * dz5
c
            end do
c
         end do
c
      end do               
c
      return
      end











