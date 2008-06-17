      PROGRAM MAIN 

! FC 24/06/96
!
! cgn_parameters.h (version 1.1)
! 96/10/16, 09:25:16
!
! as a general principle, only explicit typing
      implicit none
!
! size parameters
      integer n, ndiag, degre
      parameter (n=4500, ndiag=7, degre=5)
!
! hpf parameters
!hpf$ template T(n)
!hpf$ processors P(NUMBER_OF_PROCESSORS())
!hpf$ distribute T(block) onto P
!
! obscure things for APR I guess.
!      INTEGER apr_opt_int_arg
!      CHARACTER apr_opt_char_arg
!      COMMON /apr_res1/apr_opt_int_arg/apr_res2/apr_opt_char_arg
!      INTEGER a

      REAL sol(n)
      INTEGER i

!hpf$ align with T:: SOL

!hpf$ independent
      DO i = 1, n
         sol(i) = 0.0
      ENDDO

      CALL cgc(sol)

      print *, sol(1)
      END

      SUBROUTINE cgc(sol)
! FC 24/06/96
!
! cgn_parameters.h (version 1.1)
! 96/10/16, 09:25:16
!
! as a general principle, only explicit typing
      implicit none
!
! size parameters
      integer n, ndiag, degre
      parameter (n=4500, ndiag=7, degre=5)
!
! hpf parameters
!hpf$ template T(n)
!hpf$ processors P(NUMBER_OF_PROCESSORS())
!hpf$ distribute T(block) onto P
!
! obscure things for APR I guess.
!      INTEGER apr_opt_int_arg
!      CHARACTER apr_opt_char_arg
!      COMMON /apr_res1/apr_opt_int_arg/apr_res2/apr_opt_char_arg
!      INTEGER a

      REAL sol(n)

      INTEGER i

!hpf$ align with T:: SOL

      CALL matvect(sol)
      
      sol(1) = sol(1) + 1.0

      END

      SUBROUTINE matvect(x)
! FC 24/06/96
!
! cgn_parameters.h (version 1.1)
! 96/10/16, 09:25:16
!
! as a general principle, only explicit typing
      implicit none
!
! size parameters
      integer n, ndiag, degre
      parameter (n=4500, ndiag=7, degre=5)
!
! hpf parameters
!hpf$ template T(n)
!hpf$ processors P(NUMBER_OF_PROCESSORS())
!hpf$ distribute T(block) onto P
!
! obscure things for APR I guess.
!      INTEGER apr_opt_int_arg
!      CHARACTER apr_opt_char_arg
!      COMMON /apr_res1/apr_opt_int_arg/apr_res2/apr_opt_char_arg
!      INTEGER a
      REAL x(n)
      INTEGER i

!hpf$ align with T(*):: X(*)
      
      x(1) = x(1)+2.0

      END
