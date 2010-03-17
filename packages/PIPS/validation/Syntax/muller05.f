! bug apparaissant pour le bench NAS pour FT avec un probleme de classe C

! si on diminue ntotalp, plus d'erreur
! si on supprime "a" dans la ligne du common, plus d'erreur
! si on change le type de u0, plus d'erreur

! u0 requires 2GB of memory

      program muller05
      implicit none
      parameter (ntotalp=512*512*512)

      double complex u0(ntotalp)

      integer a
      common /bigarrays/ u0,a
      end
