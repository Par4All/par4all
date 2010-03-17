      subroutine CPLCH(MODE,TI,TF,I,NOM_VAR,LEN,N,VAR_CHAINE,INFOCA)
c     **********
c
c
c.. Formal Arguments ..
      integer MODE
      real TI
      real TF
      integer I
      character*(*) NOM_VAR
      integer LEN
      integer N
      character*(*) VAR_CHAINE
      integer INFOCA
c
c.. Local Scalars ..
      integer IOS
      character LIGNE*80
c
c ... Executable Statements ...
c
c
c Traitement
c
      write (6,1020) NOM_VAR
      read (10,*,ERR = 9000,END = 9001,IOSTAT = IOS) LIGNE
      read (10,*,ERR = 9000,END = 9001,IOSTAT = IOS) VAR_CHAINE
      N = LEN
      INFOCA = CPOK
      return
c
 9000 write (6,1000) LIGNE
      write (6,*) VAR_CHAINE
      INFOCA = CPSTOP
      return
 9001 write (6,1010) LIGNE
      INFOCA = CPSTOP
      return
c
c ... Format Declarations ...
c
 1000 format (1x,'ERREUR : ',a80)
 1010 format (1x,'FIN DE FICHIER : ',a80)
 1020 format (1x,'CPLCH : lecture CHAINE --> ',a20)
      end
