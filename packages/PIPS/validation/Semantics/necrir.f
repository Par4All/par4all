	subroutine necrir(ctab,cnomf,lon,ker)
	integer *2 lon,ker,ibuf
	character*1 ctab(*)
	character *1 cbuf(2)
	character *6 cnomf
	character *80 cnombase ,cnoml          
	common/nombase/cnombase
	equivalence (ibuf,cbuf(1))
	
c
c--------------lecture des fichiers venant de l'analyse primaire
c	auteur J.Macrez
c
c------entrees--------------------------------------------------
c	cnomf : nom du fichier a lire
c	lon : longueur en mot de 2 octets a lire
c---------------------------------------------------------------
c
c--------------------------------sorties------------------------
c	itab : table remplie
c	ker : indicatif d'erreur =0 si tout va bien
c---------------------------------------------------------------
c
	ker=0
	lon2=2*lon
	lrec=8000
C	ifin=long(cnombase)***pas de source LONG
	cnoml= cnomf
	open (unit=8, file= cnoml, err= 99,form='unformatted')
	rewind(8)
	ibuf=lon+1
C	istat=fputc(8,cbuf(1))***pas de source FPUTC
C	istat=fputc(8,cbuf(2))***pas de source FPUTC
	do i=1,lon*2
C	***istrat=fputc(8,ctab(i))***pas de source FPUTC
		x=ctab(i)
	enddo
	close(8)
	return
99	close(8)
	ker=1
	return
	end
