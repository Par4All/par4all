Conception d’une éditeur sémantique pour infrastructure de transformation de programme source-à-source avec Eclipse
===================================================================================================================

:emphasis:`Semantics editor for a source-to-source program transformation
infrastructure with Eclipse`

Sujet de stage d’ingénieur

**Mots-clés** : éditeurs, IDE, analyse sémantique, compilation, analyse de
 programmes, transformation de programmes, Eclipse, Emacs

Développement d’un plugin Eclipse générique avec comme exemple Par4All en
collaboration avec http://obeo.fr

Contexte
--------

De nombreux projets font appel à des environnements de transformation de
programme en ingénierie informatique logicielle ou matérielle :
optimisation de programmes, parallélisation automatique, rétro-ingénierie,
transformation de modèles, refactoring, synthèse architecturale, etc.

Ces environnements doivent combiner à la fois une interface (souvent
graphique) avec le moteur interne et une interface de rendu du travail
effectué par le moteur. En plus in faut faciliter la vie de l’utilisateur
en affichant de nombreuses informations sémantiques sur les objets
manipulés par l’infrastructure (programmes avec des lignes de code, des
variables typées, des prédicats...).

D’un autre côté il existe des éditeurs de texte puissants et programmables
tels qu’Emacs avec de nombreux modes spécialisés pour traiter différents
types de textes et de données (sources en différents langages) mais aussi
interagir avec des processus, des protocoles réseaux (courrier
électronique...).

Le développement d’IDE en logiciel libre tel qu’Eclipse permet d’avoir des
outils de travail logiciels ouverts dotés d’énormes capacités et
extensibles pour différents usages.


PIPS
----

PIPS (http://pips4u.org) est un environnement de d’analyse et de
transformations de programmes développé aux MINES ParisTech qui prend en
entrée des programmes écrits en différents dialectes de Fortran ou de C et
génère du Fortran, du C ou du SmallTalk dans des dialectes plus ou moins
parallèles (MPI, PVM, HPF, OpenMP...). PIPS comprend de nombreuses phases
d’analyses telles que le calcul de graphes de dépendances
interprocéduraux, de prédicats sur les variables entières des programmes,
permettant de détecter certaines propriétés et d’appliquer des phases
d’optimisations de programmes, de parallélisation, etc.

L’interface utilisateur la plus expressive actuellement est une interface
de type shell ``tpips`` qui est un peu brut de fonderie pour les
utilisateurs lambda.


EcliPIPS
--------

Eclipse est la référence en terme d’IDE portable capable de gérer de
nombreux langages. En particulier, l’extension PTP permet de la
programmation parallèle et de la mise au point sur de nombreux processeurs
à distance.

L’intégration de PIPS dans Eclipse permettrait de rajouter des fonctions
de parallélisation et d’analyse aux utilisateurs d’Eclipse PTP et
réciproquement rajouterait une interface graphique aux utilisateurs de
PIPS.

Il s’agira dans ce stage de développer de nouveaux plugins dans Eclipse
pour interagir avec PIPS, récupérer des informations sémantiques calculées
par PIPS et les rajouter comme décorations aux éléments des programmes
sources manipulés par les utilisateurs d’Eclipse, de pouvoir les masquer
ou les sélectionner, etc. Cela permettra de rajouter de nombreuses
fonctionnalités à Eclipse qui pourra être utilisé comme atelier logiciel
graphique de parallélisation, transformation, refactoring, analyse, etc.

Au niveau technique, PIPS possède une interface de script PyPS écrite en
Python qui servira de base au développement de l’interface avec Eclipse du
côté PIPS.

Il existe aussi un prototype EPips de mode GNU/Emacs écrit en EmacsLISP
avec des greffons dans PIPS en C et une vieille bibliothèque
graphique. L’intérêt d’EPips est de fournir une interface légère mais tout
de même puissante pour interagir avec PIPS. Elle n’est plus fonctionnelle
à cause de l’obsolescence de la bibliothèque graphique mais pourrait
servir de base pour construire EcliPiPS.

Le système sera capable d’afficher simultanément de nombreuses
informations orientées lignes qui seront calculées ou cachées à la demande
ainsi que d’attacher des informations sémantiques à tout objet textuel qui
seront affichables par menu contextuel, bulle d’aide, colorisation, etc.

Il faudra étudier l’interaction avec les modes natifs d’Emacs si on s’attaque aussi à l’interface EPips, outre EcliPIPS.

Profil recherché : passionné d’informatique voulant faire de la vraie informatique et n’ayant pas peur de réfléchir ni de programmer. Bonnes connaissances utiles : C, Java, Lisp, Eclipse, Emacs, Python, Unix, programmation parallèle, YAML, JSON…

Ce stage peut enchaîner avec une embauche et/ou une thèse et fait partie du projet Pôle de Compétitivité System@TIC OpenGPU.

Encadrant : Ronan Keryell (rk at hpc-project dot com) et collaboration avec Mines ParisTech/CRI et membres du projet OpenGPU

http://par4all.org

Lieu : HPC Project http://www.hpc-project.com start-up de 35 personnes en pleine croissance.

Meudon (92) ou Montpellier (34).

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "french"
  ### End:
