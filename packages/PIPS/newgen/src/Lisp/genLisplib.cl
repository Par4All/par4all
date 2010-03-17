#|

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

|#


;;;;
;;;; genLisplib.cl
;;;;
;;;; The CommonLISP library of Newgen functions
;;;; 
;;;; First version - 9/3/88: Pierre Jouvelot (pj)
;;;;
;;;; 15/11/88: Added switch (pj)
;;;; 03/04/89: Added tabulated domains (pj)
;;;; 14/04/89: Added the GEN-RECURSE macro (pj)
;;;; 19/09/89: Added sets (pj) --- NOT TESTED.

;;;; 20/08/93: Not up-to-date with NewGen-C (sparse arrays are not supported) 

;;; The Newgen package information

(provide :newgen)

(in-package :newgen)

(shadow '(set-difference))

(export '(or-tag
	  gen-read gen-write gen-read-tabulated gen-write-tabulated
	  gen-free gen-free-tabulated
	  gen-init-external gen-read-spec
	  gen-mapc-tabulated gen-filter-tabulated gen-find-tabulated
	  gen-true gen-false gen-true-p gen-false-p 
	  gen-switch gen-recurse
	  *warn-on-tabulated-redefinition*
	  *warn-on-ref-without-def-in-read-tabulated*
	  ))

(require "set" "set.fasl")

(use-package '(:newgen :set))

#+cltl2 (declaim (optimize (speed 3) (safety 0)))
#-cltl2 (proclaim '(optimize (speed 3) (safety 0)))

;;#+allegro (proclaim '(:explain :calls))

;;; User selectable options.

(defvar *warn-on-tabulated-redefinition* nil)
(defvar *warn-on-ref-without-def-in-read-tabulated* nil)

;;; Implementation of Newgen + domains.

(defstruct (or (:type #+akcl vector #-akcl (vector t)))
    "OR structures implement Newgen + domains"
    type
    tag
    val)

(defstruct (tabular-or (:type #+akcl vector #-akcl (vector t)))
    "OR structures implement Newgen + domains"
    type
    tabular
    tag
    val)

;;; Typed Newgen booleans

(defconstant gen-true '(:bool 1))
(defconstant gen-false '(:bool 0))
(defmacro gen-true-p (x) `(equal ,x TRUE))
(defmacro gen-false-p (x) `(equal ,x FALSE))

;;; The #] macro is used in Newgen generated files

(eval-when (load compile)
	   (when (get-dispatch-macro-character #\# #\])
		 (warn "Redefining #] dispatch macro"))
	   (set-dispatch-macro-character
	    #\# #\]
	    #'(lambda (stream sub-char infix)
		  (declare (ignore infix))
		  (format stream "~C~C" #\# sub-char))))

;;; Description of NewGen node implementation

(defmacro newgen-vector-p (v)
    `(and (arrayp ,v)
	  (consp (aref ,v 0)) 
	  (eq (first (aref ,v 0)) :newgen)))
(defmacro node-domain (node) `(second (aref ,node 0)))
(defmacro node-alloc (node) `(aref ,node 1))
(defmacro node-hash (node) `(aref ,node 2))

;; These constants are here to manage tabulated domains.

(defconstant max-tabulated 2 "Maximum number of tabulated domains")
(defconstant max-tabulated-elements 1000
    "Maximum number of tabulated elements per domain")
(defconstant separation-char #\| 
    "Used to separate index from string id in Newgen output format")
(defconstant min-tabulated-chunk-size 3 "Size of fake chunks for read-tabulated")
(defconstant tabulated-bp 6 "Hard coded constant for Tabulated_bp in Newgen-C") 

(defvar *write-tabulated* nil
    "Fluid-bind it to t when dealing with GEN-WRITE-TABULATED")

(proclaim '(type (simple-array t (*)) *gen-tabulated*))
(defvar *gen-tabulated* nil
    "Maps any tabulated domain index to its tabulation table")

(defvar *gen-tabulated-names* '()
    "Maps tabulated names to their index in the tabulation table")

(proclaim '(type (simple-array t (*)) *gen-tabulated-index*))
(defvar *gen-tabulated-index* '()
    "Defined by Newgen. Maps domains to their index in the *GEN-TABULATED*
table of tabulation tables")

(proclaim '(type (simple-array t (*)) *gen-tabulated-alloc*))
(defvar *gen-tabulated-alloc* '()
    "Defined by Newgen. Maps domains to their current alloc index in their 
tabulation table (or -1 if it's not a tabulated domain")

(defconstant no-attribut 0
    "Don't do anything speaical when printing tabulated objects")
(defconstant a-def 1
    "Write a #]def token and proceed as for a non-tabulated object")

(defmacro tabulated-p (domain)
    `(not (= (svref *gen-tabulated-index* ,domain) -1)))

;;; Parameters to manage external types

(defparameter max-externals 100 "Maximum number of external types")

(defvar *external-cpt* 'undefined)
(defvar *external-read* nil)
(defvar *external-write* nil)

;;; Parameters to manage sharing

(defconstant unshared-pointer 0)

(defvar *hash-counter* 0 "Number of shared pointers in the data structure")

(defvar *current-shared* '() "Hash table of shared pointers for GEN-WRITE")

;;; The inlined types by NewGen

(deftype base-type ()
  '(or integer float character string))


;;; Ok, let's begin !

(defun gen-write (stream node)
    "Prints a NODE on the STREAM"
    (setf *hash-counter* 0)
    (get-shared node)
    (let ((*standard-output* stream))
	(princ *hash-counter*) (princ " ")
	(gen-write-1 node))
    (values))

(defun gen-write-1 (node)
    "Prints a NODE on the *STANDARD-OUTPUT*.
See READ-SHARING for an explanation of the format and GET-SHARED-1
for the meaning of hash values (as soon as a shared structure has been
printed, its hash value is changed to the opposite value so that subsequent
apparitions are printed as references to this pointer"
    (flet ((print (node)
		  (etypecase 
		   node
		   (symbol (cond ((eq node :undefined) (princ "#]null "))
				 ((eq node :unit) (princ "#]unit "))
				 ((eq node nil) (princ "()"))
				 (t (error "~%Unknown symbol ~S" node))))
		   (base-type 
		    (prin1 node) (princ " "))
		   (cons
		    (cond ((eq node :list-undefined)
			   (princ "#]list "))
			  ((eq (car node) :bool)
			   (princ "#]bool ")
			   (princ (cadr node)) (princ " "))
			  ((eq (car node) :external)
			   (princ "#]external ")
			   (princ (cadr node)) (princ " ")
			   (funcall (svref *external-write* (cadr node))
				    *standard-output*
				    (caddr node)))
			  (t (princ "(")
			     (mapc #'gen-write-1 node)
			     (princ ")"))))
		   (set 
		    (cond ((eq node :set-undefined)
			   (princ "#]set "))
			  (t
			   (princ "{ ")
			   (princ (set::set-type node))
			   (maphash #'(lambda (key val)
					  (declare (ignore key))
					  (princ " ") (gen-write-1 val))
				    (set::set-table node))
			   (princ "}"))))
		   (vector
		    (cond ((eq node :array-undefined)
			   (princ "#]array "))
			  (t
			   (gen-write-vector node nil)))))))
	(let ((info (gethash node *current-shared* unshared-pointer)))
	    (declare (fixnum info))
	    (cond ((= info unshared-pointer)
		   (print node))
		  ((minusp info)
		   (princ "#]shared ")
		   (princ (- info)) (princ " "))
		  ((plusp info)
		   (unless (and (newgen-vector-p node)
				(tabulated-p (node-domain node)))
		       (princ "[")
		       (princ (- (setf (gethash node *current-shared*) 
				       (- info))))
		       (princ " "))
		   (print node))
		  (t (print node))))))

(defun gen-write-vector (v is-tabulated-def)
    "Prints a vector V on the *STANDARD-OUTPUT*. The
trick is to check whether the first elt is a symbol, in which case this is
an object (and not a vanilla vector). In this case we print the type first."
    (declare (vector v))
    (let ((gen-object (newgen-vector-p v))
	  (vector-length (array-total-size v)))
	(cond (gen-object
	       (let ((domain (node-domain v)))
		   (when (tabulated-p domain)
		       (princ (if is-tabulated-def "#]def " "#]ref "))
		       (princ (svref *gen-tabulated-index* domain))
		       (princ " ")
		       (princ #\")
		       (princ domain)
		       (princ separation-char)
		       (princ (node-hash v))
		       (princ "\" ")
		       (unless is-tabulated-def
			   (return-from gen-write-vector)))
		   (princ "#(#]type ")
		   (princ domain) (princ " ")
		   (do ((i 1 (1+ i)))
		       ((= i vector-length))
		       (declare (fixnum i))
		       (gen-write-1 (aref v i)))))
	      (t
	       (princ "#(")
	       (do ((i 0 (1+ i)))
		   ((= i vector-length))
		   (declare (fixnum i))
		   (gen-write-1 (aref v i)))))
	(format t ")~@[~%~]" gen-object)))

(defun get-shared (node)
    "Computes a hash table of shared nodes from NODE"
    (setf *current-shared* (make-hash-table))
    (get-shared-1! node)
    (values))

(defun get-shared-1! (node)
  "Nothing is done for a base-type NODE.  If the NODE is already in the 
*CURRENT-SHARED* table as an UNSHARED-POINTER, its value is updated 
to the incremented *HASH-COUNTER*. If its value is already a number, nothing 
is done, else GET-SHARED-1 structurally recurses. List, vectors and sets
aren't managed."
  (etypecase node
    ((or symbol base-type))
    (cons 
     (get-shared-1! (car node))
     (get-shared-1! (cdr node)))
    (set
     (maphash #'(lambda (key val)
		  (declare (ignore key))
		  (get-shared-1! val))
	      (set::set-table node)))
    (vector 
     (if (newgen-vector-p node)
	 (let ((info (gethash node *current-shared* '())))
	   (when info
	     (when (eql info unshared-pointer)
	       (setf (gethash node *current-shared*) (incf *hash-counter*)))
	     (return-from get-shared-1!))
	   (setf (gethash node *current-shared*) unshared-pointer)
	   (unless (tabulated-p (node-domain node))
	     (map nil #'get-shared-1! node)))
       (map nil #'get-shared-1! node)))))

(defun gen-read (&optional (stream *standard-input*))
  "Returns an sexp from the optional
STREAM. The number of shared pointers is followed by the shared sexp. The
sharing is reconstructed, using [ as a macro character which is used to
record the up-to-be-read node.  Shared nodes are recorded in the SHARED
array, which binds shared pointers numbers to their pointer value; they are
formatted as a pair with SHARED-POINTER and the pointer number in the
SHARED array"
  (let ((old-bracket-macro (get-macro-character #\[))
	(old-dispatch-bracket-macro (get-dispatch-macro-character #\# #\]))
	(old-left-brace-macro (get-macro-character #\{))
	(old-right-brace-macro (get-macro-character #\}))
	(*package* (find-package :newgen))
	(shared-nodes (lisp:make-array (read stream) :initial-element '())))
      (declare (type (simple-array t (*)) shared-nodes))
      (unwind-protect
	      (progn
		  (set-macro-character
		   #\[
		   #'(lambda (stream char)
			 (declare (ignore char))
			 (let ((shared-index (1- (the fixnum (read stream)))))
			     (setf (svref shared-nodes shared-index)
				   (lisp:make-array 
				    `(,min-tabulated-chunk-size)
				    :adjustable t))
			     (let ((chunk (read stream)))
				 (adjust-array (svref shared-nodes 
						      shared-index)
					       (array-dimensions chunk)
					       :initial-contents chunk)
				 (svref shared-nodes shared-index)))))
		  (set-dispatch-macro-character 
		   #\# #\] 
		   #'(lambda (stream subchar number)
			 (declare (ignore subchar number))
			 (ecase (read stream)
				(null :undefined)
				(set :set-undefined)
				(list :list-undefined)
				(array :array-undefined)
				(def (gen-read-def stream))
				(ref (gen-read-ref stream))
				(type `(:newgen ,(read stream)))
				(shared
				 (let ((which 
					(svref shared-nodes 
					       (1- (the fixnum 
							(read stream))))))
				     (assert which)
				     which))
				(external
				 (let ((which (read stream)))
				     `(:external 
				       ,which
				       ,(funcall (svref *external-read* which)
						 #'(lambda ()
						       (read-char stream))))))
				(bool `(:bool ,(read stream)))
				(unit :unit))))
		  (set-macro-character 
		   #\{
		   #'(lambda (stream char)
			 (declare (ignore char))
			 (let* ((elts (read-delimited-list #\} stream t))
				(set (set::set-make (car elts))))
			     (mapc #'(lambda (elt)
					 (setf (gethash elt 
							(set::set-table set))
					       elt))
				   (cdr elts))
			     set)))
		  (set-macro-character #\} (get-macro-character #\) nil))
		  (read stream))
	  (set-macro-character #\[ old-bracket-macro t)
	  (set-macro-character #\{ old-left-brace-macro t)
	  (set-macro-character #\} old-right-brace-macro t)
	  (set-dispatch-macro-character #\# #\] old-dispatch-bracket-macro))))

(defun gen-read-def (stream)
    "Returns the #]def'ed expression from the STREAM"
    (let* ((index (read stream))
	   (key (read stream))
	   (chunk (read stream))
	   (sep-position (position separation-char key))
	   (domain (parse-integer key :start 0 :end sep-position)))
	(enter-tabulated-def index
			     domain
			     (subseq key (1+ sep-position))
			     chunk
			     :allow-ref-p t)))

(defun enter-tabulated-def (index domain key chunk &key allow-ref-p)
    "Enters the CHUNK in the tabulation table of DOMAIN (with INDEX). The KEY
doesn't include yet the domain. If ALLOW-REF-P, then previous refs are 
allowed"
    (assert (svref *gen-tabulated* index))
    (assert *gen-tabulated-names*)
    (let* ((full-key (format nil "~D~C~A" domain separation-char key))
	   (where (gethash full-key *gen-tabulated-names* 0)))
	(declare (fixnum where))
	(cond ((zerop where)
	       (setf where (find-free-tabulated domain))
	       (setf (gethash full-key *gen-tabulated-names*) where)
	       (setf (svref (svref *gen-tabulated* index) where) chunk)
	       (setf (node-alloc chunk) where))
	      ((and allow-ref-p
		    (minusp where))
	       (setf (node-alloc chunk) (- where))
	       (adjust-array (svref (svref *gen-tabulated* index) (- where))
			     (array-dimensions chunk)
			     :initial-contents chunk)
	       (setf (gethash full-key *gen-tabulated-names*) (- where)))
	      (t 
	       (when *warn-on-tabulated-redefinition*
		   (warn "gen-read-def: ~A redefined, updating" key))
	       (setf (svref (svref *gen-tabulated* index) where) chunk)))
	chunk))

(defun find-free-tabulated (domain)
    "Looks for a free slot in the tabulation table of the domain of INDEX"
    (let* ((first  (svref *gen-tabulated-alloc* domain))
	   (index (svref *gen-tabulated-index* domain))
	   (table (svref *gen-tabulated* index)))
	(assert (not (= first -1)))
	(assert table)
	(do ((i (1+ (if (= first (1- max-tabulated-elements)) 1 first))
		(if (= i (1- max-tabulated-elements)) 1 (1+ i))))
	    ((= i first)
	     (error "~%Too many elements in tabulated domain ~D" domain))
	    (when (null (svref table i))
		(setf (svref *gen-tabulated-alloc* domain) i)
		(return-from find-free-tabulated i)))))

(defun gen-read-ref (stream)
    "Returns the #]ref'ed expression from STREAM"
    (let* ((index (read stream))
	   (key (read stream))
	   (sep-position (position separation-char key))
	   (domain (parse-integer key :start 0 :end sep-position))
	   (where (gethash key *gen-tabulated-names* 0))) 
	(declare (fixnum where))
	(when (zerop where)
	    (setf where (- (find-free-tabulated domain)))
	    (setf (gethash key *gen-tabulated-names*) where)
	    (assert (null (svref (svref *gen-tabulated* index) (- where))))
	    (setf (svref (svref *gen-tabulated* index) (- where))
		  (lisp:make-array 
		   `(,min-tabulated-chunk-size)
		   :adjustable t
		   :initial-contents `((:newgen ,domain)
				       ,(- where)
				       ,(subseq key (1+ sep-position))))))
	(svref (svref *gen-tabulated* index) (abs where))))

(defun gen-write-tabulated (stream domain)
    "Write the tabulation table of DOMAIN on the STREAM"
    (let ((node (svref *gen-tabulated* (svref *gen-tabulated-index* domain)))
	  (*standard-output* stream)
	  (*write-tabulated* t))
	(princ domain) (princ " ")
	(setf *hash-counter* 0)
	(get-shared node)
	(princ *hash-counter*) (princ " ")
	(princ "#(#]type ") (princ tabulated-bp) (princ " #(")
	(map  nil
	      #'(lambda (obj) 
		    (when obj (gen-write-vector obj t)))
	      node)
	(princ "))") (terpri)
	domain))

(defun gen-read-tabulated (stream create-p)
    "Read a tabulation table on STREAM, performing an update if not CREATE-P"
    (let* ((domain (read stream))
	   (index  (svref *gen-tabulated-index* domain)))
	(when create-p
	    (assert (svref *gen-tabulated* index))
	    (gen-free-tabulated domain))
	(gen-read stream)
	(when *warn-on-ref-without-def-in-read-tabulated*
	    (map nil
		 #'(lambda (chunk)
		       (when (and chunk
				  (= (length chunk) min-tabulated-chunk-size))
			   (warn "Undefined reference to ~A~%"
				 (node-hash chunk))))
		 (svref *gen-tabulated* index)))
	domain))

(defun gen-filter-tabulated (filter domain)
    "Returns a list of all the objects in the tabulation table of DOMAIN for
which the application of FILTER returns true."
    (let* ((table (svref *gen-tabulated* (svref *gen-tabulated-index* domain)))
	   (max (length table)))
	(do ((i 0 (+ i 1))
	     (l '()
		(let ((x (svref table i)))
		    (if (and x (funcall filter x)) `(,x ,@l) l))))
	    ((= i max) l))))

(defun gen-find-tabulated (key domain)
    "Returns the object in the tabulation table of DOMAIN with the KEY (or 
nil if not there)"
    (let* ((full-key (format nil "~D~C~A" domain separation-char key))
	   (index (gethash full-key *gen-tabulated-names* nil)))
	(and index
	     (svref (svref *gen-tabulated*
			   (svref *gen-tabulated-index* domain))
		    (abs index)))))

(defun gen-init-external (which read write)
    "Initialize the *EXTERNAL-READ* and *EXTERNAL-WRITE* table for external
type WHICH. READ and WRITE are closures to be called on external values"
    (unless *external-read*
	    (setf *external-read* (lisp:make-array max-externals))
	    (setf *external-write* (lisp:make-array max-externals))
	    (setf *external-cpt* -1))
    (when (>= (incf *external-cpt*) max-externals)
	  (error "Too many external types: ~D" which))
    (when (svref *external-read* which)
	  (warn "Reinitializing external type: ~D" which))
    (setf (svref *external-read* which) read)
    (setf (svref *external-write* which) write))

(defun gen-mapc-tabulated (f domain)
    "Maps the function F on each entity of the tabulated DOMAIN"
    (map nil
	 #'(lambda (obj) 
	       (when obj (funcall f obj)))
	 (svref *gen-tabulated* (svref *gen-tabulated-index* domain))))

(defun gen-free (object)
    "Frees the OBJECT"
    (let ((domain (node-domain object))) 
	(when (tabulated-p domain)
	    (let ((key (format nil "~S~C~A" 
			       domain separation-char (node-hash object))))
		(assert (remhash key *gen-tabulated-names*))
		(setf (svref (svref *gen-tabulated* 
				    (svref *gen-tabulated-index* domain))
			     (abs (node-alloc object)))
		      nil)))))

(defun gen-free-tabulated (domain)
    "Frees the tabulation table of the DOMAIN"
    (assert (tabulated-p domain))
    (let ((index (svref *gen-tabulated-index* domain)))
	(map nil
	     #'(lambda (object)
		   (when object (gen-free object)))
	     (svref *gen-tabulated* index))
	(setf (svref *gen-tabulated-alloc* domain) 1)
	(setf (svref *gen-tabulated* index)
	      (lisp:make-array max-tabulated-elements :initial-element '()))
	domain))

;;; The GEN-SWITCH macro to dispatch on or-tags.

(defconstant default-tag :DEFAULT)

;; These macros have to be used with pure expressions only.

(defmacro tag (exp) `(aref ,exp (if (= (length ,exp) 4) 2 1)))
(defmacro val (exp) `(aref ,exp (if (= (length ,exp) 4) 3 2)))

(defmacro gen-switch (exp &rest clauses)
    "To dispatch on Newgen tags:

(gen-switch <exp>
	    (<tag> <exp>*)
	    ((<tag> <var>) <exp>*)...)

where <tag> is a symbol (defined by Newgen) or :DEFAULT. Inside of the clauses
bodies, <var> is bound to the value of the or-object"
    (let ((exp-or (gensym "switch-or-"))
	  (exp-tag (gensym "switch-tag-")))
	`(let* ((,exp-or ,exp)
		(,exp-tag (tag ,exp-or)))
	     ,(reduce 
	       #'(lambda (clause rest)
		     (cond ((eq (car clause) default-tag)
			    `(progn ,@(cdr clause)))
			   ((consp (car clause))
			    (unless (and (consp (cdar clause))
					 (null (cddar clause)))
				(error "~%Incorrect SWITCH clause ~S" clause))
			    `(if (= ,(caar clause) ,exp-tag)
				     (let ((,(cadar clause)
					    (val ,exp-or)))
					 ,@(cdr clause))
				 ,rest))
			   (t `(if (= ,(car clause) ,exp-tag)
				       (progn ,@(cdr clause))
				   ,rest))))
	       clauses
	       :initial-value 
	       `(error "~%Unmatched SWITCH: got ~S instead of ones of ~S" 
		       (cadr (assoc ,exp-tag newgen::*tag-names*))
		       ',(mapcar #'(lambda (c)
				       (if (consp (car c))
					       (caar c)
					   (car c)))
				 clauses))
	       :from-end t))))

;;; The GEN-RECURSE macro to recurse on Newgen objects.

(defmacro gen-recurse (object &rest clauses)
    "To recurse on Newgen data structures:

(gen-recurse <exp>
 ((<type> <member1> ... <membern>) <body>)...)

Inside <body>, <type> is bound to the current Newgen object and <memberi> to
to i-th recursive call of gen-recurse of <type>-<memberi> on the current
Newgen object. If <member1> is \"tag\", then the object is an or-object and
its value is bound to \"tag\". If the member is a list, a vector or a set, then
a list, a vector or a set of results is returned. Note that the package
in which the data structures are defined has to be USED"
    (let ((loop (gensym))
	  (obj (gensym))
          (set (gensym))
          (elt (gensym)))
	`(labels ((,loop (,obj)
			 (etypecase
			  ,obj
			  (symbol ,obj)
			  (base-type ,obj)
			  (cons (mapcar #',loop ,obj))
                          (set (let ((,set (set-make (set::set-type ,obj))))
                                  (set-map #'(lambda (,elt)
                                                (set-add-element ,set ,set 
                                                                 (,loop ,elt)))
                                           ,obj)
                                  ,set))
			  (vector
			   (if (newgen-vector-p ,obj)
				   (cond ,@(recurse-clauses obj clauses loop))
			       (map 'vector #',loop ,obj))))))
	     (,loop ,object))))

(defun recurse-clauses (obj clauses loop)
    `(,@(mapcar #'(lambda (clause)
		      (let* ((pattern (car clause))
			     (head (car pattern)))
			  `((= (node-domain ,obj) 
			       ,(intern (concatenate 'string
						     (symbol-name head)
						     "-DOMAIN")))
			    (let ((,head ,obj)
				  ,@(recurse-members head obj (cdr pattern) loop))
				,@(if (cdr clause)
					  (cdr clause)
				      `((declare (ignore ,@pattern))))))))
							  
		clauses)
	(t (error "Unknown Newgen object of domain ~D"
		  (node-domain ,obj)))))

(defun recurse-members (head obj members loop)
    (mapcar #'(lambda (member)
		  (if (string-equal (symbol-name member) "tag")
			  `(,member (,loop (newgen::or-val ,obj)))
		      `(,member
			(,loop (,(intern 
				  (concatenate 'string
					       (symbol-name head)
					       "-"
					       (symbol-name member)))
				,obj)))))
	    members))
