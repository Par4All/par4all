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
;;;; set.cl
;;;;
;;;; The CommonLISP library of set for Newgen
;;;; 
;;;; First version - 20/09/89: Pierre Jouvelot (pj) -- NOT TESTED
;;;;
;;;; A module that wants to use this package has to shadow SET-DIFFERENCE. 

(provide :set)

(in-package :set)

;;shadow

(export '(set-string set-int set-pointer
	  set-make set-singleton set-assign set-add-element set-belong-p
	  set-union set-intersection set-difference set-equal set-clear
	  set-map))

;;require

;;use-package

(proclaim '(optimize (speed 3) (safety 0)))

;;; Implementation of Newgen sets.

(defstruct set
    type				;kind of set
    test				;membership function
    table)				;hashtable to store elements

;;; The possible type values have to be consistant with the set.h C
;;; version; see this file.

(defparameter set-string 0)
(defparameter set-int 1)
(defparameter set-pointer 2)

;;; Abstract functions

(defun set-make (type)
    (let ((test  (if (eq type set-string) #'equal #'eql)))
	(make-set :type type 
		  :table (make-hash-table :test test)
		  :test test)))

(defun set-singleton (type val)
    (let ((set (set-make type)))
	(setf (gethash val (set-table set)) val)
	set))

(defun set-assign (s1 s2)
    (unless (eq s1 s2) 
	(clrhash (set-table s1))
	(maphash #'(lambda (key val)
		       (setf (gethash key (set-table s1)) val))
		 (set-table s2)))
    s1)

(defun set-add-element (s1 s2 e)
    (unless (eq s1 s2)
	(clrhash (set-table s1))
	(maphash #'(lambda (key val)
		       (setf (gethash key (set-table s1)) val))
		 (set-table s2)))
    (setf (gethash e (set-table s1)) e)
    s1)

(defun set-belong-p (s e)
    (funcall (set-test s) (gethash e (set-table s)) e))

(defun set-union (s1 s2 s3)
    (set-assign s1 s2)
    (maphash #'(lambda (key val)
		   (setf (gethash key (set-table s1)) val))
	     (set-table s3))
    s1)

(defun set-intersection (s1 s2 s3)
    (cond ((and (not (eq s1 s2)) (not (eq s1 s3)))
	   (clrhash s1)
	   (maphash #'(lambda (key val)
			  (when (set-belong-p (set-table s2) key)
			      (setf (gethash key (set-table s1)) val)))
		    (set-table s3)))
	  (t (let ((tmp (set-make (set-type s1))))
		 (maphash #'(lambda (key val)
				(when (set-belong-p (set-table s1) key)
				    (setf (gethash key (set-table tmp)) val)))
			  (set-table (if (eq s1 s2) s3 s2)))
		 (set-assign s1 tmp))))
    s1)

(defun set-difference (s1 s2 s3)
    (set-assign s1 s2)
    (maphash #'(lambda (key val)
		   (declare (ignore val))
		   (remhash (set-table s1) key))
	     (set-table s3))
    s1)

(defun set-equal (s1 s2)
    (maphash #'(lambda (key val)
		   (declare (ignore val))
		   (unless (set-belong-p s2 key)
		       (return-from set-equal nil)))
	     (set-table s1))	
    (maphash #'(lambda (key val)
		   (declare (ignore val))
		   (unless (set-belong-p s2 key)
		       (return-from set-equal nil)))
	     (set-table s2))	
    t)

(defun set-clear (s)
    (clrhash (set-table s)))

(defun set-map (f s)
    (maphash #'(lambda (key val)
		   (declare (ignore key))
		   (funcall f val))
	     (set-table s)))
