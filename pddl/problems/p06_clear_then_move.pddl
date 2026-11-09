;; Task 6: blue sits on red. Red must end in the bin and blue on the table at t2.
;; Blue has to be lifted off first, so the order of steps matters.
(define (problem p06-clear-then-move)
  (:domain tabletop-single-arm)
  (:objects red blue - item  t1 t2 t3 bin1 bin2 - location)
  (:init (hand-empty) (table-spot t1) (table-spot t2) (table-spot t3) (reachable t1) (reachable t2) (reachable t3) (reachable bin1) (reachable bin2)
         (at red t1) (on blue red) (clear blue)
         (free t2) (free t3) (free bin1) (free bin2))
  (:goal (and (at red bin1) (at blue t2))))
