;; Task 5: swap the positions of red and blue (t3 is free to use as a parking spot).
(define (problem p05-swap-positions)
  (:domain tabletop-single-arm)
  (:objects red blue - item  t1 t2 t3 bin1 bin2 - location)
  (:init (hand-empty) (table-spot t1) (table-spot t2) (table-spot t3) (reachable t1) (reachable t2) (reachable t3) (reachable bin1) (reachable bin2)
         (at red t1) (clear red) (at blue t2) (clear blue)
         (free t3) (free bin1) (free bin2))
  (:goal (and (at red t2) (at blue t1))))
