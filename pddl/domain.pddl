;; Tabletop manipulation with ONE arm and hand (Unitree G1, fixed base).
;; Small, fixed, hand-authored domain: 5 actions.
;;
;; Items are small blocks. Places are spots on the table (t1..t3) or bin slots.
;; Every place holds at most one item; items can also be stacked on other items.
;; "reachable" encodes the arm's comfortable workspace (see Month 1 results):
;; only reachable places can be used. All current places are reachable.
(define (domain tabletop-single-arm)
  (:requirements :strips :typing)
  (:types item place)

  (:predicates
    (at ?i - item ?p - place)      ; item rests on the place
    (on ?i - item ?b - item)       ; item rests on top of another item
    (clear ?i - item)              ; nothing on top of the item (and it is not held)
    (free ?p - place)              ; nothing rests on the place
    (holding ?i - item)            ; the hand holds the item
    (hand-empty)
    (table-spot ?p - place)        ; a spot on the table surface (bin slots are not)
    (reachable ?p - place))

  ;; Grasp an item that rests directly on a place.
  (:action pick
    :parameters (?i - item ?p - place)
    :precondition (and (at ?i ?p) (clear ?i) (hand-empty) (reachable ?p))
    :effect (and (holding ?i) (free ?p)
                 (not (at ?i ?p)) (not (clear ?i)) (not (hand-empty))))

  ;; Put the held item down on a free place.
  (:action place
    :parameters (?i - item ?p - place)
    :precondition (and (holding ?i) (free ?p) (reachable ?p))
    :effect (and (at ?i ?p) (clear ?i) (hand-empty)
                 (not (holding ?i)) (not (free ?p))))

  ;; Put the held item on top of another item.
  (:action stack
    :parameters (?i - item ?b - item)
    :precondition (and (holding ?i) (clear ?b))
    :effect (and (on ?i ?b) (clear ?i) (hand-empty)
                 (not (holding ?i)) (not (clear ?b))))

  ;; Lift the top item off a stack.
  (:action unstack
    :parameters (?i - item ?b - item)
    :precondition (and (on ?i ?b) (clear ?i) (hand-empty))
    :effect (and (holding ?i) (clear ?b)
                 (not (on ?i ?b)) (not (clear ?i)) (not (hand-empty))))

  ;; Slide an item along the table to a free table spot without grasping it.
  ;; (Pushing into a bin is not possible, so bin slots are not table spots.)
  (:action push
    :parameters (?i - item ?p - place ?q - place)
    :precondition (and (at ?i ?p) (clear ?i) (hand-empty)
                       (free ?q) (table-spot ?p) (table-spot ?q)
                       (reachable ?p) (reachable ?q))
    :effect (and (at ?i ?q) (free ?p)
                 (not (at ?i ?p)) (not (free ?q))))
)
