:- use_module(library(lists)).

%ranks([2, 3, 4, 5, 6, 7, 8, 9, 10, jack, queen, king, ace]).
ranks([poop, jack, queen, king, ace]).
%suits([hearts, diamonds, clubs, spades]).

%% Define the cards
card(Value, Rank) :-
    ranks(RankList),
    member(Rank, RankList),
    %suits(SuitList),
    %member(Suit, SuitList), 
    nth0(Value, RankList, Rank).

%% Define the poker hands
hand(Cards, Hand) :-
    quick_sort(Cards, Sorted, card_comp), % Needs to be sorted for pattern matching to work
    poker_hands(Sorted, Hand).


card_comp(C1, C2):-
     card(V1, C1), card(V2, C2), V1=<V2.
     
%A straight is a hand that contains five cards of sequential rank.
poker_hands(Cards, straight(R1, R2, R3, R4, R5)) :- 
    single_card(Cards, R1, V1), 
    single_card(Cards, R2, V2), 
    single_card(Cards, R3, V3), 
    single_card(Cards, R4, V4), 
    single_card(Cards, R5, V5),
    V1<V2, V2<V3, V3<V4, V4<V5, Diff is V5-V1, Diff == 4.

single_card(Cards, Rank, Value) :-
    single(Cards, [Rank]), 
    card(Value, Rank).

% A full house is a hand that contains three cards of one rank and two cards of 
% another rank    
poker_hands(Cards, full_house([Rank1, Rank2])) :-
    poker_hands(Cards, three_of_a_kind([Rank1])), 
    poker_hands(Cards, one_pair([Rank2])), 
    Rank1 \== Rank2.

poker_hands(Cards, four_of_a_kind([Rank])) :-
    quadruplet(Cards, Rank).

poker_hands(Cards, three_of_a_kind([Rank])) :-
    triplet(Cards, Rank).

poker_hands(Cards, two_pair([Rank1, Rank2])) :-
    poker_hands(Cards, one_pair([Rank1])),
    poker_hands(Cards, one_pair([Rank2])),
    card(Value1, Rank1),
    card(Value2, Rank2),
    Value1 > Value2, % make sure that order does not mature
    Rank1 \== Rank2.
    
poker_hands(Cards, one_pair([Rank])) :-
    tuplet(Cards, Rank).
    
poker_hands(Cards, high_card([Rank])) :-
    single(Cards, Rank).

% Use pattern matching to check if the first N elements in the list are the 
% same, and if they are, it returns a Rank term. If they are not the same, 
% it removes the first element of the list and recursively calls itself with the 
% remaining list until it termination.
quadruplet([Rank, Rank, Rank, Rank | _], Rank).
quadruplet([_ | Tail], Rank) :- quadruplet(Tail, Rank).

triplet([Rank, Rank, Rank | _], Rank).
triplet([_ | Tail], Rank) :- triplet(Tail, Rank).

tuplet([Rank, Rank | _], Rank).
tuplet([_ | Tail], Rank) :- tuplet(Tail, Rank).

single([Rank | _], Rank).
single([_ | Tail], Rank) :- single(Tail, Rank).

% stole from from http://kti.mff.cuni.cz/~bartak/prolog/sorting.html and adapted 
% to cards since built i sort removes duplicates 
quick_sort(List,Sorted,Comp):- q_sort(List,[],Sorted,Comp).

q_sort([],Acc,Acc, _).
q_sort([H|T],Acc,Sorted, Comp):- pivoting(H,T,L1,L2, Comp), 
    q_sort(L1,Acc,Sorted1, Comp),q_sort(L2,[H|Sorted1],Sorted, Comp).
   
pivoting(H,[],[],[],_).
pivoting(H,[X|T],[X|L],G,Comp):- call(Comp, H, X), pivoting(H,T,L,G, Comp).
pivoting(H,[X|T],L,[X|G],Comp):- \+call(Comp, H, X), pivoting(H,T,L,G, Comp).

     
%query(quick_sort([jack, king, jack, jack, queen, ace, jack,jack, ace, poop], _, card_comp)).
%query(hand([jack, king, jack, jack, queen, ace, jack,jack, ace, 10], _)).

hand_ranks([
    high_card(_), 
    one_pair(_), 
    two_pair(_, _), 
    three_of_a_kind(_), 
    straight(_,_,_,_,_), 
    full_house(_, _), 
    four_of_a_kind(_)
    ]).
    
    
best_hand_rank(Cards, Best) :-
    findall(Hand, hand(Cards, Hand), Hands), 
    quick_sort(Hands, [Best|_], hand_ranks_comp).

hand_ranks_comp(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 > Value2.
    
hand_ranks_comp(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 == Value2, 
    compare_in_rank(Hand1, Hand2).

compare_in_rank(high_card(R1), high_card(R2)) :-   . 
compare_in_rank(one_pair(_), one_pair(_)) :-  .
compare_in_rank(two_pair(_, _), two_pair(_, _)) :-   .
compare_in_rank(three_of_a_kind(_), three_of_a_kind(_)) :-  .
compare_in_rank(straight(_,_,_,_,_), straight(_,_,_,_,_)) :-  . 
compare_in_rank(full_house(_, _), full_house(_, _)) :-   . 
compare_in_rank(four_of_a_kind(_)) :- .
    
    
hand_value(Hand, Value):-
    hand_ranks(HandRankList),
    member(Hand, HandRankList),
    nth0(Value, HandRankList, Hand).

better(BetterHand,WorseHand, HandValue1, HandValue2) :-
    best_hand_rank(BetterHand, HandRank1), 
    best_hand_rank(WorseHand, HandRank2),
    HandRank1 > HandRank2.

%better(BetterHand,WorseHand) :-
%    hand(BetterHand, one_pair(R1)), hand(WorseHand, high_card(R2)),
%    card(V1, R1), card(V2, R2), 
%    V1>V2.

query(best_hand_rank([jack, king, jack, ace, ace, jack], _)).

%query(better([jack, king, jack], [jack, jack, jack], _, _)).

%%%% Provided code


game_outcome(Cards1,Cards2,Outcome) :-
    best_hand(Cards1,Hand1),
    best_hand(Cards2,Hand2),
    outcome(Hand1,Hand2,Outcome).

outcome(Hand1,Hand2,player1) :- better(Hand1,Hand2).
outcome(Hand1,Hand2,player2) :- better(Hand2,Hand1).
outcome(Hand1,Hand2,tie) :- \+better(Hand1,Hand2), \+better(Hand2,Hand1).

best_hand(Cards,Hand) :-
    hand(Cards,Hand),
    \+ (hand(Cards,Hand2), better(Hand2,Hand)).
