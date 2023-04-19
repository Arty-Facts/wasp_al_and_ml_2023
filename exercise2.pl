:- use_module(library(lists)).

%ranks([2, 3, 4, 5, 6, 7, 8, 9, 10, jack, queen, king, ace]).
ranks([9, 10, jack, queen, king, ace]).
%suits([hearts, diamonds, clubs, spades]).

%% Define the cards
card(Value, Rank) :-
    ranks(RankList),
    member(Rank, RankList),
    %suits(SuitList),
    %member(Suit, SuitList), 
    nth0(Value, RankList, Rank).

% this is the order from the weakest to the strongest hand
hand_ranks([
    high_card(_), 
    one_pair(_), 
    two_pair(_), 
    three_of_a_kind(_), 
    straight(_), 
    full_house(_), 
    four_of_a_kind(_)
    ]).

%% Define the poker hands
hand(Cards, Hand) :-
    merge_sort(Cards, Sorted, card_comp), % Needs to be sorted for pattern matching to work
    poker_hands(Sorted, Hand).

card_comp(C1, C2):-
     card(V1, C1), card(V2, C2), V1=<V2.
     
%A straight is a hand that contains five cards of sequential rank.
poker_hands(Cards, straight([R1, R2, R3, R4, R5])) :-
    single_card(Cards, R1, V1), 
    single_card(Cards, R2, V2),
    V1 + 1 =:= V2, %early terminate if possible to improve performance
    single_card(Cards, R3, V3), 
    V2 + 1 =:= V3,
    single_card(Cards, R4, V4), 
    V3 + 1 =:= V4,
    single_card(Cards, R5, V5),
    V4 + 1 =:= V5.

single_card(Cards, Rank, Value) :-
    single(Cards, Rank), 
    card(Value, Rank).

%query(hand([9, 10, jack, queen, king, ace, jack], _)).

% A full house is a hand that contains three cards of one rank and two cards of 
% another rank    
poker_hands(Cards, full_house([Rank1, Rank2])) :-
    poker_hands(Cards, three_of_a_kind([Rank1])), 
    poker_hands(Cards, one_pair([Rank2])), 
    Rank1 \= Rank2.

poker_hands(Cards, four_of_a_kind([Rank])) :-
    quadruplet(Cards, Rank).

poker_hands(Cards, three_of_a_kind([Rank])) :-
    triplet(Cards, Rank).

poker_hands(Cards, two_pair([Rank1, Rank2])) :-
    poker_hands(Cards, one_pair([Rank1])),
    poker_hands(Cards, one_pair([Rank2])),
    Rank1 \= Rank2,
    card(Value1, Rank1),
    card(Value2, Rank2),
    Value1 > Value2. % make sure that order does not mature
    
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
merge_sort([], [], _).
merge_sort([X], [X], _).
merge_sort(List, Sorted, Comp) :-
    length(List, Len),
    Len > 1,
    split(List, Left, Right),
    merge_sort(Left, SortedLeft, Comp),
    merge_sort(Right, SortedRight, Comp),
    merge(SortedLeft, SortedRight, Sorted, Comp),
    append(Sorted, [], Sorted).
    
split(List, Left, Right) :-
    length(List, Len),
    HalfLen is Len // 2,
    length(Left, HalfLen),
    append(Left, Right, List).
    
merge([], List, List, _).
merge(List, [], List, _).
merge([H1|T1], [H2|T2], [H1|T], Comp) :-
    call(Comp, H1, H2),
    merge(T1, [H2|T2], T, Comp).
merge([H1|T1], [H2|T2], [H2|T], Comp) :-
    \+call(Comp, H1, H2),
    merge([H1|T1], T2, T, Comp).

%query(merge_sort([jack, king, jack, jack, queen, ace, jack,jack, ace, 10], _, card_comp)).
%query(hand([jack, king, jack, jack, queen, ace, jack,jack, ace, 10], _)).
    
    
best_hand_rank(Cards, Best) :-
    ordered_hand(Cards, [Best|_]).
    
ordered_hand(Cards, Sorted) :-
    findall(Hand, hand(Cards, Hand), Hands), % this is a performance bottleneck 
    remove_duplicates(Hands, SetOfHands), 
    merge_sort(SetOfHands, Sorted, hand_ranks_comp). 

remove_duplicates([],[]).
remove_duplicates([H | T], List) :-    
     member(H, T),
     remove_duplicates( T, List).

remove_duplicates([H | T], [H | List]) :- 
      \+member(H, T),
      remove_duplicates(T, List).

hand_ranks_comp(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 > Value2.
    
hand_ranks_comp(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 =:= Value2, 
    compare_in_rank(Hand1, Hand2).

compare_in_rank(HandRank1, HandRank2) :-  
       HandRank1 =.. [_|Ranks1], % decompose ex full_house([king, jack]) to [full_house, [king, jack]]
       HandRank2 =.. [_|Ranks2],
       compare_cards(Ranks1, Ranks2).

compare_cards([Rank1],[Rank2]) :-
       card(Value1, Rank1), 
       card(Value2, Rank2),
       Value1 > Value2.
compare_cards([Rank1|Tail1],[Rank2|Tail1]) :-
       card(Value1, Rank1), 
       card(Value2, Rank2),
       Value1 =:= Value2,
       compare_cards(Tail1,Tail2).
        
%query(ordered_hand([9, 10, jack, queen, king, ace], _)).
    
hand_value(Hand, Value):-
    hand_ranks(HandRankList),
    member(Hand, HandRankList),
    nth0(Value, HandRankList, Hand).

better(BetterHand,WorseHand) :-
    best_hand_rank(BetterHand, HandRank1), 
    best_hand_rank(WorseHand, HandRank2),
    hand_value(HandRank1, Value1),
    hand_value(HandRank2, Value2),
    Value1 > Value2.

%query(best_hand_rank([jack, king, jack], _)).
%query(best_hand_rank([jack, king, queen, ace, 10], _)).
% query(better([jack, king, jack, king, king], [jack, jack, jack])).
% query(better([jack, king, jack, king, king], [jack, king, queen, ace, 10])).
% query(better([jack, king, jack, king], [jack, king, queen, ace, 10])).

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
