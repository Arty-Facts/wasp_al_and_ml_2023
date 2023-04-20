:- use_module(library(lists)).

% this is the order for card ranks from the weakest to the strongest hand
ranks([jack, queen, king, ace]).
% this is the order for hand ranks from the weakest to the strongest hand
hand_ranks([highcard, onepair, twopair, threeofakind, straight, fullhouse, fourofakind]).


%% Define the cards
card(Value, Rank) :-
    ranks(RankList),
    member(Rank, RankList),
    nth0(Value, RankList, Rank).

valid_card(Rank):-
    ranks(RankList),
    member(Rank, RankList).

%% Define the poker hands
     
% A straight is a hand that contains five cards of sequential rank.
% but according to the test its only four
% hand(Cards, straight(R1, R2, R3, R4)) :-
hand(Cards, straight) :-
    select(R1, Cards, Rest1), card(V1, R1),
    select(R2, Rest1, Rest2), card(V2, R2),
    V1 + 1 =:= V2, %early terminate if possible to improve performance
    select(R3, Rest2, Rest3), card(V3, R3),
    V2 + 1 =:= V3,
    member(R4, Rest3), card(V4, R4),
    V3 + 1 =:= V4.

% A full house is a hand that contains three cards of one rank and two cards of 
% another rank    
hand(Cards, fullhouse(Rank1, Rank2)) :-
    hand(Cards, threeofakind(Rank1)), 
    hand(Cards, onepair(Rank2)), 
    Rank1 \= Rank2, 
    valid_card(Rank1),valid_card(Ran2).

hand(Cards, fourofakind(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    select(Rank, Rest2, Rest3),
    member(Rank, Rest3).

hand(Cards, threeofakind(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    member(Rank, Rest2).

hand(Cards, twopair(Rank1, Rank2)) :-
    hand(Cards, onepair(Rank1)),
    hand(Cards, onepair(Rank2)),
    Rank1 \= Rank2, 
    valid_card(Rank1),valid_card(Ran2).
    
hand(Cards, onepair(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    member(Rank, Rest1).
    
hand(Cards, highcard(Rank)) :-
    valid_card(Rank),
    member(Rank, Cards).

%query(hand([poop, poop, jack, king, jack, jack, queen, ace, jack,jack, ace, queen, queen, king, ace], _)).

better(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 > Value2.
    
better(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 =:= Value2, 
    decompose(Hand1, _, Ranks1),
    decompose(Hand2, _,  Ranks2),
    compare_in_rank(Ranks1, Ranks2).

hand_value(Hand, Value):-
    hand_ranks(HandRankList),
    decompose(Hand, HandRank, _),
    member(HandRank, HandRankList),
    nth0(Value, HandRankList, HandRank).

compare_in_rank([Rank1| _], [Rank2| _]) :-
       card(Value1, Rank1), 
       card(Value2, Rank2),
       Value1 > Value2.

compare_in_rank([Rank1| Tail1], [Rank2| Tail2]) :-
       card(Value1, Rank1), 
       card(Value2, Rank2),
       Value1 =:= Value2,
       compare_in_rank(Tail1, Tail2).

decompose(Hand, HandRank, Ranks) :-  
       Hand =.. [HandRank| Ranks]. % decompose ex fullhouse([king, jack]) to [fullhouse, [king, jack]]

% query(decompose(fullhouse(king,queen),_, _)).
% query(compare_in_rank([king,queen],[king,jack])).
% query(better(fullhouse(king,queen),fullhouse(king,jack))).
% query(best_hand([jack, king, jack, ace],_)).

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

% query(best_hand([jack ,king, ace], _)).
% query(best_hand([jack ,jack, ace], _)).
% query(best_hand([jack ,king, jack, jack], _)).
% query(best_hand([jack ,jack, jack, jack], _)).
% query(best_hand([jack ,king, king, jack], _)).
% query(best_hand([jack ,king, king, jack, jack], _)).
% query(best_hand([queen, ace, king, jack], _)).

% query(game_outcome([jack, jack], [king], _)).
% query(game_outcome([jack, jack], [king, king], _)).
% query(game_outcome([jack, king, ace], [king, queen, jack], _)).
% query(game_outcome([jack, queen, king, queen, ace], [king,queen, queen ,jack ,jack], _)).
% query(game_outcome([jack, jack, jack, jack], [king, queen, queen ,queen ,queen], _)).
% query(game_outcome([jack, king, king, king, jack], [king,king, queen ,king ,queen], _)).
% query(game_outcome([jack, king, king, queen, queen], [king,queen, queen ,king ,ace], _)).