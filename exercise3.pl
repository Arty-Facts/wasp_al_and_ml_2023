:-use_module(library(apply)).
:-use_module(library(lists)).

%%%% Insert and modify the ProbLog code from Exercise 1 here

% Encode the different cards as follows: card(Player,N,Rank)
% This means that the N-th card drawn by Player is of the given Rank.

% Cheater -> Draw2, Cheater -> Coin,
% Draw1 -> Highest
% Draw2 -> Highest
% Highest -> Coin
% Highest -> Winner
% Coin -> Winner

%%%%%%%%%
% Facts %
%%%%%%%%%

% this is the order for card ranks from the weakest to the strongest hand
ranks([jack, queen, king, ace]).

cards_in_hand([1, 2, 3, 4]).

players([player1, player2]).
% this is the order for hand ranks from the weakest to the strongest hand
hand_ranks([highcard, onepair, twopair, threeofakind, straight, fullhouse, fourofakind]).

%the cheeper has one in five probability of cheating 
1/5::cheater.

%% Define the cards
card(Player, N, Rank) :-
    player(Player),
    draw(Player, N, Rank).

player(Player) :-
    players(PlayerList),
    member(Player, PlayerList).

1/NbCards::draw(player1, N, Rank):-
    ranks(RankList),
    member(Rank, RankList), 
    length(RankList, NbCards),
    nb_cards(N).

1/NbCards::draw(player2, N, Rank):-
    valid_card(Rank),
    ranks(RankList),
    length(RankList, NbCards), 
    nb_cards(N), 
    \+ cheater.

2/NbCards::draw(player2, N, ace):-
    ranks(RankList),
    length(RankList, NbCards),
    nb_cards(N),
    cheater.

1/NbCards::draw(player2, N, Rank) :-
    valid_card(Rank),
    ranks(RankList),
    Rank \= jack, % jack and ace are accounted for in statement above  
    Rank \= ace, 
    length(RankList, NbCards),
    nb_cards(N),
    cheater.

nb_cards(N):-
    cards_in_hand(Nth), 
    member(N, Nth).

card_value(Value, Rank):-
    valid_card(Rank),
    ranks(RankList),
    nth0(Value, RankList, Rank).

valid_card(Rank):-
    ranks(RankList),
    member(Rank, RankList).

% evidence(cheater).
% evidence(card(player2, 1, jack)).
% query(card(player1, _, _)).
% query(card(player2, _, _)).

%% Define the poker hands
     
% A straight is a hand that contains five cards of sequential rank.
% but according to the test its only four
% hand(Cards, straight(R1, R2, R3, R4)) :-
hand(Cards, straight) :-
    select(R1, Cards, Rest1), card_value(V1, R1),
    select(R2, Rest1, Rest2), card_value(V2, R2),
    V1 + 1 =:= V2, %early terminate if possible to improve performance
    select(R3, Rest2, Rest3), card_value(V3, R3),
    V2 + 1 =:= V3,
    member(R4, Rest3), card_value(V4, R4),
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
       card_value(Value1, Rank1), 
       card_value(Value2, Rank2),
       Value1 > Value2.

compare_in_rank([Rank1| Tail1], [Rank2| Tail2]) :-
       card_value(Value1, Rank1), 
       card_value(Value2, Rank2),
       Value1 =:= Value2,
       compare_in_rank(Tail1, Tail2).

decompose(Hand, HandRank, Ranks) :-  
       Hand =.. [HandRank| Ranks]. % decompose ex fullhouse([king, jack]) to [fullhouse, [king, jack]]

% query(decompose(fullhouse(king,queen),_, _)).
% query(compare_in_rank([king,queen],[king,jack])).
% query(better(fullhouse(king,queen),fullhouse(king,jack))).
% query(best_hand([jack, king, jack, ace],_)).


%%%% Provided code

% The following predicate will sample a Hand as a list of ranks for the given player.
% It expects that there are probabilistic facts of the form card(Player,N,Rank) as specified above

draw_hand(Player,Hand) :- maplist(card(Player),[1, 2, 3],Hand).

query(draw_hand(player1,_)).

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



%%%% What’s the probability that player2 draws the hand [ace, king, queen, ace].
%%%% Your answer : 



%%%%  Given that player2 draws the hand [ace, king, queen, ace], and that the coin comes up tails, what is the posterior belief that your opponent is cheating?
%%%% Your answer : 



%%%%  What is the prior probability that player 1 wins?1 Why does this query take so long to answer? What is the probability that player 1 wins, given that you know that player 2 is a cheater?
%%%% Your answer : 
