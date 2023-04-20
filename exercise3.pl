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
% there are two decks  
1/4::deck1(jack);
1/4::deck1(queen);
1/4::deck1(king);
1/4::deck1(ace).

1/4::deck2(jack);
1/4::deck2(queen);
1/4::deck2(king);
1/4::deck2(ace).

% cards have values

% this is the order for card ranks from the weakest to the strongest hand
1/4::ranks([jack, queen, king, ace]).

cards_in_hand([1, 2, 3, 4]).

players([player1, player2]).
% this is the order for hand ranks from the weakest to the strongest hand
hand_ranks([highcard, onepair, twopair, threeofakind, straight, fullhouse, fourofakind]).


%% Define the cards
card(Player, N, Rank) :-
    players(PlayerList),
    member(Player, PlayerList),
    cards_in_hand(Nth),
    member(N, Nth),
    ranks(RankList),
    member(Rank, RankList),
    nth0(Value, RankList, Rank).

card_value(Rank, Value):-
    ranks(RankList),
    member(Rank, RankList),
    nth0(Value, RankList, Rank).


%the cheeper has one in five probability of cheating 
1/5::cheater.

%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
% There exists a draw action that can take a card of any value with 1/4 probability
draw1(X) :- deck1(X).

% if player2 is fair use default probs 
% There exists a draw action that can take a card of any value with 1/4 probability if there is no cheater
draw2(X) :- deck2(X), \+ cheater.

% if player2 cheats replaced all the jack cards with ace 
% There exists a draw action that can take a card of ace with probability of jack
draw2(ace) :- deck2(jack), cheater.

% There exists a draw action that can take a card of any value that is grater then one
draw2(X) :- deck2(X), card(V, X) , V>1, cheater.

% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

% coin is fair if nobody cheats
1/2::coin(heads); 1/2::coin(tails) :- \+ cheater.


%%%% Insert Prolog code from Exercise 2



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
    Rank1 \= Rank2.

hand(Cards, fourofakind(Rank)) :-
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    select(Rank, Rest2, Rest3),
    member(Rank, Rest3).

hand(Cards, threeofakind(Rank)) :-
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    member(Rank, Rest2).

hand(Cards, twopair(Rank1, Rank2)) :-
    hand(Cards, onepair(Rank1)),
    hand(Cards, onepair(Rank2)),
    Rank1 \= Rank2.
    
hand(Cards, onepair(Rank)) :-
    select(Rank, Cards, Rest1),
    member(Rank, Rest1).
    
hand(Cards, highcard(Rank)) :-
    member(Rank, Cards).

% query(hand([jack, king, jack, jack, queen, ace, jack,jack, ace, queen, queen, king, ace], _)).

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

hand(Cards,Hand).
better(BetterHand,WorseHand).

%%%% Provided code

% The following predicate will sample a Hand as a list of ranks for the given player.
% It expects that there are probabilistic facts of the form card(Player,N,Rank) as specified above

draw_hand(Player,Hand) :- maplist(card(Player),[1,2,3,4],Hand).

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
