from typing import Union, Iterable

from dataclasses import dataclass
from enum import Enum, auto, unique
import random

from environment import Environment, Reward

@dataclass
class BlackJackState:
    current_sum: int
    usable_ace: bool
    dealer_card: int

@unique
class FinalBlackJackState(Enum):
    win = auto()
    lose = auto()

@unique
class BlackJackAction(Enum):
    stick = auto()
    hit = auto()

State = Union[BlackJackState, FinalBlackJackState]

class BlackJack(Environment[State, BlackJackAction]):

    _states: list[State] = [BlackJackState(s, a, d) for s in range(12, 22) for a in [True, False] for d in range(0, 10)]

    def states(self) -> list[State]:
        return self._states

    def initial_state(self) -> State:
        return random.choice(self._states)

    def is_final(self, state: State) -> bool:
        return isinstance(state, FinalBlackJackState)

    def actions(self, state: State) -> Iterable[BlackJackAction]:
        if isinstance(state,FinalBlackJackState):
            return []
        elif state.usable_ace:
            return [BlackJackAction.hit]
        else:
            return BlackJackAction

    @staticmethod
    def draw_card() -> int:
        res = random.randrange(1, 15)
        if res <= 10:
            return res
        elif res >= 11 and res <= 13:
            return 10
        else:
            return 11

    def interaction(self, state: State, action: BlackJackAction) -> tuple[Reward, State]:
        if isinstance(state,FinalBlackJackState):
            raise Exception("Method interaction called with a final state. This should never happen")
        elif action == BlackJackAction.hit:
            player_sum = state.current_sum
            dealer_sum = state.dealer_card
            while dealer_sum < 17:
                dealer_sum += self.draw_card()
            if dealer_sum > 21 or dealer_sum < player_sum:
                return (Reward(1), FinalBlackJackState.win)
            else:
                return (Reward(-1), FinalBlackJackState.lose)
        else:
            card = self.draw_card()
            if card + state.current_sum > 21:
                if state.usable_ace:
                    return (Reward(0), BlackJackState(card + state.current_sum - 10, False, state.dealer_card ))
                else:
                    return (Reward(-1), FinalBlackJackState.lose)
            else:
                return (Reward(0), BlackJackState(card + state.current_sum, state.usable_ace, state.dealer_card ))
