import copy

import mechfight as mf
import practice_brain
import ffa_brain


class Bot:
    name = "rohaan"

    def __init__(self, hello):
        self.ffa_b = ffa_brain.Bot(copy.deepcopy(hello))
        try:
            self.prac_b = practice_brain.Bot(copy.deepcopy(hello))
        except Exception as e:
            mf.log(f"practice brain init failed: {e!r}")
            self.prac_b = None
        self.prac_fail = 0

    def act(self, obs):
        a_ffa = self.ffa_b.act(copy.deepcopy(obs))
        if self.ffa_b.ffa or self.prac_b is None or self.prac_fail > 20:
            return a_ffa
        try:
            a = self.prac_b.act(obs)
        except Exception as e:
            self.prac_fail += 1
            mf.log(f"practice brain act failed: {e!r}")
            return a_ffa
        a.pop("debug", None)
        return a


mf.run(Bot)
