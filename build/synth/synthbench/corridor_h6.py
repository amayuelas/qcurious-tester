"""Hidden-key corridor: depth=6, terminal_k=20, distractors=6.

Call s0() to enter. Each step returns the key the next step needs.
A process may advance the corridor only once."""

import os

_STATE = "/tmp/.corridor_d6_stage"
_ADVANCED = False


def _stage():
    try:
        with open(_STATE) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def _advance(to):
    global _ADVANCED
    with open(_STATE, 'w') as f:
        f.write(str(to))
    _ADVANCED = True


class Corridor:
    def __init__(self):
        self.stage = _stage()

    def s0(self):
        if self.stage != 0:
            return 'locked0'
        if _ADVANCED:
            return 'cooldown0'
        _advance(1)
        self.stage = 1
        return 'ok0:' + '4282e809'

    def s1(self, key):
        if self.stage != 1:
            return 'locked1'
        if key != '4282e809':
            return 'wrong1'
        if _ADVANCED:
            return 'cooldown1'
        _advance(2)
        self.stage = 2
        return 'ok1:' + '4f0e775c'

    def s2(self, key):
        if self.stage != 2:
            return 'locked2'
        if key != '4f0e775c':
            return 'wrong2'
        if _ADVANCED:
            return 'cooldown2'
        _advance(3)
        self.stage = 3
        return 'ok2:' + 'b4d634bc'

    def s3(self, key):
        if self.stage != 3:
            return 'locked3'
        if key != 'b4d634bc':
            return 'wrong3'
        if _ADVANCED:
            return 'cooldown3'
        _advance(4)
        self.stage = 4
        return 'ok3:' + '48b48559'

    def s4(self, key):
        if self.stage != 4:
            return 'locked4'
        if key != '48b48559':
            return 'wrong4'
        if _ADVANCED:
            return 'cooldown4'
        _advance(5)
        self.stage = 5
        return 'ok4:' + '0ac6506f'

    def s5(self, key):
        if self.stage != 5:
            return 'locked5'
        if key != '0ac6506f':
            return 'wrong5'
        if _ADVANCED:
            return 'cooldown5'
        _advance(6)
        self.stage = 6
        return 'ok5:' + '157b1704'

    def terminal(self, x):
        if self.stage >= 6:
            if x == 0:
                return 't0'
            elif x == 1:
                return 't1'
            elif x == 2:
                return 't2'
            elif x == 3:
                return 't3'
            elif x == 4:
                return 't4'
            elif x == 5:
                return 't5'
            elif x == 6:
                return 't6'
            elif x == 7:
                return 't7'
            elif x == 8:
                return 't8'
            elif x == 9:
                return 't9'
            elif x == 10:
                return 't10'
            elif x == 11:
                return 't11'
            elif x == 12:
                return 't12'
            elif x == 13:
                return 't13'
            elif x == 14:
                return 't14'
            elif x == 15:
                return 't15'
            elif x == 16:
                return 't16'
            elif x == 17:
                return 't17'
            elif x == 18:
                return 't18'
            elif x == 19:
                return 't19'
            return 'tdefault'
        return 'locked_terminal'


def dist0(x):
    if x:
        return 'd0a'
    return 'd0b'

def dist1(x):
    if x:
        return 'd1a'
    return 'd1b'

def dist2(x):
    if x:
        return 'd2a'
    return 'd2b'

def dist3(x):
    if x:
        return 'd3a'
    return 'd3b'

def dist4(x):
    if x:
        return 'd4a'
    return 'd4b'

def dist5(x):
    if x:
        return 'd5a'
    return 'd5b'
