"""Synthetic corridor: depth=4, terminal_k=20, distractors=6."""


class Corridor:
    def __init__(self):
        self.stage = 0

    def s0(self, key):
        if key == 'k0':
            self.stage = 1
            return 'ok0'
        return 'no0'

    def s1(self, key):
        if self.stage >= 1:
            if key == 'k1':
                self.stage = 2
                return 'ok1'
            return 'wrong1'
        return 'locked1'

    def s2(self, key):
        if self.stage >= 2:
            if key == 'k2':
                self.stage = 3
                return 'ok2'
            return 'wrong2'
        return 'locked2'

    def s3(self, key):
        if self.stage >= 3:
            if key == 'k3':
                self.stage = 4
                return 'ok3'
            return 'wrong3'
        return 'locked3'

    def terminal(self, x):
        if self.stage >= 4:
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
