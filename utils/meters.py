class AverageMeter(object):
    def __init__(self):
        self.accum = 0.
        self.n = 0.

    def append(self, value):
        self.accum += float(value)
        self.n += 1

    def reset(self):
        self.accum = 0.
        self.n = 0

    @property
    def value(self) -> float:
        if self.n > 0:
            return self.accum / self.n
        return 0

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.value}>'


class EMAMeter(object):
    def __init__(self, a=0.9):
        assert 0 < a < 1
        self.a = a
        self.value = None

    def append(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.value * self.a + value * (1 - self.a)

    def reset(self):
        self.value = None

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.value}>'
