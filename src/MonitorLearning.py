from aalpy import SUL
from stormpy import SparseDtmc


class FilteringSUL(SUL):
    def __init__(self, mc: SparseDtmc):
        super().__init__()

    def pre(self):
        pass

    def post(self):
        pass

    def step(self, letter):
        pass
