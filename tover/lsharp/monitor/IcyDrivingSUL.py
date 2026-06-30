from aalpy.base.SUL import SUL

class IcyDrivingSUL(SUL):
    def __init__(self):
        super().__init__()
        self.safe_words = [
            ['icy'],
            ['icy', 'dry', 'dry'],
            ['icy', 'dry', 'icy']]
        self.unsafe_words = [
            ['icy', 'icy'],
            ['icy', 'icy', 'icy'],
            ['icy', 'icy', 'dry'],
        ]

    def query(self, word: tuple):
        self.pre()
        # Empty string for DFA
        if len(word) == 0:
            out = [False] # first state always safe?
        else:
            out = []
            for i in range(1,len(word)+1):
                if word[:i] in self.safe_words:
                    out.append(False) # no alarm
                elif word[:i] in self.unsafe_words:
                    out.append(True) # alarm
                else:
                    out.append("unknown")
        self.post()
        self.num_queries += 1
        print(f"OQ {word}/{out}")
        self.num_steps += len(word)
        return out

    def pre(self):
        pass

    def step(self, letter=None):
        raise RuntimeError("Only use query please")

    def post(self):
        pass