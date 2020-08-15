from typing import Dict, Set, List
from collections import namedtuple
import itertools
import heapq

from shrdlurn.levels import get_stacks_with_color, complement, leftmost, rightmost, stack_on_top, remove_top

_LogicalForm = namedtuple("_LogicalForm", "arguments")

class LogicalForm(_LogicalForm):
# class LogicalForm(object):
    def __new__(cls, *args):
        return super().__new__(cls, args)
        # self.arguments = args

    def __repr__(self):
        if self.arguments:
            return "{}({})".format(self.predicate, ','.join(arg.__repr__() for arg in self.arguments))
        else:
            return self.predicate

    @property
    def predicate(self):
        raise NotImplementedError()

    @property
    def return_type(self):
        raise NotImplementedError()

    def size(self):
        return 1 + sum(lf.size() for lf in self.arguments)

    def denotation(self, wall):
        raise NotImplementedError()

    def featurize(self, depth):
        yield from self.rec_featurize(depth)
        for arg in self.arguments:
            yield from arg.featurize(depth)

    def rec_featurize(self, depth):
        if depth == 0:
            yield (self.predicate, )
        else:
            for index, arg in enumerate(self.arguments):
                for feat in arg.rec_featurize(depth - 1):
                    yield (self.predicate, index, ) + feat

class All(LogicalForm):
    @property
    def predicate(self):
        return "all"

    @property
    def return_type(self):
        return "set"

    def denotation(self, wall):
        return list(range(len(wall)))

class Color(LogicalForm):
    @property
    def return_type(self):
        return "color"

    @property
    def color_index(self):
        raise NotImplementedError()

    def denotation(self, wall):
        return self.color_index

class Cyan(Color):
    @property
    def predicate(self):
        return "cyan"

    @property
    def color_index(self):
        return 0

class Brown(Color):
    @property
    def predicate(self):
        return "brown"

    @property
    def color_index(self):
        return 1

class Red(Color):
    @property
    def predicate(self):
        return "red"

    @property
    def color_index(self):
        raise 2

class Orange(Color):
    @property
    def predicate(self):
        return "orange"

    @property
    def color_index(self):
        return 3

class With(LogicalForm):
    @property
    def predicate(self):
        return "with"

    @property
    def return_type(self):
        return "set"

    def denotation(self, wall):
        assert len(self.arguments) == 1
        color = self.arguments[0]
        assert isinstance(color, Color)
        return get_stacks_with_color(wall, color.denotation(wall))

class Not(LogicalForm):
    @property
    def predicate(self):
        return "not"

    @property
    def return_type(self):
        return "set_not"

    def denotation(self, wall):
        assert len(self.arguments) == 1
        sub_denote = self.arguments[0].denotation(wall)
        assert isinstance(sub_denote, list)
        return complement(wall, sub_denote)

class Leftmost(LogicalForm):
    @property
    def predicate(self):
        return "leftmost"

    @property
    def return_type(self):
        return "set_spatial"

    def denotation(self, wall):
        assert len(self.arguments) == 1
        sub_denote = self.arguments[0].denotation(wall)
        assert isinstance(sub_denote, list)
        return leftmost(wall, sub_denote)

class Rightmost(LogicalForm):
    @property
    def predicate(self):
        return "rightmost"

    @property
    def return_type(self):
        return "set_spatial"

    def denotation(self, wall):
        assert len(self.arguments) == 1
        sub_denote = self.arguments[0].denotation(wall)
        assert isinstance(sub_denote, list)
        return rightmost(wall, sub_denote)

class Add(LogicalForm):
    @property
    def predicate(self):
        return "add"

    @property
    def return_type(self):
        return "act"

    def denotation(self, wall):
        assert len(self.arguments) == 2
        arg1_d = self.arguments[0].denotation(wall)
        arg2_d = self.arguments[1].denotation(wall)
        assert isinstance(arg1_d, list)
        assert isinstance(arg2_d, int)
        return stack_on_top(wall, arg1_d, arg2_d)

class Remove(LogicalForm):
    @property
    def predicate(self):
        return "remove"

    @property
    def return_type(self):
        return "act"

    def denotation(self, wall):
        assert len(self.arguments) == 1
        d = self.arguments[0].denotation(wall)
        assert isinstance(d, list)
        return remove_top(wall, d)

unaries_by_argument_type = {
    'set': [Not, Leftmost, Rightmost, Remove],
    'set_not': [Leftmost, Rightmost, Remove],
    'set_spatial': [Not, Remove],
    'color': [With],
    'act': [],
}

def beam_candidates(beams_by_size: Dict[int, Dict[str, List[LogicalForm]]],
                    new_size: int):
    if new_size == 1:
        yield from [All(), Cyan(), Brown(), Red(), Orange()]
    else:
        assert new_size > max(beams_by_size.keys())

        # deal with unaries
        for return_type, sub_trees in beams_by_size[new_size - 1].items():
            for sub_tree in sub_trees:
                for UnaryCls in unaries_by_argument_type[sub_tree.return_type]:
                    yield UnaryCls(sub_tree)

        # deal with binaries, i.e. Add
        for size_1 in range(1, max(beams_by_size.keys()) + 1):
            size_2 = new_size - size_1 - 1
            if size_1 < size_2:
                continue
            if size_1 not in beams_by_size:
                continue
            if size_2 not in beams_by_size:
                continue
            beams_1 = beams_by_size[size_1]
            beams_2 = beams_by_size[size_2]
            for set_arg, color_arg in itertools.chain(
                itertools.product(beams_1['set'], beams_2['color']),
                itertools.product(beams_2['set'], beams_1['color']),
            ):
                yield Add(set_arg, color_arg)

def extend_beams(beams_by_size: Dict[int, Dict[str, List[LogicalForm]]],
                 new_size: int,
                 scoring_function=None,
                 pruning_k=None):

    candidates = beam_candidates(beams_by_size, new_size)
    if pruning_k is not None:
        assert scoring_function is not None
        candidates = heapq.nlargest(pruning_k, candidates, key=scoring_function)

    grouped = {'set': [], 'color': [], 'act': [], 'set_spatial': [], 'set_not': []}
    for cand in candidates:
        grouped[cand.return_type].append(cand)
        assert cand.size() == new_size
    beams_by_size[new_size] = grouped

def build_all_beams(scoring_function, pruning_k, max_size=8):
    beams_by_size = {}
    for new_size in range(1, max_size + 1):
        extend_beams(beams_by_size, new_size, scoring_function=scoring_function, pruning_k=pruning_k)
    return beams_by_size

if __name__ == "__main__":
    test = Add(
        Not(Leftmost(With(Brown()))),
        Orange()
    )
    print(test)

    for depth in range(4):
        print(f"max depth: {depth}")
        for feat in test.featurize(depth):
            print(feat)
        print()
