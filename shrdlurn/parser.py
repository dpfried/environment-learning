from typing import Dict, Set, List, Tuple
from collections import namedtuple
import itertools
import heapq
import random
from vocabulary import Index
from shrdlurn.levels import get_stacks_with_color, complement, leftmost, rightmost, stack_on_top, remove_top

MAX_SIZE = 8
MAX_FEATURE_DEPTH = 3

FeaturizedLogicalForm = namedtuple("FeaturizedLogicalForm", ["logical_form", "features", "feature_ids"])
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

    def featurize(self, max_depth=MAX_FEATURE_DEPTH):
        for depth in range(max_depth+1):
            yield from self.featurize_single_depth(depth)

    def featurize_single_depth(self, depth):
        yield from self.rec_featurize(depth)
        for arg in self.arguments:
            yield from arg.featurize_single_depth(depth)

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
        return 2

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

def beam_candidates(beams_by_size: Dict[int, Dict[str, List[Tuple[FeaturizedLogicalForm, float]]]],
                    new_size: int):
    if new_size == 1:
        yield from [All(), Cyan(), Brown(), Red(), Orange()]
    else:
        assert new_size > max(beams_by_size.keys())

        # deal with unaries
        for return_type, sub_trees_flf in beams_by_size[new_size - 1].items():
            for sub_tree_flf, _ in sub_trees_flf:
                sub_tree = sub_tree_flf.logical_form
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
            for (set_arg_flf, _), (color_arg_flf, _) in itertools.chain(
                itertools.product(beams_1['set'] + beams_1['set_not'] + beams_1['set_spatial'], beams_2['color']),
                itertools.product(beams_2['set'] + beams_2['set_not'] + beams_2['set_spatial'], beams_1['color']),
            ):
                yield Add(set_arg_flf.logical_form, color_arg_flf.logical_form)

def extend_beams(beams_by_size: Dict[int, Dict[str, List[Tuple[FeaturizedLogicalForm, float]]]],
                 new_size: int,
                 scoring_function=None,
                 pruning_k=None):

    candidates = beam_candidates(beams_by_size, new_size)
    featurized_candidates = []
    for lf in candidates:
        this_features = list(lf.featurize())
        this_ids = []
        for feature in this_features:
            this_ids.append(LOGICAL_FORM_FEATURE_INDEX.index(feature))
        featurized_candidates.append(FeaturizedLogicalForm(lf, this_features, this_ids))

    if pruning_k is not None:
        assert scoring_function is not None
        scored_candidates =[(feat_cand, scoring_function(feat_cand)) for feat_cand in featurized_candidates]
        scored_candidates = heapq.nlargest(pruning_k, scored_candidates, key=lambda t: t[1])
    else:
        scored_candidates = [(feat_cand, 0.0) for feat_cand in featurized_candidates]

    grouped = {'set': [], 'color': [], 'act': [], 'set_spatial': [], 'set_not': []}
    for cand, score in scored_candidates:
        grouped[cand.logical_form.return_type].append((cand, score))
        assert cand.logical_form.size() == new_size
    beams_by_size[new_size] = grouped

def executable(beams_by_size):
    for size, beams in beams_by_size.items():
        for return_type, candidates in beams.items():
            if return_type == 'act':
                yield from candidates

def search_over_lfs(scoring_function, pruning_k, max_size=8):
    beams_by_size = {}
    for new_size in range(1, max_size + 1):
        extend_beams(beams_by_size, new_size, scoring_function=scoring_function, pruning_k=pruning_k)
    return executable(beams_by_size)

LOGICAL_FORM_FEATURE_INDEX = Index()
FEATURIZED_LOGICAL_FORMS = list(search_over_lfs(None, None))
LOGICAL_FORM_FEATURE_INDEX.frozen = True

if __name__ == "__main__":
    test = Add(
        Not(Leftmost(With(Brown()))),
        Orange()
    )
    print(test)

    # for depth in range(4):
    #     print(f"max depth: {depth}")
    #     for feat in test.featurize_single_depth(depth):
    #         print(feat)
    #     print()

    # beams_by_size = build_all_beams(lambda _: random.random(), pruning_k=10, max_size=5)
    # for size, beams in beams_by_size.items():
    #     print(size)
    #     print(beams)
    #     print()
    beams_by_size = {}
    features = set()
    # compute sizes and num features
    candidate_count = 0
    for size in range(1, MAX_SIZE+1):
        extend_beams(beams_by_size, size)
        old_feature_count = len(features)
        old_candidate_count = candidate_count
        for cands in beams_by_size[size].values():
            for cand in cands:
                candidate_count += 1
                for feature_depth in range(MAX_FEATURE_DEPTH + 1):
                    features.update(cand.featurize_single_depth(feature_depth))
        print(f"size {size}")
        print(f"candidates: {old_candidate_count} -> {candidate_count}")
        print(f"features: {old_feature_count} -> {len(features)}")
