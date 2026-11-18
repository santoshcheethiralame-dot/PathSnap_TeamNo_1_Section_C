import numpy as np
import pandas as pd

from pathsnap.candidates import build_candidates
from pathsnap.hmm import build_graph, build_steps, decode, group_candidates
from pathsnap.segments import segment_ids
from pathsnap.trips import Trip


def road():
    node_xy = np.array([[25.0 * i, 0.0] for i in range(5)])
    u = np.array([0, 1, 2, 3, 1, 2, 3, 4])
    v = np.array([1, 2, 3, 4, 0, 1, 2, 3])
    return node_xy, u, v


def setup(xy, t, prob_for):
    node_xy, u, v = road()
    trip = Trip("syn", np.asarray(t, float), np.asarray(xy, float), node_xy, u, v)
    seg = segment_ids(trip.t, trip.xy)
    c = build_candidates(trip)
    c["prob"] = [prob_for(i, a) for i, a in zip(c.point_idx, c.arc_id)]
    return trip, seg, c


def test_viterbi_repairs_one_noisy_direction_flip():
    # drive +x along arcs 0..3 (twins 4..7). The classifier likes the forward
    # arcs except at fix 10, where it prefers the wrong-direction twin.
    xy = [[3.0 + 3 * i, -2.0] for i in range(30)]

    def prob(i, a):
        fwd = a < 4
        if i == 10:
            return 0.9 if not fwd else 0.1
        return 0.9 if fwd else 0.1

    trip, seg, c = setup(xy, np.arange(30.0), prob)
    cands = group_candidates(c)
    chosen, _ = decode(cands, build_steps(trip, build_graph(trip), cands, seg), lam=1.0, beta=5.0)
    classifier_only = [cands["arcs"][k][np.argmax(cands["prob"][k])] for k in range(len(chosen))]
    assert classifier_only[10] >= 4          # the classifier alone is wrong at fix 10
    assert chosen[10] < 4                    # the HMM fixes it
    assert (chosen < 4).all()


def test_chain_restarts_at_time_gap():
    xy = [[3.0 + 3 * i, -2.0] for i in range(10)] + [[3.0 + 3 * i, -2.0] for i in range(10, 20)]
    t = list(np.arange(10.0)) + list(1000.0 + np.arange(10.0))
    trip, seg, c = setup(xy, t, lambda i, a: 0.9 if a < 4 else 0.1)
    cands = group_candidates(c)
    steps = build_steps(trip, build_graph(trip), cands, seg)
    assert steps[9] is None and all(s is not None for s in steps[:9])
    assert (decode(cands, steps)[0] < 4).all()
