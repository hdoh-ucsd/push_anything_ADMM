"""Worker plant contexts for parallel sample evaluation must have the
SceneGraph `geometry_query` input port CONNECTED.

Root cause of the 2026-08-21 parallel-path defect: `_lazy_init_worker_kits`
built worker contexts with `plant.CreateDefaultContext()`, which returns a
STANDALONE LeafContext. The plant lives inside a Diagram alongside a
SceneGraph, and every collision/distance query the LCS build performs needs
`geometry_query` connected -- which only holds for a plant subcontext
extracted from a DIAGRAM context.

So every worker raised

    RuntimeError: InputPort::Eval(): required InputPort[0] (geometry_query)
    of System ::_::plant (MultibodyPlant<double>) is not connected

which a bare `except Exception: pass` in evaluate_sample swallowed, leaving
J_n=None and a NaN cost. The dispatcher then saw best_other=nan every tick,
never repositioned, never entered c3, and the run failed outright while
appearing 2x faster because it did no work.

These tests pin the property directly, with a minimal plant+SceneGraph
diagram (no Panda model needed, so they run in well under a second).
"""
import pytest

ad = pytest.importorskip("pydrake.all")

from control.sampling_c3.inner_solve import make_worker_plant_context  # noqa: E402


def _mini_diagram():
    builder = ad.DiagramBuilder()
    plant, _sg = ad.AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    plant.Finalize()
    return builder.Build(), plant


def test_worker_context_has_geometry_query_connected():
    """The property the LCS build actually depends on."""
    diagram, plant = _mini_diagram()
    ctx = make_worker_plant_context(plant, diagram)
    # Must not raise. This is the exact call that failed in every worker.
    plant.get_geometry_query_input_port().Eval(ctx)


def test_standalone_context_reproduces_the_original_bug():
    """Documents WHY the old code path was wrong, so nobody reintroduces it."""
    _diagram, plant = _mini_diagram()
    bad = plant.CreateDefaultContext()          # what the old code did
    with pytest.raises(RuntimeError, match="geometry_query"):
        plant.get_geometry_query_input_port().Eval(bad)


def test_each_worker_gets_a_distinct_context():
    """Workers run concurrently; sharing a context would race."""
    diagram, plant = _mini_diagram()
    a = make_worker_plant_context(plant, diagram)
    b = make_worker_plant_context(plant, diagram)
    assert a is not b
    plant.get_geometry_query_input_port().Eval(a)
    plant.get_geometry_query_input_port().Eval(b)


def test_worker_contexts_are_independent():
    """Mutating one worker's context must not touch another's."""
    diagram, plant = _mini_diagram()
    a = make_worker_plant_context(plant, diagram)
    b = make_worker_plant_context(plant, diagram)
    # An empty plant has no positions, so compare the serialized context
    # state objects rather than q: distinct objects, not aliases.
    assert a.get_state() is not b.get_state()


def test_missing_diagram_raises_rather_than_returning_a_broken_context():
    """A None diagram must fail LOUDLY. Returning a standalone context here
    is precisely how the original defect stayed invisible for a whole run."""
    _diagram, plant = _mini_diagram()
    with pytest.raises(RuntimeError, match="diagram"):
        make_worker_plant_context(plant, None)
