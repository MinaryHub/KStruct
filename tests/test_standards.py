import pytest

from kstruct import standards
from kstruct.standards import Verdict, verdict


def test_materials_are_consistent():
    mats = standards.materials()
    assert len(mats) >= 20
    for m in mats.values():
        assert m.category in standards.CATEGORY_NAMES
        assert 0 < m.poisson_ratio < 0.5
        assert 0 < m.yield_strength <= m.tensile_strength
        assert 1e-10 < m.density < 2e-8  # t/mm³
        # every pack can name every material
        for p in standards.packs().values():
            assert p.material_name(m)


def test_pack_names_the_same_steel_differently():
    s235 = standards.materials()["s235"]
    assert standards.pack("iso").material_name(s235) == "S235JR"
    assert standards.pack("jis").material_name(s235) == "SS400"
    assert standards.pack("astm").material_name(s235) == "ASTM A36"
    assert standards.pack("unknown").id == "iso"


@pytest.mark.parametrize(
    ("fos", "expected"),
    [(2.0, Verdict.OK), (1.5, Verdict.OK), (1.2, Verdict.WARN), (0.9, Verdict.FAIL)],
)
def test_verdict(fos, expected):
    assert verdict(fos, 1.5) == expected


def test_resonance_verdicts():
    from types import SimpleNamespace

    import numpy as np

    from kstruct.model import Project
    from kstruct.report.summary import summarize_modal

    result = SimpleNamespace(frequencies=np.array([100.0, 250.0, 400.0]), rigid_modes=0)
    cases = {0.0: None, 105.0: Verdict.FAIL, 118.0: Verdict.WARN, 170.0: Verdict.OK}
    for f_op, expected in cases.items():
        assert summarize_modal(Project(operating_frequency=f_op), result).verdict == expected
