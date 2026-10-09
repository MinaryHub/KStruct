import pytest

from kstruct.model import Project
from kstruct.units import fmt, quantity, system_of


def test_round_trips():
    for kind in ("length", "force", "stress", "temperature", "temperature_change"):
        q = quantity("US", kind)
        assert q.parse(q.show(123.4)) == pytest.approx(123.4)


def test_known_values():
    assert quantity("US", "length").show(25.4) == pytest.approx(1.0)
    assert quantity("US", "force").show(4.4482216152605) == pytest.approx(1.0)
    assert quantity("US", "stress").show(1.0) == pytest.approx(145.0377, rel=1e-6)
    assert quantity("US", "temperature").show(100.0) == pytest.approx(212.0)
    assert quantity("US", "temperature_change").show(10.0) == pytest.approx(18.0)
    assert fmt("SI", "stress", 250.0, ".0f") == "250 MPa"


def test_system_follows_pack_unless_overridden():
    assert system_of(Project(standard="astm")) == "US"
    assert system_of(Project(standard="ks")) == "SI"
    assert system_of(Project(standard="astm", units="SI")) == "SI"
