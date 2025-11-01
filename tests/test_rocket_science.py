import math

from rocketScience_clean import (
    trapezoid_area,
    mean_aerodynamic_chord,
    cp_from_root_le,
)


def test_trapezoid_area():
    # area for root=0.1, tip=0.05, span=0.02 => ((0.1+0.05)/2)*0.02 = 0.0015
    area = trapezoid_area(0.1, 0.05, 0.02)
    assert math.isclose(area, 0.0015, rel_tol=1e-9)


def test_mean_aerodynamic_chord():
    mac = mean_aerodynamic_chord(0.1, 0.05, 0.02)
    # compute expected using formula
    r = 0.1
    t = 0.05
    expected = (2.0 / 3.0) * (r ** 2 + r * t + t ** 2) / (r + t)
    assert math.isclose(mac, expected, rel_tol=1e-9)


def test_cp_from_root_le_increases_with_sweep():
    cp_no_sweep = cp_from_root_le(0.08, 0.03, 0.04, 0.0)
    cp_with_sweep = cp_from_root_le(0.08, 0.03, 0.04, 20.0)
    assert cp_with_sweep > cp_no_sweep
