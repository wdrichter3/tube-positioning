from tube_positioning.calibration import calibrate_planar_reference, pixel_delta_to_mm


def test_reference_calibration_known_rectangle():
    cal = calibrate_planar_reference(
        corners=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        known_width_mm=156.0,
        known_height_mm=66.3,
        reference_type="US_BILL",
    )
    assert abs(cal.mm_per_pixel_x - 0.156) < 1e-9
    assert abs(cal.mm_per_pixel_y - 0.1326) < 1e-9


def test_pixel_delta_conversion():
    cal = calibrate_planar_reference(
        corners=[(0, 0), (100, 0), (100, 100), (0, 100)],
        known_width_mm=10,
        known_height_mm=20,
    )
    dx, dy = pixel_delta_to_mm(10, 10, cal)
    assert dx == 1.0
    assert dy == 2.0
