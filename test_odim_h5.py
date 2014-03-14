""" Unit tests for odim_h5.py module. """

import odim_h5

SCAN_FILENAME = 'Example_scan.h5'
PVOL_FILENAME = 'Example_pvol.h5'


def test_read_pvol():
    radar = odim_h5.read_odim_h5(PVOL_FILENAME)

    assert round(radar.latitude['data'], 2) == 58.11
    assert round(radar.longitude['data'], 2) == 15.94
    assert round(radar.altitude['data'], 2) == 222.0
