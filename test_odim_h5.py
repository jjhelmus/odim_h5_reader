""" Unit tests for odim_h5.py module. """

import odim_h5

SCAN_FILENAME = 'Example_scan.h5'
PVOL_FILENAME = 'Example_pvol.h5'


def test_read_pvol():
    radar = odim_h5.read_odim_h5(PVOL_FILENAME)


