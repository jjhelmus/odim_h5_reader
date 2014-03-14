"""
Routines for reading ODIM_H5 files
"""

import numpy as np
import h5py


from pyart.config import FileMetadata, get_fillvalue
from pyart.io.radar import Radar
from pyart.io.common import make_time_unit_str
from pyart.io.common import radar_coords_to_cart


def read_odim_h5(filename):
    """
    Read a ODIM_H5 file

    Parameters
    ----------
    filename : str
        Name of the ODIM_H5 file to read.

    Returns
    -------
    radar : Radar
        Radar object containing data from ODIM_H5 file.

    """
    # create metadata retrieval object
    # this object is used to create generic metadata for the various
    # parameters passed to the Radar object.  Using this object allows the
    # user to adjust the default values using the Py-ART config file.
    filemetadata = FileMetadata('odim_h5')

    # open the
    hfile = h5py.File(filename, 'r')

    # The general procedure for each parameter is to create a dictionary
    # with default values from the filemetadata object. Then updating any
    # keys with data from the file, as well as recording the actual parameter
    # data in the 'data' key.

    # latitude, longitude and altitude
    # latitude and longitude are measured in degrees (north and east).
    # altitude in meters above mean sea level
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')

    h_where = hfile['where'].attrs
    latitude['data'] = np.array([h_where['lat']], dtype='float64')
    longitude['data'] = np.array([h_where['lon']], dtype='float64')
    altitude['data'] = np.array([h_where['height']], dtype='float64')


    # XXX fake data, replace
    time = filemetadata('time')
    time['data'] = np.array([0])
    _range = filemetadata('range')
    _range['data'] = np.array([0])
    fields = {}
    metadata = filemetadata('metadata')
    scan_type = 'ppi'
    sweep_number = filemetadata('sweep_number')
    sweep_number['data'] = np.array([0])
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    azimuth = filemetadata('azimuth')
    elevation = filemetadata('elevation')
    instrument_parameters = None

    return Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation,
        instrument_parameters=instrument_parameters)


    mdvfile = MdvFile(filename)

    # value attributes
    naz = len(mdvfile.az_deg)
    nele = len(mdvfile.el_deg)
    scan_type = mdvfile.scan_type

    if scan_type not in ['ppi', 'rhi']:
        raise NotImplementedError('No support for scan_type %s.' % scan_type)

    # time
    time = filemetadata('time')
    units = make_time_unit_str(mdvfile.times['time_begin'])
    time['units'] = units
    time_start = date2num(mdvfile.times['time_begin'], units)
    time_end = date2num(mdvfile.times['time_end'], units)
    time['data'] = np.linspace(time_start, time_end, naz * nele)

    # range
    _range = filemetadata('range')
    _range['data'] = np.array(mdvfile.range_km * 1000.0, dtype='float32')
    _range['meters_to_center_of_first_gate'] = _range['data'][0]
    _range['meters_between_gates'] = (_range['data'][1] - _range['data'][0])

    # fields
    fields = {}
    for mdv_field in set(mdvfile.fields):
        field_name = filemetadata.get_field_name(mdv_field)
        if field_name is None:
            continue

        # grab data from MDV object, mask and reshape
        data = mdvfile.read_a_field(mdvfile.fields.index(mdv_field))
        data[np.where(np.isnan(data))] = get_fillvalue()
        data[np.where(data == 131072)] = get_fillvalue()
        data = np.ma.masked_equal(data, get_fillvalue())
        data.shape = (data.shape[0] * data.shape[1], data.shape[2])

        # create and store the field dictionary
        field_dic = filemetadata(field_name)
        field_dic['data'] = data
        field_dic['_FillValue'] = get_fillvalue()
        fields[field_name] = field_dic

    # metadata
    metadata = filemetadata('metadata')
    for meta_key, mdv_key in MDV_METADATA_MAP.iteritems():
        metadata[meta_key] = mdvfile.master_header[mdv_key]

    # sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
    # sweep_end_ray_index
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    len_time = len(time['data'])

    if mdvfile.scan_type == 'ppi':
        nsweeps = nele
        sweep_number['data'] = np.arange(nsweeps, dtype='int32')
        sweep_mode['data'] = np.array(nsweeps * ['azimuth_surveillance'])
        fixed_angle['data'] = np.array(mdvfile.el_deg, dtype='float32')
        sweep_start_ray_index['data'] = np.arange(0, len_time, naz,
                                                  dtype='int32')
        sweep_end_ray_index['data'] = np.arange(naz-1, len_time, naz,
                                                dtype='int32')

    elif mdvfile.scan_type == 'rhi':
        nsweeps = naz
        sweep_number['data'] = np.arange(nsweeps, dtype='int32')
        sweep_mode['data'] = np.array(nsweeps * ['rhi'])
        fixed_angle['data'] = np.array(mdvfile.az_deg, dtype='float32')
        sweep_start_ray_index['data'] = np.arange(0, len_time, nele,
                                                  dtype='int32')
        sweep_end_ray_index['data'] = np.arange(nele - 1, len_time, nele,
                                                dtype='int32')

    # azimuth, elevation
    azimuth = filemetadata('azimuth')
    elevation = filemetadata('elevation')

    if scan_type == 'ppi':
        azimuth['data'] = np.tile(mdvfile.az_deg, nele)
        elevation['data'] = np.array(mdvfile.el_deg).repeat(naz)

    elif scan_type == 'rhi':
        azimuth['data'] = np.array(mdvfile.az_deg).repeat(nele)
        elevation['data'] = np.tile(mdvfile.el_deg, naz)

    # instrument parameters
    # we will set 4 parameters in the instrument_parameters dict
    # prt, prt_mode, unambiguous_range, and nyquist_velocity

    # TODO prt mode: Need to fix this.. assumes dual if two prts
    if mdvfile.radar_info['prt2_s'] == 0.0:
        prt_mode_str = 'fixed'
    else:
        prt_mode_str = 'dual'

    prt_mode = filemetadata('prt_mode')
    prt = filemetadata('prt')
    unambiguous_range = filemetadata('unambiguous_range')
    nyquist_velocity = filemetadata('nyquist_velocity')

    prt_mode['data'] = np.array([prt_mode_str] * nsweeps)
    prt['data'] = np.array([mdvfile.radar_info['prt_s']] * nele * naz,
                           dtype='float32')

    urange_m = mdvfile.radar_info['unambig_range_km'] * 1000.0
    unambiguous_range['data'] = np.array([urange_m] * naz * nele,
                                         dtype='float32')

    uvel_mps = mdvfile.radar_info['unambig_vel_mps']
    nyquist_velocity['data'] = np.array([uvel_mps] * naz * nele,
                                        dtype='float32')

    instrument_parameters = {'prt_mode': prt_mode, 'prt': prt,
                             'unambiguous_range': unambiguous_range,
                             'nyquist_velocity': nyquist_velocity}


