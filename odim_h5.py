"""
Routines for reading ODIM_H5 files
"""

import numpy as np
import h5py
import datetime

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

    # open the file, determine some key parameters
    hfile = h5py.File(filename, 'r')

    datasets = [k for k in hfile if k.startswith('dataset')]
    datasets.sort()
    nsweeps = len(datasets)

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

    # metadata
    # this dictionary is used to store any metadata not recorded
    # elsewhere, typically this is quite sparse.
    # The default keys and values are as follow:
    # 'Conventions': 'CF/Radial instrument_parameters',
    # 'comment': '',
    # 'history': '',
    # 'institution': '',
    # 'instrument_name': '',
    # 'references': '',
    # 'source': '',
    # 'title': '',
    # 'version': '1.3'}
    # See section 4.1 of the CF/Radial format for a description of these
    # attributes.  The 'what/source attribute can probably be used to
    # fill in a number of these.
    metadata = filemetadata('metadata')
    metadata['source'] = hfile['what'].attrs['source']
    metadata['original_container'] = 'odim_h5'

    # sweep_start_ray_index, sweep_end_ray_index
    # These two dictionaries contain the indices of the first and last ray
    # in each sweep (0-based indexing).
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')

    rays_per_sweep = [hfile[d]['where'].attrs['nrays'] for d in datasets]
    ssri = np.cumsum(np.append([0], rays_per_sweep[:-1])).astype('int32')
    seri = np.cumsum(rays_per_sweep).astype('int32') - 1
    sweep_start_ray_index['data'] = ssri
    sweep_end_ray_index['data'] = seri

    # sweep_number
    # sweep number in the volume, 0-based
    sweep_number = filemetadata('sweep_number')
    sweep_number['data'] = np.arange(nsweeps, dtype='int32')

    # sweep_mode
    # The scan type for each sweep, common options are:
    # 'rhi','azimuth_surveillance', 'vertical_pointing'
    # Are all ODIM_H5 files PPI volumes?  How do you check for this
    sweep_mode = filemetadata('sweep_mode')
    sweep_mode['data'] = np.array(nsweeps * ['azimuth_surveillance'])

    # scan_type
    # 'ppi', 'rhi', or 'vpt'.
    # Assuming all ODIM_H5 files are ppi scans
    scan_type = 'ppi'

    # fixed_angle
    # this is the elevation or azimuth angle that is fixed for each sweep.
    # In this case the elevation angle.
    fixed_angle = filemetadata('fixed_angle')
    sweep_el = [hfile[d]['where'].attrs['elangle'] for d in datasets]
    fixed_angle['data'] = np.array(sweep_el, dtype='float32')

    # elevation
    # elevation for each ray in the volume. Since these are PPI scans
    # the elevation angle for each sweep is repeated the number of rays
    # contained in that sweep.
    elevation = filemetadata('elevation')
    elevation['data'] = np.repeat(sweep_el, rays_per_sweep)

    # range
    # range contains the distances in meters to the center of each range
    # bin.  The 'meters_to_center_of_first_gate' and 'meters_between_gates'
    # attribute should also be set accordingly.  If the gate spacing is not
    # constant, remove the 'meters_beween_gates' key and change
    # 'spacing_is_constant' to 'false'.
    _range = filemetadata('range')

    # here we assume that the gate layout in the first sweep is
    # the same as all the sweep.  A check should be added to verify this
    # assumption.  The Radar object cannot work with radar data where the
    # gate spacing is not constant for all radials.  Data of this type
    # should raise an exception.
    first_gate = hfile['dataset1']['where'].attrs['rstart'] * 1000.
    gate_spacing = hfile['dataset1']['where'].attrs['rscale']
    nbins = hfile['dataset1']['where'].attrs['nbins']
    _range['data'] = (np.arange(nbins, dtype='float32') * gate_spacing +
                      first_gate)
    _range['meters_to_center_of_first_gate'] = first_gate
    _range['meters_between_gates'] = gate_spacing

    # azimuth
    # azimuth angle for all rays collected in the volume
    azimuth = filemetadata('azimuth')
    total_rays = np.sum(rays_per_sweep)
    az_data = np.ones((total_rays, ), dtype='float32')

    # loop over the sweeps, store the starting azimuth angles.
    # an average of the startazA and stopazA would probably be a better
    # estimate, but the discontinuity between 0 and 360 would need to be
    # addressed.
    start = 0
    for dset, rays_in_sweep in zip(datasets, rays_per_sweep):

        sweep_az = hfile[dset]['how'].attrs['startazA']
        az_data[start:start + rays_in_sweep] = sweep_az
        start += rays_in_sweep
    azimuth['data'] = az_data

    # time
    # time at which each ray was collected.  Need to define
    # starting time and the units of the data in the 'unit' dictionary
    # element.
    # Since startazT and stopazT do not appear to be present in all files
    # and the startepochs and endepochs attributes appear the same for
    # each sweep, just interpolate between these values.
    # XXX This is does not seem correct.
    # Assuming these are UTC times
    time = filemetadata('time')

    start_epoch = hfile['dataset1']['how'].attrs['startepochs']
    end_epoch = hfile['dataset1']['how'].attrs['stopepochs']
    start_time = datetime.datetime.utcfromtimestamp(start_epoch)
    delta_sec = end_epoch - start_epoch
    time['units'] = make_time_unit_str(start_time)
    time['data'] = np.linspace(0, delta_sec, total_rays).astype('float32')

    # fields
    # the radar moments or fields are stored in as a dictionary of
    # dictionaries.  The dictionary for each field, a 'field dictionary'
    # should contain any necessary metadata.  The actual data is stored in
    # the 'data' key as a 2D array of size (nrays, ngates) where nrays is the
    # total number of rays in all sweeps of the volume, and ngate is the
    # number of bins or gates in each radial.
    fields = {}
    h_field_keys = [k for k in hfile['dataset1'].keys() if
                    k.startswith('data')]
    field_names = [hfile['dataset1'][d]['what'].attrs['quantity'] for d in
                   h_field_keys]
    # loop over the fields, create the field dictionaty
    for field_name, h_field_key in zip(field_names, h_field_keys):
        # XXX still need to set metadata, some default field metadata can
        # likely be provided form the filemetadata object.
        #field_dic = filemetadata(field_name)
        field_dic = {}
        dtype = hfile['dataset1'][h_field_key]['data'].dtype
        field_dic['data'] = np.zeros((total_rays, nbins), dtype=dtype)
        start = 0
        # loop over the sweeps, copy data into correct location in data array
        for dset, rays_in_sweep in zip(datasets, rays_per_sweep):
            sweep_data = hfile[dset][h_field_key]['data'][:]
            field_dic['data'][start:start + rays_in_sweep] = sweep_data[:]
            start += rays_in_sweep
        fields[field_name] = field_dic

    # instrument_parameters
    # this is also a dictionary of dictionaries which contains
    # instrument parameter like wavelength, PRT rate, nyquist velocity, etc.
    # A full list of possible parameters can be found in section 5.1 of
    # the CF/Radial document.
    # prt, prt_mode, unambiguous_range, and nyquist_velocity are the
    # parameters which we try to set in Py-ART although a valid Radar object
    # can be created with fewer or more parameters
    instrument_parameters = None

    return Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation,
        instrument_parameters=instrument_parameters)
