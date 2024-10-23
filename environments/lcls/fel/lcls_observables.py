import logging
import time
from typing import List

import numpy as np
from badger.errors import BadgerEnvObsError
from badger.stats import percent_80

epsilon: float = 1e-8  # avoid divided by zero in relative FEL jitter

def get_fel_observables(interface, points: int, observable_pvs: dict[str,str], observable_names: List[str]):
    """
    Parameters
    ----------
    points : int
        Number of data points
    observable_pvs: dict[str, str]
        Dictionary of observable pvs. The keys are expected to be 'PV_loss' and/or 'PV_gas' depending
          on whether one or both are observed by the lcls environment, where the values are the 
          associated EPICS PV names
    observable_names: List[str]
        List of observables requested by the optimizer. Items are expected to be one or more of
          'beam_loss', 'pulse_intensity_p80', 'pulse_intensity_mean', 'pulse_intensity_median', 
          'pulse_intensity_std', and 'pulse_intensity_std_relative'
    interface
        Specified beamline interface for getting values
    
    Returns
    -------
    output_dict : dict[str, float]
        A dictionary where the keys are the selected observables in observable_names, and the values are 
          the values of those observables
            'beam_loss' : float
            'pulse_intensity_p80' : float
            'pulse_intensity_mean' : float
            'pulse_intensity_median' : float
            'pulse_intensity_std' : float
            'pulse_intensity_std_relative' : float

    Notes
    -----
    At lcls the repetition is 120 Hz and the readout buf size is 2800.
    The last 120 entries correspond to pulse energies over past 1 second.
    """

    logging.info(f"Get value of {points} points")
    output_dict = {}

    # Sleep for a while to get enough data
    try:
        rate = interface.get_value("EVNT:SYS0:1:LCLSBEAMRATE")
        logging.info(f"Beam rate: {rate}")
        nap_time = points / (rate * 1.0)
    except Exception as e:
        nap_time = 1
        logging.warn(
            "Something went wrong with the beam rate calculation. Let's sleep 1 second."
        )
        logging.warn(f"Exception was: {e}")

    time.sleep(nap_time)

    # get loss
    if 'PV_loss' in observable_pvs: # is loss observed
        PV_loss = observable_pvs['PV_loss']
        if 'beam_loss' in observable_names: # is loss signal requested
            try: 
                # get loss data
                loss_raw = interface.get_value(PV_loss)[-points:]
                ind_valid = ~np.isnan(loss_raw)
                loss_valid = loss_raw[ind_valid]
                loss_p80 = percent_80(loss_valid)
                # add loss_p80 to output_dict
                output_dict['loss_p80'] = loss_p80
            except Exception:
                raise BadgerEnvObsError
    
    # get intensity
    if 'PV_gas' in observable_pvs: # is pulse intensity observed
        PV_gas = observable_pvs['PV_gas']
        try:
            # get intensity data
            intensity_raw = interface.get_value(PV_gas)[-points:]
            ind_valid = ~np.isnan(intensity_raw)
            intensity_valid = intensity_raw[ind_valid]

            # get requested intensity signals, add them to output_dict
            if 'pulse_intensity_p80' in observable_names:
                output_dict['pulse_intensity_p80'] = percent_80(intensity_valid)
            if 'pulse_intensity_mean' in observable_names:
                output_dict['pulse_intensity_mean'] = np.mean(intensity_valid)
            if 'pulse_intensity_median' in observable_names:
                output_dict['pulse_intensity_median'] = np.median(intensity_valid)
            if 'pulse_intensity_std' in observable_names:
                output_dict['pulse_intensity_std'] = np.std(intensity_valid)
            if 'pulse_intensity_std_relative' in observable_names:
                output_dict['pulse_intensity_std_relative'] = np.std(intensity_valid) / (np.mean(intensity_valid) + epsilon)
        
        except Exception:
            # not sure what should happen here?
            raise BadgerEnvObsError
    
    return output_dict

"""
def get_intensity_and_loss(
    hxr: bool, points: int, loss_pv: str, fel_channel: str, interface
):
"""
"""
    Gets intensity and loss data from interface and returns mean, median, p80, and standard deviation for intensity and p80 for loss.

    Parameters
    ----------
    hxr : bool
        Inticates hxr (True) or sxr (False) data
    points : int
        Number of data points
    loss_pv : str
        Pv name for loss data
    fel_channel : str
        Identifier for hxr gdet pv selection
    interface
        Specified beamline interface for getting values

    Returns
    -------
    fel_stats : dict
        A dictionary with the following keys and values:
            'gas_p80' : float
                80th percentile of intensity data
            'gas_mean' : float
                Mean of intensity data
            'gas_median' : float
                Median of intensity data
            'gas_std' : float
                Standard deviation of intensity data
            'loss_p80' : float
                80th percentile of loss data

    Notes
    -----
    At lcls the repetition is 120 Hz and the readout buf size is 2800.
    The last 120 entries correspond to pulse energies over past 1 second.
"""
"""
    logging.info(f"Get value of {points} points")

    # Sleep for a while to get enough data
    try:
        rate = interface.get_value("EVNT:SYS0:1:LCLSBEAMRATE")
        logging.info(f"Beam rate: {rate}")
        nap_time = points / (rate * 1.0)
    except Exception as e:
        nap_time = 1
        logging.warn(
            "Something went wrong with the beam rate calculation. Let's sleep 1 second."
        )
        logging.warn(f"Exception was: {e}")

    time.sleep(nap_time)

    if hxr:
        PV_gas = f"GDET:FEE1:{fel_channel}:ENRCHSTCUHBR"
    else:  # SXR
        PV_gas = "EM1K0:GMD:HPS:milliJoulesPerPulseHSTCUSBR"
    try:
        results_dict = interface.get_values([PV_gas, loss_pv])
        intensity_raw = results_dict[PV_gas][-points:]
        loss_raw = results_dict[loss_pv][-points:]
        ind_valid = ~np.logical_or(np.isnan(intensity_raw), np.isnan(loss_raw))
        intensity_valid = intensity_raw[ind_valid]
        loss_valid = loss_raw[ind_valid]

        gas_p80 = percent_80(intensity_valid)
        gas_mean = np.mean(intensity_valid)
        gas_median = np.median(intensity_valid)
        gas_std = np.std(intensity_valid)

        loss_p80 = percent_80(loss_valid)

        fel_stats = {
            "gas_p80": gas_p80,
            "gas_mean": gas_mean,
            "gas_median": gas_median,
            "gas_std": gas_std,
            "loss_p80": loss_p80,
        }

        return fel_stats
    except Exception:  # if average fails use the scalar input
        if hxr:  # we don't have scalar input for HXR
            raise BadgerEnvObsError
        else:
            gas = interface.get_value("EM1K0:GMD:HPS:milliJoulesPerPulse")

            fel_stats = {
                "gas_p80": gas,
                "gas_mean": gas,
                "gas_median": gas,
                "gas_std": 0,
                "loss_p80": 0,
            }

            return fel_stats


def get_loss(points: int, loss_pv: str, interface):  # if only loss is observed
"""
"""
    Gets loss data from the beamline interface and returns the 80th percentile of the data.

    This funtion is used if only loss is observed

    Parameters
    ----------
    points : int
        Number of data points
    loss_pv : str
        Pv name for loss data
    interface
        Specified beamline interface for getting values

    Returns
    -------
    loss_p80 : float
        80th percentile of loss data

"""
"""
    logging.info(f"Get value of {points} points")

    try:
        rate = interface.get_value("EVNT:SYS0:1:LCLSBEAMRATE")
        logging.info(f"Beam rate: {rate}")
        nap_time = points / (rate * 1.0)
    except Exception as e:
        nap_time = 1
        logging.warn(
            "Something went wrong with the beam rate calculation. Let's sleep 1 second."
        )
        logging.warn(f"Exception was: {e}")

    time.sleep(nap_time)

    try:
        loss_raw = interface.get_value(loss_pv)[-points:]
        ind_valid = ~np.isnan(loss_raw)
        loss_valid = loss_raw[ind_valid]
        loss_p80 = percent_80(loss_valid)

        return loss_p80
    except Exception:  # we don't have scalar input for loss
        raise BadgerEnvObsError
"""
