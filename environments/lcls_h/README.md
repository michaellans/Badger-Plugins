# LCLS Environment w/ counter hysteresis feature

This is the LCLS environment for Badger, usually you should only use it in the LCLS ACR, and be very careful since it would change the machine parameters once you run the routine containing this env!

## Author

- Zhe Zhang, reachable at `(510) 518-0963`, or `zhezhang@slac.stanford.edu`
- Ryan Roussel, reachable at `rroussel@slac.stanford.edu`

## Parameters

For most cases the default env parameters should work but sometimes you might want to tune a few key parameters to adapt to your tuning scenario:

- `readonly`: Set to `True` to do dry run (would only read from the machine, not write to it)
- `hxr`: if tuning on HXR or SXR, will affect the PV used to get the FEL signal:
    - If set to `True`, the FEL PV would be `GDET:FEE1:{fel_channel}:ENRCHSTCUHBR`
    - Else the FEL PV would be `EM1K0:GMD:HPS:milliJoulesPerPulseHSTCUSBR`
- `fel_channel`: Determine which FEL PV to use if it's HXR, see the explaination for `hxr` param
    - Common choices are `241` or `361`. If set to `241` and `hxr` is `True`, the FEL PV to be
      used would be `GDET:FEE1:241:ENRCHSTCUHBR`
- `loss_pv`: Which beam loss monitor to use, note that it has to be a **buffer PV** (that returns a buffer of numbers instead of a single number)!
    - Common choice is `CBLM:UNDH:1375:I1_LOSSHSTBR`, note that `BR` suffix that indicates the buffer nature of this PV
    - If you put a single return value PV here you'll get an error when run the optimization, this behavior would be fixed in the future so that you can also use single return value PV here
- `use_check_var`: If check the variables reach their desired values after dialing in the solution on the machine
    - If set to `True`, Badger will check if the variables reache the desired values every `0.1s`, until all variables done changing. If for some reason some variables are not able to reach the destination values, Badger will throw an error after `check_var_timeout` seconds, and terminate the run
    - If set to `False`, Badger will not check if the variables reach the desired values, instead it would wait for `trim_delay` seconds then directly measure the observables defined in the routine
- `trim_delay`: The waiting time between setting variables and getting observables
    - If `use_check_var` is set to `True`, `trim_delay` would have no effects. This behavior would be changed in the future since sometimes we need extra settle down time even after all variables have reached their desired values
- `overshoot_fraction`: The amount Badger would overshoot if a PV w/ hysteresis is tuned down -- Badger would overshoot the target value by `overshoot_fraction` * variable full range, then recover it back to the desired values. This way we guarantee that for quads we always tune them up to the destinations, thus counter the hysteresis effect
    - Default value is `0.1`, usually it's a bit too big if we are tuning all quads within a small range, you should experiment w/ it but it's usually safe to go value around `0.02`
    - Set it to `0.0` to turn off the counter-hysteresis feature

## Notes

In most cases, when you compose a routine w/ the LCLS env, you'll only need to touch the following env params:

- `fel_channel`
- `loss_pv`
- `overshoot_fraction`

according to your needs. The most commonly used values for the params above are:

- `fel_channel`: `241`
- `loss_pv`: `CBLM:UNDH:1375:I1_LOSSHSTBR`
- `overshoot_fraction`: 0.0

If you encounter any issues in a run that uses the LCLS env, please reach out to the env author. The contact info is listed in the author section.
