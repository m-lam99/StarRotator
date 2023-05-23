import json
import logging
import warnings
from numbers import Number, Real
from os.path import splitext

import numpy as np
from scipy.constants import speed_of_light
from scipy.optimize import OptimizeWarning, least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm
from tqdm import tqdm

from pysme.solve import SME_Solver
from pysme.synthesize import Synthesizer
from pysme.sme import MASK_VALUES
from pysme.util import print_to_log, show_progress_bars
clight = speed_of_light * 1e-3  # km/s

logger = logging.getLogger(__name__)

class SR_SME_Solver(SME_Solver):
    
    def SR_solve(self, sme, param_names=None, p0=None, segments="all", bounds=None):
        """
        Find the least squares fit parameters to an observed spectrum

        NOTE: intermediary results will be saved in filename ("sme.npy")

        Parameters
        ----------
        sme : SME_Struct
            sme struct containing all input (and output) parameters
        param_names : list, optional
            the names of the parameters to fit (default: ["teff", "logg", "monh"])
        filename : str, optional
            the sme structure will be saved to this file, use None to suppress this behaviour (default: "sme.npy")

        Returns
        -------
        sme : SME_Struct
            same sme structure with fit results in sme.fitresults, and best fit spectrum in sme.smod
        """

        assert "wave" in sme, "SME Structure has no wavelength"
        assert "spec" in sme, "SME Structure has no observation"
        
        if self.restore and self.filename is not None:
            fname = self.filename.rsplit(".", 1)[0]
            fname = f"{fname}_iter.json"
            try:
                with open(fname) as f:
                    data = json.load(f)
                for fp in param_names:
                    sme[fp] = data[fp]
                logger.warning(f"Restoring existing backup data from {fname}")
            except:
                pass

        if "uncs" not in sme:
            sme.uncs = np.ones(sme.spec.size)
            logger.warning("SME Structure has no uncertainties, using all ones instead")
        if "mask" not in sme:
            sme.mask = np.full(sme.wave.size, MASK_VALUES.LINE)

        segments = Synthesizer.check_segments(sme, segments)

        # Clean parameter values
        if param_names is None:
            param_names = sme.fitparameters
        if param_names is None or len(param_names) == 0:
            logger.warning(
                "No Fit Parameters have been set. Using ('teff', 'logg', 'monh') instead."
            )
            param_names = ("teff", "logg", "monh")
        self.parameter_names = self.sanitize_parameter_names(sme, param_names)

        self.update_linelist = False
        for name in self.parameter_names:
            if name[:8] == "linelist":
                if self.update_linelist is False:
                    self.update_linelist = []
                try:
                    idx = int(name.split()[1])
                except IndexError:
                    raise ValueError(
                        f"Could not parse fit parameter {name}, expected a "
                        "linelist parameter like 'linelist n gflog'"
                    )
                self.update_linelist += [idx]
        if self.update_linelist:
            self.update_linelist = np.unique(self.update_linelist)

        # Create appropiate bounds
        if bounds is None:
            bounds = self.get_bounds(sme)
            
        if "logg" in self.parameter_names:
            logg = p0[1]-0.2, p0[1]+0.2
            bounds[:,1] = logg
        # scales = self.get_scale()
        step_sizes = self.get_step_sizes(self.parameter_names)
        # Starting values
        # p0 = self.get_default_values(sme)
        if np.any((p0 < bounds[0]) | (p0 > bounds[1])):
            logger.warning(
                "Initial values are incompatible with the bounds, clipping initial values"
            )
            p0 = np.clip(p0, bounds[0], bounds[1])
        # Restore backup
        if self.restore:
            sme = self.restore_func(sme)

        # Get constant data from sme structure
        for seg in segments:
            sme.mask[seg, sme.uncs[seg] == 0] = MASK_VALUES.BAD
        mask = sme.mask_line[segments]
        spec = sme.spec[segments][mask]
        uncs = sme.uncs[segments][mask]

        # Divide the uncertainties by the spectrum, to improve the fit in the continuum
        # Just as in IDL SME, this increases the relative error for points inside lines
        # uncs /= np.abs(spec)

        # This is the expected range of the uncertainty
        # if the residuals are larger, they are dampened by log(1 + z)
        self.f_scale = 0.2 * np.nanmean(spec.ravel()) / np.nanmean(uncs.ravel())

        logger.info("Fitting Spectrum with Parameters: %s", ",".join(param_names))
        logger.debug("Initial values: %s", p0)
        logger.debug("Bounds: %s", bounds)

        if (
            sme.wran.min() * (1 - 100 / clight) > sme.linelist.wlcent.min()
            or sme.wran.max() * (1 + 100 / clight) < sme.linelist.wlcent.max()
        ):
            logger.warning(
                "The linelist extends far beyond the requested wavelength range."
                " This will slow down the calculation, consider using only relevant lines\n"
                f"Wavelength range: {sme.wran.min()} - {sme.wran.max()} Å"
                f" ; Linelist range: {sme.linelist.wlcent.min()} - {sme.linelist.wlcent.max()} Å"
            )

        # Setup LineList only once
        dll = self.synthesizer.get_dll()
        dll.SetLibraryPath()
        dll.InputLineList(sme.linelist)

        # Do the heavy lifting
        if self.nparam > 0:
            self.progressbar = tqdm(
                desc="Iteration", total=0, disable=~show_progress_bars
            )
            self.progressbar_jacobian = tqdm(
                desc="Jacobian", total=len(p0), disable=~show_progress_bars
            )
            with print_to_log():
                res = least_squares(
                    self._residuals,
                    jac=self._jacobian,
                    x0=p0,
                    bounds=bounds,
                    loss=sme.leastsquares_loss,
                    f_scale=self.f_scale,
                    method=sme.leastsquares_method,
                    x_scale=sme.leastsquares_xscale,
                    # These control the tolerance, for early termination
                    # since each iteration is quite expensive
                    xtol=sme.accxt,
                    ftol=sme.accft,
                    gtol=sme.accgt,
                    verbose=2,
                    args=(sme, spec, uncs, mask),
                    kwargs={
                        "bounds": bounds,
                        "segments": segments,
                        "step_sizes": step_sizes,
                        "method": sme.leastsquares_jac,
                    },
                )
                # The jacobian is altered by the loss function
                # This lets us keep the original for our uncertainty estimate
                res.jac = self._latest_jacobian

            self.progressbar.close()
            self.progressbar_jacobian.close()
            for i, name in enumerate(self.parameter_names):
                sme[name] = res.x[i]
            sme = self.update_fitresults(sme, res, segments)
            logger.debug("Reduced chi square: %.3f", sme.fitresults.chisq)
            try:
                for name, value, unc in zip(
                    self.parameter_names, res.x, sme.fitresults.fit_uncertainties
                ):
                    print("%s\t%.5f +- %.5g" % (name.ljust(10), value, unc))
                logger.info("%s\t%s +- %s" % ("v_rad".ljust(10), sme.vrad, sme.vrad_unc))
            except:
                pass
        elif len(param_names) > 0:
            # This happens when vrad and/or cscale are given as parameters but nothing else
            # We could try to reuse the already calculated synthetic spectrum (if it already exists)
            # However it is much lower resolution then the newly synthethized one (usually)
            # Therefore the radial velocity wont be as good as when redoing the whole thing
            sme = self.synthesizer.synthesize_spectrum(sme, segments)
        else:
            raise ValueError("No fit parameters given")

        if self.filename is not None:
            sme.save(self.filename)

        return sme


def solve(
    sme, param_names=None, p0=None, segments="all", filename=None, restore=False, **kwargs
):
    solver = SR_SME_Solver(filename=filename, restore=restore)

    return solver.SR_solve(sme, param_names, p0, segments, **kwargs)