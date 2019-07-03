# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

_units = {}
_units['l'] = u.deg
_units['b'] = u.deg
_units['distance'] = u.kpc
_units['pm_l_cosb'] = u.mas/u.yr
_units['pm_b'] = u.mas/u.yr
_units['radial_velocity'] = u.km/u.s
component_names = list(_units.keys())

def ln_normal(x, mu, var):
    return -0.5*np.log(2*np.pi) - 0.5*np.log(var) - 0.5 * (x-mu)**2 / var


def fast_to_galcen(gal, galcen_frame):
    rep = gal.cartesian
    if 's' in rep.differentials:
        dif = rep.differentials['s']
    else:
        dif = None

    new_rep = (rep.without_differentials() +
               coord.CartesianRepresentation([-1, 0, 0] * galcen_frame.galcen_distance))

    if dif is not None:
        new_dif = dif + galcen_frame.galcen_v_sun
        new_rep = new_rep.with_differentials({'s': new_dif})

    return new_rep


class OrbitFitModel:

    _param_names = ['b', 'd', 'pml', 'pmb', 'vr',
                    'log_pos_sigma', 'log_vel_sigma',
                    'log_bg_pos_sigma', 'log_bg_vel_sigma',
                    'pot_m', 'pot_log_rs']

    def __init__(self, data, errs, l0=None,
                 galcen_frame=None, frozen=None):
        """
        Parameters
        ----------
        mgiants : `~astropy.table.Table`
        rrlyrae : `~astropy.table.Table`
        frozen : iterable (optional)
            A dictionary of parameter names to freeze and their values to freeze
            to. For example, to freeze `dv_dl` to 0.5 km/s/deg, pass
            `{'dv_dl': 0.5}`.

        **kwargs
            Any extra arguments are stored as constants as attributes of the
            instance.
        """
        if frozen is None:
            frozen = dict()
        self.frozen = dict(frozen)

        self.data = data
        self.errs = errs

        if galcen_frame is None:
            galcen_frame = coord.Galactocentric()
        self.galcen_frame = galcen_frame

        if l0 is None:
            l0 = data['l'].max() + 1
        if hasattr(l0, 'unit'):
            l0 = l0.to_value(_units['l'])
        self.l0 = l0

        # for name in potential.parameters:
        #     self._param_names.append('pot_' + name)
        # self._potential_units = potential.units
        # self.potential_cls = potential.__class__

        # TODO: might want to make this customizable too
        self._bg_mus = [np.mean(self.data[name])
                        for name in component_names[1:]]

    def get_p0(self, l_window=8*u.deg):
        # TODO: this needs to generate initial values for all non-frozen parameters, not just orbit parameters
        l_window = l_window.to_value(_units['l'])
        mask = self.data['l'] > (self.l0 - l_window)

        p0 = []
        for name in component_names[1:]:
            p0.append(np.mean(self.data[name][mask]))

        return p0

    def get_orbit_gal(self, p, dt=0.5*u.Myr, n_steps=500):
        kw = {'l': self.l0 * _units['l']}
        for k in _units:
            if k == 'l':
                continue
            kw[k] = p[k] * _units[k]
        c = coord.Galactic(**kw)

        # TODO: why is astropy so slow here?
        # cart = c.transform_to(gc_frame)
        # cart = cart.cartesian
        w0 = gd.PhaseSpacePosition(fast_to_galcen(c, self.galcen_frame))
        pot = self.get_potential(p)

        orbit = pot.integrate_orbit(w0, dt=dt, n_steps=n_steps)
        model_gal = orbit.to_coord_frame(coord.Galactic,
                                         galactocentric_frame=self.galcen_frame)
        return model_gal

    def pack_pars(self, **kwargs):
        vals = []
        for k in self._param_names:
            frozen_val = self.frozen.get(k, None)
            val = kwargs.get(k, frozen_val)
            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(val)
        return np.array(vals)

    def unpack_pars(self, p):
        key_vals = []

        j = 0
        for name in self._param_names:
            if name in self.frozen:
                key_vals.append((name, self.frozen[name]))
            else:
                key_vals.append((name, p[j]))
                j += 1

        return dict(key_vals)

    def get_potential(self, p):
        """pars is the unpacked pars dict"""
        # HACK: TODO: allow other potentials
        return gp.HernquistPotential(m=p['pot_m'],
                                     c=np.exp(p['pot_log_rs']),
                                     units=galactic)

    # =========================================================================
    # Probability functions:
    #

    def ln_prior(self, p):
        lp = 0.

        if 'f' not in self.frozen:
            if not 0 <= p['f'] <= 1:
                return -np.inf

            # lp += np.log(1.)

        # TODO: just realized that sampling in these parameters is so wrong!
        if np.abs(p['b']) > 60: # MAGIC NUMBER
            return -np.inf

        if not 5 < p['d'] < 50: # MAGIC NUMBER
            return -np.inf

        if np.abs(p['pml']) > 10 or np.abs(p['pmb']) > 10: # MAGIC NUMBER
            return -np.inf

        if np.abs(p['vr']) > 500: # MAGIC NUMBER
            return -np.inf

        if not -1 < p['log_pos_sigma'] < 3:
            return -np.inf

        if not 0 < p['log_vel_sigma'] < 4.5:
            return -np.inf

        # TODO: note, no priors on bg sigmas because I always freeze them

        if not 1e11 < p['pot_m'] < 1e13:
            return -np.inf

        if not 0 < p['pot_log_rs'] < 4.5:
            return -np.inf

        return lp

    def ln_likelihood_fg(self, p):
        model_gal = self.get_orbit_gal(p)
        model_x = model_gal.l.wrap_at(180*u.deg).degree
        if model_x[-1] < -180:
            return -np.inf

        ix = np.argsort(model_x)
        model_x = model_x[ix]

        # define interpolating functions
        order = 3
        bbox = [-180, 180]

        # Transform pos/vel stddev to observables:
        vel_sigma = np.exp(p['log_vel_sigma']) # km/s
        pos_sigma = np.exp(p['log_pos_sigma']) # kpc

        interps = dict()
        for i, name in enumerate(component_names[1:]):
            model_y = getattr(model_gal, name).to_value(_units[name])
            interps[name] = InterpolatedUnivariateSpline(model_x, model_y[ix],
                                                         k=order, bbox=bbox)

        sigma = dict()
        sigma['b'] = 180/np.pi * pos_sigma / interps['distance'](self.data['l'])
        sigma['distance'] = pos_sigma
        sigma['pm_l_cosb'] = 1/4.74 * vel_sigma / interps['distance'](self.data['l'])
        sigma['pm_b'] = sigma['pm_l_cosb']
        sigma['radial_velocity'] = vel_sigma

        ln_like = 0
        for i, name in enumerate(component_names[1:]):
            var = sigma[name]**2 + self.errs[name]**2
            ln_like += ln_normal(self.data[name],
                                 interps[name](self.data['l']),
                                 var)

        return ln_like

    def ln_likelihood_bg(self, p):
        # Transform pos/vel stddev to observables:
        vel_sigma = np.exp(p['log_bg_vel_sigma']) # km/s
        pos_sigma = np.exp(p['log_bg_pos_sigma']) # kpc

        sigma = dict()
        sigma['b'] = 180/np.pi * pos_sigma / self.data['distance']
        sigma['distance'] = pos_sigma
        sigma['pm_l_cosb'] = 1/4.74 * vel_sigma / self.data['distance']
        sigma['pm_b'] = sigma['pm_l_cosb']
        sigma['radial_velocity'] = vel_sigma

        ln_like = 0
        for i, name in enumerate(component_names[1:]):
            var = sigma[name]**2 + self.errs[name]**2
            ln_like += ln_normal(self.data[name],
                                 self._bg_mus[i],
                                 var)
        return ln_like

    def ln_likelihood(self, p):
        ll1 = self.ln_likelihood_fg(p) + np.log(p['f'])
        ll2 = self.ln_likelihood_bg(p) + np.log(1 - p['f'])
        return np.logaddexp(ll1, ll2), np.vstack((ll1, ll2)).T

    def ln_posterior(self, p):
        # unpack parameter vector, p
        kw_pars = self.unpack_pars(p)

        lnp = self.ln_prior(kw_pars)
        if not np.isfinite(lnp):
            return -np.inf, None

        lnl, blob = self.ln_likelihood(kw_pars)
        if not np.isfinite(lnl).all():
            return -np.inf, None

        return lnp + lnl.sum(), blob

    def __call__(self, p):
        return self.ln_posterior(p)
