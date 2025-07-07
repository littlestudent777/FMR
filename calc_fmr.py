import numpy as np
from math import acos, atan2
import sympy as sym
from abc import ABC, abstractmethod
from typing import List, Tuple


class Field(ABC):
    """
    Abstract base class for energy field calculations
    """
    def __init__(self):
        self._setup_symbols()
        self._setup_energy_expression()

    def _setup_symbols(self):
        """
        Initialize all necessary symbols
        """
        self.theta, self.phi = sym.symbols('theta phi', real=True)
        self.x, self.y, self.z = sym.symbols('x y z', real=True)
        self.Ms = sym.symbols('Ms', real=True, positive=True)

    @abstractmethod
    def _setup_energy_expression(self):
        """
        Set up the symbolic energy expression for this field
        """
        pass

    def _get_second_derivatives(self) -> Tuple[sym.Expr, sym.Expr, sym.Expr]:
        """
        Calculate second derivatives symbolically
        Returns: (G_theta_theta, G_phi_phi, G_theta_phi)
        """
        cartesian_to_spherical = {
            self.x: sym.sin(self.theta) * sym.cos(self.phi),
            self.y: sym.sin(self.theta) * sym.sin(self.phi),
            self.z: sym.cos(self.theta)
        }
        energy_sph = self.energy.subs(cartesian_to_spherical).simplify()

        G_theta_theta = sym.diff(energy_sph, self.theta, 2).simplify()
        G_phi_phi = sym.diff(energy_sph, self.phi, 2).simplify()
        G_theta_phi = sym.diff(energy_sph, self.theta, self.phi).simplify()

        return G_theta_theta, G_phi_phi, G_theta_phi

    def _get_subs_dict(self, theta: float, phi: float, Ms: float) -> dict:
        """
        Create substitution dictionary with common parameters
        """
        return {
            self.theta: theta,
            self.phi: phi,
            self.Ms: Ms
        }

    @abstractmethod
    def _add_field_specific_subs(self, subs_dict: dict):
        """
        Add field-specific substitutions to the dictionary
        """
        pass

    def calc_second_derivatives(self, theta: float, phi: float, Ms: float) -> Tuple[float, float, float]:
        """
        Numerically evaluate the second derivatives at given angles
        """
        subs_dict = self._get_subs_dict(theta, phi, Ms)
        self._add_field_specific_subs(subs_dict)

        G_tt, G_pp, G_tp = self._get_second_derivatives()

        try:
            G_tt_val = float(G_tt.subs(subs_dict).evalf())
            G_pp_val = float(G_pp.subs(subs_dict).evalf())
            G_tp_val = float(G_tp.subs(subs_dict).evalf())
        except TypeError as e:
            print(f"Error substituting values. Subs dict: {subs_dict}")
            print(f"Expressions: G_tt={G_tt}, G_pp={G_pp}, G_tp={G_tp}")
            raise ValueError(f"Cannot evaluate derivatives: {e}")

        return G_tt_val, G_pp_val, G_tp_val


class ExternalField(Field):
    def __init__(self, H: List[float]):
        """
        Parameters:
        ----------
        H : External field value [G]
        """
        self.H = H
        self.H1, self.H2, self.H3 = sym.symbols('H1 H2 H3', real=True)
        super().__init__()

    def _setup_energy_expression(self):
        """
        -Ms*(m·H)
        """
        self.energy = -self.Ms * (self.x * self.H1 + self.y * self.H2 + self.z * self.H3)

    def _add_field_specific_subs(self, subs_dict: dict):
        subs_dict.update({
            self.H1: self.H[0],
            self.H2: self.H[1],
            self.H3: self.H[2]
        })


class DemagnetizingField(Field):
    def __init__(self, N: List[float]):
        """
        Parameters:
        ----------
        N : Demagnetizing factors [dimensionless], sum(N[i]) = 1
        """
        self.N = N
        self.N1, self.N2, self.N3 = sym.symbols('N1 N2 N3', real=True, positive=True)
        super().__init__()

    def _setup_energy_expression(self):
        """
        Demagnetizing energy: 2*pi*Ms^2(N1x^2 + N2y^2 + N3z^2)
        """
        self.energy = 2 * sym.pi * self.Ms ** 2 * (
                self.N1 * self.x ** 2 + self.N2 * self.y ** 2 + self.N3 * self.z ** 2)

    def _add_field_specific_subs(self, subs_dict: dict):
        subs_dict.update({
            self.N1: self.N[0],
            self.N2: self.N[1],
            self.N3: self.N[2]
        })


class Anisotropy(Field):
    def __init__(self, K1_val: float = 4.2e5, K2_val: float = 1.5e5):
        """
        Parameters:
        ----------
        K1_val, K2_val : Cubic anisotropy constants [erg/cc] (default corresponds to Fe)
        """
        self.K1_val = K1_val
        self.K2_val = K2_val
        self.K1, self.K2 = sym.symbols('K1 K2', real=True)
        super().__init__()

    def _setup_energy_expression(self):
        """
        Cubic anisotropy energy: K1(x^2y^2 + y^2z^2 + z^2x^2) + K2x^2y^2z^2
        """
        self.energy = self.K1 * (self.x ** 2 * self.y ** 2 +
                                 self.y ** 2 * self.z ** 2 +
                                 self.z ** 2 * self.x ** 2) + \
                      self.K2 * self.x ** 2 * self.y ** 2 * self.z ** 2

    def _add_field_specific_subs(self, subs_dict: dict):
        subs_dict.update({
            self.K1: self.K1_val,
            self.K2: self.K2_val
        })


def calculate_total_second_derivatives(theta: float, phi: float,
                                       fields: List[Field], Ms: float) -> Tuple[float, float, float]:
    """
    Calculates the sum of the second derivatives of the energy for all fields.

    Parameters:
    ----------
    theta: Polar angle (in radians)
    phi: Azimuth angle (in radians)
    fields: List of field objects
    Ms: Saturation magnetization [emu/cc]
    Returns:
    -----------
    (G_theta_theta, G_phi_phi, G_theta_phi): Total second derivative values
    """
    G_theta_theta, G_phi_phi, G_theta_phi = 0.0, 0.0, 0.0

    for field in fields:
        g_tt, g_pp, g_tp = field.calc_second_derivatives(theta, phi, Ms)
        G_theta_theta += g_tt
        G_phi_phi += g_pp
        G_theta_phi += g_tp

    return G_theta_theta, G_phi_phi, G_theta_phi


def rotate(m: List[float], fields: List[Field]) -> Tuple[float, float, List[Field]]:
    """
    Rotates the coordinate system when m_z is close to +-1 (sin(theta) is close to 0).
    The rotation is performed by pi/2 around the y-axis, transforming z to x.

    Parameters:
    ----------
    m: Magnetization vector (dimensionless)
    fields: List of field objects (components that affect the sample)

    Returns:
    -----------
    (theta, phi, rotated_fields): New angles and fields parameters changed accordingly
    """
    # Rotate the magnetization vector: pi/2 around y-axis (z -> x), (x -> -z)
    m_new = [m[2], m[1], -m[0]]
    # Calculate new angles
    theta = acos(m_new[2])
    phi = atan2(m_new[1], m_new[0])

    # Rotate the fields
    new_fields = []
    for field in fields:
        if isinstance(field, ExternalField):
            # Rotate H: [Hx, Hy, Hz] → [Hz, Hy, -Hx]
            H_new = [field.H[2], field.H[1], -field.H[0]]
            rotated_field = ExternalField(H_new)
        elif isinstance(field, DemagnetizingField):
            # Rotate N: [N1, N2, N3] → [N3, N2, N1]
            N_new = [field.N[2], field.N[1], field.N[0]]
            rotated_field = DemagnetizingField(N_new)
        else:
            # Anisotropy remains unchanged
            rotated_field = field
        new_fields.append(rotated_field)

    return theta, phi, new_fields


def calculate_fmr_frequency(m: List[float], fields: List[Field],
                            Ms: float = 1707.0, gamma: float = 1.76e7) -> float:
    """
    Calculates the ferromagnetic resonance frequency.

    Parameters:
    ----------
    m: Magnetization vector (dimensionless)
    fields: List of field objects (components that affect the sample)
    Ms: Saturation magnetization [emu/cc] (default corresponds to Fe)
    gamma: Gyromagnetic ratio [rad/(s·G)] (default corresponds to electron)
    Returns:
    -----------
    FMR frequency in GHz
    """
    if np.isclose(abs(m[2]), 1):
        # Formula cannot be used in this case (sin(theta) = 0), so the axis rotation is applied
        theta, phi, fields = rotate(m, fields)
    else:
        theta = acos(m[2])
        phi = atan2(m[1], m[0])

    G_theta_theta, G_phi_phi, G_theta_phi = calculate_total_second_derivatives(theta, phi, fields, Ms)
    term = G_theta_theta * G_phi_phi - G_theta_phi ** 2

    if term < 0:
        raise ValueError("The radical expression is negative.")

    freq = (gamma / (Ms * np.sin(theta))) * np.sqrt(term)
    return freq * 1e-9 / (2 * np.pi)
