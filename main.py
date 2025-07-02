from abc import ABC, abstractmethod
import numpy as np


class Field(ABC):
    """
    Abstract base class to calculate energy derivatives
    """
    @abstractmethod
    def calc_second_derivatives(self, theta: float, phi: float) -> tuple[float, float, float]:
        """
        Calculates the second derivatives of energy for a given field.

        Parameters:
        ----------
        theta : Polar angle
        phi : Azimuthal angle
        Returns:
        -----------
        [G_theta_theta, G_phi_phi, G_theta_phi] : Second derivative values
        """
        pass


class ExternalField(Field):
    def __init__(self, H: np.ndarray, Ms: float = 1707):
        """
        Parameters:
        ----------
        H : External field value [G]
        Ms : Saturation magnetization [emu/cc]
        """
        self.H = H
        self.Ms = Ms

    def calc_second_derivatives(self, theta: float, phi: float) -> tuple[float, float, float]:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        H1, H2, H3 = self.H

        G_theta_theta = self.Ms*(H1*sin_theta*cos_phi + H2*sin_phi*sin_theta + H3*cos_theta)
        G_phi_phi = self.Ms*(H1*cos_phi + H2*sin_phi)*sin_theta
        G_theta_phi = self.Ms*(H1*sin_phi - H2*cos_phi)*cos_theta

        return G_theta_theta, G_phi_phi, G_theta_phi


class DemagnetizingField(Field):
    def __init__(self, N: np.ndarray, Ms: float = 1707):
        """
        Parameters:
        ----------
        N : Demagnetizing factors [dimensionless], sum(N[i]) = 1
        Ms : Saturation magnetization [emu/cc]
        """
        self.N = N
        self.Ms = Ms

    def calc_second_derivatives(self, theta: float, phi: float) -> tuple[float, float, float]:
        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)
        N1, N2, N3 = self.N

        G_theta_theta = 8*np.pi*self.Ms**2*(-N1*sin_phi**2 + N1 + N2*sin_phi**2 - N3)*np.sin(theta + np.pi/4)*np.cos(theta + np.pi/4)
        G_phi_phi = 4*np.pi*self.Ms**2*(-N1 + N2)*sin_theta**2*np.cos(2*phi)
        G_theta_phi = np.pi*self.Ms**2*(-N1 + N2)*(np.cos(2*phi - 2*theta) - np.cos(2*phi + 2*theta))

        return G_theta_theta, G_phi_phi, G_theta_phi


class Anisotropy(Field):
    def __init__(self, K1: float = 4.2e5, K2: float = 1.5e5):
        """
        Parameters:
        ----------
        K1, K2 : Cubic anisotropy constant [erg/cc] (default corresponds to Fe)
        """
        self.K1 = K1
        self.K2 = K2

    def calc_second_derivatives(self, theta: float, phi: float) -> tuple[float, float, float]:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        G_theta_theta = (self.K1 * (
                          16*sin_phi**4*sin_theta**4 - 12*sin_phi**4*sin_theta**2 - 16*sin_phi**2*sin_theta**4 + 12*sin_phi**2*sin_theta**2 + 16*sin_theta**4 - 16*sin_theta**2 + 2) +
                         self.K2 * (
                          -36*sin_phi**4*sin_theta**6 + 46*sin_phi**4*sin_theta**4 - 12*sin_phi**4*sin_theta**2 + 36*sin_phi**2*sin_theta**6 - 46*sin_phi**2*sin_theta**4 + 12*sin_phi**2*sin_theta**2))
        G_phi_phi = sin_theta**4 * (
                    self.K1 *
                     (16*(1 - cos_phi**2)**2 + 16*cos_phi**2 - 14) +
                    self.K2 *
                     (16*(1 - cos_phi**2)**2*cos_theta**2 + 16*cos_phi**2*cos_theta**2 - 14*cos_theta**2))
        G_theta_phi = 4*sin_phi*sin_theta**3*cos_phi*cos_theta * (
                      self.K1 * (4*cos_phi**2 - 2) +
                      self.K2 * (6*cos_phi**2*cos_theta**2 - 2*cos_phi**2 - 3*cos_theta**2 + 1))

        return G_theta_theta, G_phi_phi, G_theta_phi


def calculate_total_second_derivatives(theta: float, phi: float, fields: list[Field]) -> tuple[float, float, float]:
    """
    Calculates the sum of the second derivatives of the energy for all fields.

    Parameters:
    ----------
    theta: Polar angle (in radians)
    phi: Azimuth angle (in radians)
    fields: List of field objects
    Returns:
    -----------
    (G_theta_theta, G_phi_phi, G_theta_phi): Second derivative values
    """
    G_theta_theta, G_phi_phi, G_theta_phi = 0.0, 0.0, 0.0

    for field in fields:
        g_tt, g_pp, g_tp = field.calc_second_derivatives(theta, phi)
        G_theta_theta += g_tt
        G_phi_phi += g_pp
        G_theta_phi += g_tp

    return G_theta_theta, G_phi_phi, G_theta_phi


def calculate_fmr_frequency(theta: float, phi: float, fields: list[Field],
                            gamma: float = 1.76e7, Ms: float = 1707) -> float:
    """
    Calculates the ferromagnetic resonance frequency.

    Parameters:
    ----------
    gamma: Gyromagnetic ratio [rad/(s·G)] (default corresponds to electron)
    Ms: Saturation magnetization [emu/cc] (default corresponds to Fe)
    theta: Equilibrium polar angle (in radians)
    phi: Equilibrium azimuth angle (in radians)
    fields: List of field objects (components that affect the sample)
    Returns:
    -----------
    FMR frequency [rad/s]
    """
    if np.isclose(np.sin(theta), 0):
        raise ValueError("θ is close to 0 or pi (sin(θ) is close to 0).\n This formula cannot be used.")

    G_theta_theta, G_phi_phi, G_theta_phi = calculate_total_second_derivatives(theta, phi, fields)
    term = G_theta_theta * G_phi_phi - G_theta_phi * G_theta_phi

    if term < 0:
        raise ValueError("The radical expression is negative.")

    return (gamma / (Ms * np.sin(theta))) * np.sqrt(term)
