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


def rotate(m: list[float], fields: list[Field]) -> tuple[float, float, list[Field]]:
    """
    Rotates the coordinate system when m[2] is close to ±1 (sin(theta) is close to 0).
    The rotation is performed by pi/2 around the y-axis, transforming z to x.

    Parameters:
    ----------
    m: Magnetization vector
    fields: List of field objects (components that affect the sample)

    Returns:
    -----------
    (theta, phi, rotated_fields): New angles and rotated fields
    """
    # Rotate the magnetization vector: pi/2 around y-axis (z → x), (x → -z)
    m_new = np.array([m[2], m[1], -m[0]])

    # Calculate new angles
    theta = np.arccos(m_new[2])
    phi = np.arctan2(m_new[1], m_new[0])

    # Rotate the fields
    new_fields = []
    for field in fields:
        if isinstance(field, ExternalField):
            # Rotate H: [Hx, Hy, Hz] → [Hz, Hy, -Hx]
            H_new = np.array([field.H[2], field.H[1], -field.H[0]])
            rotated_field = ExternalField(H_new, field.Ms)
        elif isinstance(field, DemagnetizingField):
            # Rotate N: [N1, N2, N3] → [N3, N2, N1]
            N_new = np.array([field.N[2], field.N[1], field.N[0]])
            rotated_field = DemagnetizingField(N_new, field.Ms)
        else:
            # Anisotropy remains unchanged
            rotated_field = field
        new_fields.append(rotated_field)

    return theta, phi, new_fields


def calculate_fmr_frequency(m: list[float], fields: list[Field],
                            gamma: float = 1.76e7, Ms: float = 1707) -> float:
    """
    Calculates the ferromagnetic resonance frequency.

    Parameters:
    ----------
    gamma: Gyromagnetic ratio [rad/(s·G)] (default corresponds to electron)
    Ms: Saturation magnetization [emu/cc] (default corresponds to Fe)
    fields: List of field objects (components that affect the sample)
    Returns:
    -----------
    FMR frequency [rad/s]
    """
    if np.isclose(abs(m[2]), 1):
        # Formula cannot be used in this case (sin(theta) = 0), so the axis rotation is applied
        theta, phi, fields = rotate(m, fields)
    else:
        theta = np.arccos(m[2])
        phi = np.arctan2(m[1], m[0])

    G_theta_theta, G_phi_phi, G_theta_phi = calculate_total_second_derivatives(theta, phi, fields)
    term = G_theta_theta * G_phi_phi - G_theta_phi * G_theta_phi

    if term < 0:
        raise ValueError("The radical expression is negative.")

    return (gamma / (Ms * np.sin(theta))) * np.sqrt(term)


