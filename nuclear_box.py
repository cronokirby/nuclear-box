from dataclasses import dataclass
import pint
from typing import Any

units = pint.UnitRegistry()

BINDING_FACTOR_VOLUME = 15.835 * units.megaelectron_volt
BINDING_FACTOR_SURFACE = 18.33 * units.megaelectron_volt
BINDING_FACTOR_CHARGE = 0.714 * units.megaelectron_volt
BINDING_FACTOR_SYMMETRY = 23.20 * units.megaelectron_volt
BINDING_FACTOR_PARITY = 11.20 * units.megaelectron_volt

ATOMIC_MASS_ENERGY_EQUIVALENCE = 931.5 * units.megaelectron_volt / units.dalton
PROTON_MASS = 1.0072765 * units.dalton
NEUTRON_MASS = 1.0086649 * units.dalton
ELECTRON_MASS = 0.00054858 * units.dalton


@dataclass
class Nuclide:
    """
    Represents a Nuclide: a definite atomic isotope.

    A Nuclide is a basic kind of nucleus.
    """

    protons: int
    atomic_number: int

    def neutrons(self):
        """
        Return the number of neutrons in this nuclide.
        """
        return self.atomic_number - self.protons

    def binding_energy(self):
        """
        Returns the binding energy (in MeV) of this nuclide.

        The larger this quantity is, the more stable the nucleus is.

        The binding energy also explains additional mass not accounted
        for by counting the nucleons.
        """
        z_even = self.protons % 2 == 0
        n_even = self.neutrons() % 2 == 0
        parity = 1 if not z_even and not n_even else -1 if z_even and n_even else 0
        Z_f = float(self.protons)
        A_f = float(self.atomic_number)
        return (
            BINDING_FACTOR_VOLUME * A_f
            - BINDING_FACTOR_SURFACE * A_f ** (2 / 3)
            - BINDING_FACTOR_CHARGE * Z_f ** 2 / A_f ** (1 / 3)
            - BINDING_FACTOR_SYMMETRY * (A_f - 2 * Z_f) ** 2 / A_f
            - BINDING_FACTOR_PARITY * parity / A_f ** (1 / 2)
        )

    def binding_energy_mass(self):
        """
        The mass equivalent of the binding energy of this nuclide.
        """
        return self.binding_energy() / ATOMIC_MASS_ENERGY_EQUIVALENCE

    def atomic_mass(self):
        """
        A good estimate of the atomic mass of this nuclide, in daltons.
        """
        return (
            self.protons * (PROTON_MASS + ELECTRON_MASS)
            + self.neutrons() * NEUTRON_MASS
            # Because it takes energy to move from a grouped nucleus to separated
            # nucleons, moving them together must therefore release energy. Since
            # the total mass and energy of the system must be conserved, this released
            # energy results in a loss of mass inside the nucleus.
            - self.binding_energy_mass()
        )


def minimal_binding_energy_protons(atomic_number):
    A_f = float(atomic_number)
    Z_f = (
        atomic_number
        / 2
        * (
            1
            + (NEUTRON_MASS - PROTON_MASS)
            * ATOMIC_MASS_ENERGY_EQUIVALENCE
            / 4
            / BINDING_FACTOR_SYMMETRY
        )
        / (1 + BINDING_FACTOR_CHARGE * A_f ** (2 / 3) / 4 / BINDING_FACTOR_SYMMETRY)
    )
    return round(Z_f)


Hydrogen = Nuclide(1, 1)
Uranium235 = Nuclide(92, 235)
Uranium238 = Nuclide(92, 238)