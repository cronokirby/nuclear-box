import csv
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pint
from typing import Any

units = pint.UnitRegistry()
units.setup_matplotlib()

BINDING_FACTOR_VOLUME = 15.835 * units.megaelectron_volt
BINDING_FACTOR_SURFACE = 18.33 * units.megaelectron_volt
BINDING_FACTOR_CHARGE = 0.714 * units.megaelectron_volt
BINDING_FACTOR_SYMMETRY = 23.20 * units.megaelectron_volt
BINDING_FACTOR_PARITY = 11.20 * units.megaelectron_volt

ATOMIC_MASS_ENERGY_EQUIVALENCE = 931.49410242 * units.megaelectron_volt / units.dalton
PROTON_MASS = 1.0072765 * units.dalton
NEUTRON_MASS = 1.0086649 * units.dalton
ELECTRON_MASS = 0.00054858 * units.dalton

HYDROGEN_ISOTOPE_MASSES = [
    1.00782503223 * units.dalton,
    2.01410177812 * units.dalton,
    3.01604927790 * units.dalton,
]


@dataclass
class CalculatedType(Enum):
    Calculated = 1
    Estimated = 2
    Absent = 3

    @staticmethod
    def from_str(s):
        if s == "calculated":
            return CalculatedType.Calculated
        elif s == "estimated":
            return CalculatedType.Estimated
        else:
            return CalculatedType.Absent


@dataclass
class AtomicTableEntry:
    n: int
    z: int
    a: int
    element: str
    mass_excess: Any
    mass_excess_margin: Any
    mass_excess_calculated: CalculatedType
    binding_energy_per_nucleon: Any
    binding_energy_per_nucleon_margin: Any
    binding_energy_per_nucleon_calculated: CalculatedType
    beta_decay_energy: Any
    beta_decay_energy_margin: Any
    beta_decay_energy_calculated: CalculatedType
    atomic_mass: Any
    atomic_mass_margin: Any
    atomic_mass_margin_calculated: CalculatedType

    @staticmethod
    def from_row(row):
        energy_unit = 1e-3 * units.megaelectron_volt

        n = int(row[0])
        z = int(row[1])
        a = int(row[2])

        element = row[3]

        mass_excess = None
        mass_excess_margin = None
        mass_excess_calculated = CalculatedType.Absent
        if row[4] != "":
            mass_excess = float(row[4]) * energy_unit
            mass_excess_margin = float(row[5]) * energy_unit
            mass_excess_calculated = CalculatedType.from_str(row[6])

        binding_energy_per_nucleon = None
        binding_energy_per_nucleon_margin = None
        binding_energy_per_nucleon_calculated = CalculatedType.Absent
        if row[7] != "":
            binding_energy_per_nucleon = float(row[7]) * energy_unit
            binding_energy_per_nucleon_margin = float(row[8]) * energy_unit
            binding_energy_per_nucleon_calculated = CalculatedType.from_str(row[9])

        beta_decay_energy = None
        beta_decay_energy_margin = None
        beta_decay_energy_calculated = CalculatedType.Absent
        if row[10] != "":
            beta_decay_energy = float(row[10]) * energy_unit
            beta_decay_energy_margin = float(row[11]) * energy_unit
            beta_decay_energy_calculated = CalculatedType.from_str(row[12])

        atomic_mass = None
        atomic_mass_margin = None
        atomic_mass_calculated = CalculatedType.Absent
        if row[13] != "":
            atomic_mass = float(row[13]) * units.dalton
            atomic_mass_margin = float(row[14]) * units.dalton
            atomic_mass_calculated = CalculatedType.from_str(row[15])

        return AtomicTableEntry(
            n,
            z,
            a,
            element,
            mass_excess,
            mass_excess_margin,
            mass_excess_calculated,
            binding_energy_per_nucleon,
            binding_energy_per_nucleon_margin,
            binding_energy_per_nucleon_calculated,
            beta_decay_energy,
            beta_decay_energy_margin,
            beta_decay_energy_calculated,
            atomic_mass,
            atomic_mass_margin,
            atomic_mass_calculated,
        )


class AtomicTable:
    def __init__(self, entries):
        self.entries = entries

    @staticmethod
    def from_csv_file(file):
        entries = []
        with open(file, "r") as fp:
            reader = csv.reader(fp)
            next(reader)
            for row in reader:
                entries.append(AtomicTableEntry.from_row(row))
        return AtomicTable(entries)


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

    def approximate_binding_energy(self):
        """
        Returns the binding energy (in MeV) of this nuclide.

        The larger this quantity is, the more stable the nucleus is.

        The binding energy also explains additional mass not accounted
        for by counting the nucleons.
        """
        if self.protons == 1 and self.atomic_number <= 3:
            atomic_mass = self.neutrons() * NEUTRON_MASS + PROTON_MASS + ELECTRON_MASS
            return (
                atomic_mass - HYDROGEN_ISOTOPE_MASSES[self.atomic_number - 1]
            ) * ATOMIC_MASS_ENERGY_EQUIVALENCE
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

    def approximate_binding_energy_per_nucleon(self):
        return self.approximate_binding_energy() / self.atomic_number


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


def plot_ben():
    units.setup_matplotlib()
    atomic_numbers = np.linspace(1, 256, 256, dtype="int")
    likely_atomic_ben = (
        lambda a: Nuclide(minimal_binding_energy_protons(a), a)
        .approximate_binding_energy_per_nucleon()
        .magnitude
    )
    bens = np.vectorize(likely_atomic_ben)(atomic_numbers)
    fig, ax = plt.subplots()
    ax.plot(atomic_numbers, bens)
    plt.ylabel("Binding Energy Per Nucleon (MeV)")
    plt.xlabel("Atomic Number")

    os.makedirs("./images", exist_ok=True)
    plt.savefig("images/binding_energy_per_nucleon.png")


if __name__ == "__main__":
    plot_ben()
