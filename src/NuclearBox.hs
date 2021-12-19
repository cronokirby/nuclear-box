module NuclearBox where

import Data.Word (Word)

data Nuclide = Nuclide Word Word deriving (Eq, Show)

protons :: Nuclide -> Word
protons (Nuclide z _) = z

neutrons :: Nuclide -> Word
neutrons (Nuclide z a) = a - z

atomicNumber :: Nuclide -> Word
atomicNumber (Nuclide _ a) = a
