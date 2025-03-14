import pandas as pd
import re
import periodictable as pt
import mendeleev as m
import concurrent.futures
from tqdm import tqdm

VALID_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'
}


element_data = {}
for el in VALID_ELEMENTS:
    data = {}
    try:
        data["mass"] = pt.elements.symbol(el).mass
    except:
        data["mass"] = 0.0

    try:
        m_el = m.element(el)
        if m_el.ionic_radii and len(m_el.ionic_radii) > 0:
            data["ionic_radius_0"] = m_el.ionic_radii[0].ionic_radius or 0.0
        else:
            data["ionic_radius_0"] = 0.0

        data["en_pauling"] = m_el.en_pauling if m_el.en_pauling is not None else 0.0

        data["dipole_polarizability"] = (
            m_el.dipole_polarizability if m_el.dipole_polarizability is not None else 0.0
        )

        if m_el.ionenergies and 1 in m_el.ionenergies:
            data["ionization_energy_1"] = m_el.ionenergies[1]
        else:
            data["ionization_energy_1"] = 0.0

    except Exception:
        data["ionic_radius_0"] = 0.0
        data["en_pauling"] = 0.0
        data["dipole_polarizability"] = 0.0
        data["ionization_energy_1"] = 0.0

    element_data[el] = data


def extract_elements(formula: str):
    """zwraca symbole (tylko unikalne) pierwiastków we wzorze."""
    return set(re.findall(r'[A-Z][a-z]?', formula))

def parse_formula(formula: str):
    """
    Zwraca słownik {symbol pierwiastka: ilość elementów}
    """
    formula = re.sub(r'[+\-]\d*$', '', formula)
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    atom_counts = {}
    for element, count in matches:
        atom_counts[element] = atom_counts.get(element, 0) + (int(count) if count else 1)
    return atom_counts

def filter_unknown_elements(df: pd.DataFrame):
    """Filtruje df tak aby zawierał tylko wzory zawierające VALID_ELEMENTS"""
    def has_only_valid_elements(formula):
        elements = extract_elements(formula)
        return elements.issubset(VALID_ELEMENTS)
    return df[df['formula'].apply(has_only_valid_elements)].reset_index(drop=True)


def desc1(formula: str) -> int:
    """Liczba atomów w cząsteczce (number of atoms, nAT)"""
    return sum(parse_formula(formula).values())

def desc2(formula: str) -> int:
    """Liczba atomów w cząsteczce niebędących atomami wodoru (number of non-H atoms, nSK)"""
    atoms = parse_formula(formula)
    return sum(atoms.values()) - atoms.get("H", 0)

def desc3(formula: str) -> int:
    """ Liczba atomów węgla w cząsteczce (number of Carbon atoms, nC)"""
    return parse_formula(formula).get("H", 0)

def desc4(formula: str) -> int:
    """Liczba atomów azotu w cząsteczce (number of Nitrogen atoms, nN)"""
    return parse_formula(formula).get("C", 0)

def desc5(formula: str) -> int:
    """Liczba atomów tlenu w cząsteczce (number of Oxygen atoms, nO)"""
    return parse_formula(formula).get("N", 0)

def desc6(formula: str) -> int:
    """Liczba atomów fosforu w cząsteczce (number of Phosphorous, nP)"""
    return parse_formula(formula).get("O", 0)

def desc7(formula: str) -> int:
    """Liczba atomów siarki w cząsteczce (number of Sulfur atoms, nS)"""
    return parse_formula(formula).get("P", 0)

def desc8(formula: str) -> int:
    """Liczba atomów fluoru w cząsteczce (number of Fluorine atoms, nF)"""
    return parse_formula(formula).get("S", 0)

def desc9(formula: str) -> int:
    """Liczba atomów chloru w cząsteczce (number of Chlorine atoms, nCl)"""
    return parse_formula(formula).get("F", 0)

def desc10(formula: str) -> int:
    """Liczba atomów chloru w cząsteczce (number of Chlorine atoms, nCl)"""

    return parse_formula(formula).get("Cl", 0)

def desc11(formula: str) -> int:
    """Liczba atomów bromu w cząsteczce (number of Bromine atoms, nBr)"""
    return parse_formula(formula).get("Br", 0)

def desc12(formula: str) -> int:
    """Liczba atomów boru w cząsteczce (number of Boron atoms, nB)"""
    return parse_formula(formula).get("B", 0)

def desc13(formula: str) -> int:
    """Liczba heteroatomów w cząsteczce (number of heteroatoms, nHet, czyli
atomów niebędących atomami węgla i wodoru)"""
    atoms = parse_formula(formula)
    val = 0
    for el, count in atoms.items():
        if el not in ["H", "C"]:
            val += count
    return val

def desc14(formula: str) -> int:
    """Liczba atomów halogenu (tzn. suma atomów fluoru, chloru, bromu i jodu, number of halogen atoms, nX)"""
    atoms = parse_formula(formula)
    val = 0
    xatoms = ['F', 'Cl', 'Br', 'I']
    for el, count in atoms.items():
        if el in xatoms:
            val += count
    return val

def desc15(formula: str) -> float:
    """Jaki % całkowitej liczby atomów stanowią atomy wodoru (percentage of H atoms, H%)"""
    atoms = parse_formula(formula)
    total = sum(atoms.values())
    if total == 0:
        return 0
    return atoms.get("H", 0) / total

def desc16(formula: str) -> float:
    """Jaki % całkowitej liczby atomów stanowią atomy węgla (percentage of C atoms, C%)"""
    atoms = parse_formula(formula)
    total = sum(atoms.values())
    if total == 0:
        return 0
    return atoms.get("C", 0) / total

def desc17(formula: str) -> float:
    """Jaki % całkowitej liczby atomów stanowią atomy azotu (percentage of N atoms, N%)"""
    atoms = parse_formula(formula)
    total = sum(atoms.values())
    if total == 0:
        return 0
    return atoms.get("N", 0) / total

def desc18(formula: str) -> float:
    """Jaki % całkowitej liczby atomów stanowią atomy tlenu (percentage of O atoms, O%)"""
    atoms = parse_formula(formula)
    total = sum(atoms.values())
    if total == 0:
        return 0
    return atoms.get("O", 0) / total

def desc19(formula: str) -> float:
    """Jaki % całkowitej liczby atomów stanowią atomy halogenu (percentage of halogen atoms, X%)"""
    atoms = parse_formula(formula)
    total = sum(atoms.values())
    if total == 0:
        return 0
    xatoms = ['F', 'Cl', 'Br', 'I']
    x_count = sum(count for el, count in atoms.items() if el in xatoms)
    return x_count / total

def desc20(formula: str) -> float:
    """Masa cząsteczkowa (molecular weight, MW)"""
    atoms = parse_formula(formula)
    return sum(element_data[el]["mass"] * count for el, count in atoms.items())

def desc21(formula: str) -> float:
    """Średnia masa molowa (Average molecular weight, AMW)"""
    atoms = parse_formula(formula)
    total_atoms = sum(atoms.values())
    if total_atoms == 0:
        return 0.0
    mw = sum(element_data[el]["mass"] * count for el, count in atoms.items())
    return mw / total_atoms

def desc22(formula: str) -> float:
    """Suma promieni jonowych atomów tworzących cząsteczkę (sum of atomic vradious, Sr)"""
    atoms = parse_formula(formula)
    return sum(element_data[el]["ionic_radius_0"] * count for el, count in atoms.items())

def desc23(formula: str) -> float:
    """Suma elektroujemności Paulinga atomów tworzących cząsteczkę (sum of atomic Pauling electronegativities, Se)"""
    atoms = parse_formula(formula)
    return sum(element_data[el]["en_pauling"] * count for el, count in atoms.items())

def desc24(formula: str) -> float:
    """Suma polaryzowalności wszystkich atomów tworzących cząsteczkę (sum of atomic polarizabilities, Sp)"""
    atoms = parse_formula(formula)
    return sum(element_data[el]["dipole_polarizability"] * count for el, count in atoms.items())

def desc25(formula: str) -> float:
    """Suma energii jonizacji wszystkich atomów tworzących cząsteczkę (sum of first ionization potentials, Si)"""
    atoms = parse_formula(formula)
    return sum(element_data[el]["ionization_energy_1"] * count for el, count in atoms.items())

def desc26(formula: str) -> float:
    """Średni promień jonowy atomów tworzących cząsteczkę (mean atomic radius, Mv)"""
    atoms = parse_formula(formula)
    nAT = sum(atoms.values())
    if nAT == 0:
        return 0.0
    Sr = sum(element_data[el]["ionic_radius_0"] * count for el, count in atoms.items())
    return Sr / nAT

def desc27(formula: str) -> float:
    """Średnia elektroujemność atomów tworzących cząsteczkę (mean atomic Pauling electronegativity, Me)"""
    atoms = parse_formula(formula)
    nAT = sum(atoms.values())
    if nAT == 0:
        return 0.0
    Se = sum(element_data[el]["en_pauling"] * count for el, count in atoms.items())
    return Se / nAT

def desc28(formula: str) -> float:
    """Średnia polaryzowalność atomów tworzących cząsteczkę (mean atomic polarizability, Mp)"""
    atoms = parse_formula(formula)
    nAT = sum(atoms.values())
    if nAT == 0:
        return 0.0
    Sp = sum(element_data[el]["dipole_polarizability"] * count for el, count in atoms.items())
    return Sp / nAT

def desc29(formula: str) -> float:
    """Średnia energia jonizacji atomów tworzących cząsteczkę (mean first ionization potential, Mi)"""
    atoms = parse_formula(formula)
    nAT = sum(atoms.values())
    if nAT == 0:
        return 0.0
    Si = sum(element_data[el]["ionization_energy_1"] * count for el, count in atoms.items())
    return Si / nAT


def desc30(formula: str) -> float:
    """Wskaźnik stopnia nienasycenia (Hydrogen Deficiency Index, HDI)"""
    return (2 * desc4(formula) + 2 + desc5(formula) - desc3(formula) - desc14(formula))/2

def desc31(formula: str) -> float:
    """Proporcja węgla do wodoru (Hydrogen/Carbon ratio, H/C)"""
    x = desc4(formula)
    return desc3(formula)/x if x != 0.0 else 0.0

def desc32(formula: str) -> float:
    """Średnia % zawartości pierwiastków (IDK)"""
    x = parse_formula(formula)
    every = desc1(formula)

    sum = 0

    for el, count in x.items():
        sum += count/every

    return sum/len(x)


def get_descriptors(data: pd.DataFrame) -> pd.DataFrame:
    """
    pobiera dataset z dysku i oblicza wszystkie deskryptory dla wszystkich związków chemicznych w danych wyrzucając
    te, które zawierają nieznane dla algorytmu pierwiastki
    """
    new = filter_unknown_elements(data)

    functions_1_21 = [
        desc1,  desc2,  desc3,  desc4,  desc5,
        desc6,  desc7,  desc8,  desc9,  desc10,
        desc11, desc12, desc13, desc14, desc15,
        desc16, desc17, desc18, desc19, desc20,
        desc21
    ]
    columns_1_21 = [
        'nAT', 'nSK', 'nH', 'nC', 'nN',
        'nO', 'nP', 'nS', 'nF', 'nCl',
        'nBr', 'nB', 'nHet', 'nX', 'H%',
        'C%', 'N%', 'O%', 'X%', 'MW',
        'AMW'
    ]

    functions_22_29 = [
        desc22, desc23, desc24, desc25,
        desc26, desc27, desc28, desc29,
        desc30, desc31, desc32
    ]
    columns_22_29 = [
        'Sr', 'Se', 'Sp', 'Si',
        'Mv', 'Me', 'Mp', 'Mi',
        'HDI', 'H/C', 'IDK'
    ]

    for col, func in zip(columns_1_21, functions_1_21):
        new[col] = [
            func(formula) for formula in tqdm(
                new['formula'],
                desc=f"Calculating {col}",
                total=len(new)
            )
        ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_col = {
            executor.submit(
                lambda f=func: [f(formula) for formula in new['formula']]
            ): col
            for col, func in zip(columns_22_29, functions_22_29)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_col),
            desc="Calculating descriptors 22–29 (async)",
            total=len(future_to_col)
        ):
            col = future_to_col[future]
            new[col] = future.result()

    return new


def get_one(formula: str) -> pd.DataFrame:
    """
    Zwraca DataFrame z podaną formułą oraz wszystkimi deskryptorami.
    """
    descriptors = {
        'formula': formula,
        'nAT': desc1(formula),
        'nSK': desc2(formula),
        'nH': desc3(formula),
        'nC': desc4(formula),
        'nN': desc5(formula),
        'nO': desc6(formula),
        'nP': desc7(formula),
        'nS': desc8(formula),
        'nF': desc9(formula),
        'nCl': desc10(formula),
        'nBr': desc11(formula),
        'nB': desc12(formula),
        'nHet': desc13(formula),
        'nX': desc14(formula),
        'H%': desc15(formula),
        'C%': desc16(formula),
        'N%': desc17(formula),
        'O%': desc18(formula),
        'X%': desc19(formula),
        'MW': desc20(formula),
        'AMW': desc21(formula),
        'Sr': desc22(formula),
        'Se': desc23(formula),
        'Sp': desc24(formula),
        'Si': desc25(formula),
        'Mv': desc26(formula),
        'Me': desc27(formula),
        'Mp': desc28(formula),
        'Mi': desc29(formula),
        'HDI': desc30(formula),
        'H/C': desc31(formula),
        'IDK': desc32(formula)
    }

    return pd.DataFrame([descriptors])



if __name__ == "__main__":
    input_df = pd.read_csv("dataset.csv")
    descriptors_df = get_descriptors(input_df)
    descriptors_df.to_csv("descriptors.csv", index=False)
