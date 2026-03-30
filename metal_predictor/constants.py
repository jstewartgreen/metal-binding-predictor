RESIDUES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'UNK',
]
RES_TO_IDX = {r: i for i, r in enumerate(RESIDUES)}
IDX_TO_RES = {i: r for i, r in enumerate(RESIDUES)}

_ATOM14_COLS = [f'{c}{i}' for i in range(14) for c in ('x', 'y', 'z')]

DISTAL_CACHE_DIR = 'data/pt_cache_distal'

METAL_BINDERS = {'HIS', 'CYS', 'ASP', 'GLU'}

BARE_METAL_RESNAMES = {
    'ZN', 'MG', 'FE', 'CA', 'CU', 'MN', 'CO', 'NI', 'MO',
    'FE2', 'CD', 'V', 'W',
}
