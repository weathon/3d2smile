# From GPT's guide
from rdkit import Chem
from rdkit.Chem import SDWriter
suppl = Chem.SDMolSupplier("PubChem_compound_cache_LWqKZnPPFnMhWRRAljhdauJMfSxpD3VjD0ZuLxRXfC4UTkA_records.new.sdf")

for mol in suppl:
    writer = SDWriter(f"{mol.GetProp('_Name')}.sdf")
    writer.write(mol)