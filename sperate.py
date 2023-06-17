# From GPT's guide
from rdkit import Chem
from rdkit.Chem import SDWriter
suppl = Chem.SDMolSupplier("PubChem_compound_cache_8bZWu7oa36bojN2VX-2UvyusB8xC1tAaqj_LVrEu2VexN-U_records.sdf")

for mol in suppl:
    writer = SDWriter(f"{mol.GetProp('_Name')}.sdf")
    writer.write(mol)