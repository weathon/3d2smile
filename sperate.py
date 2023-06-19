# From GPT's guide
from rdkit import Chem
from rdkit.Chem import SDWriter
suppl = Chem.SDMolSupplier("main.sdf")

for mol in suppl:
    if len(Chem.GetMolFrags(mol)) != 1:
        print(len(Chem.GetMolFrags(mol)))
        continue
    writer = SDWriter(f"SDFs/{mol.GetProp('_Name')}.sdf")
    writer.write(mol)