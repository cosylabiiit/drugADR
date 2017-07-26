import pandas as pd
import numpy as np
import pubchempy as pcp
import pickle

def stitch_to_pubchem(_id):
    assert _id.startswith("CID")
    return int(int(_id[3:]) - 1e8)

columns = [
    'stitch_id_flat',
    'stitch_id_sterio',
    'umls_cui_from_label',
    'meddra_type',
    'umls_cui_from_meddra',
    'side_effect_name',
]
df = pd.read_table('meddra_all_se.tsv', names=columns)
df.drop(df[df.meddra_type == "LLT"].index, inplace=True)
print (df.info())

df = df.groupby('stitch_id_flat').side_effect_name.apply(list).reset_index()
df['pubchem_id'] = df.stitch_id_flat.map(stitch_to_pubchem)
print (df.head())

val = []
vals = {}
for cx in range(df.shape[0]):
    c = pcp.Compound.from_cid(int(df.pubchem_id[cx]))
    vals['pubchem_id'] = df.pubchem_id[cx]
    vals['inchikey'] = c.inchikey
    vals['inchi'] = c.inchi.split("=")[1]
    if not c.canonical_smiles == c.isomeric_smiles:
        print (cx)
        vals['SMILES'] = c.canonical_smiles
    if c.canonical_smiles == c.isomeric_smiles:
        vals['SMILES'] = c.canonical_smiles
    if cx%10==0:
        print (cx, "Done")
    val.append(vals.copy())
id_df = pd.DataFrame(val.copy())
pickle.dump(id_df,open("../data/id_df.sav","wb"))

id_df = pickle.load(open("id_df.sav","rb"))
print (id_df.head())
