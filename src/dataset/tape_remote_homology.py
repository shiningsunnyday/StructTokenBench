import os
from tqdm import tqdm
import time

import pandas as pd
import torch
import torch.distributed as dist

from tape.datasets import RemoteHomologyDataset
from dataset.base import BaseDataset


class TapeRemoteHomologyDataset(RemoteHomologyDataset, BaseDataset):
    """Extending tape.RemoteHomologyDataset to structure-based datasets
    """

    SPLIT_NAME = {
        "test": ["test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]
    }

    SCOP_CLASSIFICATION_FILE = "remote_homology/dir.cla.scop.1.75.txt"
    SCOP_CLASSIFICATION_FIELDS = ["sid", "pdb_id", "chain_description", "sccs", "sunid", "ancestor_sunid_list"]
    # sccs. SCOP(e) concise classification string. 
    # sunid. SCOP(e) unique identifier, used to reference any entry in the SCOP(e) hierarchy, from root to leaves (Fold, Superfamily, Family, etc.).
    # sid. Stable domain identifier. A 7-character sid consists of "d" followed by the 4-character PDB ID of the file of origin and the PDB chain ID.
    # cl - class; cf - fold; sf - superfamily; fa - family; dm - protein; sp - species; px - domain

    def get_target_file_name(self, ):
        return os.path.join(self.data_path, f"remote_homology/processed_structured_{self.split}")

    def __init__(self, *args, **kwargs):
        if kwargs["split"] == "validation":
            kwargs["split"] = "valid" # required by tape
        BaseDataset.__init__(self, *args, **kwargs)
    
    def process_data_from_scratch(self, *args, **kwargs):
        # calling TAPE's data processing from RemoteHomologyDataset()
        super().__init__(data_path=self.data_path, split=self.split, 
            tokenizer=kwargs["tokenizer"], in_memory=kwargs["in_memory"])
        # the current self.data is a LMDBDataset
        # type(self.data): <class 'tape.datasets.LMDBDataset'>
        # transform it to list of dicts
        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i])
        self.data = new_data

    def collate_fn(self, batch):
        return BaseDataset.collate_fn(self, batch)
    
    def update_obsolete_pdb_id(self, pdb_id):

        if not hasattr(self, "obsolete_mapping"):
            obsolete_list_file = os.path.join(self.PDB_DATA_DIR, "obsolete.dat")
            obsolete_list = pd.read_csv(obsolete_list_file, sep="    ", skiprows=1, header=None)
            obsolete_list.columns = ["obs", "date_and_old_id", "new_id"]
            obsolete_list["old_id"] = obsolete_list["date_and_old_id"].apply(lambda y: y.split(" ")[1])
            obsolete_list["new_id"] = obsolete_list["new_id"].apply(lambda y: y.strip() if y is not None else None)
            self.obsolete_mapping = obsolete_list.set_index("old_id").to_dict()["new_id"]
        
        if pdb_id.upper() in self.obsolete_mapping and self.obsolete_mapping[pdb_id.upper()] is not None:
            new_pdb_id = self.obsolete_mapping[pdb_id.upper()].lower()
            return new_pdb_id
        else:
            return pdb_id

    def load_structure(self, idx, cnt_stats):
        """Using mapping from scop_id to pdb_id, together with chain_id and 
        residue_range constraints, we retrieve the structures
        """
        # idx: the index for data in self.data
        pdb_id, chain_id, residue_range, pdb = None, None, None, None

        scop_id = str(self.data[idx]["id"], "utf-8")
        try:
            pdb_id = self.scop_pdb_mapping[scop_id]
            chain_id = self.scop_chain_mapping[scop_id]
        except:
            cnt_stats["cnt_scop_missing"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        if "," in chain_id:
            cnt_stats["cnt_fragmented"] += 1

        # chain and residue specified
        if ":" in chain_id:
            tmp = chain_id.split(",")
            chain_id = [v.split(":")[0] for v in tmp]
            assert len(set(chain_id)) == 1
            chain_id = chain_id[0]
            residue_range = [v.split(":")[1].strip("-") for v in tmp]
            
            # convert residue ranges to relative positions per chain
            if residue_range[0] != "": # length indicated by SCOP
                # redundant character attached
                try:
                    total_length = sum([eval(x.split("-")[1]) - eval(x.split("-")[0]) + 1 for x in residue_range])
                except:
                    cnt_stats["cnt_redundant_char"] += 1
                residue_range = ["-".join([x.split("-")[0].strip("ABPS"), x.split("-")[1].strip("ABPS")]) for x in residue_range]

                # length mis-match
                total_length = sum([eval(x.split("-")[1]) - eval(x.split("-")[0]) + 1 for x in residue_range])

                if total_length != self.data[idx]["protein_length"]:
                    cnt_stats["cnt_length_missmatch"] += 1
        else:
            residue_range = [""]

        pdb_id = self.update_obsolete_pdb_id(pdb_id)
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE
        
        return {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain, 
        }
        

    def prepare_structure_loading(self):
        """Get `scop_pdb_mapping` and `scop_chain_mapping`
        """

        data_file = os.path.join(self.data_path, self.SCOP_CLASSIFICATION_FILE)
        
        scop_df = pd.read_csv(data_file, sep="\t", comment="#", header=None)
        scop_df.columns = self.SCOP_CLASSIFICATION_FIELDS
        scop_dict = scop_df.set_index("sid").to_dict()
        
        self.scop_pdb_mapping = scop_dict["pdb_id"]
        self.scop_chain_mapping = scop_dict["chain_description"]
    
    def _get_init_cnt_stats(self,):
        cnt_stats = {
            "cnt_fragmented": 0,
            "cnt_scop_missing": 0,
            "cnt_length_missmatch": 0,
            "cnt_redundant_char": 0,
            "cnt_return_none": 0,
            "cnt_wrong_residue_range": 0,
        }
        return cnt_stats

    def __getitem__(self, index: int):
        assert self.target_field == "fold_label"
        return BaseDataset.__getitem__(self, index)

