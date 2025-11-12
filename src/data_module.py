import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import *
from tokenizer import *
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from tqdm import tqdm

def get_tokenizer_device(tokenizer_device) -> torch.device:
    """Get CPU or local GPU
    """
    if tokenizer_device == "cuda":
        gpu_idx = 0
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        device = torch.device(f"{tokenizer_device}:{gpu_idx}")
    elif tokenizer_device == "cpu":
        device = torch.device(tokenizer_device)
    return device

class ProteinDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer_name: str, tokenizer_device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only: bool, 
        precompute_tokens: bool, tokenizer_kwargs: dict,
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer_device = tokenizer_device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only
        self.precompute_tokens = precompute_tokens

        if self.test_only:
            self.all_split_names = []
        else:
            self.all_split_names = ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]
        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {} 

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        pass

    def _setup_tokenizer(self):
        """Initialize the tokenizer on appropriate device. 
        """
        device = get_tokenizer_device(self.tokenizer_device)
        if self.tokenizer_name.startswith("Wrapped"):
            # all Wrapped tokenizers deal with loading logic inside __init__() when built up
            # assume only this type of tokenizer needs to be device aware
            tokenizer = eval(self.tokenizer_name)(device=device, **self.tokenizer_kwargs)
        else:
            raise NotImplementedError
        
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer
    
    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.tokenizer_device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer

    def get_codebook_embedding(self,):
        tokenizer = self.get_tokenizer()
        return tokenizer.get_codebook_embedding()

    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        # need to shard the dataset here:
        assert torch.distributed.is_initialized()
        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # dataset.shard(shard_idx=process_global_rank, num_shards=world_size)
    
        if self.precompute_tokens:
            # precompute and cache the token ids:
            self.py_logger.info(
                f"Precomputing tokenized ids on {process_global_rank} with world size {world_size}..."
            )
            dataset.cache_all_tokenized()
        
        if dataset.data_name not in ["ConformationalSwitchDataset", "CASP14Dataset", "CAMEODataset"]:
            for i in tqdm(range(len(dataset.data))):
                assert len(dataset.data[i]["real_seqs"]) == len(dataset.data[i]["token_ids"])


        ## DELETE THIS. for processing   
        # PRO TIP: get token encoder to return dummy tokens without inferencing the encoder, see ./tokenizer.py
        from pathlib import Path
        import json
        data_name = self.data_args.data_name
        target_field = self.data_args.target_field
        out_path = Path(__file__).parents[2] / f"data/struct_token_bench/{data_name}_{target_field}_{split}.jsonl"
        assert len(dataset) == len(dataset.data)
        if data_name in ["ProteinGLUEEpitopeRegionDataset", "ProteinShakeBindingSiteDataset", "InterProFunctionDataset", "BioLIP2FunctionDataset", "TapeRemoteHomologyDataset", "AtlasDataset"]:
            if data_name == "ProteinGLUEEpitopeRegionDataset":
                store_dir = "proteinglue"
            elif data_name == "ProteinShakeBindingSiteDataset":
                store_dir = "proteinshake"
            elif data_name == "InterProFunctionDataset":
                if target_field == "binding_label":
                    store_dir = "interpro/binding"
                elif target_field == "activesite_label":
                    store_dir = "interpro/activesite"
                elif target_field == "conservedsite_label":
                    store_dir = "interpro/conservedsite"
                elif target_field == "repeat_label":
                    store_dir = "interpro/repeat"
                else:
                    raise NotImplementedError
            elif data_name == "BioLIP2FunctionDataset":
                if target_field == "binding_label":
                    store_dir = "biolip2/binding"
                elif target_field == "catalytic_label":
                    store_dir = "biolip2/catalytic"                
                else:
                    raise NotImplementedError
            elif data_name == "TapeRemoteHomologyDataset":
                if target_field == "fold_label":
                    store_dir = "homo"
                else:
                    raise NotImplementedError
            elif data_name == "AtlasDataset":
                store_dir = "atlas"
            else:
                raise NotImplementedError
            assert len(dataset[0]) == 3
            assert 'pdb_chain' in dataset.data[0]
            if data_name != "AtlasDataset":
                assert all(item['pdb_id'] == item['pdb_chain'].id for item in dataset.data)
                assert all(item['chain_id'] == item['pdb_chain'].chain_id for item in dataset.data)
            with open(out_path, "w+") as f:
                for item, sample in zip(dataset.data, dataset):
                    pdb_id, chain_id = item['pdb_id'], item['chain_id']
                    path = f"{Path(__file__).parents[2]}/data/struct_token_bench/{store_dir}/{pdb_id}_{chain_id}.pdb"
                    item['pdb_chain'].to_pdb(path)
                    if 'residue_index' in item:
                        if len(item['residue_index']) != len(sample[1]):
                            breakpoint()
                        fields = {
                            "pdb_path": path,
                            "residue_index": item['residue_index'],
                            target_field: sample[1]
                        }
                    else:
                        if target_field != "fold_label" and len(item['pdb_chain']) != len(sample[1]):
                            breakpoint()
                        fields = {
                            "pdb_path": path,
                            target_field: sample[1],
                            "residue_index": item['pdb_chain'].residue_index.tolist()
                        }                        
                    if item['residue_range'] != ['']:
                        fields['residue_range'] = item['residue_range']
                    try:
                        f.write(json.dumps(fields) + "\n")
                    except:
                        breakpoint()
        elif data_name == 'ConformationalSwitchDataset':
            assert all(x['prot1_residue_range'] == [''] for x in dataset.data)
            assert all(x['prot2_residue_range'] == [''] for x in dataset.data)
            assert all(item['prot1_pdb_id'] == item['prot1_pdb_chain'].id for item in dataset.data)
            assert all(item['prot1_chain_id'] == item['prot1_pdb_chain'].chain_id for item in dataset.data)
            assert all(item['prot2_pdb_id'] == item['prot2_pdb_chain'].id for item in dataset.data)
            assert all(item['prot2_chain_id'] == item['prot2_pdb_chain'].chain_id for item in dataset.data)            
            with open(out_path, "w+") as f:
                for item, sample in zip(dataset.data, dataset):
                    f.write(json.dumps({
                        "prot1_pdb_id": item["prot1_pdb_id"],
                        "prot1_chain_id": item["prot1_chain_id"],
                        "prot2_pdb_id": item["prot2_pdb_id"],
                        "prot2_chain_id": item["prot2_chain_id"],                        
                        target_field: sample[2]
                    }) + "\n")
        else:
            breakpoint()
        ## end        
        return dataset

    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None
        
        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders


class PretrainingDataModule(pl.LightningDataModule):

    def __init__(self, device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only, train_eval
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only

        self.all_split_names = []
        if not self.test_only:
            self.all_split_names += ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]
        self.all_split_names += ["train"]
        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {}
    
    def _setup_tokenizer(self):
        device = get_tokenizer_device(self.device)
        tokenizer = EsmSequenceTokenizer()
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer

    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer 
    
    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "seq_tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        ## DELETE THIS. for processing
        dest_dir = f"{Path(__file__).parents[2]}/data/vqvae_pretrain/{split}"
        os.makedirs(dest_dir, exist_ok=True)
        if split != "train":
            count = 0
            for x in tqdm(dataset.data, desc=f"writing to {dest_dir}"):
                chain = x['pdb_chain']
                key = f"{chain.id}_{chain.chain_id}"
                # 500 bad
                try:
                    chain.to_pdb(f"{dest_dir}/{key}.pdb")
                    count += 1
                except:
                    continue
            print(f"{count}/{len(dataset)} written to {dest_dir}")
        ## end
        # need to shard the dataset here:
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            # dataset.shard(shard_idx=process_global_rank, num_shards=world_size)
        return dataset
    
    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None

        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        split_names = self.all_split_names
        for split in split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=(split=="train"),
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders

    def setup(self, stage=None):
        pass
