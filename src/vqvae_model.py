import os
import math
import time
import json
import hydra
from collections import defaultdict
from pathlib import Path
import pytorch_lightning as pl
import torch.distributed as dist
import torch
from torch import nn
import safetensors
import time
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoConfig, AdamW, EsmModel
from torch.optim import Adam
import torch.nn.functional as F
import deepspeed
from pytorch_lightning.utilities import grad_norm
import shutil
import subprocess, signal
from esm.layers.structure_proj import Dim6RotStructureHead
from esm.utils.constants import esm3 as C
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)
from esm.utils.structure.protein_structure import infer_cbeta_from_atom37

from modeling_util import model_init_fn
from vqvae.quantizer_module import *
from util import get_optimizer

from vqvae.blocks import VanillaUnifiedTransformerBlock
from vqvae.transformer_stack import VanillaTransformerStack
from protein_chain import WrappedProteinChain


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)

class VanillaRelativePositionEmbedding(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L20C1-L53C1
    """

    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins

        self.embedding = torch.nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        """
        Input:
          query_residue_index: (B, ) tensor of source indices (dytpe=torch.long)
          key_residue_index: (B, L) tensor of target indices (dytpe=torch.long)
        Output:
          embeddings: B x L x embedding_dim tensor of embeddings
        """

        assert query_residue_index.dtype == torch.long
        assert key_residue_index.dtype == torch.long
        assert query_residue_index.ndim == 1
        assert key_residue_index.ndim == 2

        # key_residue_index: [B * L, 16]
        # query_residue_index: [B * L, ]

        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index
        output = self.embedding(diff) # [B * L, 16, d_model=1024]
        return output

class VanillaGeometricEncoderStack(VanillaTransformerStack):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L148C1-L166C1
    """
    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList(
            [
                VanillaUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    expansion_ratio=4,
                    bias=True,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.Identity()


class VanillaStructureTokenEncoder(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L185
    """
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        # We only support fully-geometric structure token encoders for now...
        # setting n_layers_geom to something that's not n_layers won't work because
        # sequence ID isn't supported fully in this repo for plain-old transformers
        self.transformer = VanillaGeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.relative_positional_embedding = VanillaRelativePositionEmbedding(
            32, d_model, init_std=0.02
        )
        self.knn = 16
        self.d_out = d_out

    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )
            B, L, E = knn_edges.shape
            knn_edge_mask = knn_edge_mask.view(-1, E) # (B * L, 16)

            affine_tensor = affine.tensor  # for easier manipulation # [B, L, 12]
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges) # [B, L, 16, 12]
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous() # [B * L, 16, 12]
            affine = Affine3D.from_tensor(knn_affine_tensor) # [B * L, 16]
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]

            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask)

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            ) # [B * L, 16]
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask)

            knn_chain_id = torch.zeros(
                B * L, E, dtype=torch.int64, device=coords.device
            ) # [B * L, 16]

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                ) # [B * L, 16]

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs) # [B * L, 16, d_model]

        z, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        ) # [B * L, 16, d_model]

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1) # [B, L, 16, d_model]
        z = z[:, :, 0, :] # [B, L, d_model]

        return z

    @staticmethod
    def find_knn_edges(
        coords,
        padding_mask,
        coord_mask,
        sequence_id: torch.Tensor | None = None,
        knn: int | None = None,
    ) -> tuple:
        assert knn is not None, "Must specify a non-null knn to find_knn_edges"
        # Coords are N, CA, C
        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long() # [B, L]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca,
                coord_mask,
                padding_mask,
                sequence_id,
                no_knn=knn,
            )

        return edges, edge_mask # [B, L, 16], [B, L, 16]
        # edges: residue indices whose structural distance is minimized within top 16;
        # if the structural distance is masked out, use the sequence distance
        # edge_mask: True for attending, False for not

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :] # -> [B, L, 3, 3]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords) # affine: [B, L], affine_mask: [B, L]

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        ) # [B, L, d_model]

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0) # [B, L, d_model]
        z = self.pre_vq_proj(z) # [B, L, d_out]

        return z


class VanillaCategoricalMixture:
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L120C1-L146C1
    """
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(
            start, end, bins + 1, device=self.logits.device, dtype=torch.float32
        )
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ... (floating point # for target)
        true_index = (
            (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        )
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (
            self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)
        ).squeeze(-1)

    def median(self):
        return self.v_bins[self.logits.max(-1).indices]


class VanillaPairwisePredictionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L55
    """
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        n_bins: int,
        bias: bool = True,
        pairwise_state_dim: int = 0,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(
            downproject_dim + pairwise_state_dim, hidden_dim, bias=bias
        )
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [
            prod,
            diff,
        ]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class VanillaRegressionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L104
    """
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x

class VanillaStructureTokenDecoder(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L335
    """
    def __init__(
        self,
        encoder_d_out,
        d_model,
        n_heads,
        n_layers,
    ):
        super().__init__()
        self.decoder_channels = d_model

        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN

        self.post_vq_proj = nn.Linear(encoder_d_out, d_model)
        self.decoder_stack = VanillaTransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0
        )

        self.affine_output_projection = Dim6RotStructureHead(
            self.decoder_channels, 10, predict_torsion_angles=False
        )

        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [
            64,  # distogram
            direction_loss_bins * 6,  # direction bins
            pae_bins,  # predicted aligned error
        ]
        self.pairwise_classification_head = VanillaPairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = VanillaRegressionHead(
            embed_dim=self.decoder_channels, output_dim=plddt_bins
        )

    def decode(
        self,
        quantized_z: torch.Tensor,
        structure_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ):
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        x = self.post_vq_proj(quantized_z) # [B, L, hidden_dim=128] -> [B, L, d_model=1024]
        # !!! NOTE: Attention mask is actually unused here so watch out
        x, _ = self.decoder_stack.forward(
            x, attention_mask=attention_mask, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
        ) # [B, L, d_model], [B, L, d_model]

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        ) # [B, L, 12], [B, L, 3, 3]

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x) # [B, L, L, 64 + 96 + 64]
        pairwise_dist_logits, pairwise_dir_logits, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ] # [B, L, L, 64], [B, L, L, 96], [B, L, L, 64]

        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            sequence_id=sequence_id,
            max_bin=self.max_pae_bin,
        ) # [B, L, L]
        # This might be broken for chainbreak tokens? We might align to the chainbreak
        ptm = compute_tm(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            max_bin=self.max_pae_bin,
        ) # [B,]

        plddt_logits = self.plddt_head(x) # [B, L, 50]
        plddt_value = VanillaCategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean() # [B, L]

        return dict(
            tensor7_affine=tensor7_affine,
            bb_pred=bb_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
            pairwise_dist_logits=pairwise_dist_logits,
            pairwise_dir_logits=pairwise_dir_logits,
            last_hidden_state=x,
        )


class VQVAEModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        quantizer_cfg = model_cfg.quantizer
        self.loss_weight = quantizer_cfg["loss_weight"]
        self.quantizer = eval(quantizer_cfg["quantizer_type"])(**quantizer_cfg)

        self.encoder = VanillaStructureTokenEncoder(
            **model_cfg.encoder,
            n_codes=quantizer_cfg.codebook_size
        ) # encoder_d_out not necessarily the same as self.codebook_embed_size
        model_cfg.decoder["encoder_d_out"] = model_cfg.encoder.d_out
        self.decoder = VanillaStructureTokenDecoder(**model_cfg.decoder)

        self.inverse_folding_head = VanillaRegressionHead(
            embed_dim=model_cfg.decoder.d_model, 
            output_dim=len(C.SEQUENCE_VOCAB)
        )
        self.param_size = sum((np.prod(p.shape) for p in self.decoder.parameters())) + sum((np.prod(p.shape) for p in self.quantizer.parameters()))
        self._step_count = 0

    def forward(self, input_list, use_as_tokenizer=False):
        self._step_count += 1
        coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list
        sequence_id = None
        """
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        residue_index: [B, L]
        seq_residue_tokens: [B, L]
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask # NOTE: due to data loading processing
        attention_mask = attention_mask.bool()

        # coords: torch.Tensor,
        # attention_mask: torch.Tensor | None = None,
        # sequence_id: torch.Tensor | None = None,
        # residue_index: torch.Tensor | None = None,
        z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index)
        assert self.quantizer.codebook_embed_size == self.encoder.d_out
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
        assert not z.isnan().any() and not quantized_indices.isnan().any()
        if use_as_tokenizer:
            return quantized_z, quantized_indices, z
        decoded_states = self.decoder.decode(quantized_z, quantized_indices, attention_mask, sequence_id)

        # reconstructed proteins
        bb_pred = decoded_states["bb_pred"]
        bb_rmsd_list, lddt_list = [], []
        for i in range(len(bb_pred)):
            pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[i].detach())
            pdb_chain_recon = pdb_chain_recon[:len(pdb_chain[i])]
            bb_rmsd = pdb_chain_recon.rmsd(pdb_chain[i], only_compute_backbone_rmsd=True)
            lddt = np.array(pdb_chain_recon.lddt_ca(pdb_chain[i]))
            bb_rmsd_list.append(bb_rmsd)
            lddt_list.append(lddt.mean())
        
        # reconstruction loss: 
        coords_recon = decoded_states["bb_pred"]
        # (1) backbone geometric distance loss: pairwise L2 distance matrix for 
        # the predicted and true coordinates of the 3 backbone atoms (N, Cα, C)
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            coords_recon, coords[:, :, :3, :], attention_mask) # [B, L, 3, 3]
        # (2) backbone geometric direction loss
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(
            coords_recon, coords[:, :, :3, :], attention_mask)
        # (3) backbone binned distance classification
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(
            decoded_states["pairwise_dist_logits"], coords, attention_mask)
        # (4) backbone binned direction classification
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            decoded_states["pairwise_dir_logits"], coords[:, :, :3, :], attention_mask)
        # (5) inverse folding 
        inverse_folding_loss, inverse_folding_metrics = self.compute_inverse_folding(
            decoded_states["last_hidden_state"], seq_residue_tokens, attention_mask)

        reconstruction_loss = (geom_dist_loss + geom_dir_loss + binned_dist_loss 
                                + binned_dir_loss + inverse_folding_loss).mean()
        loss = reconstruction_loss * self.loss_weight["reconstruction_loss_weight"] + partial_loss
        
        total_param_bits = torch.tensor(self.param_size*32, device=coords.device, dtype=torch.float32)
        L_avg = attention_mask.sum(axis=-1).float().mean()
        log_K = torch.log2(torch.tensor(self.model_cfg.quantizer.codebook_size, device=coords.device, dtype=torch.float32))
        
        metrics = {
            **geom_dist_metrics,
            **geom_dir_metrics,
            **binned_dist_metrics,
            **binned_dir_metrics,
            **inverse_folding_metrics,
            **partial_metrics,
            "reconstruction_loss": reconstruction_loss,
            "bb_rmsd": torch.tensor(bb_rmsd_list, device=coords.device, dtype=torch.float32).mean(),
            "lddt": torch.tensor(lddt_list, device=coords.device, dtype=torch.float32).mean(),
            "total_param_bits": total_param_bits,
            "L_avg": L_avg,
            "log_K": log_K,
            "quantized_indices": quantized_indices,
            "attention_mask": attention_mask
        }
        loss_and_metrics = (loss, metrics)
        
        return (loss_and_metrics, )
    
    def compute_geometric_distance(self, x_recon, x, attention_mask, clamp_value=25):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        """
        assert x_recon.shape[-2] == 3 and x_recon.shape[-1] == 3
        
        # ignore padding regions
        x_recon[~attention_mask] = 0
        x[~attention_mask] = 0
        B, L, E = x.shape[0], x.shape[1], x.shape[-1]
        x_recon, x = x_recon.reshape(B, -1, E), x.reshape(B, -1, E) # [B, L, 3, 3] -> [B, L * 3, 3] 

        dist_pred = torch.cdist(x_recon, x_recon, p=2.0) # [B, L * 3, L * 3]
        dist_true = torch.cdist(x, x, p=2.0)

        dist_mask = attention_mask.repeat(1, 3)
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, L * 3, L * 3]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dist_loss": loss.mean(),
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return loss.mean(), metric

    def compute_direction_vectors(self, coords,):
        """
        coords: [B, L, 3, 3]
        """
        # N -> Ca
        v1 = coords[:, :, 1, :] - coords[:, :, 0, :] # [B, 0~L, 3]
        # Ca -> C
        v2 = coords[:, :, 2, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # C -> N_next
        v3 = coords[:, 1:, 0, :] - coords[:, :-1, 2, :] # [B, 0~L-1, 3]
        # -(N -> Ca) x (Ca -> C)
        v4 = - torch.cross(v1, v2, dim=-1) # [B, 0~L, 3]
        # (C_prev -> N) x (N -> Ca)
        tmp = coords[:, 1:, 0, :] - coords[:, :-1, 2, :] # [B, 1~L, 3]
        v5 = torch.cross(tmp, v1[:, 1:], dim=-1)
        # (Ca -> C) x (C -> N_next)
        v6 = torch.cross(v2[:, :-1], v3, dim=-1) # [B, 0~L-1, 3]
        
        ret = [v1[:, 1:-1], v2[:, 1:-1], v3[:, 1:], v4[:, 1:-1], v5[:, :-1], v6[:, 1:]] # [B, L-2, 3]
        ret = torch.stack(ret, dim=1) # [B, 6, L-2, 3]
        ret = ret.reshape(ret.shape[0], -1, ret.shape[-1]) # [B, 6 * (L-2), 3]

        return ret


    def compute_geometric_direction(self, x_recon, x, attention_mask, clamp_value=20):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        vec_pred = self.compute_direction_vectors(x_recon)
        vec = self.compute_direction_vectors(x)

        # pairwise dot product
        dist_pred = torch.matmul(vec_pred, torch.transpose(vec_pred, 1, 2)) # [B, 6(L-2), 6(L-2)]
        dist_true = torch.matmul(vec, torch.transpose(vec, 1, 2)) # [B, 6(L-2), 6(L-2)]

        dist_mask = attention_mask[:, 1:-1].repeat(1, 6) # [B, 6(L-2)]
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, 6(L-2), 6(L-2)]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dir_loss": loss.mean(),
            f"geom_dir_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dir_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        return loss.mean(), metric

    def compute_binned_direction(self, pairwise_logits, coords, attention_mask):
        """
        pairwise_logits: [B, L, L, 96]
        coords: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        # compute from ground truth
        # unit vectors
        # Ca -> C
        v1 = coords[:, :, 2, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # Ca -> N
        v2 = coords[:, :, 0, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # (Ca -> C) x (Ca -> N)
        v3 = torch.cross(v1, v2, dim=-1) # [B, L, 3]
        v1 = F.normalize(v1, p=2, dim=-1)
        v2 = F.normalize(v2, p=2, dim=-1)
        v3 = F.normalize(v3, p=2, dim=-1)

        # dot products
        pairwise_prod = torch.stack([
            torch.matmul(v1, torch.transpose(v2, 1, 2)), # [B, L, L]
            torch.matmul(v1, torch.transpose(v3, 1, 2)),
            torch.matmul(v2, torch.transpose(v1, 1, 2)),
            torch.matmul(v2, torch.transpose(v3, 1, 2)),
            torch.matmul(v3, torch.transpose(v1, 1, 2)),
            torch.matmul(v3, torch.transpose(v2, 1, 2)),
        ], dim=-1) # [B, L, L, 6]
        NUM_BIN = 16
        bin_edges = [-1 + 0.125 * i for i in range(NUM_BIN)] + [1]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(pairwise_prod, bin_edges, right=True) - 1 # [B, L, L, 6]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        pairwise_logits = pairwise_logits.reshape([_ for _ in binned_labels.shape] + [-1]) # [B, L, L, 6, NUM_BIN]

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask].reshape(-1, NUM_BIN), binned_labels[mask].reshape(-1)
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)
        
        metric = {
            f"binned_dir_loss": loss.mean(),
            f"binned_dir_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_binned_distance(self, pairwise_logits, coords, attention_mask):
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        """

        # calculate Cbeta
        cbeta = infer_cbeta_from_atom37(coords) # [B, L, 3]

        # pairwise Cbeta distance
        NUM_BIN = 64
        dist_true = torch.cdist(cbeta, cbeta, p=2.0)
        bin_edges = [0] + [(2.3125 + 0.3075 * i) ** 2 for i in range(NUM_BIN)]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(dist_true, bin_edges, right=True) - 1 # [B, L, L]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        assert binned_labels.min() >= 0 and binned_labels.max() < NUM_BIN

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)
        
        metric = {
            f"binned_dist_loss": loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_inverse_folding(self, h, residue_labels, attention_mask):
        """
        h: [B, L, d_model=1024]
        residue_labels: [B, L]
        attention_mask: [B, L]
        """
        logits = self.inverse_folding_head(h) # [B, L, num_AAs]
        
        if not (logits.shape[0] == attention_mask.shape[0] and logits.shape[1] == attention_mask.shape[1]):
            raise ValueError
        
        logits, residue_labels = logits[attention_mask], residue_labels[attention_mask]
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, residue_labels)
        
        metric = {
            f"inverse_folding_loss": loss.mean(),
            f"inverse_folding_accuracy": (logits.argmax(dim=-1) == residue_labels).float().mean(),
        }
        return loss.mean(), metric
    

class LightningVQPretrainModel(pl.LightningModule):
    """
    PTL wrapper class for VQ-VAE pre-training
    """

    def __init__(
        self,
        model_cfg,
        trainer,
        py_logger,
        optimizer_cfg,
        all_split_names,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.trainer = trainer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        # for lm eval
        self.cwd = Path(__file__).parents[2]
        self.lm_every = 10
        self.valid_quantized_inds = defaultdict(list) # store quant inds
        #
        self.all_split_names = all_split_names
        for split in self.all_split_names:
            setattr(self, f"{split}_step_outputs", [])

    def setup(self, stage: str):
        """
        Set up the module, including model creation
        Args:
            stage: PTL stage train/val/test can be used to induce different
                    behavior only used for inheritance
        """
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        world   = int(os.environ.get("WORLD_SIZE", getattr(self.trainer.strategy, "world_size", 1)))
        rank    = int(os.environ.get("RANK", getattr(self.trainer.strategy, "global_rank", 0)))
        self.print(f"[setup] rank={rank} world={world} is_dist={is_dist}")
        self.trainer.strategy.config["train_micro_batch_size_per_gpu"] = self.optimizer_cfg.micro_batch_size
        self.model = model_init_fn(self.trainer, self.model_cfg)        
        # get time here for first iteration at batch 0
        # logged in on_train_batch_end
        self._last_logged_batch_start_time = time.monotonic()

        if getattr(self, "lm_eval_dir", None) is None:
            # 1) rank 0 picks a run_id; others start with None
            run_id = None
            if self.trainer.is_global_zero:
                # human-friendly; use uuid.uuid4().hex if you prefer unique IDs
                run_id = time.strftime("%Y%m%d-%H%M%S")

            # 2) broadcast run_id -> all ranks (handle backends without PL broadcast)
            if dist.is_available() and dist.is_initialized():
                try:
                    run_id = self.trainer.strategy.broadcast(run_id, src=0)
                except Exception:
                    buf = [run_id] if self.trainer.is_global_zero else [None]
                    dist.broadcast_object_list(buf, src=0)
                    run_id = buf[0]

            # Fallback in case we’re single-process (no dist)
            if run_id is None:
                run_id = time.strftime("%Y%m%d-%H%M%S")

            # 3) construct shared dir on a shared filesystem
            base = Path(self.cwd) / "ckpts" / str(run_id)
            if self.trainer.is_global_zero:
                base.mkdir(parents=True, exist_ok=True)

            # 4) ensure it exists before other ranks proceed
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            self.lm_eval_dir = base
            print(
                f"[setup] rank={getattr(self.trainer, 'global_rank', 0)} "
                f"lm_eval_dir={self.lm_eval_dir}",
                flush=True,
            )


    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_list"])
        batch_size = len(batch["input_list"][0])
        loss, metrics = outputs[0]
        quant_inds = metrics.pop('quantized_indices')
        attn_mask = metrics.pop('attention_mask')
        # end
        self.log(
            "seen_samples",
            batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            reduce_fx="sum",
            sync_dist=True,
        )
        self.log(
            "training_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size, logger=True, sync_dist=True,
        )

        return {"loss": loss}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log time/step and TFLOPS
        Args:
            outputs: outputs of train_step, not used, required for hook
            batch: use batch to get input/output sequence length for TFLOPs
            batch_idx: batch number, not used required for hook
        """

        if batch_idx > 0 and batch_idx % self.trainer.log_every_n_steps == 0:
            # get the time for this iteration
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            # start timeer for the next iteration
            self._last_logged_batch_start_time = time.monotonic()
            time_per_step = elapsed_time / self.trainer.log_every_n_steps

            # useful to log this even though PTL provides it in the progressbar
            # PTL logs provide exponential decaying average which is not useful
            # forquick benchmarking, especially for large models
            self.log(
                "sec/step", time_per_step, on_step=True, prog_bar=True, 
                logger=True, rank_zero_only=True,
            )
        
        torch.cuda.empty_cache()

    def _valid_or_test_step(self, batch, batch_idx, split="validation"):
        outputs = self.model(batch["input_list"])
        loss, metrics = outputs[0]
        batch_size = len(batch["input_list"][0])
        quant_inds = metrics.pop('quantized_indices')
        attn_mask = metrics.pop('attention_mask')
        # added this for LM eval
        for i in range(batch_size):
            self.valid_quantized_inds[split].append(quant_inds[i][attn_mask[i]].cpu().numpy().tolist())
        log_metrics = {
            f"{split}_{k}": v for k, v in metrics.items()
        }

        self.log_dict(
            {f"{split}_loss": loss, **log_metrics},
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
            add_dataloader_idx=False
        )

        return {
            f"{split}_loss": loss,
            **log_metrics,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = self.all_split_names[dataloader_idx]
        outputs = self._valid_or_test_step(batch, batch_idx, split=split)
        getattr(self, f"{split}_step_outputs").append(outputs)
        return outputs

    def on_train_start(self):
        # override the lambda schedulers
        # default configs do not adjust the schedulers
        self.lr_schedulers().lr_lambdas = [
            lambda x: self.optimizer_cfg.override.mult_factor
            * fn(x + self.optimizer_cfg.override.add_index)
            for fn in self.lr_schedulers().lr_lambdas
        ]

    def _valid_or_test_epoch_end(self, outputs, split="validation"):        
        agg_result = {k: [] for k in outputs[0].keys() if k.startswith(split)}
        for out in outputs:
            for k in out.keys():
                if k.startswith(split):
                    agg_result[k].append(out[k])

        for k in agg_result.keys():
            agg_result[k] = torch.stack(agg_result[k]).mean()

        self.log_dict(
            agg_result, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size, add_dataloader_idx=False,
        )
    
    def on_validation_epoch_start(self):
        self.valid_quantized_inds = defaultdict(list)

    def _run_small_lm(self, *, train_path: Path, valid_path: Path, out_path: Path, n_samples: int) -> list[str]:
        env = os.environ.copy()
        cwd = self.cwd
        cmd = [
            "conda", "run", "-n", "esm_env",
            "bash", f"{cwd}/scripts/train.sh", "0", "8", "1",
            str(train_path), str(valid_path), str(out_path), str(n_samples),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            self.print(f"[LM subprocess failed] rc={e.returncode}\nSTDERR:\n{e.stderr}")
            return []
        except subprocess.TimeoutExpired:
            self.print("[LM subprocess timed out]")
            return []

        if not out_path.exists():
            self.print("[LM subprocess finished but samples file missing]")
            return []

        return json.load(open(out_path))


    @staticmethod
    def _run_metrics_child(cmd, *, cwd, env, out_file: Path, log_path: Path, timeout_s: int) -> dict:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", buffering=1) as sink:  # line-buffered
            # Unbuffered python + safer defaults in the child
            env = env.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")

            # Spawn child in its own session so we can kill the whole tree on timeout
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            deadline = time.time() + timeout_s
            last_log_mtime = 0.0

            while True:
                rc = proc.poll()
                if rc is not None:
                    if rc == 0 and out_file.exists():
                        try:
                            return json.load(open(out_file, "r", encoding="utf-8"))
                        except Exception:
                            print(f"[metrics] parse error, see {log_path}")
                            return {}
                    print(f"[metrics] exited rc={rc} without {out_file.name}, see {log_path}")
                    return {}

                # progress/heartbeat: if the logfile is updating, extend patience a bit
                try:
                    m = os.stat(log_path).st_mtime
                    if m > last_log_mtime:
                        last_log_mtime = m
                        # optional: extend deadline if you observe steady progress
                except FileNotFoundError:
                    pass

                if time.time() > deadline:
                    print(f"[metrics] timeout → killing PG pid={proc.pid}, see {log_path}")
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return {}

                time.sleep(2)

    def _prepare_subset_dir(self, epoch_dir: Path, shard_idx: int, num_shards: int) -> Path:
        """
        Creates a per-rank subset directory with only files for which
        index % num_shards == shard_idx. Uses symlinks when possible.
        """
        subset = epoch_dir / f"sctm_subset_rank{shard_idx:04d}"
        if subset.exists():
            # Clean any stale files from previous attempts
            for p in subset.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
        else:
            subset.mkdir(parents=True, exist_ok=True)

        # Enumerate PDBs by index from your naming {i}.pdb
        pdbs = sorted(Path(epoch_dir).glob("*.pdb"),
                    key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

        for i, p in enumerate(pdbs):
            if i % num_shards != shard_idx:
                continue
            dst = subset / p.name
            shutil.copy2(p, dst)
        return subset

    def _eval_pdbs(self, epoch_dir: Path, *, sctm: bool,
                shard_idx: int | None, num_shards: int | None,
                out_basename: str, steal_gpu: bool,
               timeout_s: int = 24*3600) -> dict:
        """
        Runs your metrics module once. If `sctm=True`, we optionally restrict the
        input set to a per-rank shard and expose only the rank's GPU to the child.
        Returns a dict (empty dict on failure).
        """
        # Choose input dir: whole dir for cheap pass, or subset dir for sctm sharded pass
        pdb_dir = epoch_dir
        if sctm and shard_idx is not None and num_shards and num_shards > 1:
            pdb_dir = self._prepare_subset_dir(epoch_dir, shard_idx, num_shards)

        gr = int(getattr(self.trainer.strategy, "global_rank", 0))
        # Count work
        n_pdb = len(list(Path(pdb_dir).glob("*.pdb")))
        print(f"[rank{gr}] _eval_pdbs(sctm={sctm}) dir={pdb_dir} n_pdb={n_pdb}", flush=True)
        if n_pdb == 0:
            print(f"[rank{gr}] no PDBs → skipping", flush=True)
            return {}
        out_file = epoch_dir / out_basename
        log_path = epoch_dir / f"{out_basename}.rank{gr:04d}.log"

        env = os.environ.copy()
        if steal_gpu:
            # If the launcher already restricted visibility per rank (common),
            # keep that exact mapping.
            if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"]:
                # If it's multiple IDs, pick the one for our local_rank.
                cvd = env["CUDA_VISIBLE_DEVICES"]
                if "," in cvd:
                    ids = [x for x in cvd.split(",") if x != ""]
                    lr = int(getattr(self.trainer.strategy, "local_rank", 0))
                    if lr < len(ids):
                        env["CUDA_VISIBLE_DEVICES"] = ids[lr]
                    # else leave as-is
            else:
                # Fallback: restrict by local_rank (works when no mapping is set)
                lr = int(getattr(self.trainer.strategy, "local_rank", 0))
                env["CUDA_VISIBLE_DEVICES"] = str(lr)

            # free parent VRAM so child has room (optional but helpful)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Build command (fixed: use python -m not "python -m pdb")
        cmd = [
            "conda", "run", "-n", "esm_env",
            "python", "-m", "foldingdiff.metrics",
            "--gen-pdb-dir", str(pdb_dir),
            "--out-file", str(out_file),
            "--sctm", "True" if sctm else "False",
        ]
        metrics = self._run_metrics_child(
            cmd, cwd=self.cwd, env=env,
            out_file=out_file, log_path=log_path,
            timeout_s=timeout_s  # e.g., 1 hour per shard
        )
        return metrics
            
    def _gather_to_rank0(self, local_list: list[list[int]]) -> list[list[int]]:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return list(local_list)
        gathered = [None for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(gathered, list(local_list))
        merged: list[list[int]] = []
        for part in gathered:
            merged.extend(part)
        return merged

    def _merge_sctm_jsons(self, epoch_dir: Path, world_size: int) -> dict:
        import json, math
        files = [epoch_dir / f"metrics_sctm.rank{r:04d}.json" for r in range(world_size)]
        sum_sc, sum_df, n = 0.0, 0.0, 0

        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            def safe(x):
                try:
                    v = float(x)
                    return 0.0 if math.isnan(v) or math.isinf(v) else v
                except Exception:
                    return 0.0

            sum_sc += safe(d.get("scTM_mean", 0.0))
            sum_df += safe(d.get("designability_fraction", 0.0))
            n += 1

        if n == 0:
            return {"scTM_mean": 0.0, "designability_fraction": 0.0, "num_sctm_shards_merged": 0}

        return {
            "scTM_mean":               sum_sc / n,
            "designability_fraction":  sum_df / n,
            "num_sctm_shards_merged":  n,
        }

    def _mem(self, tag=""):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved()  / 1e9
            m = torch.cuda.max_memory_allocated() / 1e9
            print(f"[MEM {tag}] alloc={a:.2f}G reserved={r:.2f}G max_alloc={m:.2f}G")


    def _mark_sctm_done(self, epoch_dir: Path, rank: int):
        (epoch_dir / f"_sctm_done.rank{rank:04d}").write_text("ok")

    def _wait_all_sctm_done(self, epoch_dir: Path, world_size: int, timeout_s: int = 86400):
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if all((epoch_dir / f"_sctm_done.rank{r:04d}").exists() for r in range(world_size)):
                return True
            time.sleep(10)
        return False


    def _rank_world(self):
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = self.trainer.world_size if is_dist else 1
        global_rank = self.trainer.global_rank if is_dist else 0
        return is_dist, world_size, global_rank

    def on_validation_epoch_end(self):
        for split in self.all_split_names:
            self._valid_or_test_epoch_end(getattr(self, f"{split}_step_outputs"), split=split)
            getattr(self, f"{split}_step_outputs").clear()        
        # Optional: only run every K epochs
        if (self.current_epoch + 1) % self.lm_every != 0:
            return
        is_dist, world_size, global_rank = self._rank_world()
        train_all = self._gather_to_rank0(self.valid_quantized_inds["train"])
        valid_all = self._gather_to_rank0(self.valid_quantized_inds["validation"])
        self._mem(f"[{global_rank}] finish gather")
        samples = None
        epoch_dir = Path(self.lm_eval_dir) / f"epoch_{self.current_epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)        
        if self.trainer.is_global_zero:
            train_path = epoch_dir / "train.json"
            valid_path = epoch_dir / "valid.json"
            out_path   = epoch_dir / "samples.json"
            json.dump(train_all, open(train_path, "w+"))
            json.dump(valid_all, open(valid_path, "w+"))
            samples = self._run_small_lm(
                train_path=train_path,
                valid_path=valid_path,
                out_path=out_path,
                n_samples=100
            )            
            # decode structures
            batch_size = self.optimizer_cfg.micro_batch_size
            n_batches = (len(samples)+batch_size-1)//batch_size
            for i in range(n_batches):
                self._mem(f"[0] decode batch {i}")
                cur_bs = samples[i*batch_size:(i+1)*(batch_size)]                
                max_len = max((len(s) for s in cur_bs))
                quantized_indices = torch.stack([torch.tensor(sample+(max_len-len(sample))*[0], device=self.device) for sample in cur_bs], axis=0)
                quantized_z = torch.stack([self.model.quantizer.codebook(torch.tensor(sample+(max_len-len(sample))*[0], device=self.device)) for sample in cur_bs], axis=0)
                attn_mask = torch.tensor([[True]*len(sample)+[False]*(max_len-len(sample)) for sample in cur_bs], device=self.device)
                decoded_states = self.model.decoder.decode(quantized_z, quantized_indices, attn_mask, None)
                bb_pred = decoded_states["bb_pred"]                
                for j in range(len(cur_bs)):
                    pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[j].detach())
                    pdb_chain_recon.to_pdb(epoch_dir / f"{batch_size*i+j}.pdb")
            self._mem(f"[0] before cheap metrics")
            metrics = self._eval_pdbs(epoch_dir, sctm=False, shard_idx=None, num_shards=None, out_basename="metrics_cheap.json", steal_gpu=False)            
            for k, v in metrics.items():
                self.log(f"lm/{k}", v, prog_bar=True, logger=True, sync_dist=False)            

        self._mem(f"[{global_rank}] before sctm metrics")
        if is_dist:
            torch.distributed.barrier()                                    
        self._eval_pdbs(epoch_dir, sctm=True,
                                shard_idx=global_rank, num_shards=world_size,
                                out_basename=f"metrics_sctm.rank{global_rank:04d}.json",
                                steal_gpu=True)
        self._mark_sctm_done(epoch_dir, global_rank)
        self._mem(f"after sctm metrics")    
        # Merge SCTM shard outputs on rank 0 and log a summary
        if self.trainer.is_global_zero:
            self._mem(f"[0] before sctm merge")
            ok = self._wait_all_sctm_done(epoch_dir, world_size, timeout_s=36*3600)  # generous
            if not ok:
                self.print("[SCTM] Timeout waiting for shard markers; merging what exists")
            merged = self._merge_sctm_jsons(epoch_dir, world_size)
            self._mem(f"[0] finish sctm merge")
            # Log whatever rollups you need; here we forward keys
            for k, v in merged.items():
                self.log(f"lm/{k}", v, prog_bar=True, logger=True, sync_dist=False)                                      
                
            
        # for k, v in self.trainer.callback_metrics.items():
        #     if isinstance(v, torch.Tensor) and v.numel() == 1:
        #         print(f"[RANK {self.global_rank}] {k:<25} {v.dtype}", flush=True)    

        if is_dist:
            torch.distributed.barrier()                            
        
        self._mem(f"[{global_rank}] finish validation_epoch_end")

    def on_before_optimizer_step(self, optimizer):
        for n,p in self.model.named_parameters():
            grad_data = deepspeed.utils.safe_get_full_grad(p)
            p.grad = grad_data
        norms = grad_norm(self.model, norm_type=2)
        norms = {k:v.to(grad_data.device) for k,v in norms.items()}
        
        self.log_dict(
            norms, prog_bar=True, sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size, add_dataloader_idx=False,
            #on_step=True, #on_epoch=True,
        )

    def configure_optimizers(self):
        # hyperparameter logging needs to occur after ddp launch
        # inside config_optimizers since this occurs after ddp launch
        # use trainer logger which ensures it is mstar logger
        # self.trainer.logger.log_hyperparams(self.full_experiment_config)

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        # filter out bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # filter out layernorm with a variety of spellings
        decay_parameters = [name for name in decay_parameters if "layer_norm" not in name]
        decay_parameters = [name for name in decay_parameters if "layernorm" not in name]
        
        params_decay = [p for n, p in self.model.named_parameters() if (any(nd in n for nd in decay_parameters))]
        params_nodecay = [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in decay_parameters))]
        
        param_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.optimizer.weight_decay,
            },
            {
                "params": params_nodecay, 
                "weight_decay": 0.0
            },
        ]
        optimizer = get_optimizer(param_groups, self.optimizer_cfg.optimizer)

        scheduler = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=optimizer)
        return (
            [optimizer],
            [{
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "validation_loss",
            }],
        )
