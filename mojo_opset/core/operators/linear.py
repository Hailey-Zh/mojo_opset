from typing import Optional, Union

import torch

from ..operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Common parameter definitions for Linear operator.

        Init parameters:
        - weight (torch.Tensor): Weight tensor, shape [in_dim, out_dim].
        - bias (Optional[torch.Tensor]): Bias tensor, shape aligned with output dimension; optional.
        """
        super().__init__()

        if weight.ndim not in (2,):
            raise ValueError(f"weight should be 2-D, but got {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias should be torch.Tensor or None")
            if weight.ndim == 2:
                # Standard PyTorch Linear weight shape is [out_features, in_features]
                out_dim = weight.shape[0]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError(f"bias should be 1-D with shape [out_dim={out_dim}], but got {tuple(bias.shape)}")
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Standard PyTorch Linear weight shape is [out_features, in_features]
        in_dim = self.weight.shape[1]
        if input.shape[-1] != in_dim:
            raise ValueError(f"input should have last dim {in_dim}, but got {input.shape[-1]}")
        if input.ndim not in (3, 4):
            raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
        return torch.nn.functional.linear(input, self.weight, self.bias)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight=False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        Grouped linear forward over variable-length segments.

        Splits the 2D input into contiguous groups defined by `group_list`,
        applies a per-group weight, and concatenates outputs.

        Args:
            input (torch.Tensor): 2D tensor of shape (N, Din); rows are grouped
                contiguously. Sum(group_list) must equal N.
            group_list (torch.Tensor): 1D tensor of length G with row counts per group.

        Returns:
            torch.Tensor: 2D tensor of shape (N, Dout), concatenated per-group outputs.

        Notes:
            - Expects `self.weight` of shape (G, Din, Dout). If `trans_weight` is True,
            weights are transposed from (G, Dout, Din) to (G, Din, Dout).
            - Each group's output is computed as `input_g @ weight_g`.
        """
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            self.weight = self.weight.transpose(1, 2).contiguous()

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = self.weight[g, :, :]
            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)


class MojoRoutedLinear(MojoOperator):
    """
    Routed Linear operator for 3D weight tensors.
    
    This operator performs linear transformations with multiple weight matrices,
    routing inputs to different weights based on routing indices. It's designed for:
    - Mixture of Experts (MoE) where each expert has its own weight
    - Multi-head operations with separate weights per head
    - Any scenario requiring dynamic weight routing
    
    Key features:
    - Supports 3D weight tensors: [num_experts, out_features, in_features]
    - Efficient routed computation
    - Compatible with MojoLinear API
    - Supports both single-expert and multi-expert forward modes
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Initialize MojoRoutedLinear.
        
        Args:
            weight (torch.Tensor): Weight tensor with shape:
                - [num_experts, out_features, in_features]
                If None, will be initialized as a parameter.
            bias (Optional[torch.Tensor]): Bias tensor with shape:
                - [num_experts, out_features] for per-expert bias, or
                - [out_features] for shared bias across experts
                Optional parameter.
        
        Raises:
            TypeError: If weight is not a torch.Tensor when provided
            ValueError: If weight or bias dimensions are incorrect
        """
        super().__init__()
        
        # Validate provided weight
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight should be torch.Tensor or None")
        if weight.ndim != 3:
            raise ValueError(
                f"weight should be 3-D with shape [num_experts, out_features, in_features], "
                f"but got {tuple(weight.shape)}"
            )
        self.weight = weight
        
        # Store dimensions
        self.num_experts, self.out_features, self.in_features = self.weight.shape
        
        # Initialize or validate bias
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias should be torch.Tensor or None")
            
            # Support both per-expert bias and shared bias
            if bias.ndim == 2:
                # Per-expert bias: [num_experts, out_features]
                if bias.shape != (self.num_experts, self.out_features):
                    raise ValueError(
                        f"Per-expert bias should have shape [{self.num_experts}, {self.out_features}], "
                        f"but got {tuple(bias.shape)}"
                    )
            elif bias.ndim == 1:
                # Shared bias: [out_features]
                if bias.shape[0] != self.out_features:
                    raise ValueError(
                        f"Shared bias should have shape [{self.out_features}], "
                        f"but got {tuple(bias.shape)}"
                    )
            else:
                raise ValueError(
                    f"bias should be 1-D or 2-D, but got {bias.ndim}-D with shape {tuple(bias.shape)}"
                )
        self.bias = bias
    
    def forward(
        self,
        input: torch.Tensor,
        expert_idx: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with routed linear transformation.
        
        This method supports multiple modes:
        1. Single expert mode: When expert_idx is an int or 0-dim tensor, 
           apply single expert's weight to all inputs
        2. Multi-expert mode: When expert_idx is a 1-dim+ tensor, 
           apply per-token expert routing
        
        Args:
            input (torch.Tensor): Input tensor with shapes:[T, D]
            expert_idx (Optional[Union[int, torch.Tensor]]): 
                - If int: use single expert weight for all tokens
                - If 0-dim tensor: use single expert weight for all tokens
                - If 1-dim+ tensor: per-token expert indices with shape matching input first dim
                - If None: raises ValueError (must specify expert routing)
        
        Returns:
            torch.Tensor: Output tensor with same rank as input, last dimension = out_features
        
        Raises:
            ValueError: If input dimensions don't match expected layout or in_features
            
        Examples:
            >>> # Single expert mode
            >>> routed_linear = MojoRoutedLinear(weight)
            >>> output = routed_linear(input, expert_idx=0)
            
            >>> # Multi-expert mode
            >>> expert_indices = torch.tensor([0, 1, 2, 1, 0])
            >>> output = routed_linear(input, expert_idx=expert_indices)
        """
        # Validate input last dimension
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input last dim should be {self.in_features}, but got {input.shape[-1]}"
            )
        
        if expert_idx is None:
            raise ValueError(
                "expert_idx must be provided as int (single expert) or "
                "tensor (per-token routing)"
            )
        
        # Check if expert_idx is a tensor
        if isinstance(expert_idx, torch.Tensor):
            # Handle 0-dim tensor (scalar tensor like tensor(0))
            if expert_idx.ndim == 0:
                # Convert 0-dim tensor to int
                expert_idx_int = expert_idx.item()
                return self._single_expert_forward(input, expert_idx_int)
            else:
                # 1-dim or higher tensor - multi-expert mode
                return self._multi_expert_forward(input, expert_idx)
        
        # Handle int directly
        elif isinstance(expert_idx, int):
            return self._single_expert_forward(input, expert_idx)
        
        else:
            raise TypeError(
                f"expert_idx should be int or torch.Tensor, but got {type(expert_idx)}"
            )
    
    def _single_expert_forward(
        self,
        input: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        """
        Apply single expert's weight to all inputs.
        
        Args:
            input: Input tensor
            expert_idx: Index of expert to use (0 to num_experts-1)
        
        Returns:
            Output after linear transformation
        """
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(
                f"expert_idx should be in [0, {self.num_experts}), but got {expert_idx}"
            )
        
        # Get weight for this expert: [out_features, in_features]
        weight = self.weight[expert_idx]
        
        # Get bias if available
        if self.bias is not None:
            if self.bias.ndim == 2:
                # Per-expert bias
                bias = self.bias[expert_idx]
            else:
                # Shared bias
                bias = self.bias
        else:
            bias = None
        
        # Apply linear transformation
        return torch.nn.functional.linear(input, weight, bias)
    
    def _multi_expert_forward(
        self,
        input: torch.Tensor,
        expert_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply different expert weights to different tokens (expert routing).
        
        This uses a loop-based approach to process each expert separately.
        Optimized for sparse expert activation (common in MoE).
        
        Args:
            input: Input tensor [num_tokens, in_features] or [num_tokens, ..., in_features]
            expert_idx: Expert indices [num_tokens] with values in [0, num_experts)
        
        Returns:
            Output tensor [num_tokens, out_features] or [num_tokens, ..., out_features]
        """
        # Flatten input to [num_tokens, in_features] if needed
        original_shape = input.shape
        if input.ndim > 2:
            # Reshape to [num_tokens, in_features]
            input_flat = input.view(-1, self.in_features)
        else:
            input_flat = input
        
        num_tokens = input_flat.shape[0]
        
        # Validate expert_idx
        if expert_idx.shape[0] != num_tokens:
            raise ValueError(
                f"expert_idx first dim ({expert_idx.shape[0]}) should match "
                f"input first dim ({num_tokens})"
            )
        
        # Initialize output
        output = torch.zeros(
            num_tokens, self.out_features,
            dtype=input.dtype, device=input.device
        )

        # Find unique experts and process each
        unique_experts = torch.unique(expert_idx)
        
        for expert_id in unique_experts:
            # Find tokens assigned to this expert
            mask = expert_idx == expert_id
            token_indices = torch.where(mask)[0]
            
            # Get inputs for this expert
            expert_input = input_flat[token_indices]  # [num_selected, in_features]
            
            # Apply this expert's weight
            expert_weight = self.weight[expert_id]  # [out_features, in_features]
            expert_output = torch.nn.functional.linear(
                expert_input, expert_weight, None
            )  # [num_selected, out_features]
            
            # Add bias if available
            if self.bias is not None:
                if self.bias.ndim == 2:
                    expert_bias = self.bias[expert_id]
                else:
                    expert_bias = self.bias
                expert_output = expert_output + expert_bias
            
            # Place outputs
            output[token_indices] = expert_output
        
        # Reshape output back to original shape
        if input.ndim > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            output = output.view(output_shape)
        
        return output

class MojoLinearAllReduce(MojoOperator):
    pass


class MojoAllGatherLinear(MojoOperator):
    pass


class MojoLinearAll2All(MojoOperator):
    pass


class MojoLinearReduceScatter(MojoOperator):
    pass
