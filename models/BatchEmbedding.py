import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init, Module
import math
from typing import Optional
from einops import einsum, rearrange, repeat
class BatchEmbedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        ensemble_size (int): number of ensemble members
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (ensemble_size, num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(self, ensemble_size: int,num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.,
                 scale_grad_by_freq: bool = False, sparse: bool = False, _weight: Optional[Tensor] = None,
                 _freeze: bool = False, mode: str = 'fan_in', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
         # Initialize w/ fan_in or fan_out method?
        assert mode in ['fan_in', 'fan_out'], "Invalid mode. Mode must be 'fan_in' or 'fan_out'."
        self.mode = mode

        self.r = Parameter(torch.empty((ensemble_size, num_embeddings), **factory_kwargs)) 
        self.s = Parameter(torch.empty((ensemble_size, embedding_dim), **factory_kwargs)) 
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                                    requires_grad=not _freeze)
            self.reset_parameters()
        
        else:
            assert list(_weight.shape) == [ensemble_size, num_embeddings, embedding_dim], \
                'Shape of weight does not match ensemble_size, num_embeddings and embedding_dim'
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        # Calculate the standard deviation 
        # Torch expects weight to be of shape (out_featuers, in_features) thus we send in the transpose
        fan = init._calculate_correct_fan(self.weight.T, mode=self.mode)
        gain = init.calculate_gain(nonlinearity='relu')
        std = gain / math.sqrt(fan)

        # Initialize the weights
        with torch.no_grad():
            self.weight.normal_(0, std)
            # We change mean to 1 to not ruin pretrained weights
            self.r.normal_(1, std)
            self.s.normal_(1, std) 
        self._fill_padding_idx_with_zero()
    
    def reset_fast_weights(self,
                            mean: float = 0.0,
                            std: float = 1.0,
                            init_strategy: str = None):
        """
        Reset the fast weights of the network.

        Args:
            mean (float, optional): Mean of the normal distribution for weight initialization. Default is 0.0.
            std (float, optional): Standard deviation of the normal distribution for weight initialization. Default is 1.0.
            init_strategy (str, optional): Initialization strategy for the fast weights. 
                Supported strategies: 'xavier_normal', 'he_normal', or None (random initialization).
                Default is None.

        Note:
            If 'init_strategy' is None, weights are initialized randomly from a normal distribution 
            with mean 'mean' and standard deviation 'std'. Otherwise, initialization is based on the chosen strategy.
            Supported initialization strategies:
            - 'xavier_normal': Initializes weights using Xavier/Glorot normal initialization.
            - 'he_normal': Initializes weights using He normal initialization.

        References:
            - He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
                Proceedings of the IEEE international conference on computer vision, 2015-Decem, 1026-1034.
            - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. 
                Proceedings of the thirteenth international conference on artificial intelligence and statistics, 9-16.

        """
        if init_strategy == None:
            with torch.no_grad():
                self.r.normal_(mean, std)
                self.s.normal_(mean, std)
        else:
            # Torch expects weight to be of shape (out_featuers, in_features) thus we send in the transpose
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight.T)
            # Mistral utilizes SiLU which is similar to the ReLU funciton, hence we utilize the gain usually used for ReLU
            gain = math.sqrt(2.0)
            if init_strategy == 'xavier_normal':
                std = gain*math.sqrt(float(fan_in + fan_out))
                with torch.no_grad():
                    self.r.normal_(0, std)
                    self.s.normal_(0, std)

            elif init_strategy == 'he_normal':
                if self.mode == 'fan_in':
                    std = gain / math.sqrt(fan_in)
                elif self.mode == 'fan_out':
                    std = gain / math.sqrt(fan_out)
                with torch.no_grad():
                    self.r.normal_(0, std)
                    self.s.normal_(0, std)

            elif init_strategy == 'xavier_mean_1':
                std = gain*math.sqrt(float(fan_in + fan_out))
                with torch.no_grad():
                    self.r.normal_(1, std)
                    self.s.normal_(1, std)
            
            elif init_strategy == 'he_mean_1':
                if self.mode == 'fan_in':
                    std = gain / math.sqrt(fan_in)
                elif self.mode == 'fan_out':
                    std = gain / math.sqrt(fan_out)

                with torch.no_grad():
                    self.r.normal_(1, std)
                    self.s.normal_(1, std)
    
    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                # CHeck this part think it should be [:,self.padding_idx]
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return self._embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

  # This function replaces the functional embedding function, hence allowing for batched embeddings.
    def _embedding(
    self,
    input: Tensor,
    shared_weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    ) -> Tensor:
        r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

        This module is often used to retrieve word embeddings using indices.
        The input to the module is a list of indices, and the embedding matrix,
        and the output is the corresponding word embeddings.

        See :class:`torch.nn.Embedding` for more details.

        .. note::
            Note that the analytical gradients of this function with respect to
            entries in :attr:`weight` at the row specified by :attr:`padding_idx`
            are expected to differ from the numerical ones.

        .. note::
            Note that `:class:`torch.nn.Embedding` differs from this function in
            that it initializes the row of :attr:`weight` specified by
            :attr:`padding_idx` to all zeros on construction.

        Args:
            input (LongTensor): Tensor containing indices into the embedding matrix
            weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
                and number of columns equal to the embedding size
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                        therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                        i.e. it remains as a fixed "pad".
            max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                        is renormalized to have norm :attr:`max_norm`.
                                        Note: this will modify :attr:`weight` in-place.
            norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
            scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                    the words in the mini-batch. Default ``False``.
            sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                    :class:`torch.nn.Embedding` for more details regarding sparse gradients.

        Shape:
            - Input: LongTensor of arbitrary shape containing the indices to extract
            - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
            where V = maximum index + 1 and embedding_dim = the embedding size
            - Output: `(*, embedding_dim)`, where `*` is the input shape
        """
        # Removed has_torch_function_variadic function
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < shared_weight.size(0), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert padding_idx >= -shared_weight.size(0), "Padding_idx must be within num_embeddings"
                padding_idx = shared_weight.size(0) + padding_idx
        else:
            padding_idx = -1

        # Perform hadamard product for each fast weight i.e. W@(r_i@s_i^T)
        # Lookup table will thus be of shape (ensemble_size, num_embeddings, embed_dim)
        lookup_table = einsum(shared_weight, self.r, self.s, 'I O, E I, E O -> E I O')
        # We use unique to normalize all rows which are within one of the batched inputs
        # Validated this again torch.embedding_renorm_, it will also renormalize the row values for
        # each input in a batch
        # Renormalize the lookup_table, this replaces the _no_grad_embedding_renorm_ function
        if max_norm is not None:
            input = input.contiguous()
            with torch.no_grad():
                lookup_table[:, input.unique()] = torch.renorm(lookup_table[:,input.unique()], norm_type, 0 ,max_norm)

        # Select the embeddings for each ensemble member
        if input.dim() == 1:
            # Our input does not contain a batch of inputs hence we just return the lookup for one sequence length
            return lookup_table[:, input]
        elif input.dim() == 2:
            # Save the shape
            shape = input.shape
            # Flatten the input
            input = input.flatten()
            # Select the indeces and the reshape from (ensemble_size (batch_size seq_len) hidden_size) -> B E S H)
            return rearrange(lookup_table[:, input], 'E (B S) H -> B E S H', B = shape[0])  
        
        # We are in training and receive the token predictions for each ensemble member
        elif input.dim() == 3: 
            seq_len = input.shape[-1]
            # We first need to reshape (repeat the elements hidden_dim times) such that we can utilize it in the lookup table
            input = repeat(input, 'B E S -> E (B S) H', H = self.embedding_dim)
            # Select the indeces from the look up table
            return rearrange(lookup_table.gather(1, input), 'E (B S) H -> B E S H', S = seq_len)

        else:
            raise ValueError("Input tensor should have dimensions (sequence_length,) or (batch_size, sequence_length) or (batch_size, ensemble_size, sequence_length)")
        

