from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_text_mask(
    text_tensors: List[torch.Tensor],
    num_wrapping_dims: int = 0,
    padding_id: int = 0,
) -> torch.BoolTensor:
    """
    Takes the dictionary of tensors produced by a `TextField` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise. `padding_id` specifies the id of padding tokens.
    We also handle `TextFields` wrapped by an arbitrary number of `ListFields`, where the number of wrapping
    `ListFields` is given by `num_wrapping_dims`.

    If `num_wrapping_dims == 0`, the returned mask has shape `(batch_size, num_tokens)`.
    If `num_wrapping_dims > 0` then the returned mask has `num_wrapping_dims` extra
    dimensions, so the shape will be `(batch_size, ..., num_tokens)`.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting `num_wrapping_dims`,
    if this tensor has two dimensions we assume it has shape `(batch_size, ..., num_tokens)`,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    `(batch_size, ..., num_tokens, num_features)`, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input `text_field_tensors` contains the "mask" key, this is returned instead of inferring the mask.
    """
    tensor_dims = [
        (text_tensors.dim(), text_tensors)
    ]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return token_tensor != padding_id
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return (character_tensor != padding_id).any(dim=-1)
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))

class BowTextClassifier(nn.Module):
    def __init__(self, dict_dim, emb_dim=128, hidden_dim=128, fc_hidden_dim=96, num_labels=2):
        super().__init__()
        self.padding_idx = dict_dim - 1
        self.embedding = nn.Embedding(num_embeddings=dict_dim, embedding_dim=emb_dim, padding_idx=self.padding_idx)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.classifier = nn.Linear(fc_hidden_dim, num_labels)

    def forward(self, text, label=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedding(text)
        # Shape: (batch_size, num_tokens)
        mask = get_text_mask(text, padding_id=self.padding_idx)
        
        embedded_text = embedded_text * mask.unsqueeze(-1)
        # Shape: (batch_size, num_tokens)
        summed = embedded_text.sum(1)
        encoded_text = torch.tanh(summed)

        # Shape: (batch_size, hidden_dim)
        fc_1 = torch.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_dim)
        fc_2 = torch.tanh(self.fc2(fc_1))
        # Shape: (batch_size, num_labels)
        logits = self.classifier(fc_2)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            output['loss'] = F.cross_entropy(logits, label)
        return output





        
        
        