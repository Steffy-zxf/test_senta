from typing import Dict, List, Optional, Union, Tuple

import paddle 
import paddle.nn as nn
import paddle.nn.functional as F


def get_text_mask(
    text_tensors: List[paddle.Tensor],
    padding_id: int = 0,
) -> paddle.Tensor:
    tensor_dims = [
        (text_tensors.dim(), text_tensors)
    ]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0]
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return token_tensor != padding_id
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return (character_tensor != padding_id).any(dim=-1)
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))

class BowTextClassifier(paddle.nn.Layer):
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
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_dim)
        fc_1 = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_dim)
        fc_2 = paddle.tanh(self.fc2(fc_1))
        # Shape: (batch_size, num_labels)
        logits = self.classifier(fc_2)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits, axis=-1)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            output['loss'] = F.cross_entropy(logits, label)
        return output
