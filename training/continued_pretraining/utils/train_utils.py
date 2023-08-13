from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping

from transformers import DataCollatorForLanguageModeling

from torch.nn import CrossEntropyLoss

import torch

@dataclass
class DataCollatorSCLoss(DataCollatorForLanguageModeling):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels
    and the the provided loss mask for the sentence completion loss

    Addtional Args:
        sc_multiple ('float', defaults to 1.25):
            The multiple with which the loss values of the second half of sentence will be
            multiplied
    """
    sc_multiple: float = 2.0

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):

            for i, sample in enumerate(examples):
                
                to_add = 2048 - len(sample["loss_mask"])
                loss_mask = sample["loss_mask"]
                
                if self.sc_multiple != 2.0:
                    # adjust multiple
                    loss_mask = [self.sc_multiple if item == 2 else item for item in loss_mask]
                
                # add padding zeros
                loss_mask.extend([0] * to_add)
                
                examples[i]["loss_mask"] = loss_mask
                
            batch = self.tokenizer.pad(examples, return_tensors="pt", max_length=2048, padding="max_length")
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch



@dataclass
class DataCollatorSCLoss_Old(DataCollatorForLanguageModeling):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels
    and the the provided loss mask for the sentence completion loss

    Addtional Args:
        sc_multiple ('float', defaults to 1.25):
            The multiple with which the loss values of the second half of sentence will be
            multiplied
    """
    sc_multiple: float = 1.25

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):

            for i, sample in enumerate(examples):
                
                to_add = 2048 - len(sample["loss_mask"])
                loss_mask = sample["loss_mask"]
                
                # handle error made in tokenization; loss masks contains value 50256
                # at the and of a document; replace with 1.0
                # handle error where endoftext token has corresponding 0 in loss mask
                # replace with 1.0
                loss_mask = [1.0 if item == 50256 or item == 0 else item for item in loss_mask]
                
                # adjust multiple
                loss_mask = [self.sc_multiple if item == 1.25 else item for item in loss_mask]
                
                loss_mask.extend([0] * to_add)
                
                examples[i]["loss_mask"] = loss_mask
                
            batch = self.tokenizer.pad(examples, return_tensors="pt", max_length=2048, padding="max_length")
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def sentence_completion_loss(logits, labels, loss_mask):

    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_loss_mask = loss_mask[..., 1:].contiguous()
    
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Calculate per token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # multiple elements with loss mask
    loss = torch.mul(loss, shift_loss_mask.view(-1))

    return torch.mean(loss)



def chunks(lst, n, m=1000):
    '''
        Takes a list containing conc tokens and returns:
        sublists with a maximum len of n and minimum lenght of m
    '''
    results = []
    for i in range(0, len(lst), n):
        if len(lst[i:i + n]) > m:
            results.append(lst[i:i + n])

    return results


CONTEXT_LENGTH = 2048
def conc_together(element):

    batch_input = []
    batch_loss = []

    for input_ids, loss_mask in zip(element["input_ids"], element["loss_mask"]):
        

        batch_input.extend(input_ids)
        batch_loss.extend(loss_mask)


    # create chunks according to context size
    batch_input = chunks(batch_input, CONTEXT_LENGTH)
    batch_loss = chunks(batch_loss, CONTEXT_LENGTH)
    
    

    return {"input_ids": batch_input, "loss_mask": batch_loss}