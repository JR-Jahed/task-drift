import torch
import torch.nn as nn
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def get_embedding_matrix(model):
    # if isinstance(model, LlamaForCausalLM):
    return model.model.embed_tokens.weight

def get_embeddings(model, input_ids):
    return model.model.embed_tokens(input_ids)

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


def token_gradients(custom_model, input_ids, input_slice, target, primary_activation):
    device = custom_model.base_model.get_input_embeddings().weight.device

    embed_weights = get_embedding_matrix(custom_model.base_model)

    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=device,
        dtype=embed_weights.dtype
    )

    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype)
    )

    one_hot.requires_grad_()
    # input_embeds contains the embedding of adversarial suffix
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # embeds contains embedding of the entire prompt
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(custom_model.base_model, input_ids.unsqueeze(0)).detach()

    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],  # The part before adversarial suffix
            input_embeds,  # Adversarial suffix
            embeds[:, input_slice.stop:, :]  # The part after adversarial suffix
        ],
        dim=1
    )  # shape: (1, token_length, embedding_dimension)

    logits = custom_model(primary_activation, inputs_embeds=full_embeds)
    expanded_target = target.expand_as(logits)
    expanded_target = expanded_target.to(logits.dtype)
    expanded_target = expanded_target.to(device)

    loss = nn.BCEWithLogitsLoss()(logits, expanded_target)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad, loss, logits, one_hot


def sample_control(control_tokens, grad, batch_size, topk=256, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    # top_indices contains topk indices for a particular token position
    top_indices = (-grad).topk(topk, dim=1).indices  # shape: (num_tokens, topk)
    control_tokens = control_tokens.to(grad.device)

    original_control_tokens = control_tokens.repeat(batch_size, 1)  # shape: (batch_size, num_tokens)

    # new_token_pos contains the index at which the token will be replaced by another, for every adv suffix in the batch
    # e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
    new_token_pos = torch.arange(
        0,
        len(control_tokens),
        len(control_tokens) / batch_size,
        device=grad.device
    ).type(torch.int64)  # shape: (batch_size,)

    # new_token_val contains the token indices with which the existing tokens will be replaced
    new_token_val = torch.gather(
        # top_indices[new_token_pos] contains top candidates for every position that will be replaced
        top_indices[new_token_pos],  # shape: (batch_size, topk)
        1,
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )  # shape: (batch_size,)
    new_control_tokens = original_control_tokens.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)  # shape: (batch_size, suffix_length)

    return new_control_tokens


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):

    """
    Tokenizers are not invertible. If we have a string s,

    encoded_s = encode(s) [numbers]
    decoded_s = decode(encoded_s) [text]
    encoded_again = encode(decoded_s) [numbers]

    decoded_s might not be equal to s
    and,
    encoded_again might not be equal to encoded_s

    They might differ in length and/or content. Therefore, this function ensures each candidate has same length
    after decoding and encoding.

    """

    cands, count = [], 0

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))

    return cands


def get_logits(*, custom_model, tokenizer, input_ids, control_slice, primary_activation, test_controls=None, return_ids=False,
                               batch_size=3):
    """
    @Params
    input_ids:         input ids of entire prompt that includes the current adv suffix.
    control_slice:     input ids of adv suffix.
    test_controls:     batch of adv suffixes with only one replacement at a chosen position.
    batch_size:        It's different from the other batch_size (length of test_controls) which corresponds
                       to the number of suffixes created from (suffix_length * topk) candidates, while this
                       one corresponds to the number of suffixes to handle per iteration.
    """

    device = custom_model.base_model.get_input_embeddings().weight.device

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        # Select only the first max_len tokens. This truncation might remove the last few tokens.
        # But it's necessary to keep the shape consistent
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=device)
            for control in test_controls  # Get the encoding of all the candidate suffixes
        ]

        # Some candidates may have fewer tokens than max_len. Therefore, dummy token is added.

        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    # Locations in the prompt to replace (same for every row)
    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids;
        gc.collect()
        return forward(custom_model=custom_model, input_ids=ids, attention_mask=attn_mask, primary_activation=primary_activation, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(custom_model=custom_model, input_ids=ids, attention_mask=attn_mask, primary_activation=primary_activation, batch_size=batch_size)
        del ids;
        gc.collect()
        return logits


def forward(*, custom_model, input_ids, attention_mask, primary_activation, batch_size=3):
    logits_list = []

    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None

        logits = custom_model(primary_activation, input_ids=batch_input_ids, attention_mask=batch_attention_mask)

        logits_list.append(logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    return torch.cat(logits_list, dim=0)


def load_model_and_tokenizer(model_path, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map='auto').eval()

    return model, tokenizer


def get_prompt(index):
    data = json.load(open('/home/40456997@eeecs.qub.ac.uk/dataset_out_poisoned_v2.json', 'r'))
    return data[index]


def get_primary_activation(index):
    """
    This will return the primary activation of last layer only
    """
    activations = torch.load('/home/40456997@eeecs.qub.ac.uk/Activation/phi__3__3.8/test/poisoned_hidden_states_0_1000_20240717_115335.pt')

    return activations[0][index][-1]


def get_poisoned_activation(index):
    """
    This will return the poisoned activation of last layer only
    """
    activations = torch.load('/home/40456997@eeecs.qub.ac.uk/Activation/phi__3__3.8/test/poisoned_hidden_states_0_1000_20240717_115335.pt')

    return activations[1][index][-1]


def get_last_token_activations_single(text, tokenizer, model):

    device = model.get_input_embeddings().weight.device

    chat = [
            {
                "role": "system",
                "content": "you are a helpful assistant that will provide accurate answers to all questions.",
            },
            {"role": "user", "content": text},
        ]

    inputs = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

    inputs = inputs.to(device)

    with torch.no_grad():
        try:
            outputs = model(inputs, output_hidden_states=True)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(
                    "CUDA out of memory. Printing memory status and attempting to clear cache."
                )
                for i in range(torch.cuda.device_count()):
                    print(f"Memory summary for GPU {i}:")
                    print(torch.cuda.memory_summary(device=i))
                torch.cuda.empty_cache()
            raise e

    return outputs
