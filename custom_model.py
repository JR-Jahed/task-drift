import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, base_model, linear_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model

        W = linear_model.coef_
        b = linear_model.intercept_

        self.linear_layer = nn.Linear(in_features=W.shape[1], out_features=1)

        with torch.no_grad():
            self.linear_layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.linear_layer.bias.copy_(torch.tensor(b, dtype=torch.float32))
            self.linear_layer.weight.requires_grad = False
            self.linear_layer.bias.requires_grad = False

        # Move linear layer to base model's embedding device
        target_device = base_model.get_input_embeddings().weight.device
        self.linear_layer = self.linear_layer.to(target_device)

    def forward(self, primary_activation, input_ids=None, inputs_embeds=None, attention_mask=None):

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or input_embeds should be provided")

        if input_ids is not None:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.base_model(inputs_embeds=inputs_embeds, output_hidden_states=True)

        last_token_activation = outputs['hidden_states'][-1][:, -1]

        delta = (last_token_activation - primary_activation).float()

        logits = self.linear_layer(delta)

        return logits