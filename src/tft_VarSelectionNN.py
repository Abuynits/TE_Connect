from filepaths_constants import *
from model_constants import *
from data_constants import *
import torch.nn as nn
from tft_GRN import GRN


class VSN(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 output_size,
                 input_size=None,
                 context=None):
        super(VSN, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size = output_size
        self.input_size = input_size
        self.context = context

        # create GRN for a single input variable
        self.flat_grn = GRN(self.hidden_size,
                            input_size=self.input_size,
                            output_size=self.output_size,
                            dropout=self.dropout,
                            context=self.context)
        # create set of GRNs for each output feature present in model
        self.per_feature_grn = nn.ModuleList([GRN(self.hidden_size,
                                                  input_size=self.input_size,
                                                  output_size=self.output_size,
                                                  dropout=self.dropout,
                                                  context=self.context)
                                              for _ in range(self.output_size)])
        self.softmax = nn.Softmax()

    def forward(self, inp):
        # context is provided with static inputs
        if self.context:
            return self.forward_static(inp)
        else:
            return self.forward_non_static(inp)

    def forward_static(self, inp):
        # does not contain any static context
        embedding = inp
        # flatten all input static vars -> only have 1 dim bc they are static
        flatten = torch.flatten(embedding, start_dim=1)
        # only have 1 dim: only 1 GRN is needed
        grn_ml_out = self.flat_grn(flatten)

        v_xt = self.softmax(grn_ml_out)
        v_xt = self.unsqueeze(-1)
        # Add in another dim to loop over the time variables
        trans_emb_list = []

        for j in range(self.output_size):
            # have a static var so will be a fixed variable with fixed encodings
            # extract the grn for the specific variable used
            # extract all of the batches (first dim) extract all time steps (3rd dim)
            # j:j+1 is the col of the current var
            e_t = self.per_feature_grn[j](embedding[:, j:j + 1, :])
            trans_emb_list.append(e_t)
        # run grn through each variable through all the time steps
        # concat the list across the variable dimension
        trans_embedding = torch.cat(trans_emb_list, dim=1)
        # create final context by combining ctx for imp of each var at current ts
        # and imp of each time step for each var at diff time steps
        temporal_ctx = torch.sum(v_xt * trans_embedding, dim=1)
        return temporal_ctx, v_xt

    def forward_non_static(self, inp):
        # extract input and the static context
        # embedding contains encoded features static variables
        embedding, static_ctx = inp
        # will have batch, seq len, feat -> first dim is current ts
        time_step = embedding.shape[1]
        # embedding holds all of the input variables across all time steps leading to this pt
        # need to flatten them all
        flatten = embedding.view(-1, time_step, self.hidden_size * self.output_size)

        # add in dimension to account for matrix multiplication (to be 1 x ___) not (____
        static_ctx = static_ctx.unsqueeze(1)

        # now get the info about importance of vars for current time step for context
        grn_mlp_out = self.flat_grn((flatten, static_ctx))

        # v_xt is vector of var selection weights
        v_xt = self.softmax(grn_mlp_out)
        # v_xt holds info on each var for current time step
        # unsqueeze from timestep, var -> timestep, var, 1 to prep to loop ovr var
        v_xt = v_xt.unsqueeze(2)

        trans_emb_list = []
        # For each variable, loop over them and generate importance weights var at current time step
        for j in range(self.output_size):
            # elipsis accounts for varying variables: will change as add more time dims
            # j represents var of interest
            e_t = self.per_feature_grn[j](embedding[Ellipsis, j])
            # e_t is processed feature vector for variable j. Each e_t has own GRN
            trans_emb_list.append(e_t)
        # sack the extracted embeddings for the list
        trans_embedding = torch.stack(trans_emb_list, dim=-1)
        # combine the importance of each variable (v_xt) with
        # the importance of each variable at each time step
        temporal_ctx = torch.sum(v_xt * trans_embedding, dim=-1)

        return temporal_ctx, v_xt
