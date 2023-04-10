from model_constants import *
from tft_GRN import *
from tft_VarSelectionNN import *
from tft_InterpretableMHA import *


class TemporalFusionTransformer(nn.Module):
    def __init__(self):
        super(TemporalFusionTransformer, self).__init()
        self.time_steps = TFT_TIME_STEPS
        self.input_size = TFT_INPUT_SIZE
        self.output_size = TFT_OUTPUT_SIZE
        self.cat_counts = TFT_CATEGORY_COUNTS
        self.n_cat_vars = len(self.category_counts)
        self.n_reg_vars = self.input_size - self.n_cat_vars
        self.n_multiprocessing_workers = TFT_MULTIPROCESSING_WORKERS
        self.n_heads = TFT_N_HEADS

        # relevant indices for TFT
        self._input_obs_loc = TFT_INPUT_OBS_LOC
        self._static_input_loc = TFT_STATIC_INPUT_LOC
        self._known_regular_input_idx = TFT_KNOWN_REGULAR_INPUTS
        self._known_categorical_input_idx = TFT_KNOWN_CATEGORICAL_INPUTS

        self.num_non_static_historical_inputs = self.get_historical_num_inputs()
        self.num_non_static_future_inputs = self.get_future_num_inputs()
        self.col_def = TFT_COL_DEF
        self.quantiles = TFT_QUANTILES

        self.hidden_size = TFT_HIDDEN_SIZE
        self.dropout = TFT_DROPOUT

        self._input_place_holder = None
        self._attn_components = None
        self._pred_parts = None

        # ============ Build TFT ============ #
        # build embeddings:
        self.build_embeddings()

        # build static context networks
        self.build_static_ctx_networks()

        # build VSN
        self.build_VSN()

        # build lstm
        self.build_lstm()

        # build GLU for after lstm encoder decoder and layernorm
        self.build_post_lstm_gate_add_norm()

        # build Static Enrichment layer
        self.build_static_enrichment()

        # build decoder mha
        self.build_temporal_self_attn()

        # build positionwise dec
        self.build_position_wise_feed_forward()

        # build ouutput feed forward
        self.build_output_feed_forward()

        # ============ init weights ============ #
        self.init_weights()

    def init_weights(self):
        for name, param in self.parameters():
            # only init the lstm layers as the rest are init in other modules
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                # make lstm start out at  same place
                nn.init.xavier_uniform_(param)
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:
                nn.init.orthogonal_(param)
            elif 'lstm' in name and 'bias' in name:
                nn.init.zeros_(param)

    def get_historical_num_inputs(self):

        obs_inputs = [i for i in self._input_obs_loc]

        known_regular_inputs = [i for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]

        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.n_reg_vars not in self._static_input_loc]

        wired_embeddings = [i for i in range(self.n_cat_vars)
                            if i not in self._known_categorical_input_idx
                            and i not in self._input_obs_loc]

        unknown_inputs = [i for i in range(self.n_reg_vars)
                          if i not in self._known_regular_input_idx
                          and i not in self._input_obs_loc]

        return len(obs_inputs + known_regular_inputs + known_categorical_inputs + wired_embeddings + unknown_inputs)

    def get_future_num_inputs(self):

        known_regular_inputs = [i for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]

        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.n_reg_vars not in self._static_input_loc]

        return len(known_regular_inputs + known_categorical_inputs)

    def build_embeddings(self):
        # nn.embeddings: A simple lookup table that stores embeddings of a fixed dictionary and size.
        # for each categorical variable, create an embedding layer with the provided hidden size
        # map the categorical variable to quant var representing its unique embedding
        self.cat_var_embeddings = nn.ModuleList([nn.Embedding(self.cat_counts[i],
                                                              self.hidden_size)
                                                 for i in range(self.n_cat_vars)])

        # for regular vars, dont need to generate an embedding,
        # need to create a mapping from that one timestep value to hidden_size
        self.reg_var_embeddings = nn.ModuleList(nn.Linear(1, self.hidden_size)
                                                for i in range(self.n_reg_vars))

    def build_VSN(self):
        # input size is count of input vars * the hidden size for each
        self.static_vsn = VSN(
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            output_size=self.output_size,
            input_size=self.hidden_size * len(self._static_input_loc)
        )

        # provide input about imp for future VSN
        self.temporal_hist_vsn = VSN(
            hidden_size=self.hidden_size,
            input_size=self.num_non_static_historical_inputs * self.hidden_size,
            output_size=self.num_non_static_historical_inputs,
            dropout=self.dropout,
            context=self.hidden_size  # context of static will be in the shape of a hidden size
        )
        # provide info about imp for future VSN
        self.temporal_future_vsn = VSN(
            hidden_size=self.hidden_size,
            input_size=self.num_non_static_historical_inputs * self.hidden_size,
            output_size=self.num_non_static_future_inputs,
            dropout=self.dropout,
            context=self.hidden_size
        )

    def build_static_ctx_networks(self):
        # for all layers, init them with the hidden size (the size of the static tensors)
        self.static_var_sel_grn = GRN(self.hidden_size,
                                      self.dropout)

        self.static_ctx_enrichment_grn = GRN(self.hidden_size,
                                             self.dropout)

        self.static_ctx_state_h_grn = GRN(self.hidden_layer_size,
                                          self.dropout)

        self.static_ctx_state_c_grn = GRN(self.hidden_layer_size,
                                          self.dropout)

    def build_lstm(self):
        self.hist_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )

        self.future_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_enc_gate_add_norm = GateAddNorm(
            self.hidden_size,
            self.hidden_size,
            self.dropout,
        )

    def build_static_enrichment(self):

        self.static_enrichment = GRN(
            self.hidden_size,
            dropout=self.dropout,
            context=self.hidden_size
        )

    def build_temporal_self_attn(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(n_head=self.n_heads,
                                                               hidden_size=self.hidden_size,
                                                               dropout=self.dropout_rate)

        self.post_attn_gate_add_norm = GRN(self.hidden_layer_size,
                                           self.hidden_layer_size,
                                           self.dropout_rate)

    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GRN(self.hidden_layer_size,
                                    dropout=self.dropout)

        self.post_tfd_gate_add_norm = GateAddNorm(self.hidden_layer_size,
                                                  self.hidden,
                                                  self.dropout)

    def build_output_feed_forward(self):
        self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size,
                                                   self.output_size * len(self.quantiles))

    def get_decoder_mask(self, self_attn_inputs):
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        # torch.eye creates identity matrix
        # torch.cumsum returns the number of elements in the dimension provided
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs, 1, 1).to(torch.float32)  # creates upper triangle mask
        # for each element, take the left value, and sum it wit hthe right, and repeat with enw values
        return mask.to(DEVICE)

    def get_tft_embeddings(self, regular_inputs, categorical_inputs):
        # static input:
        if self._static_input_loc:
            # Access the static inputs for regular (quantitative) static variables
            # extracting the batch, loc and variable
            static_regular_inputs = [self.reg_var_embeddings[i](regular_inputs[:, 0, i:i + 1])
                                     for i in range(self.n_reg_vars)
                                     if i in self._static_input_loc]
            # static categorical inputs: represent each one through Elipsis (not know its representation
            # acccess the ith location corresponding with the variable location for all batches adn time steps
            static_categorical_inputs = [self.cat_var_embeddings[i](categorical_inputs[Ellipsis, i])[:, 0, :]
                                         for i in range(self.n_cat_vars)
                                         if i + self.n_reg_vars in self._static_input_loc]
            static_inputs = torch.stack(static_regular_inputs + static_categorical_inputs, dim=1)
        else:
            static_inputs = None

        # Target input
        trg_inp = torch.stack([self.reg_var_embeddings[i](regular_inputs)])