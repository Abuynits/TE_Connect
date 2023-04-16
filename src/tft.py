from tft_GRN import *
from tft_VarSelectionNN import *
from tft_InterpretableMHA import *
from seq2seq_arch import print_model


class TemporalFusionTransformer(nn.Module):
    def __init__(self):
        super(TemporalFusionTransformer, self).__init__()
        self.time_steps = TFT_TIME_STEPS
        self.input_size = TFT_INPUT_SIZE
        self.output_size = TFT_OUTPUT_SIZE
        self.cat_counts = TFT_CATEGORY_COUNTS
        self.n_cat_vars = TFT_N_CAT_VARS
        self.n_reg_vars = self.input_size - self.n_cat_vars
        self.n_multiprocessing_workers = TFT_MULTIPROCESSING_WORKERS
        self.n_enc_steps = TFT_ENC_STEPS
        self.n_stacks = TFT_STACKS
        self.n_heads = TFT_N_HEADS

        # relevant indices for TFT
        self._input_obs_loc = TFT_INPUT_OBS_LOC
        self._static_input_loc = TFT_STATIC_INPUT_LOC
        self._known_regular_input_idx = TFT_REGULAR_INPUTS
        self._known_categorical_input_idx = TFT_CATEGORICAL_INPUTS

        self.num_non_static_historical_inputs = TFT_HIST_INPUTS
        self.num_non_static_future_inputs = TFT_FUTURE_INPUTS

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
        print_model(self)

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
            # access the ith location corresponding with the variable location for all batches adn time steps
            static_categorical_inputs = [self.cat_var_embeddings[i](categorical_inputs[Ellipsis, i])[:, 0, :]
                                         for i in range(self.n_cat_vars)
                                         if i + self.n_reg_vars in self._static_input_loc]
            static_inputs = torch.stack(static_regular_inputs + static_categorical_inputs, dim=1)
        else:
            static_inputs = None

        # Target input
        obs_inputs = torch.stack([self.reg_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                                  for i in self._input_obs_loc], dim=-1)

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(self.n_cat_vars):
            if i not in self._known_categorical_input_idx \
                    and i not in self._input_obs_loc:
                e = self.cat_var_embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(self.n_reg_vars):
            if i not in self._known_regular_input_idx \
                    and i not in self._input_obs_loc:
                e = self.reg_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, dim=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [self.reg_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                                for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
        # print('known_regular_inputs')
        # print([print(emb.shape) for emb in known_regular_inputs])

        known_categorical_inputs = [self.cat_var_embeddings[i](categorical_inputs[Ellipsis, i])
                                    for i in self._known_categorical_input_idx
                                    if i + self.n_reg_vars not in self._static_input_loc]
        # print('known_categorical_inputs')
        # print([print(emb.shape) for emb in known_categorical_inputs])

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, dim=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, all_inputs):
        reg_inputs = all_inputs[:, :, :self.n_reg_vars].to(torch.float)
        cat_inputs = all_inputs[:, :, self.n_reg_vars:].to(torch.long)

        # gather all embedded inputs
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(reg_inputs, cat_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            # extract the history leading up to n_enc_steps
            hist_inp = torch.cat([
                unknown_inputs[:, :self.n_enc_steps, :],
                known_combined_layer[:, :self.n_enc_steps, :],
                obs_inputs[:, :self.n_enc_steps, :]
            ], dim=-1)
        else:
            hist_inp = torch.cat([
                known_combined_layer[:, :self.n_enc_steps, :],
                obs_inputs[:, :self.n_enc_steps, :]
            ], dim=-1)

        # extract history only after the n_enc_steps
        future_inp = known_combined_layer[:, self.n_enc_steps:, :]

        # get the static context from static vsn
        static_enc, sparse_weights = self.static_vsn(static_inputs)

        # extract static variables to be used in encder selection
        static_ctx_var_sel = self.static_var_sel_grn(static_enc)
        # extract static context for enrichment at temporal layer:
        static_context_enrich = self.static_ctx_enrichment_grn(static_enc)
        # extract variables to be used at each lstm layer
        static_ctx_state_h = self.static_ctx_state_h_grn(static_enc)
        # extract variables to be used at the grn layer in the temporal fusion decoder
        static_ctx_state_c = self.static_ctx_state_c_grn(static_enc)

        # select historical features and their flags
        hist_feat, hist_flags = self.temporal_hist_vsn(hist_inp, static_ctx_var_sel)

        # select known future inputs and their flags:
        future_feat, future_flags = self.temporal_hist_vsn(future_inp, static_ctx_var_sel)

        # pass in the historic data through the current lstm
        hist_out, (hist_out_h, hist_out_c) = self.hist_lstm(hist_feat, (static_ctx_state_h.unsqueeze(0),
                                                                        static_ctx_state_c.unsqueeze(0)))

        # pass in the future features through the future lstm:

        future_out, _ = self.future_lstm(future_feat, (hist_out_h, hist_out_c))

        # residual layer over the inputs into the model by adding the input embeddings
        # to output of historing and future data
        all_inp_embeddings = torch.cat((hist_feat, future_feat), dim=1)
        # combine all lstm outputs:
        all_lstm_out = torch.cat((hist_out, future_out), dim=1)
        # apply the skip connection (overlay map over lstm inputs / outputs)
        temporal_feature_layer = self.post_seq_enc_gate_add_norm(all_lstm_out, all_inp_embeddings)
        # enrich temporal features with static context (again)
        # expand on axis 1 to account for handling time input
        expanded_static_ctx = static_context_enrich.expand(1)

        enriched_temporal = self.static_enrichment((temporal_feature_layer, expanded_static_ctx))

        # ============ MHA =========== #
        attn_mask = self.get_decoder_mask(enriched_temporal)
        # calculate the interpolatable head pointer
        x, attn = self.self_attn_layer(enriched_temporal,
                                       enriched_temporal,
                                       enriched_temporal,
                                       mask=attn_mask)
        # apply final gate attn and normalization on the gate
        x = self.post_attn_gate_add_norm(x, attn)
        # final skip connection for transformer:
        final_skip = self.post_attn_gate_add_norm(x, temporal_feature_layer)

        outputs = self.output_feed_forward(final_skip[Ellipsis, self.n_enc_steps:, :])

        return outputs
