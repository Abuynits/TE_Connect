# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# next steps: implement a time transformer for this model

"""
Popular time series preprocessing techniques include:
Just scaling to [0, 1] or [-1, 1]
Standard Scaling (removing mean, dividing by standard deviation)
Power Transforming (using a power function to push the data to a more normal distribution, typically used on skewed data / where outliers are present)
Outlier Removal
Pairwise Diffing or Calculating Percentage Differences
Seasonal Decomposition (trying to make the time series stationary)
Engineering More Features (automated feature extractors, bucketing to percentiles, etc)
Resampling in the time dimension
Resampling in a feature dimension (instead of using the time interval, use a predicate on a feature to re-arrange your time steps — for example when recorded quantity exceeds N units)
Rolling Values
Aggregations
Combinations of these techniques

"""
from filepaths_constants import *
from model_constants import *
from dl_ds import *
from filepaths_constants import *


# need to implement the position encoder as a class for the model
class Pos_Encoder(pl.LightningModule):
    def __init__(self):
        super(Pos_Encoder, self).__init__()
        self.dim_val = TIME_DEC_DIM_VAL  # dimension of output of sublayers
        self.max_seq_len = TIME_MAX_SEQ_LEN  # max feature length of the pos encoder
        self.drop = TIME_POS_ENC_DROP  # dropout for time encoder
        self.dropout = nn.Dropout(self.drop)
        # copy pasted from PyTorch tutorial
        position = torch.arange(TIME_MAX_SEQ_LEN).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.dim_val, 2) * (-math.log(10000.0) / self.dim_val))

        pe = torch.zeros(self.max_seq_len, 1, self.dim_val)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, inp):
        # shape: batch_size, enc_seq_len,dim val
        inp = inp + self.pe[:inp.size(0)]  # return the input seq added to the position encoder
        x = self.dropout(inp)

        return x


class time_encoder(pl.LightningModule):
    def __init__(self):
        super(time_encoder, self).__init__()

        self.input_features = INPUT_DATA_FEATURES  # num vars as input to the model
        self.dim_val = TIME_ENC_DIM_VAL  # hyper parameter for input dim throughout encoder
        self.n_head = TIME_ENC_HEAD_COUNT
        self.layer_count = TIME_ENC_LAYER_COUNT
        self.dim_feed_forward = TIME_ENC_DIM_FEED_FORWARD
        self.drop = TIME_ENC_DROP

        self.enc_inp_layer = nn.Linear(  # converts the data to the dimensions for the encoder
            in_features=self.input_features,  # number of input vars
            out_features=self.dim_val  # the dim of model - all sublayers produce this dim
        )

        self.pos_enc = Pos_Encoder()
        # encoder input layers produces output size dim_val (512)

        # create encoders with n.TransformerEncoderLayer
        # will automatically have attention, feed forward and add and normalize layer
        self.enc_block = nn.TransformerEncoderLayer(
            d_model=self.dim_val,
            nhead=self.n_head,
            dim_feedforward=self.dim_feed_forward,
            dropout=self.drop,
            batch_first=True
        )
        # then need to pass this block to create multiple coppies in an encoder
        # norm: optional param - need to pass in null as enc_block already normalizes it
        self.enc = nn.TransformerEncoder(self.enc_block,
                                         num_layers=self.layer_count,
                                         norm=None)

    def forward(self, inp):
        # pass through input for the decoder
        # input size is: [batch size, inp seq len,num input features]
        if TIME_VERBOSE:
            print(f"\tenc: input shape:{inp.shape}")
        x = self.enc_inp_layer(inp)
        if TIME_VERBOSE:
            print(f"\tenc: after enc input layer:{x.shape}")
        # output is: [batch_size,source,dim_val] where dim_val is 512 (arbitrarily preset)
        x = self.pos_enc(x)
        # output is still: [batch size, inp seq len,num input features]
        if TIME_VERBOSE:
            print(f"\tenc: inp shape after positional encoder:{x.shape}")
        x = self.enc(x)
        if TIME_VERBOSE:
            print(f"\tenc: output:{x.shape}")
        # output is still: [batch size, inp seq len,num input features]
        return x


class time_decoder(pl.LightningModule):
    def __init__(self):
        super(time_decoder, self).__init__()
        self.input_features = OUTPUT_DATA_FEATURES
        self.target_input_features = INPUT_DATA_FEATURES
        self.dim_val = TIME_DEC_DIM_VAL
        self.nheads = TIME_DEC_HEAD_COUNT
        self.dec_layer_count = TIME_DEC_LAYER_COUNT
        self.feed_feed_forward_dim = TIME_DEC_DIM_FEED_FORWARD
        self.drop = TIME_DEC_DROP
        # will be passing in the variables as input to the decoder - need to get them to size 512
        # this way match output of encoder
        self.dec_inp_layer = nn.Linear(
            in_features=self.target_input_features,
            out_features=self.dim_val
        )

        # create a decoder block with number of heads and normal and attention built into it::
        self.dec_block = nn.TransformerDecoderLayer(
            d_model=self.dim_val,
            dim_feedforward=self.feed_feed_forward_dim,
            dropout=self.drop,
            nhead=self.nheads,
            batch_first=True
        )
        # create a decoder based of the decoder blocks
        self.dec = nn.TransformerDecoder(
            decoder_layer=self.dec_block,
            num_layers=self.dec_layer_count,
            norm=None
        )
        # linear mapping layer:
        # in_features: equal to output sequence length multiplied by
        # need to map from the hidden sequence length (dim_val) to the prediction sequence
        self.linear_output_mapping = nn.Linear(
            in_features=self.dim_val,
            out_features=OUTPUT_DATA_FEATURES
        )
        # multiply dim_val by OUTPUT_DATA_COL to account for final mapping
        self.linear_input_mapping = nn.Linear(
            in_features=self.dim_val,
            out_features=INPUT_DATA_FEATURES
        )

    def forward(self, enc_out, target, input_mask, target_mask):
        # the target is the last element in seq that is used for predictions - need to convert it to right size
        # x shape: target_seq_len,batch_size,features
        if TIME_VERBOSE:
            print(f"\ttarget shape: {target.shape}")
        x = self.dec_inp_layer(target)
        if TIME_VERBOSE:
            print(f"\tdec: decoder inp layer:{x.shape}")
        # x shape: target_seq_len,batch_size,dim_val
        # run the decoder, performing a mask on the target, nad on the input
        dec_out = self.dec(
            tgt=x,
            memory=enc_out,
            tgt_mask=target_mask,
            memory_mask=input_mask
        )
        if TIME_VERBOSE:
            print(f"\tdec: decoder output: {dec_out.shape}")
        # get: [batch_size,target_seq_len,dim_val]
        # now pass through linear mapping to convert back to size of features to be used
        mapped_dec_out = self.linear_output_mapping(dec_out)  # the output mapping used as the final result
        mapped_dec_inp = self.linear_input_mapping(dec_out)  # use to feed back into the model
        if TIME_VERBOSE:
            print(f"\tdec: mapped_dec output: {mapped_dec_out.shape}")
            print(f"\tdec: mapped_dec input: {mapped_dec_inp.shape}")

        return mapped_dec_out, mapped_dec_inp


# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
class time_transformer(pl.LightningModule):
    def __init__(self):
        super(time_transformer, self).__init__()
        self.enc = time_encoder()
        self.dec = time_decoder()
        print_model(self)

    def forward(self, inp, target, inp_mask, target_mask):
        # need to convert the target sequence to a dimension that can be inputed to the decoder

        # inp is all of the sequence that is taken as input
        # target is the target output but shifted over by 1
        # if you get [1,2,3,4,5] as src, then that target should be [2,3,4,5,6]
        if TIME_VERBOSE:
            print()
            print(f"input: {inp.shape}")
            print(f"target: {target.shape}")
            print(f"input mask: {inp_mask.shape}")
            print(f"target mask: {target_mask.shape}")
        enc_out = self.enc(inp)
        if TIME_VERBOSE:
            print(f"enc_out/dec_inp: {enc_out.shape}")
        dec_inp = enc_out
        out, new_inp = self.dec(dec_inp, target, inp_mask, target_mask)
        if TIME_VERBOSE:
            print(f"out shape: {out.shape}")
        return out, new_inp


def generate_mask(dim1, dim2, device):
    # dim1: for both input and output - is the target len
    # dim2: for src, this is encoder seq length
    #     : for target - this is target_seq length
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1).to(device)


# TODO: go over code and make it work better

def time_predict(model, inp, contain_batch=False, future_time_steps=PREDICT):
    if TIME_PRED_VERBOSE:
        print("\noriginal inp:", inp.shape)
    # model: the model being used
    # future_time_steps: the number of steps to predict into the future - can change them
    # the input should always be batch_first:
    if not contain_batch and inp.dim() != 3:
        if TIME_VERBOSE:
            print("adding dim 1")
        inp = inp.unsqueeze(1)  # add in dim of 1 if asked to predict on single input
    if TIME_PRED_VERBOSE:
        print("processed inp shape:", inp.shape)
    # # Take the last value of the target variable in all batches in src and make it tgt
    target = inp[:, -1, :]  # in shape [batches,last unit]
    target = target[:, None, :]  # add in single dim -> [batches,1,last unit]
    if TIME_PRED_VERBOSE:
        print("target shape:", target.shape)

    inp_mask_dim = inp.shape[1]
    for _ in range(future_time_steps - 1):
        tgt_mask_dim = target.shape[1]
        if TIME_PRED_VERBOSE:
            print(f"dim_a:{tgt_mask_dim},dim_b:{inp_mask_dim}")
        # [target sequence length, target sequence length]
        target_mask = generate_mask(
            dim1=tgt_mask_dim,
            dim2=tgt_mask_dim,
            device=DEVICE
        )
        if TIME_PRED_VERBOSE:
            print("input mask:", target_mask.shape)
        # [target sequence length, encoder sequence length]
        inp_mask = generate_mask(
            dim1=tgt_mask_dim,
            dim2=inp_mask_dim,
            device=DEVICE
        )
        if TIME_PRED_VERBOSE:
            print("input mask:", inp_mask.shape)
        model_pred, next_model_inp = model(inp, target, inp_mask, target_mask)
        if TIME_PRED_VERBOSE:
            print("All model prediction shape", model_pred.shape)
            print("All model next inp shape", next_model_inp.shape)
        # output from model is [batch_size,output_length]
        # need to unsqeeze to add in 3rd dim to make compatible with the currect dims
        last_next_inp_shape = next_model_inp[:, -1, :]
        last_next_inp_shape = last_next_inp_shape[:, None, :]
        # last_model_pred = model_pred[:, -1, :].unsqueeze(-1)
        if TIME_PRED_VERBOSE:
            """
            processed inp shape: torch.Size([2048, 10, 3])
            input target shape: torch.Size([2048, 1, 3])
            """
            print("last model prediciton shape", model_pred.shape)
            print("last model new input shape", next_model_inp.shape)
            print("last model new input shape", last_next_inp_shape.shape)
        # now add on the prediction to target:
        target = torch.cat((target, last_next_inp_shape.detach()), 1)  # concatonate along batches dimensions
        if TIME_PRED_VERBOSE:
            print("target size:", target.shape)
        # get the size of the number of input features in the data used in the input
        dim_a = target.shape[1]
        dim_b = inp.shape[1]
        if TIME_PRED_VERBOSE:
            print(f"dim_a:{dim_a},dim_b:{dim_b}")

    dim_a = target.shape[1]
    dim_b = inp.shape[1]

    final_tgt_mask = generate_mask(
        dim1=dim_a,
        dim2=dim_a,
        device=DEVICE
    )
    final_inp_mask = generate_mask(
        dim1=dim_a,
        dim2=dim_b,
        device=DEVICE
    )
    if TIME_PRED_VERBOSE:
        print("final input mask:", final_inp_mask.shape)
    if TIME_PRED_VERBOSE:
        print("final input mask:", final_tgt_mask.shape)
    final_prediction, _ = model(inp, target, final_inp_mask, final_tgt_mask)
    if TIME_PRED_VERBOSE:
        print("final prediction shape:", final_prediction.shape)
    return final_prediction
