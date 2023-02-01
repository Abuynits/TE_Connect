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


# need to implement the position encoder as a class for the model
class Pos_Encoder(pl.LightningModule):
    def __init__(self):
        super.__init__()
        self.d_model = TIME_MAX_SEQ_LEN  # dimension of output of sublayers
        self.max_seq_len = TIME_MAX_SEQ_LEN  # max feature length of the pos encoder
        self.drop = TIME_POS_ENC_DROP  # dropout for time encoder
        # copy pasted from PyTorch tutorial
        position = torch.arange(TIME_MAX_SEQ_LEN).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(self.max_seq_len, 1, self.d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, inp):
        # shape: batch_size, enc_seq_len,dim val
        inp = inp + self.pe[:inp.size(0)]  # return the input seq added to the position encoder
        return self.drop(inp)


class time_encoder(pl.LightningModule):
    def __init__(self):
        super.__init__()

        self.input_features = INPUT_DATA_FEATURES  # num vars as input to the model
        self.dim_val = TIME_ENC_DIM_VAL  # hyper parameter for input dim throughout encoder
        self.n_head = TIME_ENC_HEAD_COUNT
        self.enc_layer_count = TIME_ENC_LAYER_COUNT
        self.enc_inp_layer = nn.Linear(  # converts the data to the dimensions for the encoder
            in_features=self.input_features,
            out_features=self.dim_val
        )

        self.pos_enc = Pos_Encoder()
        # encoder input layers produces output size dim_val (512)

        # create encoders wiht n.TransfoermerEncoderLayer
        # will automatically have attention, feed forward and add and normalize layer
        self.enc_block = nn.TransformerEncoderLayer(self.dim_val, self.n_head, batch_first=True)
        # then need to pass this block to create multiple coppies in an encoder
        # norm: optional param - need to pass in null as enc_block already normalizes it
        self.enc = nn.TransformerEncoder(self.enc_block, num_layers=self.enc_layer_count, norm=None)

    def forward(self, inp):
        # pass through input for the decoder
        # input size is: [batch size, inp seq len,num input features]
        x = self.enc_inp_layer(inp)
        # output is: [batch_size,source,dim_val] where dim_val is 512 (arbitrarily preset)
        x = self.pos_enc(x)
        # output is still: [batch size, inp seq len,num input features]
        x = self.enc(x)
        # output is still: [batch size, inp seq len,num input features]
        return x


class time_decoder(pl.LightningModule):
    def __init__(self):
        super.__init__()
        self.input_features = INPUT_DATA_FEATURES
        self.dim_val = TIME_DEC_DIM_VAL
        self.nheads = TIME_DEC_HEAD_COUNT
        self.dec_layer_count = TIME_DEC_LAYER_COUNT
        # will be passing in the variables as input to the decoder - need to get them to size 512
        # this way match output of encoder
        self.dec_inp_layer = nn.Linear(
            in_features=self.input_features,
            out_features=self.dim_val
        )
        # create a decoder block with number of heads and normal and attention built into it::
        self.dec_block = nn.TransformerEncoderLayer(
            d_model=self.dim_val,
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
        self.linear_mapping = nn.Linear(
            in_features=self.dim_val,
            out_features=PREDICT
        )

    def forward(self, enc_out, target, input_mask, target_mask):
        # the target is the last element in seq that is used for predictions - need to convert it to right size
        # x shape: target_seq_len,batch_size,features
        x = self.dec_inp_layer(target)
        # x shape: target_seq_len,batch_size,dim_val
        # run the decoder, performing a mask on the target, nad on the input
        dec_out = self.decoder(
            tgt=x,
            memory=enc_out,
            tgt_mask=target_mask,
            memory_mask=input_mask
        )
        # get: [batch_size,target_seq_len,dim_val]
        # now pass through linear mapping to convert back to size of features to be used
        mapped_dec_out = self.linear_mapping(dec_out)

        return mapped_dec_out


# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
class time_transformer(pl.LightningModule):
    def __init__(self):
        super.__init__()
        self.enc = time_encoder()
        self.dec = time_decoder()

    def forward(self, inp, target, inp_mask, target_mask):
        # need to convert the target sequence to a dimension that can be inputed to the decoder
        enc_out = self.enc(inp)
        dec_inp = enc_out
        out = self.dec(dec_inp, target, inp_mask, target_mask)
        return out
