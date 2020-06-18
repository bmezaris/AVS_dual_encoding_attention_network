import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from loss import TripletLoss
from basic.bigfile import BigFile
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we))
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """

    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features


class AttentionLayer(nn.Module):
    """
        Attention Layer
    """

    def __init__(self, fc_input, hidden_size):
        super(AttentionLayer, self).__init__()

        self.input_size = fc_input
        self.hidden_size = hidden_size
        self.output_size = fc_input

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc1.bias.data.fill_(0)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.fc2.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        xavier_init_fc(self.fc1)
        xavier_init_fc(self.fc2)

    def forward(self, inputs):
        W_s1 = self.fc1(inputs)
        tanh = self.tanh(W_s1)
        W_s2 = self.fc2(tanh)
        output = self.softmax(W_s2)
        return output


class Video_multilevel_encoding(nn.Module):
    """
    """

    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate
        # self.rnn_output_size = opt.text_rnn_size * 2
        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # Attention
        self.atten = AttentionLayer(self.rnn_output_size, self.rnn_output_size)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])

        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

    def forward(self, videos, gru_text_out):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos

        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        # H_new = Variable(torch.zeros(gru_init_out.size(0), gru_init_out.size(1), self.rnn_output_size)).cuda()

        H_new_tmp_2 = self.atten(gru_init_out)
        H_new_2 = gru_init_out * H_new_tmp_2

        for i, batch in enumerate(H_new_2):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1, -1, gru_init_out.size(2))  # (N,C,F1)
        # gru_init_out = gru_init_out * vidoes_mask
        gru_init_out = H_new_2 * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)


class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate

        # visual bidirectional rnn encoder
        self.embed_w2v = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.embed_bert = BertEmbedding(opt)
        self.rnn = nn.GRU(opt.word_dim_concat, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])

        # multi fc layers
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 1268 and self.we_parameter is not None:
            self.embed_w2v.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed_w2v.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows
        # tmp = (Variable(cap_wids).data).cpu().numpy()
        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids_w2v = self.embed_w2v(cap_wids)
        cap_wids_bert = self.embed_bert(tokens_tensor_padded, segments_tensors_padded)
        cap_wids_concat = torch.cat([cap_wids_w2v, cap_wids_bert], dim=2)

        packed = pack_padded_sequence(cap_wids_concat, lengths, batch_first=True)
        # tmp = (Variable(packed).data).cpu().numpy()
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)

        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features, gru_out


class BertEmbedding(nn.Module):

    def __init__(self, opt):
        super(BertEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load pre-trained model (weights)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bert_model.eval()

    def forward(self, tokens_tensor, segments_tensors):
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers = self.bert_model(tokens_tensor, segments_tensors)

        last_hidden_state = encoded_layers[0]  # The last hidden-state is the first element of the output tuple

        return last_hidden_state


class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()

    def forward_loss(self, cap_emb, vid_emb, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss = self.criterion(cap_emb, vid_emb)
        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
            self.logger.update('Le', loss.data[0], vid_emb.size(0))
        else:
            self.logger.update('Le', loss.item(), vid_emb.size(0))
        return loss

    def train_emb(self, videos, captions, lengths, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_emb(videos, captions, False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb)

        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value


class Dual_Encoding(BaseModel):
    """
    dual encoding network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.text_encoding = Text_multilevel_encoding(opt)
        self.vid_encoding = Video_multilevel_encoding(opt)
        # print(self.vid_encoding)
        # print(self.text_encoding)
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation,
                                         cost_style=opt.cost_style,
                                         direction=opt.direction)

        params = list(self.text_encoding.parameters())
        params += list(self.vid_encoding.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0

    def forward_emb(self, videos, targets, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

        # text data
        captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert = targets
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        if tokens_tensor_padded is not None:
            tokens_tensor_padded = Variable(tokens_tensor_padded, volatile=volatile)
            if torch.cuda.is_available():
                tokens_tensor_padded = tokens_tensor_padded.cuda()

        if segments_tensors_padded is not None:
            segments_tensors_padded = Variable(segments_tensors_padded, volatile=volatile)
            if torch.cuda.is_available():
                segments_tensors_padded = segments_tensors_padded.cuda()

        text_data = (
            captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert)

        cap_emb, gru_text_out = self.text_encoding(text_data)
        vid_emb = self.vid_encoding(videos_data, gru_text_out)

        return vid_emb, cap_emb


NAME_TO_MODELS = {'dual_encoding_ATV_w2v_bert': Dual_Encoding}


def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.' % name
    return NAME_TO_MODELS[name]
