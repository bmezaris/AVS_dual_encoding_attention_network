from __future__ import print_function
import pickle
import os
import sys
import time

import torch

from ATV_w2v_bert_model import get_model, get_we_parameter
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

import logging
import json
import numpy as np
import pickle

import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from scipy.spatial import distance
from util.vocab import clean_str

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import itertools

VIDEO_MAX_LEN = 64


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


def cosine_calculate(matrix_a, matrix_b):
    result = distance.cdist(matrix_a, matrix_b, 'cosine')
    return result.tolist()


def groupc(listtest):
    for x, y in itertools.groupby(enumerate(listtest), lambda (a, b): b - a):
        y = list(y)
        yield y[0][1], y[-1][1]


def text2Berttext(caption_text, tokenizer):
    tokenized_text = tokenizer.tokenize(caption_text)
    retuned_tokenized_text = tokenized_text[:]

    res = [coun for coun, ele in enumerate(tokenized_text) if ('##' in ele)]

    res2 = list(groupc(res))

    for ree in res2:
        start = ree[0] - 1
        end_ = ree[1]
        tmp_token = ''
        for i in range(start, end_ + 1):
            # print tokenized_text[i].replace('##', '')
            tmp_token = tmp_token + tokenized_text[i].replace('##', '')
        # print tmp_token
        for i in range(start, end_ + 1):
            retuned_tokenized_text[i] = tmp_token
    return ' '.join(retuned_tokenized_text)


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('--evalpath', type=str, default=ROOT_PATH, help='path to evaluation video features. (default: %s)' % ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=1, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video (default: 1)')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    evalpath = opt.evalpath
    testCollection = opt.testCollection
    batchsize = opt.batch_size

    # n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    saveFile_AVS16 = (opt.logger_name + '/AVS16_' + testCollection + '_Dense_Dual_model_bin.txt')
    saveFile_AVS17 = (opt.logger_name + '/AVS17_' + testCollection + '_Dense_Dual_model_bin.txt')
    saveFile_AVS18 = (opt.logger_name + '/AVS18_' + testCollection + '_Dense_Dual_model_bin.txt')

    if os.path.exists(saveFile_AVS17):
        sys.exit(0)

    queriesFile = 'AVS/tv16_17_18.avs.topics_parsed.txt'
    lineList = [line.rstrip('\n') for line in open(queriesFile)]

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']

    if not hasattr(options, 'do_visual_feas_norm'):
        setattr(options, "do_visual_feas_norm", 0)

    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection)
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    # data loader prepare
    caption_files = {'test': os.path.join(evalpath, testCollection, 'TextData', '%s.caption.txt' % testCollection)}
    img_feat_path = os.path.join(evalpath, testCollection, 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(evalpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))}
    # video2frames = None

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # initialize word embedding
    options.we_parameter = None
    if options.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        options.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    # switch to evaluate mode
    model.val_start()

    video2frames = video2frames['test']
    videoIDs = [key for key in video2frames.keys()]

    # Queries embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    queryEmbeddingsTMP = []
    for quer in lineList:
        videBatch = videoIDs[0]  # a dummy video
        data = dataLoadedVideoText_one(video2frames, videBatch, visual_feats['test'], quer, bow2vec, rnn_vocab, tokenizer, options)
        videos, captions = collate_frame_gru_fn(data)
        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)
        # preserve the embeddings by copying from gpu and converting to numpy
        cap_embs = cap_emb.data.cpu().numpy().copy()
        queryEmbeddingsTMP.append(cap_embs[0])

    queryEmbeddings = np.stack(queryEmbeddingsTMP)
    # print(queryEmbeddings.shape)

    start = time.time()
    VideoIDS = []
    errorlistList = []

    for i in xrange(0, len(videoIDs), batchsize):
        videBatch = videoIDs[i:i + batchsize]
        VideoIDS.extend(videBatch)

        data = []
        for bb in videBatch:
            data.extend(dataLoadedVideoText_one(video2frames, bb, visual_feats['test'], lineList[0], bow2vec, rnn_vocab, tokenizer, options))
        videos, captions = collate_frame_gru_fn(data)

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)
        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs = vid_emb.data.cpu().numpy().copy()

        # calculate cosine distance
        errorlistList.extend(cosine_calculate(video_embs, queryEmbeddings))

        if i % 100000 == 0:
            # print (i)
            end = time.time()
            print(str(i) + ' in: ' + str(end - start))
            start = time.time()

    errorlist = np.asarray(errorlistList)
    f = open(saveFile_AVS16, "w")
    for num, name in enumerate(lineList[:30], start=1):
        queryError = errorlist[:, num - 1]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS16, "a")
        c = 0
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(1000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()

    # AVS17
    f = open(saveFile_AVS17, "w")
    for num, name in enumerate(lineList[30:60], start=31):
        queryError = errorlist[:, num - 1]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS17, "a")
        c = 0
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(1000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()

    # AVS18
    f = open(saveFile_AVS18, "w")
    for num, name in enumerate(lineList[60:90], start=61):
        queryError = errorlist[:, num - 1]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS18, "a")
        c = 0
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(1000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()

    resultAVSFile16 = saveFile_AVS16[:-4] + '_results.txt'
    command = "perl AVS/sample_eval.pl -q AVS/avs.qrels.tv16 {} > {}".format(saveFile_AVS16, resultAVSFile16)
    os.system(command)
    resultAVSFile17 = saveFile_AVS17[:-4] + '_results.txt'
    command = "perl AVS/sample_eval.pl -q AVS/avs.qrels.tv17 {} > {}".format(saveFile_AVS17, resultAVSFile17)
    os.system(command)
    resultAVSFile18 = saveFile_AVS18[:-4] + '_results.txt'
    command = "perl AVS/sample_eval.pl -q AVS/avs.qrels.tv18 {} > {}".format(saveFile_AVS18, resultAVSFile18)
    os.system(command)


def dataLoadedVideoText_one(video2frames, video_id, visual_feats, query, bow2vec, vocab, tokenizer, options):
    data = []

    videos = []

    frame_list = video2frames[video_id]
    frame_vecs = []
    for frame_id in frame_list:
        # visual_feats.read_one(frame_id)
        if options.do_visual_feas_norm:
            frame_vecs.append(do_L2_norm(visual_feats.read_one(frame_id)))
        else:
            frame_vecs.append(visual_feats.read_one(frame_id))
    # Text encoding
    cap_tensors = []
    cap_bows = []

    caption_text = query[:]
    caption_text = ' '.join(clean_str(caption_text))
    caption_text = text2Berttext(caption_text, tokenizer)
    caption_text = caption_text.encode("utf-8")

    if bow2vec is not None:
        cap_bow = bow2vec.mapping(caption_text)
        if cap_bow is None:
            cap_bow = torch.zeros(bow2vec.ndims)
        else:
            cap_bow = torch.Tensor(cap_bow)
    else:
        cap_bow = None

    if vocab is not None:
        tokens = clean_str(caption_text)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        cap_tensor = torch.Tensor(caption)
    else:
        cap_tensor = None

    # BERT
    caption_text = query[:]
    caption_text = ' '.join(clean_str(query))
    marked_text = "[CLS] " + caption_text + " [SEP]"
    # print marked_text
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segments_ids)

    caption_text = caption_text.encode("utf-8")

    data.append([torch.Tensor(frame_vecs), cap_tensor, cap_bow, tokens_tensor, segments_tensors, caption_text])

    return data


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows, tokens_tensor, segments_tensors, caption_text = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    # 'BERT Process'
    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths_bert = [len(seg) for seg in segments_tensors]
        tokens_tensor_padded = torch.zeros(len(tokens_tensor), max(lengths_bert)).long()
        segments_tensors_padded = torch.zeros(len(segments_tensors), max(lengths_bert)).long()
        words_mask_bert = torch.zeros(len(tokens_tensor), max(lengths_bert))

        for i, cap in enumerate(tokens_tensor):
            end = lengths_bert[i]
            tokens_tensor_padded[i, :end] = cap[:end]
            words_mask_bert[i, :end] = 1.0
        for i, cap in enumerate(segments_tensors):
            end = lengths_bert[i]
            segments_tensors_padded[i, :end] = cap[:end]


    else:
        lengths_bert = None
        tokens_tensor_padded = None
        segments_tensors_padded = None
        words_mask_bert = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert)

    return video_data, text_data


if __name__ == '__main__':
    main()
