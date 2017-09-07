# rnnsearch.py

import numpy
import theano
import theano.sandbox.rng_mrg
import theano.tensor as T

import nn
import ops
from bridge import map_key
from encoder import Encoder
from search import beam, select_nbest


class rnnsearch:
    def __init__(self, **option):

        # source and target embedding dim
        sedim, tedim = option["embdim"]
        # source, target and attention hidden dim
        shdim, thdim, ahdim = option["hidden"]
        # maxout hidden dim
        maxdim = option["maxhid"]
        # maxout part
        maxpart = option["maxpart"]
        # deepout hidden dim
        deephid = option["deephid"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        # source and target vocabulary size
        svsize, tvsize = len(sid2w), len(tid2w)

        if "scope" not in option or option["scope"] is None:
            option["scope"] = "rnnsearch"

        if "initializer" not in option:
            option["initializer"] = None

        if "regularizer" not in option:
            option["regularizer"] = None

        if "keep_prob" not in option:
            option["keep_prob"] = 1.0

        dtype = theano.config.floatX
        initializer = option["initializer"]
        regularizer = option["regularizer"]
        keep_prob = option["keep_prob"] or 1.0

        scope = option["scope"]
        decoder_scope = "decoder2"

        encoder = Encoder(sedim, shdim)
        import decoder2
        decoder = decoder2.DecoderGruCond(2, option['method'], tedim, thdim, ahdim, 2 * shdim+thdim, dim_readout=deephid,
                                          n_y_vocab=tvsize)

        # training graph
        with ops.variable_scope(scope, initializer=initializer,
                                regularizer=regularizer, dtype=dtype):
            src_seq = T.imatrix("source_sequence")
            src_mask = T.matrix("source_sequence_mask")
            tgt_seq = T.imatrix("target_sequence")
            tgt_mask = T.matrix("target_sequence_mask")
            byseq = T.imatrix("backward_target_sequence")

            with ops.variable_scope("source_embedding"):
                source_embedding = ops.get_variable("embedding",
                                                    [svsize, sedim])
                source_bias = ops.get_variable("bias", [sedim])

            with ops.variable_scope("target_embedding"):
                target_embedding = ops.get_variable("embedding",
                                                    [tvsize, tedim])
                target_bias = ops.get_variable("bias", [tedim])

            source_inputs = nn.embedding_lookup(source_embedding, src_seq) + source_bias
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq) + target_bias
            by_inputs = nn.embedding_lookup(target_embedding, byseq) + target_bias

            if keep_prob < 1.0:
                source_inputs = nn.dropout(source_inputs, keep_prob=keep_prob)
                target_inputs = nn.dropout(target_inputs, keep_prob=keep_prob)
                by_inputs = nn.dropout(by_inputs, keep_prob=keep_prob)

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            annotation = nn.dropout(annotation, keep_prob=keep_prob)

            import softdec
            soft_decoder = softdec.SoftDecoder(option["eosid"], option["softk"], tedim, thdim, ahdim, 2 * shdim,
                                               dim_readout=deephid, n_y_vocab=tvsize)
            with ops.variable_scope('soft_decoder'):
                initial_state = nn.feedforward(states[-1], [shdim, thdim],
                                               True, scope='initial',
                                               activation=T.tanh)
                mapped_keys = map_key(annotation, 2 * shdim, ahdim)
                soft_states, _, _, soft_mask = soft_decoder.infer(mapped_keys, src_mask,
                                                                  annotation, initial_state,
                                                                  target_embedding, target_bias,
                                                                  keep_prob)

            with ops.variable_scope('soft_decoder', reuse=True):
                _, _, soft_cost, _ = soft_decoder.forward(byseq, by_inputs, tgt_mask, mapped_keys,
                                                          src_mask, annotation, initial_state, keep_prob)

            # compute initial state for decoder
            # first state of backward encoder
            # initialize with only encoder state
            final_state = r_states[0]

            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)
                # keys for query
                with ops.variable_scope('map-key-src'):
                    mapped_keys_src = map_key(annotation, 2 * shdim, ahdim)
                with ops.variable_scope('map-key-soft'):
                    mapped_keys_soft = map_key(soft_states, thdim, ahdim)

                _, _, _, snt_cost = decoder.forward(tgt_seq, target_inputs, tgt_mask,
                                                    [mapped_keys_src, mapped_keys_soft], [src_mask, soft_mask],
                                                    [annotation, soft_states], initial_state, keep_prob)

            ce = snt_cost
            true_cost = T.mean(ce)
            lamb = theano.shared(numpy.asarray(option['lambda'], dtype), 'lambda')
            cost = lamb * soft_cost + (1 - lamb) * true_cost

        # import utils.ttensor
        # print 'true_cost %d:' % len(utils.ttensor.find_inputs_and_params(true_cost)[0])
        # for xxx in utils.ttensor.find_inputs_and_params(true_cost)[0]:
        #     print '\t', xxx
        # print 'soft_cost %d:' % len(utils.ttensor.find_inputs_and_params(soft_cost)[0])
        # for xxx in utils.ttensor.find_inputs_and_params(soft_cost)[0]:
        #     print '\t', xxx
        # print 'tot_cost: %d' % len(utils.ttensor.find_inputs_and_params(cost)[0])
        # for xxx in utils.ttensor.find_inputs_and_params(cost)[0]:
        #     print '\t', xxx
        # print 'snt_cost: %d' % len(utils.ttensor.find_inputs_and_params(snt_cost)[0])
        # for xxx in utils.ttensor.find_inputs_and_params(snt_cost)[0]:
        #     print '\t', xxx

        training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask, byseq]
        training_outputs = [cost, soft_cost, true_cost]

        # get_snt_cost = theano.function(training_inputs[:4], snt_cost)
        get_snt_cost = None

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = T.ivector("prev_words")

            # disable dropout
            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            source_inputs = source_inputs + source_bias
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)
            target_inputs = target_inputs + target_bias

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            with ops.variable_scope('soft_decoder'):
                initial_state = nn.feedforward(states[-1], [shdim, thdim],
                                               True, scope='initial',
                                               activation=T.tanh)
                mapped_keys = map_key(annotation, 2 * shdim, ahdim)
                soft_states, soft_contexts, soft_probs, soft_mask = soft_decoder.infer(mapped_keys, src_mask,
                                                                                       annotation, initial_state,
                                                                                       target_embedding, target_bias,
                                                                                       1.0)

            # decoder
            final_state = r_states[0]
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)
                # keys for query
                with ops.variable_scope('map-key-src'):
                    mapped_keys_src = map_key(annotation, 2 * shdim, ahdim)
                with ops.variable_scope('map-key-soft'):
                    mapped_keys_soft = map_key(soft_states, thdim, ahdim)

            prev_inputs = nn.embedding_lookup(target_embedding, prev_words)
            prev_inputs = prev_inputs + target_bias

            cond = T.neq(prev_words, 0)
            # zeros out embedding if y is 0, which indicates <s>
            prev_inputs = prev_inputs * cond[:, None]

            with ops.variable_scope(decoder_scope):
                mask = T.ones_like(prev_words, dtype=dtype)
                next_state, context = decoder.step(prev_inputs, mask, initial_state,
                                                   *[mapped_keys_src, mapped_keys_soft, annotation, soft_states,
                                                    src_mask, soft_mask])
                probs = decoder.prediction(prev_inputs, next_state, context)

                # encoding
        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [initial_state, annotation, soft_states, mapped_keys_src, mapped_keys_soft, soft_mask]
        encode = theano.function(encoding_inputs, encoding_outputs)

        if option["decoder"] == "GruSimple":
            raise ValueError()
            prediction_inputs = [prev_words, initial_state, annotation,
                                 mapped_keys, src_mask]
            prediction_outputs = [probs, context]
            predict = theano.function(prediction_inputs, prediction_outputs)

            generation_inputs = [prev_words, initial_state, context]
            generation_outputs = next_state
            generate = theano.function(generation_inputs, generation_outputs)

            self.predict = predict
            self.generate = generate
        elif option["decoder"] == "GruCond":
            prediction_inputs = [prev_words, initial_state, annotation,
                                 mapped_keys_src, src_mask, soft_states, mapped_keys_soft, soft_mask]
            prediction_outputs = [probs, next_state]
            predict = theano.function(prediction_inputs, prediction_outputs)
            self.predict = predict


        self.cost = cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        self.align = None
        self.sample = None
        self.encode = encode

        self.get_snt_cost = get_snt_cost
        self.option = option


# TODO: add batched decoding
def beamsearch(models, seq, mask=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None):
    dtype = dtype or theano.config.floatX

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)

    # get vocabulary from the first model
    option = models[0].option
    vocab = option["vocabulary"][1][1]
    eosid = option["eosid"]
    bosid = option["bosid"]

    if maxlen is None:
        maxlen = seq.shape[0] * 3

    if minlen is None:
        minlen = seq.shape[0] / 2

    # encoding source
    if mask is None:
        mask = numpy.ones(seq.shape, dtype)

    outputs = [model.encode(seq, mask) for model in models]

    states = [item[0] for item in outputs]
    annotations0 = [item[1] for item in outputs]
    annotations1 = [item[2] for item in outputs]
    mapped_annots0 = [item[3] for item in outputs]
    mapped_annots1 = [item[4] for item in outputs]
    soft_masks = [item[5] for item in outputs]

    # sys.stderr.write("l-src={}\nl-soft-tgt={}\n".format(numpy.sum(mask), numpy.sum(soft_masks[0])))
    # soft_prob = soft_probs[0]
    # soft_mask = soft_masks[0]
    # soft_y = numpy.argmax(soft_prob, 1)

    initial_beam = beam(beamsize)
    size = beamsize
    # bosid must be 0
    initial_beam.candidates = [[bosid]]
    initial_beam.scores = numpy.zeros([1], dtype)

    hypo_list = []
    beam_list = [initial_beam]
    done_predicate = lambda x: x[-1] == eosid

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidates = prev_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda cand: cand[-1], candidates), "int32")

        # compute context first, then compute word distribution
        batch_mask = numpy.repeat(mask, num, 1)
        batch_annots0 = map(numpy.repeat, annotations0, [num] * num_models,
                            [1] * num_models)
        batch_annots1 = map(numpy.repeat, annotations1, [num] * num_models,
                            [1] * num_models)
        batch_mannots0 = map(numpy.repeat, mapped_annots0, [num] * num_models,
                             [1] * num_models)
        batch_mannots1 = map(numpy.repeat, mapped_annots1, [num] * num_models,
                             [1] * num_models)
        batch_soft_mask = map(numpy.repeat, soft_masks, [num] * num_models, [1] * num_models)

        # predict returns [probs, context, alpha]
        outputs = [model.predict(last_words, state, annot0, mannot0, batch_mask, annot1, mannot1, softmask)
                   for model, state, annot0, annot1, mannot0, mannot1, softmask in
                   zip(models, states, batch_annots0, batch_annots1,
                       batch_mannots0, batch_mannots1, batch_soft_mask)]
        prob_dists = [item[0] for item in outputs]

        # search nbest given word distribution
        if arithmetic:
            logprobs = numpy.log(sum(prob_dists) / num_models)
        else:
            # geometric mean
            logprobs = sum(numpy.log(prob_dists)) / num_models

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf  # make sure eos won't be selected

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob  # make sure eos will be selected

        next_beam = beam(size)
        finished, remain_beam_indices = next_beam.prune(logprobs, done_predicate, prev_beam)

        hypo_list.extend(finished)  # completed translation
        size -= len(finished)

        if size == 0:  # reach k completed translation before maxlen
            break

        # generate next state
        candidates = next_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda t: t[-1], candidates), "int32")

        if option["decoder"] == "GruSimple":
            contexts = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model
            contexts = select_nbest(contexts, remain_beam_indices)

            states = [model.generate(last_words, state, context)
                      for model, state, context in zip(models, states, contexts)]
        elif option["decoder"] == "GruCond":
            states = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model

        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = numpy.array(hypo_list)[numpy.argsort(score_list)]
    score_list = numpy.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output


def batchsample(model, seq, mask, maxlen=None):
    sampler = model.sample

    vocabulary = model.option["vocabulary"]
    eosid = model.option["eosid"]
    vocab = vocabulary[1][1]

    if maxlen is None:
        maxlen = int(len(seq) * 1.5)

    words = sampler(seq, mask, maxlen)
    trans = words.astype("int32")

    samples = []

    for i in range(trans.shape[1]):
        example = trans[:, i]
        # remove eos symbol
        index = -1

        for i in range(len(example)):
            if example[i] == eosid:
                index = i
                break

        if index >= 0:
            example = example[:index]

        example = map(lambda x: vocab[x], example)

        samples.append(example)

    return samples


# used for analysis
def evaluate_model(model, xseq, xmask, yseq, ymask, alignment=None,
                   verbose=False):
    t = yseq.shape[0]
    batch = yseq.shape[1]

    vocab = model.option["vocabulary"][1][1]

    annotation, states, mapped_annot = model.encode(xseq, xmask)

    last_words = numpy.zeros([batch], "int32")
    costs = numpy.zeros([batch], "float32")
    indices = numpy.arange(batch, dtype="int32")

    for i in range(t):
        outputs = model.predict(last_words, states, annotation, mapped_annot,
                                xmask)
        # probs: batch * vocab
        # contexts: batch * hdim
        # alpha: batch * srclen
        probs, contexts, alpha = outputs

        if alignment is not None:
            # alignment tgt * src * batch
            contexts = numpy.sum(alignment[i][:, :, None] * annotation, 0)

        max_prob = probs.argmax(1)
        order = numpy.argsort(-probs)
        label = yseq[i]
        mask = ymask[i]

        if verbose:
            for i, (pred, gold, msk) in enumerate(zip(max_prob, label, mask)):
                if msk and pred != gold:
                    gold_order = None

                    for j in range(len(order[i])):
                        if order[i][j] == gold:
                            gold_order = j
                            break

                    ent = -numpy.sum(probs[i] * numpy.log(probs[i]))
                    pp = probs[i, pred]
                    gp = probs[i, gold]
                    pred = vocab[pred]
                    gold = vocab[gold]
                    print "%d: predication error, %s vs %s" % (i, pred, gold)
                    print "prob: %f vs %f, entropy: %f" % (pp, gp, ent)
                    print "gold is %d-th best" % (gold_order + 1)

        costs -= numpy.log(probs[indices, label]) * mask

        last_words = label
        states = model.generate(last_words, states, contexts)

    return costs

