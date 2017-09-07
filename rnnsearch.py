#!/usr/bin/python
import argparse
import cPickle
import datetime
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy
import yaml

import ops
from data import textreader, textiterator
from data.align import convert_align
from data.plain import convert_data, data_length
from model.rnnsearch import rnnsearch, beamsearch, batchsample, evaluate_model
from optimizer import optimizer
from progress.progress import Progress
from utils import validate, misc


def build_model(**options):
    model = rnnsearch
    return model(**options)


def load_vocab(file):
    fd = open(file, "rb")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def count_parameters(variables):
    n = 0

    for item in variables:
        v = item.get_value()
        n += v.size

    return n


def load_word2vec(f):
    word2vec = {}
    with open(f) as r:
        r.next()
        for i, l in enumerate(r):
            sp = l.split()
            word = sp[0]
            word2vec[word] = numpy.asarray(sp[1:], numpy.float32)
    return word2vec


def serialize(name, option, progress=None):
    # in order not to corrupt dumped file
    tfd, tname = tempfile.mkstemp()
    fd = open(tname, "wb")
    params = ops.trainable_variables()
    names = [p.name for p in params]
    vals = dict([(p.name, p.get_value()) for p in params])

    if progress:
        option["#progress"] = progress

    cPickle.dump(option, fd, cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(names, fd, cPickle.HIGHEST_PROTOCOL)
    # compress
    numpy.savez(fd, **vals)

    fd.close()
    os.close(tfd)
    if progress:
        del option["#progress"]
    shutil.move(tname, name)


# load model from file
def load_model(name):
    fd = open(name, "rb")
    option = cPickle.load(fd)
    names = cPickle.load(fd)
    vals = dict(numpy.load(fd))

    params = [(n, vals[n]) for n in names]

    fd.close()

    return option, params


def match_variables(variables, values, ignore_prefix=True):
    var_dict = {}
    val_dict = {}
    matched = []
    not_matched = []

    for var in variables:
        if ignore_prefix:
            name = "/".join(var.name.split("/")[1:])
        var_dict[name] = var

    for (name, val) in values:
        if ignore_prefix:
            name = "/".join(name.split("/")[1:])
        val_dict[name] = val

    # matching
    for name in var_dict:
        var = var_dict[name]

        if name in val_dict:
            val = val_dict[name]
            matched.append([var, val])
        else:
            not_matched.append(var)

    return matched, not_matched


def restore_variables(matched, not_matched):
    for var, val in matched:
        var.set_value(val)

    for var in not_matched:
        sys.stderr.write("%s NOT restored\n" % var.name)


def set_variables(variables, values):
    values = [item[1] for item in values]

    for p, v in zip(variables, values):
        p.set_value(v)


def get_variables_with_prefix(prefix):
    var_list = ops.trainable_variables()
    new_list = []

    for var in var_list:
        if var.name.startswith(prefix):
            new_list.append(var)

    return new_list


# format: source target prob
def load_dictionary(filename):
    fd = open(filename)

    mapping = {}

    for line in fd:
        sword, tword, prob = line.strip().split()
        prob = float(prob)

        if sword in mapping:
            oldword, oldprob = mapping[sword]
            if prob > oldprob:
                mapping[sword] = (tword, prob)
        else:
            mapping[sword] = (tword, prob)

    newmapping = {}
    for item in mapping:
        newmapping[item] = mapping[item][0]

    fd.close()

    return newmapping


def build_sample_space(refs, examples):
    space = {}

    for ref in refs:
        space[ref] = 1

    for example in examples:
        # remove empty
        if len(example) == 0:
            continue

        example = " ".join(example)

        if example in space:
            continue

        space[example] = 1

    return list(space.iterkeys())


def _valid_skip(parser):
    def valid(arg):
        if arg is None:
            return arg
        suffix = None
        if arg.endswith('batch'):
            suffix = 'batch'
        elif arg.endswith('epc'):
            suffix = 'epc'
        elif arg.endswith('epoch'):
            suffix = 'epoch'
        if suffix:
            n = arg[:-len(suffix)]
        else:
            n = arg
        try:
            n = int(n)
        except ValueError as e:
            parser.error(e.message)
        return arg

    return valid


def parseargs_train(argv):
    msg = "training rnnsearch"
    usage = "rnnsearch.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    valid_file = validate.valid_file(parser)

    parser.add_argument("--method", choices=['concat', 'attn'], type=str)
    parser.add_argument("--softk", type=int)
    parser.add_argument("--lambda", type=float)

    data = parser.add_argument_group('data sets; model loading and saving')
    msg = 'source and target corpus'
    data.add_argument('--corpus', nargs=2, help=msg)
    msg = 'source and target vocabulary'
    data.add_argument('--vocab', nargs=2, help=msg)
    msg = 'model name to save or saved model to initialize, required'
    data.add_argument('--model', required=True, help=msg)
    msg = 'save frequency, default 1000'
    data.add_argument('--freq', type=int, help=msg)
    msg = 'reset progress'
    data.add_argument('--reset', action='store_true', help=msg)
    msg = 'do not overwrite the previous model'
    data.add_argument('--no-overwrite', type=int, help=msg)

    network = parser.add_argument_group('network parameters')
    msg = 'decoder version, GruCond for dl4nmt and GruSimple for groundhog'
    network.add_argument('--decoder', type=str, choices=['GruCond', 'GruSimple'], help=msg)
    msg = 'source and target embedding size, default 620'
    network.add_argument('--embdim', nargs=2, type=int, help=msg)
    msg = 'source, target and alignment hidden size, default 1000'
    network.add_argument('--hidden', nargs=3, type=int, help=msg)
    msg = 'maxout hidden dimension, default 500'
    network.add_argument('--maxhid', type=int, help=msg)
    msg = 'maxout number, default 2'
    network.add_argument('--maxpart', type=int, help=msg)
    msg = 'deepout hidden dimension, default 620'
    network.add_argument('--deephid', type=int, help=msg)
    msg = 'dropout keep probability'
    network.add_argument('--keep-prob', type=float, help=msg)

    training = parser.add_argument_group('training')
    msg = 'random seed'
    training.add_argument('--seed', type=int, help=msg)
    msg = 'sort batches'
    training.add_argument('--sort', type=int, help=msg)
    msg = 'shuffle every epcoh'
    training.add_argument('--shuffle', type=int, help=msg)
    msg = 'source and target sentence limit, default 50 (both), 0 to disable'
    training.add_argument('--limit', type=int, nargs='+', help=msg)
    msg = 'L1 regularizer scale'
    training.add_argument('--l1-scale', type=float, help=msg)
    msg = 'L2 regularizer scale'
    training.add_argument('--l2-scale', type=float, help=msg)
    msg = 'maximum training epoch, default 5'
    training.add_argument('--maxepoch', type=int, help=msg)
    msg = 'learning rate, default 5e-4'
    training.add_argument('--alpha', type=float, help=msg)
    msg = 'momentum, default 0.0'
    training.add_argument('--momentum', type=float, help=msg)
    msg = 'batch size, default 128'
    training.add_argument('--batch', type=int, help=msg)
    msg = 'optimizer, default rmsprop'
    training.add_argument('--optimizer', type=str, help=msg)
    msg = 'gradient clipping, default 1.0'
    training.add_argument('--norm', type=float, help=msg)
    msg = 'early stopping iteration, default 0'
    training.add_argument('--stop', type=int, help=msg)
    msg = 'decay factor, default 0.5'
    training.add_argument('--decay', type=float, help=msg)
    msg = 'initialization scale, default 0.08'
    training.add_argument('--scale', type=float, help=msg)

    validation = parser.add_argument_group('validation')
    msg = 'external validation script'
    validation.add_argument('--ext-val-script', type=str, help=msg)
    msg = 'validation dataset'
    validation.add_argument('--validation', type=valid_file, help=msg)
    msg = 'reference data'
    validation.add_argument('--references', type=valid_file, nargs='+', help=msg)
    msg = 'validation frequency, default 1000'
    validation.add_argument('--vfreq', type=int, help=msg)
    msg = 'skip validation phase, e.g, 15/1500batch/1epc'
    validation.add_argument('--skip-val', type=_valid_skip(parser), help=msg)
    msg = 'use delayed validation'
    validation.add_argument('--delay-val', type=int, help=msg)

    display = parser.add_argument_group('display')
    msg = 'sample frequency, default 50'
    display.add_argument('--sfreq', type=int, help=msg)
    msg = 'printing frequency, default 1000'
    display.add_argument('--pfreq', type=int, help=msg)

    misc = parser.add_argument_group('misc')
    msg = 'load args from a .args or .yaml file except those present in command line'
    misc.add_argument('--config', type=str, help=msg)
    msg = 'initialize from another model'
    misc.add_argument('--initialize', type=str, help=msg)
    msg = 'fine tune model'
    misc.add_argument('--finetune', action='store_true', help=msg)

    args = parser.parse_args(argv)
    # load from file
    config = args.config
    if config:
        if args.config.endswith('.yaml'):
            with open(args.config) as r:
                opt = yaml.load(r)
        elif config.endswith('.args'):
            with open(config) as r:
                opt = parser.parse_args(r.read().split())
                opt = opt.__dict__
        else:
            raise ValueError('arg config should be a file path with suffix .yaml or .args')
        # override, args has higher priority
        for k, v in args.__dict__.iteritems():
            if v is None and k in opt:
                args.__dict__[k] = opt[k]

    # dump settings
    # .args for command line args
    modelname = get_filename(os.path.basename(args.model))
    init = not os.path.exists(args.model)
    path = modelname + '.settings.args'
    if init or not os.path.exists(path):
        with open(path, 'w') as w:
            for arg in argv:
                if arg.startswith('-'):
                    w.write('\n')
                w.write(arg + ' ')

    return args


def parseargs_decode(args):
    msg = "translate using exsiting nmt model"
    usage = "rnnsearch.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", nargs="+", required=True, help=msg)
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    msg = "normalize probability by the length of candidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "use arithmetic mean instead of geometric mean"
    parser.add_argument("--arithmetic", action="store_true", help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_sample(args):
    msg = "sample sentence from exsiting nmt model"
    usage = "rnnsearch.py sample [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    msg = "sample batch examples"
    parser.add_argument("--batch", default=1, type=int, help=msg)
    msg = "max sentence length"
    parser.add_argument("--maxlen", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_replace(args):
    msg = "replace unk symbol"
    usage = "rnnsearch.py replace [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained models"
    parser.add_argument("--model", required=True, nargs="+", help=msg)
    msg = "source text and translation file"
    parser.add_argument("--text", required=True, nargs=2, help=msg)
    msg = "dictionary used to replace unk"
    parser.add_argument("--dictionary", type=str, help=msg)
    msg = "replacement heuristic (0: copy, 1: replace, 2: heuristic replace)"
    parser.add_argument("--heuristic", type=int, default=1, help=msg)
    msg = "batch size"
    parser.add_argument("--batch", type=int, default=128, help=msg)
    msg = "use arithmetic mean instead of geometric mean"
    parser.add_argument("--arithmetic", action="store_true", help=msg)

    return parser.parse_args(args)


def parseargs_evaluate(args):
    msg = "evaluate a given model"
    usage = "rnnsearch.py evaluate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    msg = "batch size"
    parser.add_argument("--batch", default=128, type=int, help=msg)
    msg = "source file"
    parser.add_argument("--source", type=str, required=True, help=msg)
    msg = "target file"
    parser.add_argument("--target", type=str, required=True, nargs="+", help=msg)
    msg = "alignment file"
    parser.add_argument("--align", type=str, help=msg)
    msg = "sentence cost only"
    parser.add_argument("--sntcost", action="store_true", help=msg)
    msg = "normalize sentence cost"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "print more informations"
    parser.add_argument("--verbose", action="store_true", help=msg)

    return parser.parse_args(args)


def default_option():
    option = {}

    option['method'] = 'concat'
    option['softk'] = 1
    option['lambda'] = 0.5
    option["ext_val_script"] = ""
    option["skip_val"] = None

    option["decoder"] = "GruCond"

    option["delay_val"] = None
    # training corpus and vocabulary
    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = [620, 620]
    option["hidden"] = [1000, 1000, 1000]
    option["maxpart"] = 2
    option["maxhid"] = 500
    option["deephid"] = 620

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["momentum"] = 0.0
    option["optimizer"] = "rmsprop"
    option["norm"] = 1.0
    option["stop"] = 0
    option["decay"] = 0.5
    option["scale"] = 0.08
    option["l1_scale"] = None
    option["l2_scale"] = None
    option["keep_prob"] = None

    # batch/reader count
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["sfreq"] = 50
    option["seed"] = 1234
    option["validation"] = None
    option["references"] = None

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    # special symbols
    option["unk"] = "UNK"
    option["eos"] = "<eos>"

    return option


def args_to_dict(args):
    return args.__dict__


def override_if_not_none(opt1, opt2, key):
    val1 = opt1[key] if key in opt1 else None
    val2 = opt2[key] if key in opt2 else None
    opt1[key] = val2 if val2 is not None else val1


# override existing options
def override(opt1, opt2):
    # training corpus
    if opt2["corpus"] is None and opt1["corpus"] is None:
        raise ValueError("error: no training corpus specified")

    # vocabulary
    if opt2["vocab"] is None and opt1["vocab"] is None:
        raise ValueError("error: no training vocabulary specified")

    if opt2["limit"] and len(opt2["limit"]) > 2:
        raise ValueError("error: invalid number of --limit argument (<=2)")

    if opt2["limit"] and len(opt2["limit"]) == 1:
        opt2["limit"] = opt2["limit"] * 2

    # vocabulary and model paramters cannot be overrided
    exclusion = ["vocab", "eos", "bosid", "eosid", "vocabulary", "embdim", "hidden", "maxhid", "maxpart", "deephid"]

    for key in opt2:
        if key not in exclusion:
            override_if_not_none(opt1, opt2, key)

    if opt1["vocab"] is None:
        opt1["vocab"] = opt2["vocab"]
        svocab = load_vocab(opt2["vocab"][0])
        tvocab = load_vocab(opt2["vocab"][1])
        isvocab = invert_vocab(svocab)
        itvocab = invert_vocab(tvocab)

        # append a new symbol "<eos>" to vocabulary, it is not necessary
        # because we can reuse "</s>" symbol in vocabulary
        # but here we retain compatibility with GroundHog
        svocab[opt1["eos"]] = len(isvocab)
        tvocab[opt1["eos"]] = len(itvocab)
        isvocab[len(isvocab)] = opt1["eos"]
        itvocab[len(itvocab)] = opt1["eos"]

        # <s> and </s> have the same id 0, used for decoding (target side)
        opt1["bosid"] = 0
        opt1["eosid"] = len(itvocab) - 1

        opt1["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(opt1, opt2, "embdim")
        override_if_not_none(opt1, opt2, "hidden")
        override_if_not_none(opt1, opt2, "maxhid")
        override_if_not_none(opt1, opt2, "maxpart")
        override_if_not_none(opt1, opt2, "deephid")


def print_option(option):
    isvocab = option["vocabulary"][0][1]
    itvocab = option["vocabulary"][1][1]

    print "options"

    print "\n[runtime]"
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "sfreq:", option["sfreq"]
    print "pfreq:", option["pfreq"]
    print "seed:", option["seed"]
    print "overwrite:", not option["no_overwrite"]

    print "\n[corpus]"
    print "corpus:", option["corpus"]
    print "vocab:", option["vocab"]
    print "vocabsize:", [len(isvocab), len(itvocab)]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]

    print "\n[validation]"
    print "validation:", option["validation"]
    print "references:", option["references"]
    print "delay-val:", option["delay_val"]
    print "skip-val:", option["skip_val"]

    print "\n[model]"
    print "decoder:", option["decoder"]
    print "embdim:", option["embdim"]
    print "hidden:", option["hidden"]
    print "maxhid:", option["maxhid"]
    print "maxpart:", option["maxpart"]
    print "deephid:", option["deephid"]
    print "L1-scale:", option["l1_scale"]
    print "L2-scale:", option["l2_scale"]
    print "keep-prob:", option["keep_prob"]

    print "\n[optimizer]"
    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "momentum:", option["momentum"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]
    print "scale:", option["scale"]

    print "\n[decoding]"
    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]

    print "tokens"
    print "unk:", option["unk"]
    print "eos:", option["eos"]

    print


def get_filename(name):
    if name.endswith('.pkl'):
        s = name.rsplit('.', 2)
        return s[0]
    else:
        return name


def evaluate_snt_cost(model, option, src, refs, batch, normalize):
    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs
    unk_sym = option["unk"]
    eos_sym = option["eos"]

    reader = textreader([src] + refs)
    stream = textiterator(reader, [batch, batch])
    get_cost = model.get_snt_cost
    for data in stream:
        xdata, xmask = convert_data(data[0], svocab, unk_sym, eos_sym)
        for i, y in enumerate(data[1:]):
            ydata, ymask = convert_data(y, tvocab, unk_sym, eos_sym)
            snt_cost = get_cost(xdata, xmask, ydata, ymask)
            if normalize:
                # per word cost
                lens = numpy.array([len(item) for item in y])
                snt_cost /= lens
            yield snt_cost


def should_skip_val(skip_val, vfreq, cur_epoch, batch):
    def _parse(skip_val):
        if skip_val is None:
            return 0, None
        suffix = None
        if skip_val.endswith("batch"):
            suffix = "batch"
        elif skip_val.endswith("epc"):
            suffix = "epc"
        elif skip_val.endswith("epoch"):
            suffix = "epoch"
        if suffix:
            n = skip_val[:-len(suffix)]
        else:
            n = skip_val
        n = int(n)

        return n, suffix

    n, suffix = _parse(skip_val)
    if n <= 0:
        return False
    if suffix == "batch":
        return batch <= n
    elif suffix == "epc" or suffix == "epoch":
        return cur_epoch < n
    else:
        return (batch / vfreq) <= n


def train(args):
    option = default_option()

    # predefined model names
    pathname, basename = os.path.split(args.model)
    modelname = get_filename(basename)
    autoname_format = os.path.join(pathname, modelname + ".iter{epoch}-{batch}.pkl")
    bestname = os.path.join(pathname, modelname + ".best.pkl")

    # load models
    if os.path.exists(args.model):
        opt, params = load_model(args.model)
        override(option, opt)
        init = False
    else:
        init = True

    if args.initialize:
        pretrain_params = load_model(args.initialize)
        pretrain_params = pretrain_params[1]
        pretrain = True
    else:
        pretrain = False

    override(option, args_to_dict(args))

    # check external validation script
    ext_val_script = option['ext_val_script']
    if not os.path.exists(ext_val_script):
        raise ValueError('File doesn\'t exist: %s' % ext_val_script)
    elif not os.access(ext_val_script, os.X_OK):
        raise ValueError('File is not executable: %s' % ext_val_script)
    # check references format
    ref_stem = None
    if option['validation'] and option['references']:
        ref_stem = misc.infer_ref_stem([option['validation']], option['references'])
        ref_stem = ref_stem[0]

    # .yaml for ultimate options
    yaml_name = "%s.settings.yaml" % modelname
    if init or not os.path.exists(yaml_name):
        with open(yaml_name, "w") as w:
            _opt = args.__dict__.copy()
            for k, v in _opt.iteritems():
                if k in option:
                    _opt[k] = option[k]
            yaml.dump(_opt, w,
                      default_flow_style=False)
            del _opt

    print_option(option)

    # reader
    batch = option["batch"]
    sortk = option["sort"]
    shuffle = option["shuffle"]
    reader = textreader(option["corpus"], shuffle)
    processor = [data_length, data_length]

    stream = textiterator(reader, [batch, batch * sortk], processor,
                          option["limit"], option["sort"])

    # progress
    # initialize before building model
    progress = Progress(option["delay_val"], stream, option["seed"])

    # create model
    regularizer = []

    if option["l1_scale"]:
        regularizer.append(ops.l1_regularizer(option["l1_scale"]))

    if option["l2_scale"]:
        regularizer.append(ops.l2_regularizer(option["l2_scale"]))

    scale = option["scale"]
    initializer = ops.random_uniform_initializer(-scale, scale)
    regularizer = ops.sum_regularizer(regularizer)

    option["scope"] = "rnnsearch"

    model = build_model(initializer=initializer, regularizer=regularizer,
                        **option)

    variables = None

    if pretrain:
        print "using pretrain"
        _pp1= {}
        for name, val in pretrain_params:
            names = name.split('/')[1:]
            if "embedding" in names[0]:
                _pp1['/'.join(names)] = val
            else:
                _pp1['/'.join(names[1:])] = val
        matched = []
        not_matched = []
        for var in ops.trainable_variables():
            names = var.name.split('/')[1:]
            if "decoder2" in var.name:
                not_matched.append((var.name, var.get_value().size))
                continue

            if "embedding" in names[0]:
                match_name = '/'.join(names)
                var.set_value(_pp1[match_name])
            else:
                match_name = '/'.join(names[1:])
                var.set_value(_pp1[match_name])
            matched.append((var.name, var.get_value().size))
        print "------------------- matched -------------------"
        for name, size in matched:
            print name, size
        print "------------------- not matched -------------------"
        for name, size in not_matched:
            print name, size
        print "------------------- end -------------------\n"


    if not init:
        set_variables(ops.trainable_variables(), params)

    print "parameters: %d\n" % count_parameters(ops.trainable_variables())

    # tuning option
    tune_opt = {}
    tune_opt["algorithm"] = option["optimizer"]
    tune_opt["constraint"] = ("norm", option["norm"])
    tune_opt["norm"] = True
    tune_opt["variables"] = variables

    # create optimizer
    scopes = [".*"]

    trainer = optimizer(model.inputs, model.outputs, model.cost, scopes, **tune_opt)

    # vocabulary and special symbol
    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs
    unk_sym = option["unk"]
    eos_sym = option["eos"]

    alpha = option["alpha"]

    maxepoch = option["maxepoch"]

    # restore right before training to avoid randomness changing when trying to resume progress
    if not args.reset:
        if "#progress" in option:
            print 'Restore progress >>'
            progress = (option["#progress"])
            stream = progress.iterator
            stream.set_processor(processor)
            for ttt in progress.task_manager.tasks:
                ttt.status=4
                ttt.result=0.0
        else:
            print 'New progress >>'
    else:
        print 'Discard progress >>'

    # setup progress
    progress.oldname = args.model
    progress.serializer = serialize

    stream = progress.iterator
    overwrite = not args.no_overwrite

    if progress.task_manager:
        print progress.task_manager

    try:
        while progress.epoch < maxepoch:
            epc = progress.epoch
            for data in stream:
                progress.tic()
                if progress.failed():
                    raise RuntimeError("progress failure")
                xdata, xmask = convert_data(data[0], svocab, unk_sym, eos_sym)
                ydata, ymask = convert_data(data[1], tvocab, unk_sym, eos_sym)
                bydata, _ = convert_data(data[1], tvocab, unk_sym, eos_sym, True)

                t1 = time.time()
                tot_cost, soft_cost, true_cost, norm = trainer.optimize(xdata, xmask, ydata, ymask, bydata)
                trainer.update(alpha=alpha)
                t2 = time.time()

                # per word cost
                w_cost = true_cost * ymask.shape[1] / ymask.sum()

                progress.batch_count += 1
                progress.batch_total += 1
                progress.loss_hist.append(w_cost)

                count = progress.batch_count

                if not args.pfreq or count % args.pfreq == 0:
                    print epc + 1, progress.batch_count, w_cost,tot_cost,soft_cost,true_cost, norm, t2 - t1

                if count % option["vfreq"] == 0 and not should_skip_val(args.skip_val, option["vfreq"], epc,
                                                                        progress.batch_total):
                    if option["validation"] and option["references"]:
                        progress.add_valid(option['scope'], option['validation'], ref_stem, ext_val_script, __file__,
                                           option, modelname, bestname, serialize)

                # save after validation
                progress.toc()
                if count % option["freq"] == 0:
                    progress.save(option, autoname_format, overwrite)

                progress.tic()
                if count % option["sfreq"] == 0:
                    n = len(data[0])
                    ind = numpy.random.randint(0, n)
                    sdata = data[0][ind]
                    tdata = data[1][ind]
                    xdata = xdata[:, ind: ind + 1]
                    xmask = xmask[:, ind: ind + 1]
                    hls = beamsearch(model, xdata, xmask)
                    best, score = hls[0]
                    print "--", sdata
                    print "--", tdata
                    print "--", " ".join(best[:-1])
                progress.toc()
            print "--------------------------------------------------"
            progress.tic()
            if option["validation"] and option["references"]:
                progress.add_valid(option['scope'], option['validation'], ref_stem, ext_val_script, __file__, option,
                                   modelname, bestname, serialize)
            print "--------------------------------------------------"

            progress.toc()
            # early stopping
            if epc + 1 >= option["stop"]:
                alpha = alpha * option["decay"]

            stream.reset()

            progress.epoch += 1
            progress.batch_count = 0
            # update autosave
            option["alpha"] = alpha
            progress.save(option, autoname_format, overwrite)

        stream.close()

        progress.tic()
        print "syncing ..."
        progress.barrier()  # hangup and wait
        progress.toc()

        best_valid = max(progress.valid_hist, key=lambda item: item[1])
        (epc, count), score = best_valid

        print "best bleu {}-{}: {:.4f}".format(epc + 1, count, score)

        if progress.delay_val:
            task_elapse = sum([task.elapse for task in progress.task_manager.tasks])
            print "training finished in {}({})".format(datetime.timedelta(seconds=int(progress.elapse)),
                                                       datetime.timedelta(seconds=int(progress.elapse + task_elapse)))
        else:
            print "training finished in {}".format(datetime.timedelta(seconds=int(progress.elapse)))
        progress.save(option, autoname_format, overwrite)


    except KeyboardInterrupt:
        traceback.print_exc()
        progress.terminate()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        progress.terminate()
        sys.exit(1)


def decode(args):
    num_models = len(args.model)
    models = [None for i in range(num_models)]

    for i in range(num_models):
        option, params = load_model(args.model[i])
        scope = "rnnsearch_%d" % i
        option['scope'] = scope
        model = build_model(**option)
        var_list = get_variables_with_prefix(scope)
        set_variables(var_list, params)
        models[i] = model

    # use the first model
    svocabs, tvocabs = models[0].option["vocabulary"]
    unk_sym = models[0].option["unk"]
    eos_sym = models[0].option["eos"]

    count = 0

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    option = {}
    option["maxlen"] = args.maxlen
    option["minlen"] = args.minlen
    option["beamsize"] = args.beamsize
    option["normalize"] = args.normalize
    option["arithmetic"] = args.arithmetic

    while True:
        line = sys.stdin.readline()

        if line == "":
            break

        data = [line]
        seq, mask = convert_data(data, svocab, unk_sym, eos_sym)
        t1 = time.time()
        tlist = beamsearch(models, seq, **option)
        t2 = time.time()

        if len(tlist) == 0:
            translation = ""
            score = -10000.0
        else:
            best, score = tlist[0]
            translation = " ".join(best[:-1])

        sys.stdout.write(translation)
        sys.stdout.write("\n")

        count = count + 1
        sys.stderr.write(str(count) + " ")
        sys.stderr.write(str(score) + " " + str(t2 - t1) + "\n")


def sample(args):
    option, values = load_model(args.model)
    model = build_model(**option)
    set_variables(ops.trainable_variables(), values)

    svocabs, tvocabs = model.option["vocabulary"]
    unk_symbol = model.option["unk"]
    eos_symbol = model.option["eos"]

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    batch = args.batch

    while True:
        line = sys.stdin.readline()

        if line == "":
            break

        data = [line]
        seq, mask = convert_data(data, svocab, unk_symbol, eos_symbol)
        t1 = time.time()
        seq = numpy.repeat(seq, batch, 1)
        mask = numpy.repeat(mask, batch, 1)
        tlist = batchsample(model, seq, mask, maxlen=args.maxlen)
        t2 = time.time()

        count = count + 1

        if len(tlist) == 0:
            sys.stdout.write("\n")
        else:
            for i in range(min(args.batch, len(tlist))):
                example = tlist[i]
                sys.stdout.write(" ".join(example))
                sys.stdout.write("\n")

        sys.stderr.write(str(count) + " " + str(t2 - t1) + "\n")


# unk replacement
def replace(args):
    num_models = len(args.model)
    models = [None for i in range(num_models)]
    alignments = [None for i in range(num_models)]

    if args.dictionary:
        mapping = load_dictionary(args.dictionary)
        heuristic = args.heuristic
    else:
        if args.heuristic > 0:
            raise ValueError("heuristic > 0, but no dictionary available")
        heuristic = 0

    for i in range(num_models):
        option, params = load_model(args.model[i])
        scope = "rnnsearch_%d" % i
        option["scope"] = scope
        model = build_model(**option)
        var_list = get_variables_with_prefix(scope)
        set_variables(var_list, params)
        models[i] = model

    # use the first model
    svocabs, tvocabs = models[0].option["vocabulary"]
    unk_symbol = models[0].option["unk"]
    eos_symbol = models[0].option["eos"]

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    reader = textreader(args.text, False)
    stream = textiterator(reader, [args.batch, args.batch])

    for data in stream:
        xdata, xmask = convert_data(data[0], svocab, unk_symbol, eos_symbol)
        ydata, ymask = convert_data(data[1], tvocab, unk_symbol, eos_symbol)

        for i in range(num_models):
            # compute attention score
            alignments[i] = models[i].align(xdata, xmask, ydata, ymask)

        # ensemble, alignment: tgt_len * src_len * batch
        if args.arithmetic:
            alignment = sum(alignments) / num_models
        else:
            alignments = map(numpy.log, alignments)
            alignment = numpy.exp(sum(alignments) / num_models)

        # find source word to which each target word was most aligned
        indices = numpy.argmax(alignment, 1)

        # write to output
        for i in range(len(data[1])):
            source_words = data[0][i].strip().split()
            target_words = data[1][i].strip().split()
            translation = []

            for j in range(len(target_words)):
                source_length = len(source_words)
                word = target_words[j]

                # found unk symbol
                if word == unk_symbol:
                    source_index = indices[j, i]

                    if source_index >= source_length:
                        translation.append(word)
                        continue

                    source_word = source_words[source_index]

                    if heuristic and source_word in mapping:
                        if heuristic == 1:
                            translation.append(mapping[source_word])
                        else:
                            # source word begin with lower case letter
                            if source_word.decode("utf-8")[0].islower():
                                translation.append(mapping[source_word])
                            else:
                                translation.append(source_word)
                    else:
                        translation.append(source_word)

                else:
                    translation.append(word)

            sys.stdout.write(" ".join(translation))
            sys.stdout.write("\n")

    stream.close()


def evaluate(args):
    option, params = load_model(args.model)
    model = build_model(**option)
    var_list = ops.trainable_variables()
    set_variables(var_list, params)

    if args.sntcost:
        for costs in evaluate_snt_cost(model, option, args.source, args.target, args.batch, args.normalize):
            for cost in costs:
                sys.stdout.write("{}\n".format(cost))
    else:
        # use the first model
        svocabs, tvocabs = model.option["vocabulary"]
        unk_symbol = model.option["unk"]
        eos_symbol = model.option["eos"]

        svocab, isvocab = svocabs
        tvocab, itvocab = tvocabs
        if args.align:
            inputs = [args.source, args.target[0], args.align]
        else:
            inputs = [args.source, args.target[0]]

        reader = textreader(inputs, False)
        stream = textiterator(reader, [args.batch, args.batch])

        for data in stream:
            xdata, xmask = convert_data(data[0], svocab, unk_symbol, eos_symbol)
            ydata, ymask = convert_data(data[1], tvocab, unk_symbol, eos_symbol)

            if not args.align:
                align = None
            else:
                align = convert_align(data[0], data[1], data[2])

            cost = evaluate_model(model, xdata, xmask, ydata, ymask, align,
                                  verbose=args.verbose)

            for i in range(len(cost)):
                if args.verbose:
                    sys.stdout.write("src: %s\n" % data[0][i])
                    sys.stdout.write("tgt: %s\n" % data[1][i])
                sys.stdout.write("cost: %f\n" % cost[i])

        stream.close()


def helpinfo():
    print "usage:"
    print "\trnnsearch.py <command> [<args>]"
    print "use 'rnnsearch.py train --help' to see training options"
    print "use 'rnnsearch.py translate' --help to see decoding options"
    print "use 'rnnsearch.py sample' --help to see sampling options"
    print "use 'rnnsearch.py replace' --help to see UNK replacement options"
    print "use 'rnnsearch.py evaluate --help' to see evaluation options"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == "train":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_train(sys.argv[2:])
            train(args)
        elif command == "translate":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_decode(sys.argv[2:])
            decode(args)
        elif command == "sample":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_sample(sys.argv[2:])
            sample(args)
        elif command == "replace":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_replace(sys.argv[2:])
            replace(args)
        elif command == "evaluate":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_evaluate(sys.argv[2:])
            evaluate(args)
        else:
            helpinfo()

