import theano
import theano.tensor as T

import decoder
import nn
import ops


def max_k(x, k):
    assert k > 1
    idx = T.argsort(x, axis=-1)
    idx = idx[..., ::-1]  # max2min
    idx = idx[..., :k]
    tmp_idx = T.arange(x.shape[0])

    r_val = x[tmp_idx[:, None], idx]
    return r_val, idx


class SoftDecoder(decoder.DecoderGruCond):
    def __init__(self, eosid, softk, dim_y, dim_hid, dim_key, dim_value, dim_readout, n_y_vocab, *args, **kwargs):
        decoder.DecoderGruCond.__init__(self, dim_y, dim_hid, dim_key, dim_value, dim_readout, n_y_vocab, *args,
                                        **kwargs)
        self.eosid = eosid
        self.softk = softk

    def _infer_step(self, y_prev, mask, state, keys, values, key_mask, embedding, embedding_bias, keep_prob):
        softk = self.softk
        next_state, context = self.step(y_prev, mask, state, keys, values, key_mask)
        probs = self.prediction(y_prev[None, :, :], next_state[None, :, :], context[None, :, :], keep_prob)
        pred_idx = T.argmax(probs, 1)  # (batch,)
        iseos = T.eq(pred_idx, self.eosid)
        curr_mask = 1.0 - iseos
        mask = mask[:, 0]
        curr_mask = mask * curr_mask
        # greedy search
        if softk == 1:  # greedy
            next_inputs = nn.embedding_lookup(embedding, pred_idx) + embedding_bias  # (batch, emb_dim)
        elif softk == 0:  # soft over all
            next_inputs = T.dot(probs, embedding) + embedding_bias
        elif softk > 1:  # soft over k-best
            # (batch, k), (batch, k)
            max_k_probs, max_k_idx = max_k(probs, softk)
            # (batch, k, dim)
            max_k_probs = max_k_probs / T.sum(max_k_probs, 1)[:, None]
            max_k_embeddings = nn.embedding_lookup(embedding, max_k_idx) + embedding_bias
            next_inputs = T.sum(max_k_probs[:, :, None] * max_k_embeddings, 1)
        else:
            raise RuntimeError()
        curr_mask = curr_mask[:, None]
        return (next_inputs, curr_mask, next_state, context, probs), theano.scan_module.until(
            T.eq(T.sum(curr_mask), 0.0))

    def infer(self, keys, key_mask, values, initial_state, target_embedding, target_bias, keep_prob):
        def infer_step(y_prev, mask, state, keys, values, key_mask, embedding, embedding_bias):
            return self._infer_step(y_prev, mask, state, keys, values, key_mask, embedding, embedding_bias, keep_prob)

        n_steps, batch_size = key_mask.shape
        seq = None
        initial_inputs = T.zeros((batch_size, target_embedding.shape[1]), "float32")
        initial_mask = T.ones((batch_size, 1), "float32")
        outputs_info = [initial_inputs, initial_mask, initial_state, None, None]
        non_seq = [keys, values, key_mask, target_embedding, target_bias]


        # max length is len_src*3
        inputs, mask, states, contexts, probs = ops.scan(infer_step, seq, outputs_info, non_seq,
                                                           n_steps=n_steps * 2)
        mask = T.reshape(mask, mask.shape[:-1])
        mask = T.roll(mask, 1, 0)
        mask = T.set_subtensor(mask[0, :], initial_mask[:, 0])
        # (step, batch, n_voc)->(step*batch, n_voc)
        probs = T.reshape(probs, (probs.shape[0] * probs.shape[1], probs.shape[2]))
        return states, contexts, probs, mask

