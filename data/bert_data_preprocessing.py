import json
import math
import pickle
from data.bert_tokenization import BasicTokenizer, BertTokenizer, whitespace_tokenize
import collections

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class BertDataPreprocessing(object):
    """Data prepared for multi GPU training.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer(self.args.vocab_file, do_lower_case=self.args.do_lower_case, max_len=512)
        self.eval_examples = None
        self.train_features = self.get_train_features()
        self.eval_features = self.get_eval_features()
        #self.num_train_features = len(self.train_features)

    def get_train_features(self):
        if not self.args.eval_only:
            if not self.args.synthetic_data:
                return self.get_features(is_training=True)

    def get_eval_features(self):
        if self.args.eval:
            if not self.args.synthetic_data:
                return self.get_features(is_training=False)

    def get_features(self, is_training):
        
        examples = self.read_squad_examples(is_training)
        if is_training:
            cached_feature_file = self.args.cached_train_features_file
            mode = 'training'
        else:
            cached_feature_file = self.args.cached_eval_features_file
            self.eval_examples = examples
            mode = 'evaluation'
        if not self.args.renew_cache:
            try:
                with open(cached_feature_file, "rb") as reader:
                    print(f'Read cache-file for {mode} data from {cached_feature_file}...', end='', flush=True)
                    features = pickle.load(reader)
                    print('Done')
            except FileNotFoundError:
                print(f'No cache-file for {mode} data found. Making training features...', end='', flush=True)
                features = self.convert_examples_to_features(
                    examples=examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.args.max_seq_length,
                    doc_stride=self.args.doc_stride,
                    max_query_length=self.args.max_query_length,
                    is_training=is_training)
                print('Done')
                if not self.args.skip_cache:
                    print(f'Build cache-file for {mode} data in {cached_feature_file}...', end='', flush=True)
                    with open(cached_feature_file, "wb") as writer:
                        pickle.dump(features, writer)
                    print('Done')
        else:
            print(f'Skipped reading cache file for {mode} data. Making {mode} features...', end='', flush=True)
            features = self.convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.args.max_seq_length,
                doc_stride=self.args.doc_stride,
                max_query_length=self.args.max_query_length,
                is_training=is_training)
            print('Done')

            if not self.args.skip_cache:
                print(f'Rebuild cache file for {mode} data in {cached_feature_file}...', end='', flush=True)
                with open(cached_feature_file, "wb") as writer:
                    pickle.dump(features, writer)
                print('Done')
        return features


    def read_squad_examples(self, is_training):
        """Read a SQuAD json file into a list of SquadExample."""
        if is_training:
            input_file=self.args.train_data_file
        else:
            input_file=self.args.eval_data_file

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:

                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:
                        if self.args.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
        
        if is_training:
            self.args.num_train_optimization_steps = int(
                len(examples) / self.args.batch_size / self.args.gradient_accumulation_steps) * self.args.num_epochs // self.args.num_gpus
        return examples


    def convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                    doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 0

        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=example.is_impossible))
                unique_id += 1
        return features


    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                            orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs


    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


    def get_answers(self, results):
        predictions = collections.defaultdict(list) #it is possible that one example corresponds to multiple features
        Prediction = collections.namedtuple('Prediction', ['text', 'start_logit', 'end_logit'])

        if self.args.version_2_with_negative:
            null_vals = collections.defaultdict(lambda: (float("inf"),0,0))
        for ex, feat, result in self.match_results(self.eval_examples, self.eval_features, results):
            start_indices = self._get_best_indices(result.start_logits, self.args.n_best_size)
            end_indices = self._get_best_indices(result.end_logits, self.args.n_best_size)
            prelim_predictions = self.get_valid_prelim_predictions(start_indices, end_indices, feat, result)
            prelim_predictions = sorted(
                                prelim_predictions,
                                key=lambda x: (x.start_logit + x.end_logit),
                                reverse=True)
            if self.args.version_2_with_negative:
                score = result.start_logits[0] + result.end_logits[0]
                if score < null_vals[ex.qas_id][0]:
                    null_vals[ex.qas_id] = (score, result.start_logits[0], result.end_logits[0])

            curr_predictions = []
            seen_predictions = []
            for pred in prelim_predictions:
                if len(curr_predictions) == self.args.n_best_size:
                    break
                if pred.start_index > 0:  # this is a non-null prediction TODO: this probably is irrelevant
                    final_text = self.get_answer_text(ex, feat, pred)
                    if final_text in seen_predictions:
                        continue
                else:
                    final_text = ""

                seen_predictions.append(final_text)
                curr_predictions.append(Prediction(final_text, pred.start_logit, pred.end_logit))
            predictions[ex.qas_id] += curr_predictions

        #Add empty prediction
        if self.args.version_2_with_negative:
            for qas_id in predictions.keys():
                predictions[qas_id].append(Prediction('',
                                                    null_vals[ex.qas_id][1],
                                                    null_vals[ex.qas_id][2]))


        nbest_answers = collections.defaultdict(list)
        answers = {}
        for qas_id, preds in predictions.items():
            nbest = sorted(
                    preds,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)[:self.args.n_best_size]

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if not nbest:                                                    
                nbest.append(Prediction(text="empty", start_logit=0.0, end_logit=0.0))

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry and entry.text:
                    best_non_null_entry = entry
            probs = self._compute_softmax(total_scores)
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_answers[qas_id].append(output)
            if self.args.version_2_with_negative:
                score_diff = null_vals[qas_id][0] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                if score_diff > self.args.null_score_diff_threshold:
                    answers[qas_id] = ""
                else:
                    answers[qas_id] = best_non_null_entry.text
            else:
                answers[qas_id] = nbest_answers[qas_id][0]['text']

        return answers, nbest_answers

    def get_answer_text(self, example, feature, pred):
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = self.get_final_text(tok_text, orig_text, self.args.do_lower_case)
        return final_text

    def get_valid_prelim_predictions(self, start_indices, end_indices, feature, result):
        
        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = []
        for start_index in start_indices:
            for end_index in end_indices:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > self.args.max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))
        return prelim_predictions

    def match_results(self, examples, features, results):
        unique_f_ids = set([f.unique_id for f in features])
        unique_r_ids = set([r.unique_id for r in results])
        matching_ids = unique_f_ids & unique_r_ids
        features = [f for f in features if f.unique_id in matching_ids]
        results = [r for r in results if r.unique_id in matching_ids]
        features.sort(key=lambda x: x.unique_id)
        results.sort(key=lambda x: x.unique_id)

        for f, r in zip(features, results): #original code assumes strict ordering of examples. TODO: rewrite this
            yield examples[f.example_index], f, r

    def get_final_text(self, pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info(
                    "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text


    def _get_best_indices(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indices = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indices.append(index_and_score[i][0])
        return best_indices

   