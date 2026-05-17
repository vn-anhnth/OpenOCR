import copy
import random
import string
import numpy as np
from openrec.preprocess.smtr_label_encode import SMTRLabelEncode

class LicensePlateSMTRLabelEncode(SMTRLabelEncode):
    """
    Hacked SMTRLabelEncode for License Plates.
    Replaces real characters with format tokens (e.g., 'A' for letters, '0' for digits)
    for the Semantic Guidance Module (SGM/GTC) branch.
    """

    def __call__(self, data):
        text = data['label']
        
        # Generate format string
        format_text = ""
        for char in text:
            if char in string.digits:
                format_text += "0"
            elif char in string.ascii_letters or char == 'Đ':
                format_text += "A"
            else:
                format_text += char
        
        # Encode both
        real_text_encoded = self.encode(text)
        format_text_encoded = self.encode(format_text)
        
        if real_text_encoded is None or format_text_encoded is None:
            return None
        if len(real_text_encoded) > self.max_text_len:
            return None

        data['length'] = np.array(len(real_text_encoded))
        
        # For SGM, we use format_text_encoded
        text_in = [self.dict[self.IN_F]] * (self.substr_len) + format_text_encoded + [
            self.dict[self.IN_B]
        ] * (self.substr_len)

        sub_string_list_pre = []
        next_label_pre = []
        sub_string_list = []
        next_label = []
        for i in range(self.substr_len, len(text_in) - self.substr_len):
            sub_string_list.append(text_in[i - self.substr_len:i])
            next_label.append(text_in[i])

            if self.substr_len - i == 0:
                sub_string_list_pre.append(text_in[-i:])
            else:
                sub_string_list_pre.append(text_in[-i:self.substr_len - i])

            next_label_pre.append(text_in[-(i + 1)])

        sub_string_list.append(
            [self.dict[self.IN_F]] *
            (self.substr_len - len(format_text_encoded[-self.substr_len:])) +
            format_text_encoded[-self.substr_len:])
        next_label.append(self.dict[self.EOS])
        
        sub_string_list_pre.append(
            format_text_encoded[:self.substr_len] + [self.dict[self.IN_B]] *
            (self.substr_len - len(format_text_encoded[:self.substr_len])))
        next_label_pre.append(self.dict[self.EOS])

        # Random perturbation for robustness (using format tokens only)
        # We only use indices of '0' and 'A' for perturbation
        format_indices = [self.dict['0'], self.dict['A']]
        
        for sstr, l in zip(sub_string_list[self.substr_len:],
                           next_label[self.substr_len:]):
            id_shu = np.random.choice(self.rang_subs, 2)
            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.choice(format_indices)
            if sstr1 not in sub_string_list:
                sub_string_list.append(sstr1)
                next_label.append(l)
            sstr[id_shu[1] - 1] = random.choice(format_indices)

        for sstr, l in zip(sub_string_list_pre[self.substr_len:],
                           next_label_pre[self.substr_len:]):
            id_shu = np.random.choice(self.rang_subs, 2)
            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.choice(format_indices)
            if sstr1 not in sub_string_list_pre:
                sub_string_list_pre.append(sstr1)
                next_label_pre.append(l)
            sstr[id_shu[1] - 1] = random.choice(format_indices)

        # Padding and return
        data['length_subs'] = np.array(len(sub_string_list))
        sub_string_list = sub_string_list + [
            [self.dict[self.PAD]] * self.substr_len
        ] * ((self.max_text_len * 2) + 2 - len(sub_string_list))
        next_label = next_label + [self.dict[self.PAD]] * (
            (self.max_text_len * 2) + 2 - len(next_label))
        data['label_subs'] = np.array(sub_string_list)
        data['label_next'] = np.array(next_label)

        data['length_subs_pre'] = np.array(len(sub_string_list_pre))
        sub_string_list_pre = sub_string_list_pre + [
            [self.dict[self.PAD]] * self.substr_len
        ] * ((self.max_text_len * 2) + 2 - len(sub_string_list_pre))
        next_label_pre = next_label_pre + [self.dict[self.PAD]] * (
            (self.max_text_len * 2) + 2 - len(next_label_pre))
        data['label_subs_pre'] = np.array(sub_string_list_pre)
        data['label_next_pre'] = np.array(next_label_pre)

        # Main label (used for CTC) stays REAL
        text = [self.dict[self.BOS]] + real_text_encoded + [self.dict[self.EOS]]
        text = text + [self.dict[self.PAD]
                       ] * (self.max_text_len + 2 - len(text))
        data['label'] = np.array(text)
        
        # Add ctc_label if needed (some configs use it)
        data['ctc_label'] = real_text_encoded
        data['ctc_length'] = np.array(len(real_text_encoded))
        
        return data
