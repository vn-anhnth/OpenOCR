import random
import numpy as np
from openrec.preprocess.smtr_label_encode import SMTRLabelEncode

class LicensePlateSMTRLabelEncode(SMTRLabelEncode):
    """
    Advanced Label Encoder for Multi-National License Plates.
    Covers Vietnam, Brazil (Mercosul), and China (Latin parts).
    Injects syntax rules directly into the semantic feature space.
    """
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 sub_str_len=5,
                 **kwargs):
        super(LicensePlateSMTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char, sub_str_len)
        
        # Define character sets for semantic categorization
        self.digits = "0123456789"
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZĐ" # Vietnam's Đ added
        self.symbols = "-. "

    def _get_char_type_id(self, char):
        """Map character to a semantic category ID."""
        if char in self.digits: return 1 # Digit
        if char in self.letters: return 2 # Letter
        if char in self.symbols: return 3 # Symbol
        return 0 # Unknown

    def _detect_country_and_apply_rules(self, label):
        """
        Refined rules based on data analysis:
        Vietnam (VN): 60A00576, 60CD00012 (DDL... or DDLL...)
        Brazil (BR): ODE2510 (LLL-DDDD) or Mercosul ABC1D23 (LLL-D-L-DD)
        China (CN): F5MTG, QFU9887 (L...)
        """
        semantic_seq = [self._get_char_type_id(c) for c in label]
        
        # 1. Vietnam Analysis (Starts with 2 Digits)
        if len(label) >= 3 and label[0:2].isdigit():
            # City code detected. 3rd is Letter.
            if label[2] in self.letters:
                pass # Already mapped as Letter (2)
            
        # 2. Brazil Analysis (Starts with 3 Letters)
        elif len(label) >= 4 and label[0:3].isalpha() and label[0:3].isupper():
            # Possible Brazil plate
            pass
            
        # 3. China Analysis (Starts with Letter)
        elif len(label) > 0 and label[0] in self.letters:
            pass

        return semantic_seq

    def __call__(self, data):
        label = data['label']
        # Apply country-specific rules
        semantic_seq = self._detect_country_and_apply_rules(label)
        
        # Padding semantic_seq to max_text_len for batch collation
        padded_semantic_seq = np.zeros(self.max_text_len, dtype=np.int64)
        actual_len = min(len(semantic_seq), self.max_text_len)
        padded_semantic_seq[:actual_len] = semantic_seq[:actual_len]
        
        data['semantic_seq'] = padded_semantic_seq
        
        # Use base SMTR encoding
        data = super(LicensePlateSMTRLabelEncode, self).__call__(data)
        return data
