import re
import pandas as pd 
import numpy as np

data = pd.read_csv("./base_english.csv")

def get_bi_char_words(input_txt, custom_period='%', pad_char_reg_exp='[a-zA-Z0-9]'):
    """
    get_bi_char_words('abcd다0')
    'abcd%다0%'
    """
    def is_pad_character(char):

        if not isinstance(char, str):
            raise ValueError('str type char should be given.')
        elif len(char) != 1:
            raise ValueError('char should be given.')

        else:
            pass

        if re.search(pad_char_reg_exp, char):
            return True
        else:
            return False

    tmp_output_txt = []
    for idx, current_char in enumerate(input_txt):
        if idx == len(input_txt) - 1:
            next_char = None
        else:
            next_char = input_txt[idx + 1]

        if is_pad_character(current_char):  # 현재 문자가 2글자 토큰 대상이고
            if next_char is None or not is_pad_character(next_char):  # 현재 문자가 마지막 문자이거나, 다음문자가 대상이 아니면
                tmp_output_txt.append(current_char + custom_period)  # 현재 문자에 pad_char 붙여서 넣는다.
            else:
                tmp_output_txt.append(current_char + next_char)  # 현재문자와 다음 문자 모두 대상이므로 붙여서 넣는다.
        else:  # 한글 문자 같은 경우 문자 하나만 넣는다.
            tmp_output_txt.append(current_char)

        if next_char is not None:
            tmp_output_txt.append(' ')  # 단어 구분자를 넣는다. 마지막이 아닌때만 넣는다.

    return ''.join(tmp_output_txt)

def stride_w_custom_period(
        string,
        max_len=64,
        filters=r'[\?\.\,\<\>\\\|\=\+\-\_\(\)\{\}\[\]\&\^\#\!\`\~\'\"\:\;\/\%]',
        custom_period='%',
        timing=False):
    """
    stride_w_custom_period('[Abc0다abc')
    returns 'ab bc c0 0% %다'
    """

    if len(custom_period) != 1:
        raise ValueError('custom_period must be a letter, not a word.')
    if custom_period not in filters:
        raise ValueError('custom_period must be in the filters.')

    string = string[:max_len]  # maxlen 만큼만 사용
    string = re.sub(filters, r'', string)  # filter에 대당하는 문자는 ''으로 대체
    string = re.sub(r'(.)\1{2,}', r'\1\1', string)  # ???
    string = re.sub(r' ', '', string)  # 빈칸을 없앤다.
    string = get_bi_char_words(string, custom_period, pad_char_reg_exp='[a-zA-Z0-9]')

    return string

txt = data['txt']
label = data['badword']
txt_len = len(txt)
new_frame = []

for cnt in range(txt_len):
        raw_string = txt[cnt]
        result = stride_w_custom_period(raw_string, timing=False)
        
        new_frame.extend([result])
 
data = pd.DataFrame(new_frame)
data.columns = ['txt']
data = pd.concat([label, data], axis=1)
data.columns = ['badword', 'txt']

data.to_csv("data_processed.csv")

