# encoding=utf-8

"""
nltk: nature language Toolkit
     1) NLTK 分析单词和句子:
        from nltk.tokenize import sent_tokenize, word_tokenize
     2) NLTK 与停止词:
        from nltk.corpus import stopwords
        set(stopwords.words('english'))
     3) NLTK 词干提取: 词干的概念是一种规范化方法。 除涉及时态之外，许多词语的变体都具有相同的含义。
            from nltk.stem import PorterStemmer
            from nltk.tokenize import sent_tokenize, word_tokenize
            ps = PorterStemmer()
            ps.stem(w)
     4)


"""

import re
import nltk
from typing import List

'''
 初步分析为NLP的Token；Javatokenizer为Code Token
'''

class Tokenizer:
    @classmethod
    def camel_case_split(cls, identifier):
        return re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()

    @classmethod
    def tokenize_identifier_raw(cls, token, keep_underscore=True):
        '''
        标识符存在_下划线，对其进行处理，下划线保留，驼峰再切割
        '''
        regex = r'(_+)' if keep_underscore else r'_+'    # split函数：如果有(), 则同时返回()，若没有 咋不返回
        id_tokens = []
        for t in re.split(regex, token):
            if t:
                id_tokens += cls.camel_case_split(t)
        return list(filter(lambda x: len(x) > 0, id_tokens))

    @classmethod
    def tokenize_desc_with_con(cls, desc: str) -> List[str]:
        '''
        自然语言的分割： 1) 空格分割 2) nltk.word_tokenize再分割 3) 驼峰法则再分割
        '''
        def _tokenize_word(word):
            new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', word)
            subwords = nltk.word_tokenize(new_word)
            new_subwords = []
            for w in subwords:
                new_subwords += cls.tokenize_identifier_raw(w, keep_underscore=True)
            return new_subwords

        tokens = []
        for word in desc.split():
            if not word:
                continue
            tokens += " <con> ".join(_tokenize_word(word)).split()
        return tokens

if __name__ =="__main__":
    mytext = "Bonjour M. Adam, comment allez-vous? J'espère que tout va bien. Aujourd'hui est un bon jour."
    mytext="testGetLibrariesDoesDeDuplication ( ) { when ( design . getContentResource ( ) ) . thenReturn ( designContentResource ) ;"
    print(Tokenizer.tokenize_desc_with_con(mytext))
    print(nltk.tokenize.word_tokenize(mytext))