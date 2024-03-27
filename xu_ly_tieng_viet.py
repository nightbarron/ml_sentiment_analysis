import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
import re

def load_lst():
    # Load Emojicon
    file = open('data/files/emojicon.txt', 'r', encoding='utf-8')
    emoji_lst = file.read().split('\n')
    emoji_dict = {}
    for line in emoji_lst:
        key, value = line.split('\t')
        emoji_dict[key] = value
    file.close()

    # Load Teencode
    file = open('data/files/teencode.txt', 'r', encoding='utf-8')
    teencode_lst = file.read().split('\n')
    teencode_dict = {}
    for line in teencode_lst:
        key, value = line.split('\t')
        teencode_dict[key] = value
    file.close()

    # Load Translate English to Vietnamese
    file = open('data/files/english-vnmese.txt', 'r', encoding='utf-8')
    eng_vnmese_lst = file.read().split('\n')
    eng_vnmese_dict = {}
    for line in eng_vnmese_lst:
        key, value = line.split('\t')
        eng_vnmese_dict[key] = value
    file.close()

    # Load Wrong Words
    file = open('data/files/wrong-word.txt', 'r', encoding='utf-8')
    wrong_word_lst = file.read().split('\n')
    file.close()

    # Load stop words
    file = open('data/files/vietnamese-stopwords.txt', 'r', encoding='utf-8')
    stop_word_lst = file.read().split('\n')
    file.close()

    return emoji_dict, teencode_dict, eng_vnmese_dict, wrong_word_lst, stop_word_lst



def process_text(text, emoji_dict, teen_dict, wrong_lst, eng_vnmese_dict):    
    document = text.lower()    
    document = document.replace("’",'')    
    document = regex.sub(r'\.+', ".", document)    
    new_sentence =''    
    for sentence in sent_tokenize(document):        
        # if not(sentence.isascii()):
        # Add space before emoji
        sentence = regex.sub(r'(?<=[^\W\d_])\b', ' ', sentence)

        ###### CONVERT EMOJICON        
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence)) 

        ###### Remove emoji
        sentence = demoji.replace(sentence, '') 

        ###### CONVERT ENGLISH TO VIETNAMESE
        sentence = ' '.join(eng_vnmese_dict[word] if word in eng_vnmese_dict else word for word in sentence.split())

        ###### CONVERT TEENCODE        
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())        
        
        ###### DEL Punctuation & Numbers        
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'        
        sentence = ' '.join(regex.findall(pattern,sentence))        
        # ...        
        ###### DEL wrong words        
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())    
            
        new_sentence = new_sentence+ sentence + '. '    
    document = new_sentence    
    #print(document)    
    ###### DEL excess blank space    
    document = regex.sub(r'\s+', ' ', document).strip()    
    #...    
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()


def normalize_repeated_characters(text):
    # xử lý các từ có ký tự lặp lại nhiều lần
    return re.sub(r'(.+?)\1+', r'\1', text)

# https://viettelgroup.ai/document/part-of-speech-tagging
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        ## POS tag
        lst_word_type = ['N', 'NP', 'A', 'AB' , 'AY' , 'V', 'VB', 'VY', 'R', 'M' ]
        # lst_word_type = [ 'N', 'NP', 'A', 'AB' , 'AY' , 'ABY', 'V', 'VB', 'VY', 'R' , 'M', 'I']
        # print(pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    # Delete excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def process_postag_thesea_adj(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        ## POS tag
        lst_word_type = ['A', 'AB' , 'AY' , 'ABY', 'M' , 'R']
        # lst_word_type = [ 'N', 'NP', 'A', 'AB' , 'AY' , 'ABY', 'V', 'VB', 'VY', 'R' , 'M', 'I']
        # print(pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    # Delete excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopwords(text, stop_word_lst):
    doc= ' '.join('' if word in stop_word_lst else word for word in text.split())
    doc = regex.sub(r'\s+', ' ', doc).strip()
    return doc

def preprocess_text_all_together(text):
    emoji_dict, teen_dict, eng_vnmese_dict, wrong_lst, stop_word_lst = load_lst()
    # print("# Original text: ")
    # print(text)
    text = process_text(text, emoji_dict, teen_dict, wrong_lst, eng_vnmese_dict)
    # print("# After process_text: ")
    # print(text)
    # text = convert_unicode(text)
    # print("# After convert_unicode: ")
    # print(text)
    text = process_special_word(text)
    # print("# After process_special_word: ")
    # print(text)

    text = normalize_repeated_characters(text)
    # print("# After normalize_repeated_characters: ")
    # print(text)

    text = process_postag_thesea(text)
    # print("# After process_postag_thesea: ")
    # print(text)
    text = remove_stopwords(text, stop_word_lst)
    # print("# After remove_stopwords: ")
    # print(text)
    return text


text = "Đã thư rất ngon"
text2 = '''21h30...2 đứa nhỏ kêu đói, sau 1 hồi bình loạn là chốt McDonal!
Đang háo hức vì 2 đứa nó vừa ăn vừa khen lấy khen để, vậy mà nỡ lòng nào đến lượt mình nó lại thành ra thế nàyyyy😭😭😭
Cơm sống toàn tập, ăn cơm mà cứ như nhai gạo... Chán chịu ko nổi😤😤😤'''
text3 = "Đã ăn qua rồi, ăn ok ạ	"
preprocess_text_all_together("Gà chiên còn sống, rất tanh. Khủng khiép	")
