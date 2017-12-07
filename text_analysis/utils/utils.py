# -*- coding: utf-8 -*-
import emoji 
import re
import string 
import keras.preprocessing.text as Text

def add_age_category_to_df(df):
    df['age_category'] = 'X'
    #df.loc[df['age'] <= 17, 'age_category'] = 'A'
    df.loc[(df['age'] > 17) & (df['age'] <= 24), 'age_category'] = 'A'
    df.loc[(df['age'] > 24) & (df['age'] <= 30), 'age_category'] = 'B'
    df.loc[(df['age'] > 30) & (df['age'] <= 39), 'age_category'] = 'C'
    df.loc[(df['age'] > 40), 'age_category'] = 'D'
    return df


def process(text):
    for e in emoji.UNICODE_EMOJI:
        if e in text:
            text = text.replace(e, ' emoji_icon ')
    list_icons = [':o', ':/', ':(', ':))', '=))', '>:o', ':v', '(:', ':)', '>.<', '-_-', ':-)', ':P', '>:(', ':D']
    list_punc = ['\\', '!', '#', '$', '%', '&', '*', '+', '-', '.', '^', '_', '`', '|', '~', ':', '\"', '\'', '?', ',']
    for i in range(len(list_icons)):         
        text = text.replace(list_icons[i], ' emoji_icon ')
    for p in list_punc:
        text = text.replace(p, '')
    text = re.sub("(\(\+?0?84\))?(09|012|016|018|019)((\d(\s|\.|\,)*){8})", "", text)

    #text.translate(None, string.punctuation)

    #string = re.sub(r'[()\[\]{}.,;:!?\<=>?@^_`~#$%"&*-]', ' ', string)
    #print string.strip().lower()
    return text.strip().lower()



if __name__ == '__main__':
    print icons(u':) Ha ha vui quá đi thôi :D')