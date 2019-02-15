import re
import nltk

from urllib import request
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

TICKER = 'Python'
URL_TEMPLATE = 'https://blog.jetbrains.com/?s=%s'

nltk.download('stopwords')


def get_article_urls(ticker):
    """해당 주식과 관련된 기사를 반환한다."""
    link_pattern = re.compile(r'<link>[^<]*</link>')
    xml_url = URL_TEMPLATE % ticker
    xml_data = request.urlopen(xml_url).read().decode('utf-8')
    link_hits = re.findall(link_pattern, xml_data)
    return [h[6:-7] for h in link_hits]


def get_article_content(url):
    """
    :param url: 신문기사 url
    :return: 신문 기사 전처리 결과
    HTML 파일을 내려받은 뒤 각 문단의 내용을 정리한다.
    """
    paragraph_re = re.compile(r'<p>.*</p>')
    tag_re = re.compile(r'<[^>]*>')
    raw_html = request.urlopen(url).read().decode('utf-8')
    paragraphs = re.findall(paragraph_re, raw_html)
    all_text = ''.join(paragraphs)
    content = re.sub(tag_re, '', all_text)
    return content


def text_to_bag(txt):
    """
    :param txt: 문자열
    :return: BoW(Bag-of-words) 특징값
    불용어(stop words)는 무시하고 처리한다.
    """
    lemmatizer = WordNetLemmatizer()
    txt_as_ascii = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt_as_ascii)
    words = [t for t in tokens if t.isalnum()]
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    stop = set(stopwords.words('english'))
    nostops = [l for l in lemmas if l not in stop]
    return nltk.FreqDist(nostops)


def count_good_bad(bag):
    """
    :param bag: BoW 특징값
    :return: 긍정/부정적 단어의 갯수
    """
    good_synsets = set(wn.synsets('good') + wn.synsets('up'))
    bad_synsets = set(wn.synsets('bad') + wn.synsets('down'))
    n_good, n_bad = 0, 0
    for lemma, ct in bag.items():
        ss = wn.synsets(lemma)
        if good_synsets.intersection(ss):
            n_good += 1
        if bad_synsets.intersection(ss):
            n_bad += 1
    return n_good, n_bad


urls = get_article_urls(TICKER)
contents = [get_article_content(u) for u in urls]
bags = [text_to_bag(txt) for txt in contents]
counts = [count_good_bad(txt) for txt in bags]
n_good_articles = len([g for g, b in counts if g > b])  # 긍정적인 기사 갯수
n_bad_articles = len([b for g, b in counts if g < b])  # 부정적인 기사 갯수

print('긍정적인 기사: %i개, 부정적인 기사: %i개' % (n_good_articles, n_bad_articles))
