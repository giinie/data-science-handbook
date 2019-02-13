import urllib.request
from html.parser import HTMLParser

TOPIC = "Dangiwa_Umar"
url = "https://en.wikipedia.org/wiki/%s" % TOPIC


class LinkCountingParser(HTMLParser):
    def __init__(self):
        """HTMLParser를 상속한 클래스를 정의한다."""
        self.in_paragraph = False
        self.link_count = 0
        HTMLParser.__init__(self)

    def handle_starttag(self, tag, attrs):
        """시작 태그를 처리하는 방법을 정의한다."""
        if tag == 'p':
            self.in_paragraph = True
        elif tag == 'a' and self.in_paragraph:
            self.link_count += 1

    def handle_endtag(self, tag):
        """종료 태그를 처리하는 방법을 정의한다."""
        if tag == 'p':
            self.in_paragraph = False


html = urllib.request.urlopen(url).read()
html = html.decode('utf-8')
parser = LinkCountingParser()
parser.feed(html)
print('이 문서에는 %d개의 링크가 존재합니다.' % parser.link_count)
