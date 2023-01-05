import paddle.utils
from paddlenlp import Taskflow

# paddle.utils.run_check()
summarizer = Taskflow("text_summarization")


def get_abstract(article: str):
    print(f'Content: {article}\n')
    title = summarizer(article)
    print(f'Title: {title[0]}')
    return title[0]

#
# text = '63岁退休教师谢淑华，拉着人力板车，历时1年，走了2万4千里路，带着年过九旬的妈妈环游中国，完成了妈妈“一辈子在锅台边转，也想出去走走”的心愿。她说：“妈妈愿意出去走走，我就愿意拉着，孝心不能等，能走多远就走多远。'
# print(f'Content: {text}\n')
# title = summarizer(text)
# print(f'Title: {title[0]}')
#
# text = '本届赛事在媒体运行方面，采取很多信息化、智能化、数字化的措施，比如电子化媒体取票台、虚拟现场室、主媒体中心统一化的比赛日前新闻发布会等，理论上，记者在这里就可以掌握全部8个赛场的动态，感受新闻报道的更多便利。'
# print(f'Content: {text}\n')
# title = summarizer(text)
# print(f'Title: {title[0]}')
#
#
# text = '二十条措施发布后，民众关心的“层层加码”问题是否得到进一步整治？沈洪兵指出，从国家卫生健康委员会官方网站“落实疫情防控九不准公众留言板”信息数据来看，民众投诉量明显下降，由11月11日的3306条降至16日的2014条，投诉量降幅达到39%。'
# print(f'Content: {text}\n')
# title = summarizer(text)
# print(f'Title: {title[0]}')
#
# text = '17日晚些时候，波兰总统府国务秘书、国际政策局局长库莫赫称，乌方将获准进入爆炸现场，“但这与获准参与调查不同，后者需要单独的程序”。'
# print(f'Content: {text}\n')
# title = summarizer(text)
# print(f'Title: {title[0]}')
