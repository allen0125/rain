import time

from brain import answer, auto_prompt
from config import TOOTBOT_ACCESS_TOKEN, TOOTBOT_API_BASE_URL
from lxml import etree
from mastodon import Mastodon

m = Mastodon(
    access_token=TOOTBOT_ACCESS_TOKEN,
    api_base_url=TOOTBOT_API_BASE_URL,
)


def get_content_text(content: str) -> str:
    html = etree.HTML(content)
    return html.xpath("//p/text()")[0].strip()


while True:
    for i in m.notifications():
        if i.type == "mention":
            print(i.status.in_reply_to_id)
            query = get_content_text(i.status.content)

            if i.status.in_reply_to_id:
                status = m.status(i.status.in_reply_to_id)
                content = get_content_text(status.content)
                prompt = auto_prompt(query=query, origin_text=content)
                result = answer(auto_prompt(query=query, origin_text=content))
                print(result)
                m.status_reply(to_status=i.status, status=result)

    m.notifications_clear()
    print("Sleeping for 10 seconds")
    time.sleep(10)
